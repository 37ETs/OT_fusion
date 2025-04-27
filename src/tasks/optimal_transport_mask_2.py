import logging
from typing import Iterator, Dict, Optional
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.optim import Adam
from geomloss import SamplesLoss

# 简单定义
StateDict = Dict[str, Tensor]

log = logging.getLogger(__name__)

class MultiModelOptimalTransportMask(nn.Module):
    def __init__(
        self,
        state_dict_0: StateDict,
        state_dict_1: StateDict,
        reg: float = 0.01,
        device: Optional[torch.device] = 2,
        lr: float = 1e-2,
        num_epochs: int = 100,
        blur: float = 0.05,
        lmd: float = 0.01,
        binary_mask: bool = False
    ):
        """
        :param state_dict_0: 第一个任务向量
        :param state_dict_1: 第二个任务向量
        :param reg: 预留正则化系数（如需额外使用，可扩展）
        :param device: 指定训练设备，默认自动使用 cuda(可用) 否则 cpu
        :param lr: 优化器学习率
        :param num_epochs: 训练迭代次数
        :param blur: Sinkhorn 距离中的模糊参数
        :param lmd: L1 稀疏约束系数
        :param binary_mask: 是否在训练完成后对 mask 进行硬二值化
        """
        super().__init__()
        self.reg = reg
        self.num_epochs = num_epochs
        self.lr = lr
        self.blur = blur
        self.lmd = lmd
        self.binary_mask = binary_mask

        # 如果用户未指定 device，就自动检测
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 将输入参数都移动到同一 device
        self.state_dict_0 = {k: v.to(self.device) for k, v in state_dict_0.items()}
        self.state_dict_1 = {k: v.to(self.device) for k, v in state_dict_1.items()}

        # 为两份 state_dict 初始化多掩码：common、task0、task1
        self.common_masks, self.task0_masks, self.task1_masks = \
            self._initialize_multi_masks(self.state_dict_0)

        # 一次性训练：同时优化多掩码，让 p_fused 接近 p0 和 p1
        self._train_ot_masks()

        # 在训练结束后，如果需要，就把最终掩码做硬二值化
        if self.binary_mask:
            self._binarize_all_masks()

    def _initialize_multi_masks(self, state_dict: StateDict):
        """
        依次为每个参数 key 创建三套掩码：
          - common_masks[key]
          - task0_masks[key]
          - task1_masks[key]
        都是可以学习的 nn.Parameter。
        """
        common_masks = {}
        task0_masks = {}
        task1_masks = {}

        for k, v in state_dict.items():
            shape = v.shape
            common_masks[k] = nn.Parameter(torch.rand_like(v, device=self.device))
            task0_masks[k] = nn.Parameter(torch.rand_like(v, device=self.device))
            task1_masks[k] = nn.Parameter(torch.rand_like(v, device=self.device))

        return common_masks, task0_masks, task1_masks

    def _train_ot_masks(self, sample_size: int = 256):
        sinkhorn_loss_fn = SamplesLoss(loss="sinkhorn", blur=self.blur).to(self.device)

        all_params = list(self.common_masks.values()) \
                + list(self.task0_masks.values()) \
                + list(self.task1_masks.values())

        optimizer = Adam(all_params, lr=self.lr)

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            optimizer.zero_grad()

            for k in self.state_dict_0.keys():
                p0 = self.state_dict_0[k].view(-1)
                p1 = self.state_dict_1[k].view(-1)

                c = torch.sigmoid(self.common_masks[k]).view(-1)
                t0 = torch.sigmoid(self.task0_masks[k]).view(-1)
                t1 = torch.sigmoid(self.task1_masks[k]).view(-1)

                p_fused = c * 0.5 * (p0 + p1) + t0 * p0 + t1 * p1

                # 随机抽样一小部分元素
                num_elements = p0.shape[0]
                if num_elements > sample_size:
                    indices = torch.randperm(num_elements, device=self.device)[:sample_size]
                else:
                    indices = torch.arange(num_elements, device=self.device)

                fused_sample = p_fused[indices].unsqueeze(1)
                p0_sample = p0[indices].unsqueeze(1)
                p1_sample = p1[indices].unsqueeze(1)

                # 小规模OT距离
                dist_0 = sinkhorn_loss_fn(fused_sample, p0_sample)
                dist_1 = sinkhorn_loss_fn(fused_sample, p1_sample)

                loss_sparse = (c.abs().mean() + t0.abs().sum() + t1.abs().sum())

                param_loss = dist_0 + dist_1 + self.lmd * loss_sparse
                total_loss += param_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 显式释放显存
            torch.cuda.empty_cache()

            if epoch % 5 == 0:
                log.info(
                    f"[OT Mask Training] Epoch {epoch}/{self.num_epochs}, "
                    f"total_loss={total_loss.item():.6f}"
                )


    def _binarize_all_masks(self, threshold: float = 0.5):
        """
        将所有掩码做硬二值化 (common/task0/task1)，
        可用于在融合之后进行更明确的二值划分。
        """
        for k in self.common_masks.keys():
            self.common_masks[k].data = (torch.sigmoid(self.common_masks[k].data) > threshold).float()
            self.task0_masks[k].data = (torch.sigmoid(self.task0_masks[k].data) > threshold).float()
            self.task1_masks[k].data = (torch.sigmoid(self.task1_masks[k].data) > threshold).float()

    def fusion(self, state_dict_0: StateDict, state_dict_1: StateDict) -> StateDict:
        """
        在推理/推断阶段，可调用 fusion 得到最终的融合参数。
        注意：此处可与训练时的 state_dict_0/1 相同或不同，
        如果输入不同，则需要确保维度匹配。
        """
        fused_state_dict = {}
        for k in state_dict_0.keys():
            p0 = state_dict_0[k].to(self.device)
            p1 = state_dict_1[k].to(self.device)

            c = torch.sigmoid(self.common_masks[k])
            t0 = torch.sigmoid(self.task0_masks[k])
            t1 = torch.sigmoid(self.task1_masks[k])

            # 最终融合公式
            fused_param = c * 0.5 * (p0 + p1) + t0 * p0 + t1 * p1
            fused_state_dict[k] = fused_param.detach().cpu()  # 根据需求，也可不 .cpu()

        return fused_state_dict

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        返回本模块所有可学习参数(三类 mask)，以便外部可以访问或再次进行优化。
        """
        return list(self.common_masks.values()) \
             + list(self.task0_masks.values()) \
             + list(self.task1_masks.values())

    def to(self, device):
        """
        将本模块的所有掩码移动到指定 device。
        """
        super().to(device)
        for dct in (self.common_masks, self.task0_masks, self.task1_masks):
            for k, param in dct.items():
                dct[k].data = dct[k].data.to(device)
        return self

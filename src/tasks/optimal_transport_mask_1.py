import logging
from typing import Iterator, List, Optional, final
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.optim import Adam
from geomloss import SamplesLoss

# 简单定义 StateDict 为 dict 类型
StateDict = dict

log = logging.getLogger(__name__)

class MultiModelOptimalTransportMask(nn.Module):
    def __init__(
        self,
        state_dict_0: StateDict,
        state_dict_1: StateDict,
        reg: float = 0.01,
        n_iter: int = 10,
        device: Optional[torch.device] = 1,
        lr: float = 1e-2,
        num_epochs: int = 100,
        blur: float = 0.05,
        lmd: float = 0.01,
        binary_mask: bool = False
    ):
        """
        :param state_dict_0: 第一个任务向量
        :param state_dict_1: 第二个任务向量
        :param reg: 预留的正则化系数（若需额外使用，可自行扩展）
        :param n_iter: 预留的迭代次数（如果与 num_epochs 不同，可灵活拆分逻辑）
        :param device: 指定训练设备，若不传入则自动使用 cuda (如果可用) 否则 cpu
        :param lr: 优化器学习率
        :param num_epochs: 训练迭代次数
        :param blur: Sinkhorn 距离中的模糊参数
        :param lmd: L1 稀疏约束系数
        :param binary_mask: 是否在训练完成后对 mask 进行硬二值化
        """
        super().__init__()
        self.reg = reg
        self.n_iter = n_iter
        self.num_epochs = num_epochs
        self.lr = lr
        self.blur = blur
        self.lmd = lmd
        self.binary_mask = binary_mask

        # 如果调用者没有显式指定 device，就自动检测 cuda
        self.device = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # 初始化所有参数对应的随机 mask
        self.masks = self._initialize_masks(state_dict_0)

        # 将初始 state_dict_0, state_dict_1 记录下来
        # 这里仅保留对后续可能要做其他操作（如 fusion 等）
        self.state_dict_0 = {
            k: v.to(self.device) for k, v in state_dict_0.items()
        }
        self.state_dict_1 = {
            k: v.to(self.device) for k, v in state_dict_1.items()
        }

        # 在一次大的循环里同时优化所有 mask
        self.masks = self._find_ot_mask(self.state_dict_0, self.state_dict_1)

    def _initialize_masks(self, state_dict: StateDict) -> StateDict:
        """
        为每个参数创建一个可学习的掩码，初始随机值 (在 [-1, 1] or [0,1] 都可以)。
        """
        masks = {}
        for k, v in state_dict.items():
            # 用与原张量形状相同的随机值初始化
            # 也可以是 torch.zeros_like(v) 等方式
            rand_init = torch.rand_like(v, device=self.device)
            masks[k] = nn.Parameter(rand_init)
        return masks

    def _find_ot_mask(
        self,
        state_dict_0: StateDict,
        state_dict_1: StateDict
    ) -> StateDict:
        """
        在一次大的循环里对所有参数的 mask 进行联合优化。
        使用 Sinkhorn 距离 + L1 稀疏项 作为损失。
        """
        # 定义 sinkhorn loss
        sinkhorn_loss_fn = SamplesLoss(loss="sinkhorn", blur=self.blur).to(self.device)

        # 优化所有 mask
        optimizer = Adam(self.masks.values(), lr=self.lr)

        for epoch in range(self.num_epochs):
            total_loss = 0.0

            # 逐参数累加损失
            for k in state_dict_0.keys():
                # 这里 mask 取 sigmoid，限制其输出在 (0,1)
                mask = torch.sigmoid(self.masks[k])

                # masked_tensor_X = 原参数 * mask
                masked_tensor_0 = state_dict_0[k] * mask
                masked_tensor_1 = state_dict_1[k] * mask

                # 计算 sinkhorn loss
                ot_loss = sinkhorn_loss_fn(
                    masked_tensor_0.view(-1, 1),
                    masked_tensor_1.view(-1, 1)
                )

                # L1 稀疏约束：鼓励 mask 更加稀疏
                loss_sparse = torch.norm(mask, p=1)

                # 累加到 total_loss
                total_loss = total_loss + ot_loss + self.lmd * loss_sparse

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"[OT Mask Training] Epoch {epoch}/{self.num_epochs}, "
                      f"total_loss = {total_loss.item():.6f}")

        # 训练完成后，再给出最终的掩码
        final_ot_masks = {}
        for k in state_dict_0.keys():
            final_mask = torch.sigmoid(self.masks[k])
            if self.binary_mask:
                # 如果需要硬二值化，就做简单阈值处理
                final_mask = (final_mask > 0.5).float()
            final_ot_masks[k] = final_mask
        
        final_ot_masks = {k: v.detach() for k, v in final_ot_masks.items()}
        
        return final_ot_masks

    def fusion(self, state_dict_0: StateDict, state_dict_1: StateDict) -> StateDict:
        """
        融合函数：将两份 state_dict 用学到的 mask 进行融合。
        这里的写法是 (A + B)/2 * mask，可根据实际需求进行调整。
        """
        merged_state_dict = {}
        for key, mask_param in self.masks.items():
            # 取最终训练得到的 sigmoid mask
            mask = torch.sigmoid(mask_param)
            merged_state_dict[key] = mask * (state_dict_0[key] + state_dict_1[key]) / 2
        return merged_state_dict

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        返回本模块所有可学习参数，这里就是所有 mask。
        以便外部如果需要，也可以调用优化器对其进行更新。
        """
        return self.masks.values()

    def to(self, device):
        """
        将本模块的所有 mask 移动到指定 device。
        """
        super().to(device)
        for k, mask in self.masks.items():
            self.masks[k].data = self.masks[k].data.to(device)
        return self

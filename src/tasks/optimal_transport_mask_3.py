import logging
from typing import Iterator, Dict, Optional
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.optim import Adam
from geomloss import SamplesLoss

# 类型定义
StateDict = Dict[str, Tensor]
log = logging.getLogger(__name__)

class SingleMaskOptimalTransport(nn.Module):
    def __init__(
        self,
        state_dict_0: StateDict,
        state_dict_1: StateDict,
        reg: float = 0.01,
        device: Optional[torch.device] = None,
        lr: float = 1e-2,
        num_epochs: int = 5,
        blur: float = 0.05,
        entropy_coeff: float = 0.1,
        binary_mask: bool = False,
        task_specific_model: int = 0  # 0或1，指定任务特定模型
    ):
        super().__init__()
        self.reg = reg
        self.num_epochs = num_epochs
        self.lr = lr
        self.blur = blur
        self.entropy_coeff = entropy_coeff
        self.binary_mask = binary_mask
        self.task_specific_model = task_specific_model

        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        
        # 将输入参数移动到设备
        self.state_dict_0 = {k: v.to(self.device) for k, v in state_dict_0.items()}
        self.state_dict_1 = {k: v.to(self.device) for k, v in state_dict_1.items()}

        # 初始化单组logits
        self.mask_logits = self._initialize_single_logits(self.state_dict_0)
        
        # 训练OT掩码
        self._train_ot_masks()

        # 二值化掩码
        if self.binary_mask:
            self._binarize_masks()

    def _initialize_single_logits(self, state_dict: StateDict):
        """为每个参数初始化单组logits"""
        mask_logits = {}
        for k, v in state_dict.items():
            mask_logits[k] = Parameter(torch.randn_like(v, device=self.device))
        return mask_logits

    def _train_ot_masks(self, sample_size: int = 64):
        sinkhorn_loss_fn = SamplesLoss(loss="sinkhorn", blur=self.blur).to(self.device)
        optimizer = Adam(self.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            total_loss, total_entropy = 0.0, 0.0
            optimizer.zero_grad()

            for k in self.state_dict_0.keys():
                # 生成sigmoid掩码
                m = torch.sigmoid(self.mask_logits[k])
                p0 = self.state_dict_0[k]
                p1 = self.state_dict_1[k]
                avg = 0.5 * (p0 + p1)
                
                # 选择任务特定参数
                p_task = p0 if self.task_specific_model == 0 else p1
                
                # 融合公式：m越大越偏向共同部分（平均），越小越偏向任务特定模型
                p_fused = m * avg + (1 - m) * p_task

                # 随机采样
                num_elements = p0.numel()
                indices = torch.randperm(num_elements, device=self.device)[:sample_size] if num_elements > sample_size else slice(None)
                
                # 计算OT损失（与两个原始模型保持分布一致）
                fused_sample = p_fused.view(-1)[indices].unsqueeze(1)
                p0_sample = p0.view(-1)[indices].unsqueeze(1)
                p1_sample = p1.view(-1)[indices].unsqueeze(1)
                
                loss_ot = sinkhorn_loss_fn(fused_sample, p0_sample) + sinkhorn_loss_fn(fused_sample, p1_sample)
                
                # 计算熵正则化（鼓励掩码二值化）
                entropy = -(m * torch.log(m + 1e-8) + (1 - m) * torch.log(1 - m + 1e-8)).mean()
                
                total_loss += loss_ot + self.entropy_coeff * entropy
                total_entropy += entropy.item()

            total_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            if epoch % 10 == 0:
                log.info(f"Epoch {epoch}/{self.num_epochs}, Loss: {total_loss.item():.4f}, Entropy: {total_entropy:.4f}")

    def _binarize_masks(self):
        """硬二值化：根据sigmoid阈值0.5生成二值掩码"""
        self.binary_masks = {}
        for k in self.mask_logits:
            m = torch.sigmoid(self.mask_logits[k])
            self.binary_masks[k] = (m > 0.5).float()

    def fusion(self, state_dict_0: StateDict, state_dict_1: StateDict) -> StateDict:
        """最终融合：应用训练好的掩码"""
        fused_dict = {}
        for k in state_dict_0:
            m = torch.sigmoid(self.mask_logits[k])
            p0 = state_dict_0[k].to(self.device)
            p1 = state_dict_1[k].to(self.device)
            avg = 0.5 * (p0 + p1)
            p_task = p0 if self.task_specific_model == 0 else p1
            fused_dict[k] = (m * avg + (1 - m) * p_task).detach().cpu()
        return fused_dict

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return list(self.mask_logits.values())

    def to(self, device):
        super().to(device)
        for k in self.mask_logits:
            self.mask_logits[k] = self.mask_logits[k].to(device)
        return self
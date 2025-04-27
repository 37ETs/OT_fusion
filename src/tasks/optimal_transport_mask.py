from calendar import c
from fcntl import LOCK_MAND
import logging
from typing import Iterator, List, Optional
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.optim import Adam
from geomloss import SamplesLoss
from tqdm import tqdm  # 确保正确导入 tqdm


# 简单定义 StateDict 为 dict 类型
StateDict = dict

log = logging.getLogger(__name__)

import torch
from torch.optim import Adam
from geomloss import SamplesLoss
import random
from copy import deepcopy

def compute_ot_mask(
    state_dict_0: dict,
    state_dict_1: dict,
    masks_pre: dict,
    masks_post: dict,
    lr: float = 0.01,
    blur: float = 0.08,
    lmd: float = 0.01,
    max_epochs: int = 100,
    chunk_size: int = 200000,
    early_stop_patience: int = 10,
):
    # 准备 sinkhorn loss
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur)

    # 确保所有 masks 都是 nn.Parameter，方便优化器管理
    for k in masks_pre.keys():
        if not isinstance(masks_pre[k], torch.nn.Parameter):
            masks_pre[k] = torch.nn.Parameter(masks_pre[k])
        if not isinstance(masks_post[k], torch.nn.Parameter):
            masks_post[k] = torch.nn.Parameter(masks_post[k])

    # 为 masks_pre 和 masks_post 分别创建独立的优化器
    optimizer_pre = Adam(masks_pre.values(), lr=lr)
    optimizer_post = Adam(masks_post.values(), lr=lr)

    best_loss = float("inf")
    no_improve_count = 0  # 用于早停

    # 开始训练
    for epoch in range(max_epochs):
        # 判断当前 epoch 训练哪一部分
        train_pre = (epoch % 2 == 0)  # 偶数 epoch 训练 pre；奇数 epoch 训练 post

        if train_pre:
            optimizer_pre.zero_grad()
        else:
            optimizer_post.zero_grad()

        total_loss = 0.0

        # 逐层（逐 key）处理，做分块/随机采样
        for k in state_dict_0.keys():
            p0 = state_dict_0[k].view(-1)
            p1 = state_dict_1[k].view(-1)

            length = p0.shape[0]
            if length <= chunk_size:
                idx = torch.arange(length, device=p0.device)
            else:
                idx = torch.randperm(length, device=p0.device)[:chunk_size]

            sample_p0 = p0[idx]
            sample_p1 = p1[idx]

            # 同样对掩码进行 flatten + 采样
            # 注意：只计算当前要训练的那部分损失
            if train_pre:
                # 只训练 masks_pre => masks_post 暂时固定，仅在计算对齐时可能会用到固定值(如1)
                #mask_pre_sigmoid = torch.sigmoid(masks_pre[k].view(-1)[idx])
                mask_pre = masks_pre[k].view(-1)[idx]
                masked_tensor_0_pre = sample_p0 * mask_pre
                masked_tensor_1_pre = sample_p1

                # 计算 OT
                ot_loss_pre = sinkhorn_loss(
                    masked_tensor_0_pre.view(-1, 1),
                    masked_tensor_1_pre.view(-1, 1)
                )
                # 稀疏正则
                #reg_pre = torch.norm(mask_pre_sigmoid, p=1)
                reg_pre = torch.norm(mask_pre, p=1)

                # post 相关的正则可选做/不做
                # 如果希望完全固定 post，就不加 reg_post
                # 若想保持原策略，也可加上:
                reg_post = 0.0

                layer_loss = ot_loss_pre + lmd * (reg_pre + reg_post)

            else:
                # 只训练 masks_post => masks_pre 暂时固定
                #mask_post_sigmoid = torch.sigmoid(masks_post[k].view(-1)[idx])
                mask_post = masks_post[k].view(-1)[idx]
                masked_tensor_0_post = sample_p0
                masked_tensor_1_post = sample_p1 * mask_post

                ot_loss_post = sinkhorn_loss(
                    masked_tensor_0_post.view(-1, 1),
                    masked_tensor_1_post.view(-1, 1)
                )
                #reg_post = torch.norm(mask_post_sigmoid, p=1)
                reg_post = torch.norm(mask_post, p=1)

                # pre 不参与更新时，可不加 reg_pre
                reg_pre = 0.0

                layer_loss = ot_loss_post + lmd * (reg_pre + reg_post)

            total_loss += layer_loss

        # 反向传播 + 更新（只针对当前所训练的掩码）
        total_loss.backward()
        if train_pre:
            optimizer_pre.step()
        else:
            optimizer_post.step()

        # early stopping 检查
        curr_loss_val = total_loss.item()
        if curr_loss_val < best_loss - 1e-6:
            best_loss = curr_loss_val
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 日志打印
        if epoch % 20 == 0:
            which_mask = "masks_pre" if train_pre else "masks_post"
            print(f"[Epoch {epoch} - Train {which_mask}] total_loss = {curr_loss_val:.4f} (best={best_loss:.4f})")

        # 如果超过 patience 次没有改进，则提前结束
        if no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch={epoch}")
            break

    # 最后将参数 sigmoid 化得到最终的掩码
    #ot_masks_pre = {k: torch.sigmoid(v) for k, v in masks_pre.items()}
    #ot_masks_post = {k: torch.sigmoid(v) for k, v in masks_post.items()}
    ot_masks_pre = {k: v for k, v in masks_pre.items()}
    ot_masks_post = {k: v for k, v in masks_post.items()}
    return ot_masks_pre, ot_masks_post

def fusion(state_dict_0: StateDict, state_dict_1: StateDict, ot_masks_pre: StateDict, ot_masks_post: StateDict) -> StateDict:
    merged_state_dict = {}
    for key in state_dict_0.keys():
        ot_mask_pre = ot_masks_pre[key]
        ot_mask_post = ot_masks_post[key]
        # Combine the two masks for fusion
        
        with torch.no_grad():
                merged_state_dict[key] = (
                    0.5*ot_mask_pre * state_dict_0[key] + 0.5*ot_mask_post * state_dict_1[key]
                ) 
                
    return merged_state_dict


class OptimalTransportMask(nn.Module):
    def __init__(
        self,
        temperature: float,
        state_dict: StateDict,
        init_value: float = 3.0,
        draw_sample: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        masks = {}
        for k, v in state_dict.items():
            masks[k] = nn.Parameter(torch.ones_like(v) * 0, requires_grad=True)
            init_device = v.device
        self.masks = masks
        self.draw_sample = draw_sample
        
    def _draw_mask(self, binary_mask: Optional[bool] = False):
        ot_masks = {}
        for k in self.masks.keys():
            concrete_dist = torch.distributions.RelaxedBernoulli(
                self.temperature,
                logits=self.masks[k],
            )
            if binary_mask == True:
                ot_masks[k] = (concrete_dist.sample()).detach_() > 0.5
            else:
                if self.draw_sample:
                    #ot_masks[k] = concrete_dist.rsample()
                    #全部置为0
                    ot_masks[k] = torch.ones_like(concrete_dist.probs)
                else:
                    ot_masks[k] = torch.ones_like(concrete_dist.probs)
                    #ot_masks[k] = concrete_dist.probs
        
        return ot_masks
            

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.masks.values()

    def to(self, device):
        super().to(device)
        for mask in self.masks.values():
            mask.data = mask.data.to(device)
        return self

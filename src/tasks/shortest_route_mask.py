import logging
from operator import ge
from pyexpat import model
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.parameter import Parameter
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, Iterator, List, Optional
from src.datasets.common import maybe_dictionarize
from src.route_merged_model import RouteMergedModel
import geomloss


log = logging.getLogger(__name__)


# 简单定义 StateDict 为 dict 类型
StateDict = dict


class ShortestRouteMask(nn.Module):
    """
    一个示例的可学习 mask 模块。
    """
    def __init__(
        self,
        temperature: float,
        state_dict: StateDict,
        init_value: float = 3.0,
        draw_sample: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.draw_sample = draw_sample
        # 将 state_dict 中的权重形状对应地设置可学习参数
        masks = {}
        for k, v in state_dict.items():
            # 也可以根据 init_value 来初始化
            masks[k] = nn.Parameter(torch.ones_like(v) * init_value, requires_grad=True)
        self.masks = masks
        
    def _draw_mask(self, binary_mask: bool = False):
        """
        如果要进行 sample，可以考虑用 Gumbel-Softmax 或 Concrete 分布；
        这里仅做示例，也可返回概率。
        """
        ot_masks = {}
        for k, param in self.masks.items():
            # 构造一个 RelaxedBernoulli 分布
            concrete_dist = torch.distributions.RelaxedBernoulli(
                self.temperature,
                logits=param
            )
            if binary_mask:
                # 训练后期可能想要离散化
                ot_masks[k] = (concrete_dist.sample() > 0.5).float().detach()
            else:
                if self.draw_sample:
                    ot_masks[k] = concrete_dist.rsample()
                else:
                    # 只取分布的期望
                    ot_masks[k] = concrete_dist.probs
        return ot_masks

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return self.masks.values()

    def to(self, device):
        super().to(device)
        for k in self.masks:
            self.masks[k].data = self.masks[k].data.to(device)
        return self


def compute_sr_mask(
    task_vector_pre: StateDict,
    task_vector_post: StateDict,
    pretrained_model: nn.Module,
    masks_pre: Dict[str, Tensor],
    masks_post: Dict[str, Tensor],
    lr: float = 0.01,
    max_epochs: int = 100,
    pre_task_dataloader: Optional[torch.utils.data.DataLoader] = None,
    post_task_dataloader: Optional[torch.utils.data.DataLoader] = None,
    mask_alpha: float = 0.5,
    device: str = "cuda:1",
):
    
    for p in pretrained_model.parameters():
            p.detach_().requires_grad_(False)
            
    # # 1. 将原始 mask 张量变为可训练的 Parameter (初始值为 1)
    for k in masks_pre.keys():
        masks_pre[k] = Parameter(torch.ones_like(masks_pre[k]), requires_grad=True)
        masks_post[k] = Parameter(torch.ones_like(masks_post[k]), requires_grad=True)
    
    # 2. 构造辅助模型
    model_pre = build_model(pretrained_model, task_vector_pre, device)
    model_post = build_model(pretrained_model, task_vector_post, device)
    
    # 这一步是融合模型对象，会在 forward 里按照 masks 来进行加权
    model_merged = RouteMergedModel(
        pretrained_model, 
        task_vector_pre, 
        task_vector_post, 
        masks_pre, 
        masks_post, 
        mask_alpha,
        device
    )
 
    # 3. 移动到 GPU/CPU
    model_pre.to(device).eval()   # 教师模型不训练，eval 模式
    model_post.to(device).eval()  # 教师模型不训练，eval 模式
    model_merged.to(device).train()  # 学生(融合)模型训练
    
    # 4. 优化器只优化两个任务对应的 mask
    optimizer = Adam(
        [
            {'params': model_merged.masks_pre.values(), 'lr': lr, 'betas': (0.9, 0.999), 'weight_decay': 0.},
            {'params': model_merged.masks_post.values(), 'lr': lr, 'betas': (0.9, 0.999), 'weight_decay': 0.},
        ]
    )
    
    # 6. 初始化数据迭代器
    pre_iter = iter(pre_task_dataloader)
    post_iter = iter(post_task_dataloader)
    
    # 7. 记录最佳
    best_loss = float('inf')
    best_masks_pre = None
    best_masks_post = None
    
    pbar = tqdm(range(max_epochs), desc="Training masks")
    for epoch in pbar:
        model_merged.train()
        
        # 交替训练：偶数 epoch 用 pre_task 数据，奇数 epoch 用 post_task 数据
        train_pre = (epoch % 2 == 0)
        
        # 冻结非当前任务的 mask
        for param in masks_pre.values():
            param.requires_grad = train_pre
        for param in masks_post.values():
            param.requires_grad = not train_pre
        
        # 从相应数据集中取 batch
        try:
            batch = next(pre_iter if train_pre else post_iter)
        except StopIteration:
            # 如果迭代器到了末尾，则重新初始化
            pre_iter = iter(pre_task_dataloader)
            post_iter = iter(post_task_dataloader)
            batch = next(pre_iter if train_pre else post_iter)
        
        batch = maybe_dictionarize(batch)
        x, y = batch["images"].to(device), batch["labels"].to(device)
        
        # 先将mask应用到融合模型权重(合并参数)
        model_merged.merge_weights()
        # 学生模型前向
        logits_student = model_merged(x)
        
        # 教师模型前向 (不计算梯度)
        if train_pre:
            logits_teacher = model_pre(x).detach()
        else:
            logits_teacher = model_post(x).detach()
        
        #loss = nn.L1Loss()(logits_student, logits_teacher)
        loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.05)(logits_student, logits_teacher)

        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #print(model_merged.alpha.item(), model_merged.beta.item())
        
        # 记录、打印
        pbar.set_postfix({
            "epoch": epoch,
            "loss": f"{loss.item():.4f}",
        })
        
        # 如果有需要，可以做一个简单的“最优loss保存”
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_masks_pre = {k: v.detach().clone() for k,v in masks_pre.items()}
            best_masks_post = {k: v.detach().clone() for k,v in masks_post.items()}
    
    # ============ 训练结束后 ============
    # 如果想用最佳结果，可以还原回最佳 mask 参数
    if best_masks_pre is not None:
        for k in model_merged.masks_pre:
            model_merged.masks_pre[k].data = best_masks_pre[k]
        for k in model_merged.masks_post:
            model_merged.masks_post[k].data = best_masks_post[k]
    
    # 最后再做一次 merge
    model_merged.merge_weights()

    # 最终融合后的参数 (merged_state_dict)，可直接拿来推断
    merged_state_dict = {
        k: v.detach().cpu() for k, v in model_merged.merged_state_dict.items()
    }

    return merged_state_dict


def build_model(
    pretrained_model: nn.Module,
    task_state_dict: StateDict,
    device: str = "cuda:1",
):
    """
    根据预训练模型 + 一个任务的向量，构建对应的模型。
    通常做法是把 pretrained_model 的 state_dict 复制，然后加上 task_state_dict。
    """
    model = deepcopy(pretrained_model)
    model_sd = model.state_dict()
    
    for n, p in task_state_dict.items():
        if n in model_sd:
            # 直接加上去，或者看实际需求：是否相当于 residual 的方式
            model_sd[n] = model_sd[n].to(device) + p.to(device)
    
    model.load_state_dict(model_sd)
    model = model.to(device)
    return model
from ast import arg
import logging
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
from src.tasks.shortest_route_mask import build_model

log = logging.getLogger(__name__)

# 简单定义 StateDict 为 dict 类型
StateDict = dict

def compute_sr_classification_heads_single(
    task_vector_pre: StateDict,
    pretrained_model: nn.Module,
    merged_state_dict: StateDict,
    classification_head_pre: nn.Module,
    lr: float = 0.01,
    max_epochs: int = 100,
    pre_task_dataloader: Optional[torch.utils.data.DataLoader] = None,
    device: str = "cuda:1",
):
    
    for p in pretrained_model.parameters():
            p.detach_().requires_grad_(False)    
    
    # 2. 构造辅助模型
    model_pre = build_model(pretrained_model, task_vector_pre, device)

    model_merged = deepcopy(pretrained_model)
    model_merged.load_state_dict(merged_state_dict)

    classification_head_pre_student = deepcopy(classification_head_pre)
    # 将融合后模型的权重冻结
    for param in model_merged.parameters():
        param.requires_grad = False

    # 3. 将学生分类头的参数设置为可训练
    classification_head_pre_student.weight.requires_grad = True

    # 4. 将教师模型的分类头参数设置为不可训练
    classification_head_pre.weight.requires_grad = False

    # 4. 优化器只优化两个任务对应的分类头
    optimizer = Adam(
        [
            {'params': classification_head_pre_student.parameters(), 'lr': lr, 'betas': (0.9, 0.999), 'weight_decay': 0.},
        ]
    )
    # 5. 移动到 GPU/CPU
    model_pre.to(device)
    model_merged.to(device)

    classification_head_pre.to(device)

    model_pre.eval()
    model_merged.eval()
    classification_head_pre.train()

    # 6. 初始化数据迭代器
    pre_iter = iter(pre_task_dataloader)

    criterion = nn.CrossEntropyLoss()
    
    # 7. 记录最佳
    best_loss = float('inf')
    
    pbar = tqdm(range(max_epochs), desc="Training classification heads")
    for epoch in pbar:
        
        # 从相应数据集中取 batch
        try:
            batch = next(pre_iter)
        except StopIteration:
            # 如果迭代器到了末尾，则重新初始化
            pre_iter = iter(pre_task_dataloader)
            batch = next(pre_iter)
        
        batch = maybe_dictionarize(batch)
        x, y = batch["images"].to(device), batch["labels"].to(device)
        
        # 学生模型前向
        with torch.no_grad():
            distribution_pre = model_merged(x)
        logits_student = classification_head_pre_student(distribution_pre)
        
        # 教师模型前向
        with torch.no_grad():
            logits_teacher = classification_head_pre(model_pre(x))

        loss = criterion(logits_student, y)
        # # 与教师模型的输出进行对比
        # loss = F.kl_div(
        #     F.log_softmax(logits_student, dim=1),
        #     F.softmax(logits_teacher, dim=1),
        #     reduction='batchmean'
        # )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录、打印
        pbar.set_postfix({
            "epoch": epoch,
            "loss": f"{loss.item():.4f}",
        })
        
        # 如果有需要，可以做一个简单的“最优loss保存”
        if loss.item() < best_loss:
            best_loss = loss.item()
    
    # 训练完成后，冻结分类头的参数，保存
    classification_head_pre_student.weight.requires_grad = False
    
    return classification_head_pre_student
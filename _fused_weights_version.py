from typing import List, Optional
from torch import Tensor


# 版本1：使用task-wise权重进行融合
def _fuse_weights(task_wise_weight: Tensor,
                  tensors: List[Tensor], 
                  salient_mask: Optional[List[Tensor]] = None
                  ) -> Tensor:
    device = task_wise_weight.device
    # Regular fused result
    base = sum(task_wise_weight[i] * w.to(device) for i, w in enumerate(tensors))

    if salient_mask is not None:
        fused = base.clone()
        for i, (mask_i, tensor_i) in enumerate(zip(salient_mask, tensors)):
            mask_i = mask_i.to(device)
            tensor_i = tensor_i.to(device)
            fused[mask_i == 1] = tensor_i[mask_i == 1]
        return fused
    else:
        return base
    
    
# 版本2：基于任务权重优先级公平地解决overlap问题
def _fuse_weights(task_wise_weight: Tensor,
                  tensors: List[Tensor], 
                  salient_mask: Optional[List[Tensor]] = None
                  ) -> Tensor:
    """
    优化版本：基于任务权重优先级公平地解决overlap问题。
    """
    #device = task_wise_weight.device
    device = torch.device("cuda:1")
    base = sum(task_wise_weight[i] * tensors[i].to(device) for i in range(len(tensors)))

    if salient_mask is None:
        return base

    # 如果有salient_mask，则公平处理冲突
    masks_stack = torch.stack([mask.to(device) for mask in salient_mask], dim=0)  # [num_models, ...]
    fused = base.clone()

    # 找出所有位置的重叠情况
    overlap_positions = masks_stack.sum(dim=0) > 1  # 重叠位置 (bool tensor)

    # 无重叠的位置直接赋值
    for i in range(len(tensors)):
        exclusive_mask = (masks_stack[i] == 1) & (~overlap_positions)
        # 确保被索引张量和掩码在同一设备
        tensor_i = tensors[i].to(device)  # 先移动权重到目标设备
        mask_i = exclusive_mask.to(device)
        fused[mask_i] = tensor_i[mask_i]  # 现在两者都在device上
        
        
    # 重叠位置，使用task_wise_weight优先级
    if overlap_positions.any():
        # 获得最高优先级模型的索引
        _, max_idx = task_wise_weight.max(dim=0)
        max_tensor = tensors[max_idx].to(device)
        fused[overlap_positions] = max_tensor[overlap_positions]

    return fused


# 版本3：基于overlap数量进行平均合并
def _fuse_weights(task_wise_weight: Tensor,
                  tensors: List[Tensor], 
                  salient_mask: Optional[List[Tensor]] = None
                  ) -> Tensor:
    """
    优化版本：如果发生overlap，则根据overlap的数量进行平均合并。
    """
    device = torch.device("cuda:1")
    base = sum(task_wise_weight[i] * tensors[i].to(device) for i in range(len(tensors)))

    if salient_mask is None:
        return base

    masks_stack = torch.stack([mask.to(device) for mask in salient_mask], dim=0)  # [num_models, ...]
    fused = base.clone()

    overlap_count = masks_stack.sum(dim=0)  # 计算每个位置的overlap数量
    overlap_positions = overlap_count > 1   # 重叠位置 (bool tensor)

    # 处理无重叠位置
    for i in range(len(tensors)):
        exclusive_mask = (masks_stack[i] == 1) & (~overlap_positions)
        tensor_i = tensors[i].to(device)
        fused[exclusive_mask] = tensor_i[exclusive_mask]

    # 处理重叠位置
    if overlap_positions.any():
        stacked_tensors = torch.stack([t.to(device) for t in tensors], dim=0)  # [num_models, ...]
        # 仅在掩码为1的位置考虑tensor值，其余置0
        masked_tensors = stacked_tensors * masks_stack
        # 求和再除以overlap数量，获得平均值
        #averaged_tensor = masked_tensors.sum(dim=0) / overlap_count.clamp(min=1)
        # 根据task_wise_weight加权平均
        averaged_tensor = (masked_tensors.sum(dim=0) / overlap_count.clamp(min=1)).sum(dim=0)
        fused[overlap_positions] = averaged_tensor[overlap_positions]

    return fused


# 版本4：overlap位置使用task_wise_weight在有效mask位置的重新归一化权重进行加权融合
def _fuse_weights(task_wise_weight: Tensor,
                  tensors: List[Tensor], 
                  salient_mask: Optional[List[Tensor]] = None
                  ) -> Tensor:
    """
    优化版本：overlap位置使用task_wise_weight在有效mask位置的重新归一化权重进行加权融合。
    """
    device = torch.device("cuda:1")
    base = sum(task_wise_weight[i] * tensors[i].to(device) for i in range(len(tensors)))

    if salient_mask is None:
        return base

    masks_stack = torch.stack([mask.to(device) for mask in salient_mask], dim=0)  # [num_models, ...]
    fused = base.clone()

    overlap_count = masks_stack.sum(dim=0)  # 每个位置的overlap数量
    overlap_positions = overlap_count > 1   # 重叠位置 (bool tensor)

    # 处理无重叠位置
    for i in range(len(tensors)):
        exclusive_mask = (masks_stack[i] == 1) & (~overlap_positions)
        tensor_i = tensors[i].to(device)
        fused[exclusive_mask] = tensor_i[exclusive_mask]

    if overlap_positions.any():
        stacked_tensors = torch.stack([t.to(device) for t in tensors], dim=0)  # [num_models, ...]
        
        # 提取mask为1的位置的task权重
        task_weights_expanded = task_wise_weight.view(-1, *[1]*(masks_stack.dim()-1)).to(device)
        
        # 仅保留mask=1位置的权重，mask=0的任务权重置0
        effective_weights = masks_stack * task_weights_expanded
        
        # 在overlap位置归一化权重之和为1
        weight_sum = effective_weights.sum(dim=0, keepdim=True).clamp(min=1e-8)
        normalized_weights = effective_weights / weight_sum
        
        # 根据重新归一化的权重加权融合tensor
        weighted_tensor = (normalized_weights * stacked_tensors).sum(dim=0)
        
        fused[overlap_positions] = weighted_tensor[overlap_positions]
    
    return fused


# 版本5：overlap位置使用task_wise_weight在有效mask位置取最大权重的任务值
def _fuse_weights(task_wise_weight: Tensor,
                  tensors: List[Tensor], 
                  salient_mask: Optional[List[Tensor]] = None
                  ) -> Tensor:
    """
    优化版本：overlap位置使用task_wise_weight在有效mask位置的重新归一化权重进行加权融合。
    """
    device = torch.device("cuda:1")
    base = sum(task_wise_weight[i] * tensors[i].to(device) for i in range(len(tensors)))

    if salient_mask is None:
        return base

    masks_stack = torch.stack([mask.to(device) for mask in salient_mask], dim=0)  # [num_models, ...]
    fused = base.clone()

    overlap_count = masks_stack.sum(dim=0)  # 每个位置的overlap数量
    overlap_positions = overlap_count > 1   # 重叠位置 (bool tensor)

    # 处理无重叠位置
    for i in range(len(tensors)):
        exclusive_mask = (masks_stack[i] == 1) & (~overlap_positions)
        tensor_i = tensors[i].to(device)
        fused[exclusive_mask] = tensor_i[exclusive_mask]

    # 处理重叠位置 (修改为直接取mask=1中权重最大的那个)
    if overlap_positions.any():
        stacked_tensors = torch.stack([t.to(device) for t in tensors], dim=0)  # [num_models, ...]

        task_weights_expanded = task_wise_weight.view(-1, *[1]*(masks_stack.dim()-1)).to(device)

        # mask=0位置权重设为-inf，以确保不会被选中
        masked_task_weights = masks_stack * task_weights_expanded + (1 - masks_stack) * (-float('inf'))

        # 选出在每个位置上权重最大的任务索引
        _, max_weight_task_indices = masked_task_weights.max(dim=0)

        # 根据索引选择对应tensor中的值
        fused_overlap = torch.gather(
            stacked_tensors,
            dim=0,
            index=max_weight_task_indices.unsqueeze(0)
        ).squeeze(0)

        fused[overlap_positions] = fused_overlap[overlap_positions]


    return fused

# 版本6：overlap位置使用task_wise_weight直接加权融合
def _fuse_weights(task_wise_weight: Tensor,
                  tensors: List[Tensor], 
                  salient_mask: Optional[List[Tensor]] = None
                  ) -> Tensor:
    """
    优化版本：
    - 无重叠(mask=1)位置直接赋值。
    - 重叠位置直接根据task_wise_weight加权融合，与mask=0情况一致。
    """
    device = torch.device("cuda:1")
    base = sum(task_wise_weight[i] * tensors[i].to(device) for i in range(len(tensors)))

    if salient_mask is None:
        return base

    masks_stack = torch.stack([mask.to(device) for mask in salient_mask], dim=0)  # [num_models, ...]
    fused = base.clone()

    overlap_positions = masks_stack.sum(dim=0) > 1   # 重叠位置 (bool tensor)

    # 处理无重叠的位置 (mask=1时直接赋值)
    for i in range(len(tensors)):
        exclusive_mask = (masks_stack[i] == 1) & (~overlap_positions)
        tensor_i = tensors[i].to(device)
        fused[exclusive_mask] = tensor_i[exclusive_mask]

    # 重叠位置保持base的值（即与mask=0一样直接加权融合），无需额外处理
    # 因为 base 已经是加权融合后的结果，因此下面的操作可以省略，但为明确起见可显式注明：
    # fused[overlap_positions] = base[overlap_positions]  # 其实fused已经是base的拷贝，所以可省略

    return fused
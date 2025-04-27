R"""
```python
# Get the task-wise weights
task_wise_weights = get_task_wise_weights(num_models)

# Define the task vectors (in this case, we'll use the state_dict of the pretrained model)
task_vectors = ...

# Initialize the TaskWiseMergedModel
merged_model = TaskWiseMergedModel(pretrained_model, task_wise_weights, task_vectors)

# Now you can use the merged_model like a regular PyTorch model
outputs = merged_model(inputs)
```
"""
import logging
from re import L
import types
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .ties_merging_utils import check_parameterNamesMatch
from .type import StateDict
from .utils import timeit_context

log = logging.getLogger(__name__)

__all__ = ["get_task_wise_weights", "fuse_weights", "TaskWiseMergedModel"]


def get_task_wise_weights(num_models: int, init_values: float = None):
    """
    This function generates a tensor of weights for each model.

    Args:
        num_models (int): The number of models.
        init_values (float, optional): The initial value for each weight. Defaults to None.

    Returns:
        Tensor: A tensor of weights for each model.
    """
    assert num_models >= 1, f"num_models must be >= 1, got {num_models}"
    if init_values is None:
        init_values = 1.0 / num_models
    return torch.full((num_models,), init_values, dtype=torch.float32)


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
        averaged_tensor = masked_tensors.sum(dim=0) / overlap_count.clamp(min=1)
        fused[overlap_positions] = averaged_tensor[overlap_positions]

    return fused





# def fuse_weights(task_wise_weight: Tensor, state_dicts: List[StateDict], salient_masks: List[StateDict]) -> StateDict:
#     num_models = len(state_dicts)
#     assert task_wise_weight.dim() == 1, f"task_wise_weight must be a 1D tensor, got {task_wise_weight.dim()}"
#     assert num_models == task_wise_weight.size(
#         0
#     ), f"num_models must be equal to the number of state_dicts, got {num_models} and {task_wise_weight.size(0)}"
#     #return {k: _fuse_weights(task_wise_weight, [sd[k] for sd in state_dicts]) for k in state_dicts[0].keys()}
#     #return {k: _fuse_weights(task_wise_weight, [sd[k] for sd in state_dicts], [sm[k] for sm in salient_masks]) for k in state_dicts[0].keys()}
#     fused_state_dict = {}
#     v = salient_masks[0].keys()
    
#     for k in state_dicts[0].keys():
#         if k in v:
#             fused_state_dict[k] = _fuse_weights(task_wise_weight, [sd[k] for sd in state_dicts], [salient_masks[sm][k] for sm in salient_masks])
#         else:
#             fused_state_dict[k] = _fuse_weights(task_wise_weight, [sd[k] for sd in state_dicts])
#     return fused_state_dict

def fuse_weights(task_wise_weight: Tensor, state_dicts: List[StateDict], salient_masks: List[StateDict]) -> StateDict:
    fused_state_dict = {}
    salient_keys = salient_masks[0].keys()

    for k in state_dicts[0].keys():
        if k in salient_keys:
            masks_k = [salient_masks[sm][k] for sm in salient_masks]
            tensors_k = [sd[k] for sd in state_dicts]
            fused_state_dict[k] = _fuse_weights(task_wise_weight, tensors_k, masks_k)
        else:
            tensors_k = [sd[k] for sd in state_dicts]
            fused_state_dict[k] = _fuse_weights(task_wise_weight, tensors_k)

    return fused_state_dict


class TaskWiseMergedModel(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module,
        task_wise_weight: Tensor,
        task_vectors: List[StateDict],
        salient_masks: List[StateDict],
        clamp_weights: bool = True,
    ):
        super().__init__()
        self._model = (pretrained_model,)  # self._model should be on cpu

        self.task_wise_weight = nn.Parameter(task_wise_weight, requires_grad=True)
        self.task_vectors = task_vectors  # should be on cpu
        self.pretrained_state_dict: StateDict = self.model.state_dict(keep_vars=False)
        check_parameterNamesMatch(self.task_vectors)
        self.clamp_weights = clamp_weights
        self.salient_masks = salient_masks
        self.merged_state_dict = None

    @property
    def model(self):
        return self._model[0]
    
    def compute_overlap_ratio(self):
        salient_masks = self.salient_masks
        for i in range(len(salient_masks)):
            for j in range(i+1, len(salient_masks)):
                for k in salient_masks[i].keys():
                    overlap = torch.sum(salient_masks[i][k] * salient_masks[j][k])
                    num1_of_1 = torch.sum(salient_masks[i][k])
                    num2_of_1 = torch.sum(salient_masks[j][k])
                    print(f"Number1 of 1 in {k} of {i} is {num1_of_1}")
                    print(f"Number1 of 2 in {k} of {j} is {num2_of_1}")
                    print(f"Overlap of {k} between {i} and {j} is {overlap}")
                    union = salient_masks[i][k].shape[0] * salient_masks[i][k].shape[1]
                    print(f"Overlap ratio of {k} between {i} and {j} is {overlap/union}")

    def merge_weights(self):
        if self.clamp_weights:
            task_wise_weight = self.task_wise_weight.clamp(0, 1)
        else:
            task_wise_weight = self.task_wise_weight
        device = task_wise_weight.device
        task_vector = fuse_weights(task_wise_weight, self.task_vectors, self.salient_masks)
        self.merged_state_dict = {k: self.pretrained_state_dict[k].to(device, non_blocking=True) for k in self.pretrained_state_dict.keys()}
        for k in task_vector.keys():
            self.merged_state_dict[k] += task_vector[k]

    def forward(self, *args, **kwargs):
        if self.merged_state_dict is None:
            self.merge_weights()
        return functional_call(
            self.model,
            self.merged_state_dict,
            args=args,
            kwargs=kwargs,
            tie_weights=False,
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            attr = getattr(self.model, name)
            if isinstance(attr, Callable):
                warnings.warn(f"forwarding `{name}` to the underlying model", UserWarning)
            return attr

    def __setattr__(self, name: str, value: Any) -> None:
        try:
            super().__setattr__(name, value)
        except AttributeError:
            setattr(self.model, name, value)

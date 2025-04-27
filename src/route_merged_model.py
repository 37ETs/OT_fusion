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
from re import S
import types
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .ties_merging_utils import check_parameterNamesMatch
from .type import StateDict
from .utils import timeit_context

log = logging.getLogger(__name__)

__all__ = ["fuse_weights", "RouteMergedModel"]

def _fuse_weights(
    mask_pre: Tensor,
    mask_post: Tensor,
    task_vectors: List[Tensor],
    mask_alpha: float = 0.5,
    device: str = "cuda:1",
) -> Tensor:
    """
    This function fuses the weights of the models.

    Args:
        mask_pre (Tensor): The mask for the pre-task.
        mask_post (Tensor): The mask for the post-task.
        task_vectors (List[Tensor]): The task vectors.

    Returns:
        Tensor: The fused weights.
    """
    assert len(task_vectors) == 2
    return mask_pre.to(device) * task_vectors[0].to(device) * (mask_alpha) + mask_post.to(device) * task_vectors[1].to(device) * (1-mask_alpha)


def fuse_weights(
    task_vector_pre: StateDict,
    task_vector_post: StateDict,
    masks_pre: Dict[str, Tensor],
    masks_post: Dict[str, Tensor],
    mask_alpha: float = 0.5,
    device: str = "cuda:1",
) -> StateDict:
    """
    This function fuses the weights of the models.

    Args:
        task_vector_pre (StateDict): The weights for the pre-task.
        task_vector_post (StateDict): The weights for the post-task.
        masks_pre (Dict[str, Tensor]): The masks for the pre-task.
        masks_post (Dict[str, Tensor]): The masks for the post-task.

    Returns:
        StateDict: The fused weights.
    """
    task_vector = {}
    for k in task_vector_pre.keys():
        task_vector[k] = _fuse_weights(masks_pre[k], masks_post[k], [task_vector_pre[k], task_vector_post[k]], mask_alpha, device=device)
    return task_vector


class RouteMergedModel(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module,
        task_vector_pre: StateDict,
        task_vector_post: StateDict,
        masks_pre: Dict[str, Tensor],
        masks_post: Dict[str, Tensor],
        mask_alpha: float,
        device: str,
    ):
        super().__init__()
        self._model = (pretrained_model,)  # self._model should be on cpu
        self.pretrained_model = pretrained_model
        self.task_vector_pre = task_vector_pre
        self.task_vector_post = task_vector_post
        self.masks_pre = masks_pre
        self.masks_post = masks_post
        self.mask_alpha = mask_alpha
        self.device = device
        self.pretrained_state_dict: StateDict = self.model.state_dict(keep_vars=False)
        self.merged_state_dict = None

    @property
    def model(self):
        return self._model[0]

    def merge_weights(self):
        task_vector = fuse_weights(self.task_vector_pre, self.task_vector_post, self.masks_pre, self.masks_post, mask_alpha=self.mask_alpha, device=self.device)
        device = self.device
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

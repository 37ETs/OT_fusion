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


def fuse_weights(
    task_vector_pre: StateDict,
    task_vector_post: StateDict,
    masks_pre: Dict[str, Tensor],
    masks_post: Dict[str, Tensor],
    alpha: Tensor,
    beta: Tensor,
) -> StateDict:
    """
    这里把原先写死的 0.5 用 alpha 和 beta 两个可训练参数替代。
    alpha、beta 均为 nn.Parameter，因此可以在优化时被更新。
    """
    task_vector = {}
    for k in task_vector_pre.keys():
        # 根据 mask & task_vector 做融合
        task_vector[k] = (masks_pre[k] * task_vector_pre[k] * alpha
                          + masks_post[k] * task_vector_post[k] * beta)/(alpha + beta)
    return task_vector


class RouteMergedModel(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module,
        task_vector_pre: StateDict,
        task_vector_post: StateDict,
        masks_pre: Dict[str, Tensor],
        masks_post: Dict[str, Tensor],
        device: str,
    ):
        super().__init__()
        self._model = (pretrained_model,)  # self._model should be on cpu
        self.pretrained_model = pretrained_model
        self.task_vector_pre = task_vector_pre
        self.task_vector_post = task_vector_post
        self.masks_pre = masks_pre
        self.masks_post = masks_post
        self.device = device

        # 定义可训练参数 alpha 和 beta，用于替换原先的两个0.5
        # 这里的 alpha 和 beta 是可训练的参数，初始值为 1
        # 这两个参数会在训练过程中被更新
        self.alpha = nn.Parameter(torch.tensor(0.7, device=device))
        self.beta = nn.Parameter(torch.tensor(0.3, device=device))

        # 缓存一下预训练模型的参数，后续做融合时会在此基础上加
        self.pretrained_state_dict: StateDict = self.model.state_dict(keep_vars=False)
        self.merged_state_dict = None

    @property
    def model(self):
        return self._model[0]

    def merge_weights(self):
        # 使用上方的 fuse_weights，并传入 alpha、beta
        task_vector = fuse_weights(
            self.task_vector_pre,
            self.task_vector_post,
            self.masks_pre,
            self.masks_post,
            self.alpha,
            self.beta
        )
        device = self.device
        # 拷贝一份预训练权重，然后在其基础上加上融合后的增量
        self.merged_state_dict = {
            k: self.pretrained_state_dict[k].to(device, non_blocking=True)
            for k in self.pretrained_state_dict.keys()
        }
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

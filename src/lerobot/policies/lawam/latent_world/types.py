from __future__ import annotations

from typing import Union
from collections.abc import Sequence

import numpy as np
import torch
from typing_extensions import TypedDict
from typing import NotRequired, Required

TensorLike2D = Union[torch.Tensor, np.ndarray, Sequence[float], Sequence[Sequence[float]]]
FrameArray = np.ndarray
ImageViews = Sequence[FrameArray]
VideoViews = Sequence[Sequence[FrameArray]]


class LatentWorldPolicyTrainRawSample(TypedDict):
    primary_videos: torch.Tensor
    wrist_images: torch.Tensor
    lang: str
    state: torch.Tensor
    action: torch.Tensor
    embodiment_id: int
    action_hz: float


class LatentWorldPolicyInferExample(TypedDict, total=False):
    primary_image: Required[ImageViews]
    lang: Required[str]
    embodiment_id: Required[int]
    action_hz: Required[float]
    state: NotRequired[TensorLike2D]
    state_mask: NotRequired[TensorLike2D]
    wrist_image: NotRequired[ImageViews]


class LatentWorldPolicyInferBatch(TypedDict):
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    act_placeholder_mask: torch.Tensor
    flow_placeholder_mask: torch.Tensor
    primary_image: torch.Tensor
    state: torch.Tensor
    state_mask: torch.Tensor
    embodiment_id: torch.Tensor
    action_hz: torch.Tensor
    image_grid_thw: torch.Tensor | None


class LatentWorldPolicyTrainBatch(TypedDict):
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    act_placeholder_mask: torch.Tensor
    flow_placeholder_mask: torch.Tensor
    primary_video: torch.Tensor
    state: torch.Tensor
    state_mask: torch.Tensor
    embodiment_id: torch.Tensor
    action_hz: torch.Tensor
    image_grid_thw: torch.Tensor | None
    actions: torch.Tensor
    actions_mask: torch.Tensor

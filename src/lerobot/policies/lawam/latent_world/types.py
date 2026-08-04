# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence
from typing import NotRequired, Required

import numpy as np
import torch
from typing_extensions import TypedDict

TensorLike2D = torch.Tensor | np.ndarray | Sequence[float] | Sequence[Sequence[float]]
FrameArray = np.ndarray
ImageViews = Sequence[FrameArray]


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

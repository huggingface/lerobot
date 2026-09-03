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

"""Typed tensor batch contracts consumed by the LaWAM backend."""

from __future__ import annotations

import torch
from typing_extensions import TypedDict


class LatentWorldPolicyInferBatch(TypedDict):
    """Tensor batch consumed by the native LaWAM inference backend."""

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
    """Tensor batch consumed by the native LaWAM training backend."""

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

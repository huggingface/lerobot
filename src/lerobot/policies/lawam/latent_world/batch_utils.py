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

"""Tensor helpers shared by the LaWAM processor and backend."""

from __future__ import annotations

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_placeholder_masks(
    input_ids: torch.Tensor,
    *,
    act_queries: int,
    flow_queries: int,
    placeholder_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ordered placeholder tokens into latent-action and flow masks."""
    expected = int(act_queries + flow_queries)
    placeholder = input_ids == int(placeholder_id)
    order = placeholder.cumsum(dim=1)
    act_mask = placeholder & (order <= int(act_queries))
    flow_mask = placeholder & (order > int(act_queries)) & (order <= expected)
    return act_mask, flow_mask


def imagenet_normalize_video_(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a video batch in place with ImageNet channel statistics."""
    if tensor.ndim != 5 or int(tensor.shape[2]) != 3:
        raise ValueError(f"Expected video tensor with shape [B, T, 3, H, W], got {tuple(tensor.shape)}.")
    mean = tensor.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor

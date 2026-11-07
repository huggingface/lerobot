#!/usr/bin/env python

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

"""Run a transformers Qwen2.5-VL vision tower on a fixed image grid (see `qwen_vision.py`).

`Qwen2_5VisionGrid` derives the rope positions and the attention windows of a grid with transformers'
`vision_utils`, as the stock forward does.
"""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Self

import torch
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

from .qwen_vision import FixedImageGrid, replay_vision_block

if TYPE_CHECKING or _transformers_available:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )
    from transformers.vision_utils import get_vision_position_ids, get_vision_window_index
else:
    apply_rotary_pos_emb_vision = None
    eager_attention_forward = None
    get_vision_position_ids = None
    get_vision_window_index = None


@dataclass
class Qwen2_5VisionGrid(FixedImageGrid):  # noqa: N801
    """What a Qwen2.5-VL vision tower derives from one image grid, as its stock forward computes it."""

    position_embeddings: tuple[Tensor, Tensor]
    window_index: Tensor
    reverse_index: Tensor
    full_lengths: list[int]
    window_lengths: list[int]
    window_size: int
    patch_size: int

    @classmethod
    def _from_grid(cls, visual: nn.Module, grid_thw: list[list[int]]) -> Self:
        grid = torch.tensor(grid_thw, device=visual.device)
        position_ids = get_vision_position_ids(grid, visual.spatial_merge_size)
        window_index, cu_window_seqlens = get_vision_window_index(
            grid, visual.spatial_merge_size, visual.window_size, visual.patch_size
        )
        window_index = window_index.to(visual.device)
        # The rotary embedding only takes the device from its first argument.
        cos, sin = (
            visual.permute_input_for_window_attn(emb, window_index, position_ids.shape[0])
            for emb in visual.rotary_pos_emb(position_ids, position_ids)
        )
        return cls(
            grid_thw=grid_thw,
            device=visual.device,
            dtype=visual.dtype,
            position_embeddings=(cos, sin),
            window_index=window_index,
            reverse_index=torch.argsort(window_index),
            full_lengths=[h * w for t, h, w in grid_thw for _ in range(t)],
            window_lengths=[b - a for a, b in pairwise(cu_window_seqlens.tolist())],
            window_size=visual.window_size,
            patch_size=visual.patch_size,
        )


def qwen2_5_vision_forward(
    visual: nn.Module,
    pixel_values: Tensor,
    grid_thw: Tensor,
    grid: Qwen2_5VisionGrid,
    attention: Callable | None = None,
) -> Tensor:
    """`visual(pixel_values, grid_thw).pooler_output`, computed from `grid` so that `torch.export` can trace it.

    `grid` is `Qwen2_5VisionGrid.build(visual, grid_thw)`; a grid built for another image grid is refused.
    `attention` replaces the chunked attention of every block (see `replay_vision_block`).
    """
    grid.check(visual, grid_thw)
    # Both set the attention windows of a grid.
    if (grid.window_size, grid.patch_size) != (visual.window_size, visual.patch_size):
        raise ValueError(
            f"The vision grid was built for window_size {grid.window_size} and patch_size "
            f"{grid.patch_size}, but the tower has {visual.window_size} and {visual.patch_size}."
        )
    hidden_states = visual.patch_embed(pixel_values)
    hidden_states = visual.permute_input_for_window_attn(
        hidden_states, grid.window_index, hidden_states.shape[0]
    )
    for layer_num, block in enumerate(visual.blocks):
        lengths = grid.full_lengths if layer_num in visual.fullatt_block_indexes else grid.window_lengths
        hidden_states = replay_vision_block(
            block,
            hidden_states,
            grid.position_embeddings,
            lengths,
            apply_rotary_pos_emb_vision,
            eager_attention_forward,
            attention,
        )
    return visual.merger(hidden_states)[grid.reverse_index]

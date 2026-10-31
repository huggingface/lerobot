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

"""Run a transformers Qwen2.5-VL vision tower on one fixed image grid (see `qwen_vision.py`).

`Qwen2_5VisionGrid` calls the tower's `rot_pos_emb` and `get_window_index`. Transformers removed both in
5.17, while LeRobot pins it below 5.6, so this module needs updating before the pin moves past 5.16.
"""

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
else:
    apply_rotary_pos_emb_vision = None
    eager_attention_forward = None


def qwen2_5_image_temporal_scale(model: nn.Module) -> int:
    """The `image_temporal_scale` of `qwen_image_rope_index` that the installed `model.get_rope_index` uses.

    It asks `get_rope_index` for the positions of one text token and one image token on tiny CPU tensors.
    """
    merge = model.config.vision_config.spatial_merge_size
    input_ids = torch.tensor([[0, model.config.image_token_id]])
    positions, _ = model.get_rope_index(
        input_ids,
        mm_token_type_ids=input_ids.new_tensor([[0, 1]]),
        image_grid_thw=input_ids.new_tensor([[1, merge, merge]]),
        attention_mask=torch.ones_like(input_ids),
    )
    # The image starts at position 1, so its temporal position is the scale itself.
    return int(positions[0, 0, 1])


@dataclass
class Qwen2_5VisionGrid(FixedImageGrid):  # noqa: N801
    """What a Qwen2.5-VL vision tower derives from one image grid, as its stock forward computes it."""

    position_embeddings: tuple[Tensor, Tensor]
    window_index: Tensor
    reverse_index: Tensor
    full_lengths: list[int]
    window_lengths: list[int]

    @classmethod
    def _from_grid(cls, visual: nn.Module, grid_thw: list[list[int]]) -> Self:
        grid = torch.tensor(grid_thw, device=visual.device)
        unit = visual.spatial_merge_unit
        rotary_pos_emb = visual.rot_pos_emb(grid)
        window_index, cu_window_seqlens = visual.get_window_index(grid)
        seq_len = rotary_pos_emb.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // unit, unit, -1)[window_index].reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        window_index = window_index.to(emb.device)
        return cls(
            grid_thw=grid_thw,
            device=visual.device,
            dtype=visual.dtype,
            position_embeddings=(emb.cos(), emb.sin()),
            window_index=window_index,
            reverse_index=torch.argsort(window_index),
            full_lengths=[h * w for t, h, w in grid_thw for _ in range(t)],
            # The stock forward drops the repeated boundaries with unique_consecutive.
            window_lengths=[b - a for a, b in pairwise(cu_window_seqlens) if b > a],
        )


def qwen2_5_vision_forward(visual: nn.Module, pixel_values: Tensor, grid: Qwen2_5VisionGrid) -> Tensor:
    """`visual(pixel_values, grid_thw).pooler_output` for the prepared `grid`, traceable by `torch.export`."""
    hidden_states = visual.patch_embed(pixel_values)
    seq_len, unit = hidden_states.shape[0], visual.spatial_merge_unit
    hidden_states = hidden_states.reshape(seq_len // unit, unit, -1)[grid.window_index].reshape(seq_len, -1)
    for layer_num, block in enumerate(visual.blocks):
        lengths = grid.full_lengths if layer_num in visual.fullatt_block_indexes else grid.window_lengths
        hidden_states = replay_vision_block(
            block,
            hidden_states,
            grid.position_embeddings,
            lengths,
            apply_rotary_pos_emb_vision,
            eager_attention_forward,
        )
    return visual.merger(hidden_states)[grid.reverse_index]

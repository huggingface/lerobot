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

"""Run a transformers Qwen3-VL model on one fixed image grid (see `qwen_vision.py`).

`Qwen3VisionGrid` calls the tower's `rot_pos_emb`, which transformers removed in 5.17, while LeRobot pins
it below 5.6, so this module needs updating before the pin moves past 5.16.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import torch
from torch import Tensor

from lerobot.utils.import_utils import _transformers_available

from .input_checks import check_input
from .qwen_vision import FixedImageGrid, replay_vision_block

if TYPE_CHECKING or _transformers_available:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLVisionModel,
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )
else:
    Qwen3VLModel = None
    Qwen3VLVisionModel = None
    apply_rotary_pos_emb_vision = None
    eager_attention_forward = None


def _bilinear_indices_and_weights(
    grid_thw: list[list[int]], num_grid_per_side: int, merge_size: int
) -> tuple[Tensor, Tensor]:
    """The weight-independent part of `Qwen3VLVisionModel.fast_pos_embed_interpolate`.

    Returns the four `pos_embed` rows each patch interpolates and their weights, both `(4, patches)`, in the
    tower's patch order (frames repeated, then grouped by merge window).
    """
    side = num_grid_per_side
    indices, weights = [], []
    for t, h, w in grid_thw:
        h_idxs = torch.linspace(0, side - 1, h)
        w_idxs = torch.linspace(0, side - 1, w)
        h_floor, w_floor = h_idxs.int(), w_idxs.int()
        h_ceil, w_ceil = (h_floor + 1).clip(max=side - 1), (w_floor + 1).clip(max=side - 1)
        dh, dw = h_idxs - h_floor, w_idxs - w_floor
        index = torch.stack(
            [(a[:, None] * side + b[None]).flatten() for a in (h_floor, h_ceil) for b in (w_floor, w_ceil)]
        )
        weight = torch.stack([(a[:, None] * b[None]).flatten() for a in (1 - dh, dh) for b in (1 - dw, dw)])
        order = (
            torch.arange(h * w)
            .repeat(t)
            .view(t, h // merge_size, merge_size, w // merge_size, merge_size)
            .permute(0, 1, 3, 2, 4)
            .flatten()
        )
        indices.append(index[:, order].long())
        weights.append(weight[:, order])
    return torch.cat(indices, dim=1), torch.cat(weights, dim=1)


@dataclass
class Qwen3VisionGrid(FixedImageGrid):
    """What a Qwen3-VL vision tower derives from one image grid, as its stock forward computes it.

    The learned position embeddings are not cached, only which rows they interpolate, so training and
    weight updates always use the current `pos_embed`.
    """

    pos_embed_indices: Tensor
    pos_embed_weights: Tensor
    position_embeddings: tuple[Tensor, Tensor]
    chunk_lengths: list[int]

    @classmethod
    def _from_grid(cls, visual: "Qwen3VLVisionModel", grid_thw: list[list[int]]) -> Self:
        indices, weights = _bilinear_indices_and_weights(
            grid_thw, visual.num_grid_per_side, visual.spatial_merge_size
        )
        rotary_pos_emb = visual.rot_pos_emb(torch.tensor(grid_thw, device=visual.device))
        rotary_pos_emb = rotary_pos_emb.reshape(indices.shape[1], -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        device = visual.pos_embed.weight.device
        return cls(
            grid_thw=grid_thw,
            device=visual.device,
            dtype=visual.dtype,
            pos_embed_indices=indices.to(device),
            pos_embed_weights=weights.to(device),
            position_embeddings=(emb.cos(), emb.sin()),
            chunk_lengths=[h * w for t, h, w in grid_thw for _ in range(t)],
        )


def qwen3_vision_forward(
    visual: "Qwen3VLVisionModel", pixel_values: Tensor, grid: Qwen3VisionGrid
) -> tuple[Tensor, list[Tensor]]:
    """The merged image embeddings and DeepStack features of `visual(pixel_values, grid_thw)`.

    They are computed from `grid`, the prepared cache of `grid_thw`, so that `torch.export` can trace them.
    """
    weights = grid.pos_embed_weights.to(visual.pos_embed.weight.dtype)
    corners = visual.pos_embed(grid.pos_embed_indices) * weights[..., None]
    pos_embeds = corners[0] + corners[1] + corners[2] + corners[3]
    hidden_states = visual.patch_embed(pixel_values.type(visual.dtype)) + pos_embeds
    deepstack_features = []
    for layer_num, block in enumerate(visual.blocks):
        hidden_states = replay_vision_block(
            block,
            hidden_states,
            grid.position_embeddings,
            grid.chunk_lengths,
            apply_rotary_pos_emb_vision,
            eager_attention_forward,
        )
        if layer_num in visual.deepstack_visual_indexes:
            merger = visual.deepstack_merger_list[visual.deepstack_visual_indexes.index(layer_num)]
            deepstack_features.append(merger(hidden_states))
    return visual.merger(hidden_states), deepstack_features


def qwen3_vl_forward(
    model: "Qwen3VLModel",
    inputs_embeds: Tensor,
    attention_mask: Tensor | None,
    pixel_values: Tensor,
    grid: Qwen3VisionGrid,
    image_mask: Tensor,
    position_ids: Tensor | None = None,
) -> Tensor:
    """`Qwen3VLModel.forward` for images and no cache, on the prepared `grid`, traceable by `torch.export`.

    `image_mask` is True at the image placeholder tokens. `position_ids` are the 3D positions from
    `qwen_image_rope_index`; when None, positions are 0, 1, 2, ... as in the stock forward without
    `input_ids`. Returns the last hidden state.
    """
    image_embeds, deepstack_features = qwen3_vision_forward(model.visual, pixel_values, grid)
    check_input(image_mask.sum() == image_embeds.shape[0], "Image features and image tokens do not match.")
    inputs_embeds = inputs_embeds.masked_scatter(
        image_mask[..., None].expand_as(inputs_embeds), image_embeds.to(inputs_embeds.dtype)
    )
    return model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_pos_masks=image_mask,
        deepstack_visual_embeds=deepstack_features,
    ).last_hidden_state

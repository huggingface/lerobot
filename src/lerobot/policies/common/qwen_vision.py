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

"""Shared pieces for running transformers Qwen-VL vision towers under `torch.export` on a fixed image grid.

The stock towers turn the image grid (`image_grid_thw`) into Python ints on every call, which
`torch.export` cannot trace. When every batch uses the same camera grid, what a tower derives from the
grid can be computed once in eager mode and passed to the policy. `FixedImageGrid` holds that data, with
one subclass per tower:

    grid = Qwen2_5VisionGrid.build(policy.model.visual, batch["image_grid_thw"])

    class Predict(torch.nn.Module):
        def forward(self, batch):
            return policy.predict_action_chunk(batch, vision_grid=grid)

    with torch.no_grad():
        program = torch.export.export(Predict(), (batch,), strict=False)

A policy given a grid runs its vision tower and rope positions from it, in eager mode too, and checks
that the call's `image_grid_thw` is the grid it was built for (a runtime assert in an exported program).
Without one, it runs the stock tower. `replay_vision_block` runs each vision block from its submodules
with the attention chunk lengths as Python ints, because the stock attention reads them back from a
tensor.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import torch
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

from .input_checks import check_input

if TYPE_CHECKING or _transformers_available:
    from transformers import PretrainedConfig
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
else:
    PretrainedConfig = None
    ALL_ATTENTION_FUNCTIONS = None


def is_exporting() -> bool:
    """Whether non-strict `torch.export` is tracing."""
    # torch.compile also reports is_exporting() on some releases, 2.11 among them; non-strict export runs
    # without Dynamo.
    return torch.compiler.is_exporting() and not torch.compiler.is_dynamo_compiling()


@dataclass
class FixedImageGrid:
    """What a vision tower derives from one image grid, built in eager mode by `build`.

    Subclasses add the tower's metadata and implement `_from_grid`. Only weight-independent tensors are
    kept, built outside inference mode, so a grid never holds stale weights and an export can trace a
    trainable tower.
    """

    grid_thw: list[list[int]]
    device: torch.device
    dtype: torch.dtype

    @classmethod
    def _from_grid(cls, visual: nn.Module, grid_thw: list[list[int]]) -> Self:
        raise NotImplementedError

    @classmethod
    def build(cls, visual: nn.Module, grid_thw: Tensor) -> Self:
        """Compute the metadata of `grid_thw` for `visual`; this reads the grid on the host."""
        # Inference tensors cannot be saved for backward when the export traces a trainable tower.
        with torch.inference_mode(False), torch.no_grad():
            return cls._from_grid(visual, grid_thw.tolist())

    def check(self, visual: nn.Module, grid_thw: Tensor) -> None:
        """Refuse a call whose vision tower or image grid is not the one this grid was built for.

        The image count, device and dtype are compared in Python, the grid values with `check_input`.
        """
        if len(self.grid_thw) != grid_thw.shape[0]:
            raise ValueError(
                f"The vision grid was built for {len(self.grid_thw)} images, but the call has {grid_thw.shape[0]}."
            )
        if (self.device, self.dtype) != (visual.device, visual.dtype):
            raise ValueError(
                f"The vision grid was built for {self.device} {self.dtype}, but the vision tower is on "
                f"{visual.device} {visual.dtype}."
            )
        check_input(
            grid_thw == grid_thw.new_tensor(self.grid_thw),
            "The image grid differs from the one the vision grid was built for.",
        )


def check_visible_images(input_ids: Tensor, attention_mask: Tensor, config: PretrainedConfig) -> None:
    """Refuse an attention mask that hides more than padding at the ends of a row, or hides a vision token.

    `qwen_image_rope_index` keeps every image on its own grid, while eager rope positions give the grids to the
    visible images only, join image runs that a hidden token separated, and fail on a row with no visible token.
    The processors of the policies that call this hide only left or right padding, where the two agree. So each
    row's visible tokens must be one run, and every image, vision start and vision end token must be visible.
    `config` is the Qwen-VL config with the token ids. A refusal is a `ValueError` in eager mode and a runtime
    assert in an exported program.
    """
    valid = attention_mask.bool()
    run_starts = valid & ~torch.cat([torch.zeros_like(valid[:, :1]), valid[:, :-1]], dim=1)
    vision_token_ids = [config.image_token_id, config.vision_start_token_id, config.vision_end_token_id]
    hidden_vision = ~valid & torch.isin(input_ids, input_ids.new_tensor(vision_token_ids))
    check_input(
        (run_starts.sum(dim=1) == 1) & ~hidden_vision.any(dim=1),
        "To run on a fixed image grid, the attention mask must hide only padding at the ends of each row, and no "
        "image, vision start or vision end token: eager rope positions differ or fail otherwise.",
    )


def chunked_vision_attention(
    attn: nn.Module,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    lengths: list[int],
    apply_rotary_pos_emb_vision: Callable,
    eager_attention_forward: Callable,
) -> Tensor:
    """A stock Qwen-VL vision attention, run chunk by chunk over `lengths` (Python ints).

    The stock attention takes the chunk boundaries from the `cu_seqlens` tensor; this calls the configured
    attention function once per chunk instead. The last two arguments come from the tower's transformers
    modeling module.
    """
    seq_length = hidden_states.shape[0]
    query, key, value = (
        attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)
    query, key, value = (state.transpose(0, 1).unsqueeze(0) for state in (query, key, value))
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        attn.config._attn_implementation, eager_attention_forward
    )
    outputs = [
        attention_interface(
            attn, q, k, v, attention_mask=None, scaling=attn.scaling, dropout=0.0, is_causal=False
        )[0]
        for q, k, v in zip(
            *(torch.split(state, lengths, dim=2) for state in (query, key, value)), strict=True
        )
    ]
    return attn.proj(torch.cat(outputs, dim=1).reshape(seq_length, -1).contiguous())


def replay_vision_block(
    block: nn.Module,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    lengths: list[int],
    apply_rotary_pos_emb_vision: Callable,
    eager_attention_forward: Callable,
    attention: Callable | None = None,
) -> Tensor:
    """Run a Qwen-VL vision block with its attention split into chunks of `lengths` (Python ints).

    The attention is `chunked_vision_attention`, or `attention` when given, called with the same first four
    arguments.
    """
    attn, normed = block.attn, block.norm1(hidden_states)
    if attention is None:
        attended = chunked_vision_attention(
            attn, normed, position_embeddings, lengths, apply_rotary_pos_emb_vision, eager_attention_forward
        )
    else:
        attended = attention(attn, normed, position_embeddings, lengths)
    hidden_states = hidden_states + attended
    return hidden_states + block.mlp(block.norm2(hidden_states))


def qwen_image_rope_index(
    input_ids: Tensor,
    attention_mask: Tensor,
    image_grid_thw: list[list[int]],
    spatial_merge_size: int,
    image_token_id: int,
    vision_start_token_id: int | None = None,
) -> Tensor:
    """The 3D rope positions of Qwen-VL `get_rope_index` for still images, with tensor ops only.

    Text tokens count up by one. An image's tokens get `(start, start + row, start + col)` and move the
    next start by `max(rows, cols)`. Padding gets 0. `image_grid_thw` lists the
    (t, h, w) grid of every image in `input_ids`, in order, as Python ints, and every image keeps its own
    grid, hidden by `attention_mask` or not. That equals eager rope positions when only padding is hidden,
    the masks `check_visible_images` allows. It also equals eager code that drops the grids of hidden
    images, as long as the visible images stay apart: each after its own vision start token, or one visible
    image per sample, which such a caller must check. Each image must be one run of tokens inside one
    sample, right after a `vision_start_token_id` token when that is given, as the processors write images:
    eager rope positions place other layouts differently or fail on them. Only still images (t == 1) are
    supported, as the image processors produce them; video grids raise.
    """
    if any(t != 1 for t, _, _ in image_grid_thw):
        raise ValueError(f"qwen_image_rope_index supports still images only (t == 1), got {image_grid_thw}.")
    device = input_ids.device
    valid = attention_mask.bool()
    is_image = input_ids == image_token_id

    # Per image token, in batch order: its (t, h, w) offset inside the image, and how far the image
    # moves the text position, put on the image's last token so earlier tokens keep the image's start.
    offsets, advances, image_ends = [], [], [0]
    for _, h, w in image_grid_thw:
        h, w = h // spatial_merge_size, w // spatial_merge_size
        rows = torch.arange(h, device=device).repeat_interleave(w)
        cols = torch.arange(w, device=device).repeat(h)
        offsets.append(torch.stack([torch.zeros_like(rows), rows, cols]))
        advance = torch.zeros(h * w, dtype=torch.long, device=device)
        advance[-1] = max(h, w)
        advances.append(advance)
        image_ends.append(image_ends[-1] + h * w)
    offsets = torch.cat(offsets, dim=1)
    advances = torch.cat(advances)
    # Before masked_scatter: with too few source values it fails on CUDA with a device-side assert, which
    # breaks the CUDA context.
    check_input(is_image.sum() == advances.numel(), "The image tokens do not match the image grid.")
    # The images take the image tokens in batch order. An image token that does not start an image must follow
    # another image token in its row.
    tokens_before = is_image.flatten().cumsum(0).view_as(is_image) - 1
    starts_image = is_image & torch.isin(tokens_before, input_ids.new_tensor(image_ends[:-1]))
    previous = torch.cat([input_ids.new_full((input_ids.shape[0], 1), -1), input_ids[:, :-1]], dim=1)
    check_input(
        ~is_image | starts_image | (previous == image_token_id),
        "Each image of the image grid must be one run of image tokens inside one sample.",
    )
    if vision_start_token_id is not None:
        check_input(
            ~starts_image | (previous == vision_start_token_id),
            "Each image must follow a vision start token.",
        )

    steps = torch.ones_like(input_ids).masked_scatter(is_image, advances) * valid
    start = steps.cumsum(dim=1) - steps
    position_ids = torch.stack([start, start, start])
    image_offsets = torch.zeros_like(position_ids).masked_scatter(is_image.expand_as(position_ids), offsets)
    return (position_ids + image_offsets).masked_fill(~valid, 0)

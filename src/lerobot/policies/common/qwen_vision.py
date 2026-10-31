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
grid can be computed once in eager mode and reused by the exported program. `FixedImageGrid` holds that
cache, with one subclass per tower.

The cache is prepared explicitly, by one eager call on the export inputs inside `prepare_image_grid()`:

    with torch.no_grad():
        with prepare_image_grid():
            policy.predict_action_chunk(batch)
        program = torch.export.export(module, args, strict=False)

Outside that block the policies run their stock code, and the cache helper neither reads the grid nor
caches it. Under export, `replay_vision_block` runs each vision block from its submodules with the
attention chunk lengths as Python ints, because the stock attention reads them back from a tensor.

Under export the cached grid is checked by a runtime assert in the program (see `check_input`).
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
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

_preparing_image_grid: ContextVar[bool] = ContextVar("preparing_image_grid", default=False)


@contextmanager
def prepare_image_grid() -> Iterator[None]:
    """Inside this block, eager calls cache the image grid metadata that `torch.export` needs.

    Run the policy once on the export inputs, on the export device and dtype, inside the block; then export.
    """
    token = _preparing_image_grid.set(True)
    try:
        yield
    finally:
        _preparing_image_grid.reset(token)


def is_exporting() -> bool:
    """Whether non-strict `torch.export` is tracing."""
    # torch.compile on PyTorch 2.11 also reports is_exporting(); non-strict export runs without Dynamo.
    return torch.compiler.is_exporting() and not torch.compiler.is_dynamo_compiling()


def is_preparing_image_grid() -> bool:
    """Whether an eager call runs inside `prepare_image_grid()`."""
    return _preparing_image_grid.get() and not is_exporting()


@dataclass
class FixedImageGrid:
    """What a vision tower derives from one image grid, built in eager mode.

    Subclasses add the tower's metadata and implement `_from_grid`. Only weight-independent tensors are
    cached, built outside inference mode, so the cache never holds stale weights and an export can trace a
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
        # Preparation usually runs a prediction in inference mode, but the cache must hold normal tensors:
        # inference tensors cannot be saved for backward when the export traces a trainable tower.
        with torch.inference_mode(False), torch.no_grad():
            return cls._from_grid(visual, grid_thw.tolist())

    @classmethod
    def track(cls, cache: Self | None, visual: nn.Module, grid_thw: Tensor | None) -> Self | None:
        """Return the cache a policy keeps for `grid_thw`.

        Inside `prepare_image_grid()` an eager call builds it, and rebuilds it whenever the grid, the device or
        the dtype changed. Under `torch.export` the grid cannot be read, so the prepared cache must fit: the
        image count, device and dtype are checked while tracing, and the grid values by a runtime assert in
        the program. Outside preparation and export the cache is returned as it is and the grid is not read.
        A call without images (`grid_thw` None) has no grid: None. A policy stores the returned cache only when
        its call succeeds, so a call without images, or one that fails, keeps the prepared cache.
        """
        if grid_thw is None:
            return None
        if is_exporting():
            if cache is None:
                problem = "no image grid was prepared"
            elif len(cache.grid_thw) != grid_thw.shape[0]:
                problem = (
                    f"the prepared grid has {len(cache.grid_thw)} images but the export inputs have "
                    f"{grid_thw.shape[0]} (another batch size or camera count)"
                )
            elif (cache.device, cache.dtype) != (visual.device, visual.dtype):
                problem = (
                    f"the grid was prepared for {cache.device} {cache.dtype}, but the tower is now on "
                    f"{visual.device} {visual.dtype}"
                )
            else:
                check_input(
                    grid_thw == grid_thw.new_tensor(cache.grid_thw),
                    "The image grid differs from the one prepared for torch.export.",
                )
                return cache
            raise RuntimeError(
                f"Cannot export on a fixed image grid: {problem}. Run the policy once on the export inputs "
                "inside `prepare_image_grid()` before torch.export."
            )
        if is_preparing_image_grid() and (
            cache is None
            or (cache.device, cache.dtype) != (visual.device, visual.dtype)
            or cache.grid_thw != grid_thw.tolist()
        ):
            return cls.build(visual, grid_thw)
        return cache


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

    The stock attention splits by `cu_seqlens`, which it reads back from a tensor; this calls the configured
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
) -> Tensor:
    """Under `torch.export`, what a Qwen-VL vision block computes when its attention runs per chunk of `lengths`.

    The attention is `chunked_vision_attention`, unless the block's attention module has its own
    `replay_on_lengths(hidden_states, position_embeddings, lengths)` (WALL-X's packed varlen attention does).
    """
    attn, normed = block.attn, block.norm1(hidden_states)
    replay = getattr(attn, "replay_on_lengths", None)
    if replay is None:
        attended = chunked_vision_attention(
            attn, normed, position_embeddings, lengths, apply_rotary_pos_emb_vision, eager_attention_forward
        )
    else:
        attended = replay(normed, position_embeddings, lengths)
    hidden_states = hidden_states + attended
    return hidden_states + block.mlp(block.norm2(hidden_states))


def qwen_image_rope_index(
    input_ids: Tensor,
    attention_mask: Tensor,
    image_grid_thw: list[list[int]],
    spatial_merge_size: int,
    image_token_id: int,
    image_temporal_scale: int = 1,
) -> Tensor:
    """The 3D rope positions of Qwen-VL `get_rope_index` for still images, with tensor ops only.

    Text tokens count up by one. An image's tokens get `(start * image_temporal_scale, start + row,
    start + col)` and move the next start by `max(rows, cols)`. Padding gets 0. `image_grid_thw` lists the
    (t, h, w) grid of every image in `input_ids`, in order, as Python ints, and every image keeps its own
    grid, hidden by `attention_mask` or not. That equals eager rope positions when only padding is hidden, and
    when the eager code drops the grids of hidden images; `check_visible_images` refuses the other masks. Only
    still images (t == 1) are supported, as the image processors produce them; video grids raise.

    For Qwen2.5-VL, transformers 5.4.0 to 5.5.3 multiply the temporal position of an image's start by
    `tokens_per_second` (fixed upstream in 5.5.4); `image_temporal_scale` reproduces that, and its default of
    1 matches transformers 4.x and 5.5.4 and later.
    """
    if any(t != 1 for t, _, _ in image_grid_thw):
        raise ValueError(f"qwen_image_rope_index supports still images only (t == 1), got {image_grid_thw}.")
    device = input_ids.device
    valid = attention_mask.bool()
    is_image = input_ids == image_token_id

    # Per image token, in batch order: its (t, h, w) offset inside the image, and how far the image
    # moves the text position, put on the image's last token so earlier tokens keep the image's start.
    offsets, advances = [], []
    for _, h, w in image_grid_thw:
        h, w = h // spatial_merge_size, w // spatial_merge_size
        rows = torch.arange(h, device=device).repeat_interleave(w)
        cols = torch.arange(w, device=device).repeat(h)
        offsets.append(torch.stack([torch.zeros_like(rows), rows, cols]))
        advance = torch.zeros(h * w, dtype=torch.long, device=device)
        advance[-1] = max(h, w)
        advances.append(advance)
    offsets = torch.cat(offsets, dim=1)
    advances = torch.cat(advances)

    steps = torch.ones_like(input_ids).masked_scatter(is_image, advances) * valid
    start = steps.cumsum(dim=1) - steps
    temporal = torch.where(is_image, start * image_temporal_scale, start)
    position_ids = torch.stack([temporal, start, start])
    image_offsets = torch.zeros_like(position_ids).masked_scatter(is_image.expand_as(position_ids), offsets)
    return (position_ids + image_offsets).masked_fill(~valid, 0)

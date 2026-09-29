# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Block-causal attention for the first-party DINO-video runtime.

Mask semantics (extracted from the published teacher's behaviour and kept
byte-for-byte compatible):

- The packed clip is a sequence of per-frame blocks. Every frame block holds
  ``1 + n_storage_tokens + Hp * Wp`` tokens in the order
  ``[cls, storage..., patches...]`` (row-major over the patch grid).
- A query in frame ``f`` attends to **every** token (cls, storage and patch)
  of frames ``0..f`` inclusive, and to nothing from later frames:
  ``allow[i, j] = block_id[j] <= block_id[i]`` where ``block_id`` is the
  frame index carried by :class:`~lerobot.policies.lingbot_vla_v2.teachers.
  dino_video.backbone.PackedVideoTokens`.
- Within a single frame block attention is fully bidirectional; the cls and
  storage prefix tokens are ordinary members of their frame block (they see
  exactly the same frames the frame's patches see). There are no fully masked
  query rows because every query at least sees its own frame.

Two backends share this single entry point:

- ``"sdpa"`` (default): explicit additive mask +
  :func:`torch.nn.functional.scaled_dot_product_attention`. Zero extra
  dependencies; this is the correctness reference.
- ``"flex"`` (optional): ``torch.nn.attention.flex_attention`` when the local
  torch/GPU support check passes. It encodes the identical predicate, and it
  is only selectable after :func:`verify_flex_backend` confirms it matches the
  SDPA reference; it is never an install requirement.

No xformers / flash-attn / upstream attention wrappers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from .backbone import PackedVideoTokens

AttentionBackend = Literal["sdpa", "flex"]

_ADDITIVE_MASK_CACHE: dict[tuple, torch.Tensor] = {}
_FLEX_BLOCK_MASK_CACHE: dict[tuple, object] = {}


def _frame_signature(layout: PackedVideoTokens) -> tuple[int, ...]:
    """Run-length frame sizes from ``layout.block_id``.

    The packed layout is contiguous per frame, so ``block_id`` must start at 0
    and be non-decreasing; anything else means the layout is malformed.
    """
    block_id = layout.block_id.to(device="cpu", dtype=torch.long)
    if block_id.numel() == 0:
        raise ValueError("packed layout is empty; cannot build an attention mask.")
    if int(block_id[0]) != 0 or bool((block_id[1:] < block_id[:-1]).any()):
        raise ValueError("layout.block_id must start at 0 and be non-decreasing.")
    counts = torch.bincount(block_id)
    return tuple(int(count) for count in counts if count > 0)


def _block_id_tensor(signature: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.cat(
        [torch.full((size,), index, dtype=torch.long, device=device) for index, size in enumerate(signature)]
    )


def _additive_mask(signature: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Additive frame-causal mask (0 where allowed, -inf elsewhere)."""
    key = (signature, device.type, device.index if device.index is not None else 0, dtype)
    if key not in _ADDITIVE_MASK_CACHE:
        block_id = _block_id_tensor(signature, device)
        allow = block_id[None, :] <= block_id[:, None]
        mask = torch.zeros(len(block_id), len(block_id), dtype=dtype, device=device)
        mask.masked_fill_(~allow, float("-inf"))
        _ADDITIVE_MASK_CACHE[key] = mask
    return _ADDITIVE_MASK_CACHE[key]


def flex_available() -> bool:
    """Whether ``torch.nn.attention.flex_attention`` is importable here."""
    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401

        return True
    except Exception:
        return False


def _flex_block_mask(layout: PackedVideoTokens, *, batch: int, device: torch.device) -> object:
    """Cached ``BlockMask`` encoding the same predicate as the additive mask."""
    from torch.nn.attention.flex_attention import create_block_mask

    signature = _frame_signature(layout)
    key = (signature, batch, device.type, device.index if device.index is not None else 0)
    if key not in _FLEX_BLOCK_MASK_CACHE:
        block_id = _block_id_tensor(signature, device)
        total = int(block_id.numel())

        def mask_mod(_b, _h, q_idx, kv_idx):
            return block_id[q_idx] <= block_id[kv_idx]

        _FLEX_BLOCK_MASK_CACHE[key] = create_block_mask(
            mask_mod, B=batch, H=None, Q_LEN=total, KV_LEN=total, device=device, BLOCK_SIZE=128
        )
    return _FLEX_BLOCK_MASK_CACHE[key]


def block_causal_attention(
    q: torch.Tensor,  # [B, heads, tokens, head_dim]
    k: torch.Tensor,
    v: torch.Tensor,
    layout: PackedVideoTokens,
    *,
    backend: AttentionBackend = "sdpa",
) -> torch.Tensor:
    """Attend within the frame blocks permitted by ``layout.block_id``."""
    num_tokens = q.shape[-2]
    if num_tokens != layout.block_id.numel():
        raise ValueError(
            f"q carries {num_tokens} tokens but the packed layout has {layout.block_id.numel()}."
        )
    if backend == "sdpa":
        mask = _additive_mask(_frame_signature(layout), q.device, q.dtype)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if backend == "flex":
        from torch.nn.attention.flex_attention import flex_attention

        block_mask = _flex_block_mask(layout, batch=q.shape[0], device=q.device)
        return flex_attention(q, k, v, block_mask=block_mask)
    raise ValueError(f"unknown attention backend {backend!r}; expected 'sdpa' or 'flex'.")


def verify_flex_backend(
    *,
    num_heads: int,
    head_dim: int,
    device: torch.device | str,
    batch: int = 2,
    frames: int = 3,
    tokens_per_frame: int = 32,
    tolerance: float = 2e-2,
) -> None:
    """Gate the flex backend on a randomized SDPA parity probe.

    Raises ``RuntimeError`` when flex attention is unavailable or disagrees
    with the SDPA reference beyond ``tolerance`` on the probe; this is what
    keeps ``backend="flex"`` an optional accelerator instead of a silent
    numerical change.
    """
    from .backbone import pack_video_tokens

    if not flex_available():
        raise RuntimeError(
            "attention_backend='flex' requires torch.nn.attention.flex_attention, which is not "
            "importable in this environment; keep the default 'sdpa' backend."
        )
    target = torch.device(device)
    if target.type != "cuda":
        raise RuntimeError("attention_backend='flex' requires a CUDA device.")
    torch.manual_seed(0)
    embed_dim = num_heads * head_dim
    patches_per_frame = max(tokens_per_frame - 2, 1)  # + 1 cls + 1 storage token
    patch_tokens = torch.randn(1, frames, patches_per_frame, embed_dim)
    cls_token = torch.randn(1, 1, embed_dim)
    storage_tokens = torch.randn(1, 1, embed_dim)
    layout = pack_video_tokens(patch_tokens, cls_token, storage_tokens, grid_size=(1, patches_per_frame))
    num_tokens = layout.block_id.numel()
    q = torch.randn(batch, num_heads, num_tokens, head_dim, device=target, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sdpa_output = block_causal_attention(q, k, v, layout, backend="sdpa")
    flex_output = block_causal_attention(q, k, v, layout, backend="flex")
    if not torch.allclose(flex_output.float(), sdpa_output.float(), atol=tolerance, rtol=tolerance):
        max_diff = (flex_output.float() - sdpa_output.float()).abs().max().item()
        raise RuntimeError(
            "flex attention backend does not match the SDPA reference "
            f"(max abs diff {max_diff:.3e} > tolerance {tolerance:.1e}); keep 'sdpa'."
        )

# Vendored from runwayml/praxis-1-oss at commit 8dad1a4, original path praxis/action_expert.py.
# Import paths are rewritten to the LeRobot package; the class PraxisPolicy of praxis.model
# is renamed PraxisModel here. Keep this file in sync with the upstream file.
"""Action expert: a small transformer that predicts the action velocity.

The expert has its own weights. It reads hidden states of the video backbone
encoder ("taps") and the encoder input sequence. It never writes back into the
backbone, so the video stream has no path from the action tokens.

The layers alternate between two types:

* Even layers are cross-attention layers. The queries are the expert tokens.
  The keys and values are the global tokens and the frame tokens of one tap.
* Odd layers are self-attention layers among the expert tokens.

Expert token sequence (Q)::

    [ proprio (1 per chunk) | future tokens (F per chunk) | actions (T) ]

The output head reads only the action tokens.

Cross-attention key/value sequence (KV)::

    [ global tokens | clean frame tokens | noised frame tokens ]

* The global tokens are the registers, the anchor frame tokens and the text
  tokens of the encoder INPUT sequence (the block 0 input), in
  ``global_order``. They enter through ``text_proj``.
* The frame tokens are encoder block outputs at the tap blocks. Each
  cross-attention layer reads one tap. The tap goes through an RMS norm with a
  learned scale (``tap_norm_scale``) and ``tap_proj``.

Trunk-rope layout (``rope={"kind": "trunk", ...}``, the checkpoint):

* The expert has the ``future_chunk_embed`` table and reads it from row 0.
* The queries and keys turn by the trunk's own rotary position ids, before
  the q/k RMS norm (see ``trunk_query_rope_ids``).
* The global tokens are ordered registers, anchor, text.
* The window holds K' noised chunks and K' - 1 clean chunks.
* The attention can run as compiled flex attention with a block mask
  (``attention="flex_dynamic"``).
* ``head_plus_zero`` adds ``+0.0`` to the head output (``-0`` becomes ``+0``).

Numerics: the weights and activations are bfloat16. The norms, the AdaLN
scale and the timestep features run in float32 and cast back. The op order
follows the original implementation exactly, so the output is bitwise equal for the
same weights and inputs.

``run_expert_on_taps_k`` runs the expert on the taps of one chunk-causal
encoder forward.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, FlexKernelOptions

from lerobot.policies.praxis.praxis_core.layers import (
    RotaryEmbedding,
    _compiled_flex_attention_dynamic,
    apply_rope,
)

__all__ = [
    "ATTENTION_MODES",
    "FLEX_DYNAMIC_KERNEL_OPTIONS",
    "GLOBAL_KINDS",
    "GLOBAL_ORDER_TRUNK",
    "ActionExpert",
    "ActionExpertBlock",
    "build_trunk_rope_masks",
    "noised_frame_offset",
    "run_expert_on_taps_k",
    "tap_blocks_for",
    "trunk_mask_mods",
    "trunk_query_rope_ids",
]

# Global token kinds and their order in the expert KV.
GLOBAL_KINDS = frozenset({"register", "text", "anchor"})
GLOBAL_ORDER_TRUNK = ("register", "anchor", "text")

ATTENTION_MODES = ("sdpa", "flex_dynamic")

# Default template parameters of the compiled flex attention kernel in the
# "flex_dynamic" mode. They are pinned, so the kernel compiles without
# autotuning.
FLEX_DYNAMIC_KERNEL_OPTIONS: dict[str, int] = {
    "fwd_BLOCK_M": 32,
    "fwd_BLOCK_N": 128,
    "num_warps": 4,
    "num_stages": 3,
}

# Mask tags are padded to a multiple of this (the flex block size), so a mask
# read past the sequence end stays in bounds.
_FLEX_TAG_PAD = 128


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """RMS norm over the last dim, without a weight.

    The square runs in the input dtype. The mean, the scale and the product run
    in float32. The result casts back to the input dtype.
    """
    mean_sq = torch.mean(x**2, dim=-1, keepdim=True, dtype=torch.float32)
    scale = torch.rsqrt(mean_sq + eps)
    output = x.to(torch.float32) * scale
    return output.to(x.dtype)


def geglu(x: torch.Tensor) -> torch.Tensor:
    """Gated GELU: ``gelu_tanh(first half) * second half``."""
    value, gate = torch.chunk(x, 2, dim=-1)
    return F.gelu(value, approximate="tanh") * gate


class TimestepEmbedder(nn.Module):
    """Sinusoidal features of a flow time, then a two-layer MLP.

    The features are ``[cos(t * f), sin(t * f)]`` with ``f`` in float32. The
    features cast to the weight dtype before the MLP.
    """

    def __init__(self, dim: int, frequency_dim: int = 256, max_period: float = 10_000) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(nn.Linear(frequency_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in (0, 2):
            layer = self.mlp[i]
            assert isinstance(layer, nn.Linear)
            nn.init.normal_(layer.weight, std=0.02)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``t (...)`` -> ``(..., dim)``."""
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32)
            / half
        ).to(dtype=t.dtype)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        first = self.mlp[0]
        assert isinstance(first, nn.Linear)
        features = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype=first.weight.dtype)
        return self.mlp(features)


class AdaLN(nn.Module):
    """Scale-only adaptive RMS norm: ``(1 + proj(silu(cond))) * rms_norm(x)``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``tokens (B, S, D)``, ``cond (B, S, D)`` -> ``(B, S, D)``."""
        scale = self.proj(F.silu(cond))
        return ((1 + scale.float()) * rms_norm(tokens)).to(dtype=tokens.dtype)


# ---------------------------------------------------------------------------
# Layout: tap blocks, timeline offset, query chunks
# ---------------------------------------------------------------------------


def tap_blocks_for(num_blocks: int, num_taps: int) -> tuple[int, ...]:
    """Encoder blocks that feed the cross-attention layers.

    The taps spread evenly over the depth. The last tap is the last block. For
    14 blocks and 8 taps the result is ``(0, 2, 4, 6, 7, 9, 11, 13)``.
    """
    if not 0 < num_taps <= num_blocks:
        raise ValueError(f"bad tap count: {num_taps} taps for {num_blocks} blocks")
    stride = num_blocks / num_taps
    return tuple(sorted({min(num_blocks - 1, int((i + 1) * stride) - 1) for i in range(num_taps)}))


def noised_frame_offset(num_noised_frames: int, num_clean_frames: int, chunk_size: int) -> int:
    """Timeline position of noised frame 0.

    * No clean frames: 0.
    * One noised frame: ``num_clean_frames``.
    * One noised chunk (the serve window): ``num_clean_frames``.
    * Noised frames = clean frames + 1: 0.
    * Noised frames = clean frames + one chunk (the training window): 0. Clean
      chunk k and noised chunk k cover the same frames.
    """
    if num_clean_frames == 0:
        return 0
    if num_noised_frames == 1:
        return num_clean_frames
    if chunk_size > 1 and num_noised_frames == chunk_size:
        return num_clean_frames
    if num_noised_frames == num_clean_frames + 1:
        return 0
    if chunk_size > 1 and num_noised_frames == num_clean_frames + chunk_size:
        return 0
    raise ValueError(
        f"unsupported layout: noised={num_noised_frames}, clean={num_clean_frames}, chunk={chunk_size}"
    )


def _chunk_of_query(
    num_action_tokens: int,
    num_future_per_chunk: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Chunk index of each expert token, in the order of the Q layout."""
    num_chunks = max(1, num_action_tokens // chunk_size)
    chunks = torch.arange(num_chunks, device=device)
    proprio = chunks
    future = chunks.repeat_interleave(num_future_per_chunk)
    actions = torch.arange(num_action_tokens, device=device) // chunk_size
    return torch.cat([proprio, future, actions])


# ---------------------------------------------------------------------------
# Trunk-rope layout: query rope ids, flex masks
# ---------------------------------------------------------------------------


def trunk_query_rope_ids(
    kv_rope_ids: torch.Tensor,
    *,
    num_global_tokens: int,
    num_clean_frames: int,
    num_noised_frames: int,
    tokens_per_frame: int,
    num_action_tokens: int,
    num_future_per_chunk: int,
    chunk_size: int,
) -> torch.Tensor:
    """``(B, Q)`` trunk rope ids of the expert tokens.

    ``kv_rope_ids (B, G + frames * tokens_per_frame)`` are the trunk rope ids
    of the expert KV in the order ``[global | clean frames | noised frames]``,
    frame-major. The id of a frame is its smallest token id (the band start,
    the ``h = 0, w = 0`` token).

    * action token i: the id of noised frame i
    * proprio and future tokens of chunk k: the id of noised frame
      ``k * chunk_size``

    There is one action token per noised frame.
    """
    b = kv_rope_ids.shape[0]
    total_frames = num_clean_frames + num_noised_frames
    video = kv_rope_ids[:, num_global_tokens:]
    if video.shape[1] != total_frames * tokens_per_frame:
        raise ValueError(
            f"{video.shape[1]} video rope ids for {total_frames} frames of {tokens_per_frame} tokens"
        )
    if num_action_tokens != num_noised_frames:
        raise ValueError(
            f"one action token per noised frame: {num_action_tokens} tokens, "
            f"{num_noised_frames} noised frames"
        )
    frame_ids = video.view(b, total_frames, tokens_per_frame).amin(dim=-1)
    noised_ids = frame_ids[:, num_clean_frames:]
    q_chunk = _chunk_of_query(num_action_tokens, num_future_per_chunk, chunk_size, kv_rope_ids.device)
    n_prefix = q_chunk.numel() - num_action_tokens
    q_frame = torch.cat(
        [
            (q_chunk[:n_prefix] * chunk_size).clamp(min=0),
            torch.arange(num_action_tokens, device=kv_rope_ids.device),
        ]
    )
    return noised_ids.gather(1, q_frame[None].expand(b, -1))


MaskMod = Callable[[object, object, torch.Tensor, torch.Tensor], torch.Tensor]


def trunk_mask_mods(
    *,
    num_action_tokens: int,
    num_future_per_chunk: int,
    chunk_size: int,
    num_clean_frames: int,
    num_noised_frames: int,
    tokens_per_frame: int,
    window_size: int,
    num_global_tokens: int,
    device: torch.device,
    include_noised: bool = True,
) -> tuple[MaskMod, MaskMod, int, int]:
    """Flex ``mask_mod`` functions ``(cross, self, Q, KV)`` of one expert row.

    The rules:

    * A query can read all global tokens.
    * A query can read a clean frame strictly before the start of its chunk
      and at most ``window_size`` frames before it.
    * A query can read the noised frames of its own chunk only
      (``include_noised=False`` removes them).
    * Self attention: one block per chunk.

    They are read from per-token int32 tags. The tags are padded to whole
    flex blocks: padded queries get the row tag -1 and padded keys -2, so a
    padded position never matches. The expressions follow the original implementation op
    by op, so the compiled kernel is the same.
    """
    f = num_future_per_chunk
    cs = chunk_size
    n_c = num_clean_frames
    tpf = tokens_per_frame
    k = max(1, num_action_tokens // cs)
    q_len = k * (1 + f) + num_action_tokens
    n_g = num_global_tokens
    kv_len = n_g + (n_c + num_noised_frames) * tpf
    off = noised_frame_offset(num_noised_frames, n_c, cs)

    def tag(valid: torch.Tensor, value: torch.Tensor, fill: int) -> torch.Tensor:
        return torch.where(valid, value, fill).to(torch.int32)

    qi = torch.arange(q_len + -q_len % _FLEX_TAG_PAD, device=device)
    qv = qi < q_len
    q_row = torch.zeros_like(qi)
    qp = qi
    q_chunk = torch.where(
        qp < k,
        qp,
        torch.where(qp < k * (1 + f), (qp - k) // max(f, 1), (qp - k * (1 + f)) // cs),
    )
    qr, qch = tag(qv, q_row, -1), tag(qv, q_chunk, -1)
    qst = tag(qv, q_chunk * cs + off, -1)
    qw = tag(qv, torch.full_like(qi, window_size), -1)

    ki = torch.arange(kv_len + -kv_len % _FLEX_TAG_PAD, device=device)
    kv, glob = ki < kv_len, ki < n_g
    k_row = torch.zeros_like(ki)
    sp = ki - n_g
    frame = sp // tpf
    clean = frame < n_c
    noised_f = frame - n_c
    kr = tag(kv, k_row, -2)
    kk = tag(kv, torch.where(glob, 0, torch.where(clean, 1, 2)), -1)
    kf = tag(kv, torch.where(glob | clean, frame * ~glob, noised_f), -1)
    kc = tag(kv, torch.where(glob | clean, -1, noised_f // cs), -1)

    def cross(b: object, h: object, qi: torch.Tensor, ki: torch.Tensor) -> torch.Tensor:
        k_kind, k_f, cs0 = kk[ki], kf[ki], qst[qi]
        clean = (k_kind == 1) & (k_f < cs0) & (k_f >= cs0 - qw[qi])
        rule = (k_kind == 0) | clean
        if include_noised:
            rule = rule | ((k_kind == 2) & (kc[ki] == qch[qi]))
        return (qr[qi] == kr[ki]) & rule

    def self_(b: object, h: object, qi: torch.Tensor, ki: torch.Tensor) -> torch.Tensor:
        return (qr[qi] == qr[ki]) & (qch[qi] == qch[ki])

    return cross, self_, q_len, kv_len


def _block_mask_from_mod(
    mod: MaskMod, q_len: int, kv_len: int, device: torch.device, block: int = 128
) -> BlockMask:
    """Flex ``BlockMask`` of ``mod`` with ``block x block`` tiles.

    ``mod`` is evaluated once on the dense ``(Q, KV)`` grid. A tile is full
    when all its entries are allowed and partial when some are. Full and
    partial tiles are kept apart, so the kernel skips the mask on full tiles.
    """
    dense = mod(
        None,
        None,
        torch.arange(q_len, device=device)[:, None],
        torch.arange(kv_len, device=device)[None],
    )
    padded = F.pad(dense[None, None], (0, -kv_len % block, 0, -q_len % block))
    _, _, q_pad, kv_pad = padded.shape
    tiles = padded.view(1, 1, q_pad // block, block, kv_pad // block, block)
    sums = tiles.permute(0, 1, 2, 4, 3, 5).sum(dim=[-2, -1])
    full = (sums == block * block).to(torch.int8)
    partial = ((sums > 0) & (sums < block * block)).to(torch.int8)

    def ordered(tile_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tile_mask = tile_mask.to(torch.int32)
        count = tile_mask.sum(dim=-1).to(torch.int32)
        index = torch.argsort(tile_mask, dim=-1, descending=True, stable=True)
        return count.contiguous(), index.to(torch.int32).contiguous()

    p_count, p_index = ordered(partial)
    f_count, f_index = ordered(full)
    return BlockMask.from_kv_blocks(
        p_count,
        p_index,
        f_count,
        f_index,
        BLOCK_SIZE=(block, block),
        mask_mod=mod,
        seq_lengths=(q_len, kv_len),
    )


def build_trunk_rope_masks(
    *,
    num_action_tokens: int,
    num_future_per_chunk: int,
    chunk_size: int,
    num_clean_frames: int,
    num_noised_frames: int,
    tokens_per_frame: int,
    window_size: int,
    num_global_tokens: int,
    device: torch.device,
    include_noised: bool = True,
    block: bool = True,
) -> tuple[torch.Tensor | BlockMask, torch.Tensor | BlockMask]:
    """``(cross, self)`` masks of one expert row in the trunk-rope layout.

    ``block=True`` gives flex ``BlockMask`` objects (128 x 128 tiles).
    ``block=False`` gives the dense ``(Q, KV)`` and ``(Q, Q)`` bool masks.
    """
    cross, self_, q_len, kv_len = trunk_mask_mods(
        num_action_tokens=num_action_tokens,
        num_future_per_chunk=num_future_per_chunk,
        chunk_size=chunk_size,
        num_clean_frames=num_clean_frames,
        num_noised_frames=num_noised_frames,
        tokens_per_frame=tokens_per_frame,
        window_size=window_size,
        num_global_tokens=num_global_tokens,
        device=device,
        include_noised=include_noised,
    )
    if block:
        return (
            _block_mask_from_mod(cross, q_len, kv_len, device),
            _block_mask_from_mod(self_, q_len, q_len, device),
        )
    q = torch.arange(q_len, device=device)[:, None]
    return (
        cross(None, None, q, torch.arange(kv_len, device=device)[None]),
        self_(None, None, q, torch.arange(q_len, device=device)[None]),
    )


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class ActionTauEncoder(nn.Module):
    """Action embedding with the flow time at the input.

    ``w3(silu(w2([w1(a) ++ timestep_features(tau)])))``.
    """

    def __init__(self, action_patch_dim: int, dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(action_patch_dim, dim)
        self.tau = TimestepEmbedder(dim)
        self.w2 = nn.Linear(2 * dim, dim)
        self.w3 = nn.Linear(dim, dim)

    def forward(self, actions: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """``actions (B, T, action_patch_dim)``, ``tau (B, T)`` -> ``(B, T, dim)``."""
        a = self.w1(actions)
        t = self.tau(tau).to(a.dtype)
        if t.shape[:2] != a.shape[:2]:
            raise ValueError(f"tau shape {tuple(t.shape)} != {tuple(a.shape)}")
        x = F.silu(self.w2(torch.cat([a, t], dim=-1)))
        return self.w3(x)


class ActionExpertBlock(nn.Module):
    """One expert layer: attention (cross or self), then a GEGLU MLP.

    Both halves use a scale-only AdaLN on the flow time. The trunk RoPE
    rotates the queries and keys, then they go through an RMS norm per head.
    """

    def __init__(self, *, dim: int, num_heads: int, head_dim: int, is_cross: bool, kv_dim: int) -> None:
        super().__init__()
        self.is_cross = is_cross
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim

        self.adaln_attn = AdaLN(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(kv_dim if is_cross else dim, 2 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

        self.adaln_mlp = AdaLN(dim)
        hidden = 4 * dim
        self.mlp_in = nn.Linear(dim, 2 * hidden)
        self.mlp_out = nn.Linear(hidden, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.to_out.weight, std=0.02)
        nn.init.normal_(self.mlp_out.weight, std=0.02)
        nn.init.zeros_(self.mlp_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        *,
        kv_source: torch.Tensor | None,
        mask: torch.Tensor | BlockMask,
        q_table: torch.Tensor,
        k_table: torch.Tensor,
        kernel_options: Mapping[str, int] | None = None,
    ) -> torch.Tensor:
        """``x (B, Q, D)``, ``cond (B, Q, D)``, ``kv_source (B, KV, D)`` or None.

        The layer runs ``forward_trunk_rope``. The module call lets FSDP and
        activation checkpointing of the block apply.
        """
        return self.forward_trunk_rope(
            x,
            cond,
            kv_source=kv_source,
            mask=mask,
            q_table=q_table,
            k_table=k_table,
            kernel_options=kernel_options,
        )

    def forward_trunk_rope(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        *,
        kv_source: torch.Tensor | None,
        mask: torch.Tensor | BlockMask,
        q_table: torch.Tensor,
        k_table: torch.Tensor,
        kernel_options: Mapping[str, int] | None = None,
    ) -> torch.Tensor:
        """The layer in the trunk-rope layout.

        ``q_table`` and ``k_table`` are trunk rotary tables ``(B, N, D/2, 2)``.
        The rotation runs BEFORE the q/k RMS norm. ``mask`` is a flex
        ``BlockMask`` (compiled dynamic flex attention with
        ``kernel_options``) or a dense ``(Q, KV)`` bool mask (SDPA).
        """
        b, q_len, _ = x.shape
        h = self.adaln_attn(x, cond)
        q = self.to_q(h)
        kv_in = kv_source if self.is_cross else h
        if kv_in is None:
            raise ValueError("a cross-attention block needs kv_source")
        k, v = self.to_kv(kv_in).chunk(2, dim=-1)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, t.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        q, k = split(q), split(k)
        # One table per side: the cross-attention keys carry other ids than
        # the queries. ``apply_rope`` rotates two tensors with one table, so
        # each side passes itself twice and keeps one output.
        q = apply_rope(q, q, q_table)[0]
        k = apply_rope(k, k, k_table)[1]
        qh, kh, vh = rms_norm(q), rms_norm(k), split(v)
        if isinstance(mask, BlockMask):
            attn = _compiled_flex_attention_dynamic(
                qh,
                kh,
                vh,
                block_mask=mask,
                scale=None,
                enable_gqa=False,
                kernel_options=(
                    None if kernel_options is None else cast(FlexKernelOptions, dict(kernel_options))
                ),
            )
            assert isinstance(attn, torch.Tensor)
        else:
            attn = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask[None, None])
        x = x + self.to_out(attn.transpose(1, 2).reshape(b, q_len, -1))
        m = self.adaln_mlp(x, cond)
        return x + self.mlp_out(geglu(self.mlp_in(m)))


RopeConfig = Mapping[str, int | float | str]


class ActionExpert(nn.Module):
    """Action transformer that reads video backbone taps.

    ``forward_trunk_rope`` returns the action velocity
    ``(B, T, action_patch_dim)``.

    The parameter names are equal to the checkpoint keys after the
    ``action_expert.`` prefix (for example ``blocks.0.to_q.weight``).

    ``tap_norm=False`` removes ``tap_norm_scale`` and the tap RMS norm.
    ``max_action_tokens`` is a field of the checkpoint config. No table uses
    it.

    * ``rope={"kind": "trunk", "dim", "effective_dim", "base", "table_size",
      "order": "before_norm"}`` sets the trunk rotary positions.
    * The ``future_chunk_embed`` table is read from row 0.
    * ``global_order`` is the order of the global token kinds in the KV.
    * ``attention="flex_dynamic"`` runs compiled dynamic flex attention with
      a block mask. ``"sdpa"`` runs SDPA with a dense mask.
    * ``head_plus_zero`` adds ``+0.0`` to the head output.
    * ``layout_stamp`` is recorded only.
    """

    def __init__(
        self,
        *,
        action_patch_dim: int,
        proprio_patch_dim: int,
        trunk_dim: int,
        dim: int = 1024,
        num_layers: int = 16,
        num_heads: int = 16,
        head_dim: int = 64,
        num_future_per_chunk: int = 8,
        max_chunks: int = 16,
        max_action_tokens: int = 64,
        tap_norm: bool = True,
        clean_only_conditioning: bool = False,
        rope: RopeConfig,
        global_order: Sequence[str] = GLOBAL_ORDER_TRUNK,
        attention: str = "sdpa",
        flex_kernel_options: Mapping[str, int] | None = None,
        head_plus_zero: bool = False,
        layout_stamp: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers % 2:
            raise ValueError(f"num_layers must be even, got {num_layers}")
        self.rope_table = self._build_trunk_rope(rope, head_dim)
        if attention not in ATTENTION_MODES:
            raise ValueError(f"bad attention {attention!r}, use {ATTENTION_MODES}")
        if sorted(global_order) != sorted(GLOBAL_KINDS):
            raise ValueError(f"global_order must permute {sorted(GLOBAL_KINDS)}")
        if flex_kernel_options is not None and attention != "flex_dynamic":
            raise ValueError("flex_kernel_options needs attention='flex_dynamic'")
        self.global_order = tuple(global_order)
        self.attention = attention
        self.flex_kernel_options: dict[str, int] | None = None
        if attention == "flex_dynamic":
            self.flex_kernel_options = dict(
                FLEX_DYNAMIC_KERNEL_OPTIONS if flex_kernel_options is None else flex_kernel_options
            )
        self.head_plus_zero = bool(head_plus_zero)
        self.layout_stamp = None if layout_stamp is None else int(layout_stamp)
        self.dim = dim
        self.num_layers = num_layers
        self.num_cross_layers = num_layers // 2
        self.num_future_per_chunk = num_future_per_chunk
        self.action_patch_dim = action_patch_dim
        self.clean_only_conditioning = bool(clean_only_conditioning)
        self.head_dim = head_dim

        self.action_encoder = ActionTauEncoder(action_patch_dim, dim)
        self.timestep_embedder = TimestepEmbedder(dim)
        self.future_tokens = nn.Parameter(torch.zeros(num_future_per_chunk, dim))
        self.future_chunk_embed = nn.Embedding(max_chunks, dim)
        self.proprio_token = nn.Parameter(torch.zeros(1, dim))
        self.embed_proprio = nn.Linear(proprio_patch_dim, dim) if proprio_patch_dim > 0 else None
        # One projection for all taps and one for the global tokens.
        self.tap_proj = nn.Linear(trunk_dim, dim, bias=False)
        self.text_proj = nn.Linear(trunk_dim, dim, bias=False)
        self.tap_norm_scale: nn.Parameter | None = nn.Parameter(torch.ones(trunk_dim)) if tap_norm else None
        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    is_cross=(i % 2 == 0),
                    kv_dim=dim,
                )
                for i in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.head = nn.Linear(dim, action_patch_dim)
        self.reset_parameters()

    # Init std of the output head. It is small but not zero, so that all
    # expert weights get a gradient at the first step.
    HEAD_INIT_STD = 1e-3

    @staticmethod
    def _build_trunk_rope(rope: RopeConfig, head_dim: int) -> RotaryEmbedding:
        """The trunk rotary table of the trunk-rope layout."""
        expected = {"kind", "dim", "effective_dim", "base", "table_size", "order"}
        if set(rope) != expected:
            raise ValueError(f"rope needs the keys {sorted(expected)}, got {sorted(rope)}")
        if rope["kind"] != "trunk" or rope["order"] != "before_norm":
            raise ValueError(f"unsupported rope {dict(rope)}")
        if int(rope["dim"]) != head_dim:
            raise ValueError(f"rope dim {rope['dim']} != head_dim {head_dim}")
        return RotaryEmbedding(
            dim=int(rope["dim"]),
            effective_dim=int(rope["effective_dim"]),
            base=int(rope["base"]),
            table_size=int(rope["table_size"]),
        )

    def reset_parameters(self) -> None:
        """Initialize the expert-level weights.

        The construction order and the init calls follow the original implementation.
        The same seed thus gives the same random weights.
        """
        nn.init.normal_(self.future_tokens, std=0.02)
        nn.init.normal_(self.proprio_token, std=0.02)
        nn.init.normal_(self.future_chunk_embed.weight, std=0.02)
        if self.tap_norm_scale is not None:
            nn.init.ones_(self.tap_norm_scale)
        nn.init.normal_(self.head.weight, std=self.HEAD_INIT_STD)
        nn.init.zeros_(self.head.bias)

    def forward_trunk_rope(
        self,
        *,
        noised_action_latents: torch.Tensor,
        action_timesteps: torch.Tensor,
        trunk_features: Sequence[torch.Tensor],
        global_tokens: torch.Tensor | None,
        proprio_latents: torch.Tensor | None,
        kv_rope_ids: torch.Tensor,
        chunk_size: int,
        num_clean_frames: int,
        num_noised_frames: int,
        tokens_per_frame: int,
        window_size: int,
    ) -> torch.Tensor:
        """Predict the action velocity ``(1, T, action_patch_dim)``.

        Args:
            noised_action_latents: ``(1, T, action_patch_dim)`` noised actions.
                The tokens are chunk-major. Every noised chunk has its own
                action chunk (``T = num_noised_frames``).
            action_timesteps: ``(1,)`` or ``(1, T)`` flow time of the actions.
            trunk_features: one tensor per cross-attention layer, in tap order.
                Each is ``(1, (clean + noised) * tokens_per_frame, trunk_dim)``.
                It holds frame tokens only, clean frames first, frame-major.
            global_tokens: ``(1, G, trunk_dim)`` encoder input tokens that all
                chunks can read (in ``global_order``), or None.
            proprio_latents: ``(1, T, proprio_patch_dim)``. The expert reads the
                first latent of each chunk (the chunk boundary pose).
            kv_rope_ids: ``(1, G + frames * tokens_per_frame)`` trunk rope ids
                of the KV tokens, gathered like the features.
            chunk_size: latent frames (and action tokens) per chunk.
            num_clean_frames: clean latent frames in ``trunk_features``.
            num_noised_frames: noised latent frames in ``trunk_features``.
            tokens_per_frame: tokens per latent frame.
            window_size: how many frames before its chunk a query can read.
        """
        rope = self.rope_table
        if len(trunk_features) != self.num_cross_layers:
            raise ValueError(f"expected {self.num_cross_layers} taps, got {len(trunk_features)}")
        b, t_lat, _ = noised_action_latents.shape
        if b != 1:
            raise ValueError(f"the trunk-rope layout runs one sample, got {b}")
        device = noised_action_latents.device
        num_chunks = max(1, t_lat // chunk_size)
        if action_timesteps.ndim == 1:
            a_t = action_timesteps[:, None].expand(-1, t_lat)
        else:
            if action_timesteps.shape[1] != t_lat:
                raise ValueError(
                    f"action_timesteps has {action_timesteps.shape[1]} entries for {t_lat} action tokens"
                )
            a_t = action_timesteps

        # Expert sequence: [proprio | future tokens | actions].
        act = self.action_encoder(noised_action_latents, a_t)
        # The future_chunk_embed rows start at row 0.
        chunk_rows = range(num_chunks)
        if chunk_rows.stop > self.future_chunk_embed.num_embeddings:
            raise ValueError(f"future_chunk_embed rows {chunk_rows} out of range")
        fut = self.future_tokens[None, None].expand(b, num_chunks, -1, -1)
        fut = fut + self.future_chunk_embed.weight[chunk_rows.start : chunk_rows.stop][None, :, None, :].to(
            fut.dtype
        )
        fut = fut.reshape(b, num_chunks * self.num_future_per_chunk, self.dim)
        prop = self.proprio_token[None].expand(b, num_chunks, -1).to(act.dtype)
        if self.embed_proprio is not None and proprio_latents is not None:
            idx = torch.arange(num_chunks, device=device) * chunk_size
            prop = prop + self.embed_proprio(proprio_latents[:, idx].to(prop.dtype))
        x = torch.cat([prop, fut.to(act.dtype), act], dim=1)

        q_chunk = _chunk_of_query(t_lat, self.num_future_per_chunk, chunk_size, device)
        tok_frame = (q_chunk * chunk_size).clamp(min=0, max=t_lat - 1)
        tok_t = a_t.gather(1, tok_frame[None].expand(b, -1))
        cond = self.timestep_embedder(tok_t).to(x.dtype)

        num_global = 0 if global_tokens is None else global_tokens.shape[1]
        cross_mask, self_mask = build_trunk_rope_masks(
            num_action_tokens=t_lat,
            num_future_per_chunk=self.num_future_per_chunk,
            chunk_size=chunk_size,
            num_clean_frames=num_clean_frames,
            num_noised_frames=num_noised_frames,
            tokens_per_frame=tokens_per_frame,
            window_size=window_size,
            num_global_tokens=num_global,
            device=device,
            include_noised=not self.clean_only_conditioning,
            block=self.attention == "flex_dynamic",
        )

        kv_ids = kv_rope_ids.to(device).long()
        expected_kv = num_global + (num_clean_frames + num_noised_frames) * (tokens_per_frame)
        if tuple(kv_ids.shape) != (b, expected_kv):
            raise ValueError(f"kv_rope_ids {tuple(kv_ids.shape)} != {(b, expected_kv)}")
        q_ids = trunk_query_rope_ids(
            kv_ids,
            num_global_tokens=num_global,
            num_clean_frames=num_clean_frames,
            num_noised_frames=num_noised_frames,
            tokens_per_frame=tokens_per_frame,
            num_action_tokens=t_lat,
            num_future_per_chunk=self.num_future_per_chunk,
            chunk_size=chunk_size,
        )
        q_table = rope(q_ids)
        k_table = rope(kv_ids)

        text_kv = None if global_tokens is None else self.text_proj(global_tokens)
        options = self.flex_kernel_options
        cross_idx = 0
        for block in self.blocks:
            assert isinstance(block, ActionExpertBlock)
            if block.is_cross:
                feat = trunk_features[cross_idx]
                if self.tap_norm_scale is not None:
                    feat = (rms_norm(feat) * self.tap_norm_scale).to(feat.dtype)
                feat = self.tap_proj(feat)
                kv = feat if text_kv is None else torch.cat([text_kv.to(feat.dtype), feat], dim=1)
                x = block(
                    x,
                    cond,
                    kv_source=kv,
                    mask=cross_mask,
                    q_table=q_table,
                    k_table=k_table,
                    kernel_options=options,
                )
                cross_idx += 1
            else:
                x = block(
                    x,
                    cond,
                    kv_source=None,
                    mask=self_mask,
                    q_table=q_table,
                    k_table=q_table,
                    kernel_options=options,
                )

        out = self.head(self.norm_out(x[:, -t_lat:]))
        if self.head_plus_zero:
            out = out + 0.0
        return out.view(b, t_lat, self.action_patch_dim)


# ---------------------------------------------------------------------------
# Glue from the encoder taps to the expert
# ---------------------------------------------------------------------------


def run_expert_on_taps_k(
    expert: ActionExpert,
    taps: Sequence[torch.Tensor],
    embeddings: torch.Tensor,
    rope_ids: torch.Tensor,
    *,
    global_spans: Mapping[str, tuple[int, int]],
    noised_span: tuple[int, int],
    clean_span: tuple[int, int] | None,
    n_chunks: int,
    frames_per_chunk: int,
    tokens_per_frame: int,
    noised_actions: torch.Tensor,
    action_t_per_chunk: torch.Tensor,
    proprio_per_chunk: torch.Tensor,
) -> torch.Tensor:
    """Run a trunk-rope expert on the taps of one chunk-causal encoder forward.

    The encoder sequence is ``[registers | anchor | text | noised chunks |
    clean chunks]``. Spans are ``(start, length)`` token ranges of that
    sequence.

    Args:
        expert: an expert with ``rope.kind == "trunk"``.
        taps: encoder block outputs at the tap blocks, each ``(1, S, C)``.
        embeddings: the encoder input sequence (the block 0 input), ``(1, S, C)``.
        rope_ids: ``(1, S)`` int64 trunk rope ids, as the trunk used them.
        global_spans: span of each global kind (``register``, ``anchor``,
            ``text``). The expert reads them in ``expert.global_order``.
        noised_span: the noised frames, ``frames_per_chunk * n_chunks`` frames.
        clean_span: the clean frames, ``frames_per_chunk * (n_chunks - 1)``
            frames, or None when ``n_chunks == 1``.
        n_chunks: chunks in the window (K' = context depth + 1).
        frames_per_chunk: latent frames (and action tokens) per chunk.
        tokens_per_frame: tokens per latent frame.
        noised_actions: ``(1, frames_per_chunk * n_chunks, action_patch_dim)``.
        action_t_per_chunk: ``(1, n_chunks)`` flow time of each chunk.
        proprio_per_chunk: ``(1, n_chunks, proprio_patch_dim)`` boundary poses.

    Returns:
        The action velocity ``(1, T, action_patch_dim)`` in the taps dtype.
    """
    device = taps[0].device
    seq = embeddings.shape[1]
    if embeddings.shape[0] != 1 or rope_ids.shape != (1, seq):
        raise ValueError(
            f"one sample: embeddings {tuple(embeddings.shape)}, rope_ids {tuple(rope_ids.shape)}"
        )
    if set(global_spans) != set(expert.global_order):
        raise ValueError(f"global spans {sorted(global_spans)} != {sorted(expert.global_order)}")
    n_noised = frames_per_chunk * n_chunks
    n_clean = frames_per_chunk * (n_chunks - 1)
    if noised_span[1] != n_noised * tokens_per_frame:
        raise ValueError(f"noised span {noised_span} for {n_noised} frames")
    clean_len = 0 if clean_span is None else clean_span[1]
    if clean_len != n_clean * tokens_per_frame:
        raise ValueError(f"clean span {clean_span} for {n_clean} frames")

    def positions(span: tuple[int, int] | None) -> list[int]:
        if span is None:
            return []
        start, length = span
        if start < 0 or start + length > seq:
            raise ValueError(f"span {span} outside the sequence of {seq} tokens")
        return list(range(start, start + length))

    g_pos = [p for kind in expert.global_order for p in positions(global_spans[kind])]
    s_pos = positions(clean_span) + positions(noised_span)
    g_idx = torch.tensor(g_pos, dtype=torch.long).to(device)
    s_idx = torch.tensor(s_pos, dtype=torch.long).to(device)
    ids = rope_ids.to(device)
    kv_rope_ids = torch.cat([ids.index_select(1, g_idx), ids.index_select(1, s_idx)], 1)
    dtype = embeddings.dtype
    noised = noised_actions.to(device, dtype)
    if noised.shape[:2] != (1, n_noised):
        raise ValueError(f"noised_actions {tuple(noised.shape)} for {n_noised} frames")
    a_t = action_t_per_chunk.to(device).repeat_interleave(frames_per_chunk, dim=1)
    proprio = proprio_per_chunk.to(device).to(dtype)
    proprio = proprio.repeat_interleave(frames_per_chunk, dim=1)
    return expert.forward_trunk_rope(
        noised_action_latents=noised,
        action_timesteps=a_t,
        trunk_features=[f.index_select(1, s_idx) for f in taps],
        global_tokens=embeddings.index_select(1, g_idx) if g_pos else None,
        proprio_latents=proprio,
        kv_rope_ids=kv_rope_ids,
        chunk_size=frames_per_chunk,
        num_clean_frames=n_clean,
        num_noised_frames=n_noised,
        tokens_per_frame=tokens_per_frame,
        window_size=n_clean + n_noised,
    )

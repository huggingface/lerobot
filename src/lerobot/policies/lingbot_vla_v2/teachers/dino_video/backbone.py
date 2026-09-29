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

"""Video ViT-L/16 backbone and explicit token layout for the DINO-video teacher.

Stock PyTorch only: Conv2d patch embedding, fused-QKV attention blocks with
LayerScale and MLP FFN, final norm, and per-block outputs. The published
checkpoint geometry (from its state dict + config.yaml) is:

- hidden 1024, 24 blocks, 16 heads, MLP 4096, patch 16;
- ``cls_token`` / ``mask_token`` / ``storage_tokens`` (4);
- ``rope_embed.periods`` (spatial) and ``rope_embed.periods_t`` (temporal).

Frame bookkeeping must go through :class:`PackedVideoTokens` — attention code
must never re-derive frame indices from magic offsets. Module names deliberately
mirror the checkpoint keys (``patch_embed.proj``, ``blocks.N.attn.qkv``,
``blocks.N.ls1.gamma``, ...) so :func:`.checkpoint.load_backbone_strict` can
restore the backbone without any key translation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from .attention import AttentionBackend, block_causal_attention
from .rope import VideoRoPE

_ARCH_SIZES: dict[str, tuple[int, int, int]] = {
    # arch: (embed_dim, depth, num_heads)
    "vit_small": (384, 12, 6),
    "vit_base": (768, 12, 12),
    "vit_large": (1024, 24, 16),
    "vit_giant2": (1536, 40, 24),
}
_FFN_RATIO = 4.0
_NORM_EPS: dict[str, float] = {"layernorm": 1e-6, "layernormbf16": 1e-5}


@dataclass(frozen=True)
class PackedVideoTokens:
    """Explicit per-token layout for one packed clip.

    Attributes:
        flat: ``[B, tokens, D]`` token tensor.
        frame_index: per-token frame id (storage/cls tokens use their
            attending frame convention, documented at pack time).
        row, col: patch-grid coordinates (``-1`` for non-patch tokens).
        kind: per-token tag (``cls`` / ``storage`` / ``patch``).
        block_id: causal-attention block id per token.
        current_index: frame slice holding the current frame.
        future_index: frame slice holding the future frame.
        extras: layout metadata: ``grid_hw`` (patch grid), ``num_frames``,
            ``tokens_per_frame``, ``patch_tokens_per_frame``.
    """

    flat: torch.Tensor
    frame_index: torch.Tensor
    row: torch.Tensor
    col: torch.Tensor
    kind: list[str]
    block_id: torch.Tensor
    current_index: int
    future_index: int
    extras: dict = field(default_factory=dict)


def pack_video_tokens(
    patch_tokens: torch.Tensor,  # [B, T, H*W, D]
    cls_token: torch.Tensor,  # [1, 1, D]
    storage_tokens: torch.Tensor | None,  # [1, S, D]
    *,
    grid_size: tuple[int, int] | None = None,
    current_index: int = 0,
) -> PackedVideoTokens:
    """Assemble the packed clip layout used by attention and RoPE.

    Each frame contributes one contiguous block of ``[cls, storage..., patch...]``
    tokens (patches row-major over the grid), and blocks are stacked in frame
    order. Metadata tensors describe one such block broadcast over the batch
    dimension of ``flat``.
    """
    if patch_tokens.ndim != 4:
        raise ValueError(f"patch_tokens must be [B, T, H*W, D], got {tuple(patch_tokens.shape)}.")
    batch, frames, patches, embed_dim = patch_tokens.shape
    num_storage = 0 if storage_tokens is None else int(storage_tokens.shape[1])
    if cls_token.shape != (1, 1, embed_dim):
        raise ValueError(f"cls_token must be [1, 1, D], got {tuple(cls_token.shape)}.")
    if storage_tokens is not None and storage_tokens.shape != (1, num_storage, embed_dim):
        raise ValueError(f"storage_tokens must be [1, S, D], got {tuple(storage_tokens.shape)}.")
    if grid_size is None:
        side = int(math.isqrt(patches))
        if side * side != patches:
            raise ValueError(f"cannot infer a square patch grid from {patches} patch tokens; pass grid_size.")
        grid_size = (side, side)
    if grid_size[0] * grid_size[1] != patches:
        raise ValueError(f"grid_size {grid_size} does not match {patches} patch tokens per frame.")
    if not 0 <= current_index < frames:
        raise ValueError(f"current_index must be in [0, {frames}), got {current_index}.")

    cls_tokens = cls_token.expand(batch, frames, 1, embed_dim)
    if storage_tokens is None:
        storage = patch_tokens.new_empty(batch, frames, 0, embed_dim)
    else:
        storage = storage_tokens.expand(batch, frames, num_storage, embed_dim)
    flat = torch.cat((cls_tokens, storage, patch_tokens), dim=2).reshape(
        batch, frames * (1 + num_storage + patches), embed_dim
    )

    prefix = 1 + num_storage
    frame_size = prefix + patches
    grid_h, grid_w = grid_size
    patch_index = torch.arange(patches)
    row = torch.cat((torch.full((prefix,), -1, dtype=torch.long), patch_index // grid_w)).repeat(frames)
    col = torch.cat((torch.full((prefix,), -1, dtype=torch.long), patch_index % grid_w)).repeat(frames)
    frame_index = torch.arange(frames).repeat_interleave(frame_size)
    kind = (["cls"] + ["storage"] * num_storage + ["patch"] * patches) * frames

    return PackedVideoTokens(
        flat=flat,
        frame_index=frame_index,
        row=row,
        col=col,
        kind=kind,
        block_id=frame_index.clone(),
        current_index=current_index,
        future_index=frames - 1,
        extras={
            "grid_hw": (grid_h, grid_w),
            "num_frames": frames,
            "tokens_per_frame": frame_size,
            "patch_tokens_per_frame": patches,
        },
    )


@dataclass(frozen=True)
class DinoVideoBackboneConfig:
    """Geometry and RoPE options for :class:`FirstPartyDinoVideoBackbone`.

    Defaults follow the published ``config.yaml`` (ViT-L/16, 4 storage tokens,
    3D RoPE with fp32 buffers, ``layernormbf16`` norms, prefix temporal RoPE,
    base fps 30).
    """

    arch: str = "vit_large"
    img_size: int = 256
    patch_size: int = 16
    in_chans: int = 3
    n_storage_tokens: int = 4
    qkv_bias: bool = True
    proj_bias: bool = True
    ffn_bias: bool = True
    layerscale_init: float = 1e-5
    norm_layer: str = "layernormbf16"
    rope_normalize_coords: str = "separate"
    rope_prefix_temporal: bool = True
    rope_base: float = 100.0
    rope_temporal_base: float = 10000.0
    rope_base_fps: float = 30.0
    rope_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.arch not in _ARCH_SIZES:
            raise ValueError(f"unknown arch {self.arch!r}; expected one of {sorted(_ARCH_SIZES)}.")
        if self.norm_layer not in _NORM_EPS:
            raise ValueError(
                f"norm_layer {self.norm_layer!r} is not implemented by the first-party runtime; "
                f"expected one of {sorted(_NORM_EPS)}."
            )
        if self.rope_normalize_coords not in {"min", "max", "separate"}:
            raise ValueError(f"unknown rope_normalize_coords {self.rope_normalize_coords!r}.")
        if self.patch_size <= 0 or self.n_storage_tokens < 0:
            raise ValueError("patch_size must be positive and n_storage_tokens non-negative.")

    @property
    def embed_dim(self) -> int:
        return _ARCH_SIZES[self.arch][0]

    @property
    def depth(self) -> int:
        return _ARCH_SIZES[self.arch][1]

    @property
    def num_heads(self) -> int:
        return _ARCH_SIZES[self.arch][2]

    @property
    def ffn_hidden_dim(self) -> int:
        return int(self.embed_dim * _FFN_RATIO)

    @property
    def norm_eps(self) -> float:
        return _NORM_EPS[self.norm_layer]


class _PatchEmbed(nn.Module):
    """2D patch embedding with the checkpoint's ``patch_embed.proj`` naming."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, C, H, W] -> [N, Hp*Wp, D]
        return self.proj(x).flatten(2).transpose(1, 2)


class _LayerScale(nn.Module):
    """Per-channel residual scaling (``blocks.N.ls*.gamma`` in the checkpoint)."""

    def __init__(self, dim: int, init_values: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim))
        self.init_values = init_values

    def reset_parameters(self) -> None:
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class _Mlp(nn.Module):
    """Linear -> GELU (exact erf) -> Linear feed-forward block."""

    def __init__(self, in_features: int, hidden_features: int, *, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _FusedAttention(nn.Module):
    """Fused-QKV attention with RoPE and frame-block-causal masking."""

    def __init__(self, dim: int, num_heads: int, *, qkv_bias: bool, proj_bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(
        self,
        x: torch.Tensor,  # [B, tokens, D]
        layout: PackedVideoTokens,
        rope: VideoRoPE,
        *,
        fps: float | torch.Tensor | None = None,
        backend: AttentionBackend = "sdpa",
    ) -> torch.Tensor:
        batch, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, num_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, heads, tokens, head_dim]
        q, k = rope(q, k, layout, fps=fps)
        out = block_causal_attention(q, k, v, layout, backend=backend)
        out = out.transpose(1, 2).reshape(batch, num_tokens, dim)
        return self.proj(out)


class _TransformerBlock(nn.Module):
    """Pre-norm block: norm1 -> attn -> ls1 (+), norm2 -> mlp -> ls2 (+)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        *,
        qkv_bias: bool,
        proj_bias: bool,
        ffn_bias: bool,
        norm_eps: float,
        layerscale_init: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = _FusedAttention(dim, num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.ls1 = _LayerScale(dim, init_values=layerscale_init)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = _Mlp(dim, ffn_hidden_dim, bias=ffn_bias)
        self.ls2 = _LayerScale(dim, init_values=layerscale_init)

    def forward(
        self,
        x: torch.Tensor,
        layout: PackedVideoTokens,
        rope: VideoRoPE,
        *,
        fps: float | torch.Tensor | None = None,
        backend: AttentionBackend = "sdpa",
    ) -> torch.Tensor:
        x_attn = self.attn(self.norm1(x), layout, rope, fps=fps, backend=backend)
        x = x + self.ls1(x_attn)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class FirstPartyDinoVideoBackbone(nn.Module):
    """Inference-only video ViT; weights restored by :mod:`.checkpoint`."""

    def __init__(
        self, config: DinoVideoBackboneConfig, *, attention_backend: AttentionBackend = "sdpa"
    ) -> None:
        super().__init__()
        self.config = config
        self.attention_backend = attention_backend
        self.patch_embed = _PatchEmbed(config.in_chans, config.embed_dim, config.patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.embed_dim))
        self.mask_token = nn.Parameter(torch.empty(1, config.embed_dim))
        if config.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, config.n_storage_tokens, config.embed_dim))
        else:
            self.storage_tokens = None
        self.rope_embed = VideoRoPE(
            config.embed_dim,
            num_heads=config.num_heads,
            base=config.rope_base,
            temporal_base=config.rope_temporal_base,
            normalize_coords=config.rope_normalize_coords,
            prefix_temporal=config.rope_prefix_temporal,
            base_fps=config.rope_base_fps,
            dtype=config.rope_dtype,
        )
        self.blocks = nn.ModuleList(
            _TransformerBlock(
                config.embed_dim,
                config.num_heads,
                config.ffn_hidden_dim,
                qkv_bias=config.qkv_bias,
                proj_bias=config.proj_bias,
                ffn_bias=config.ffn_bias,
                norm_eps=config.norm_eps,
                layerscale_init=config.layerscale_init,
            )
            for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Deterministic init; the published checkpoint overwrites all of it."""
        self.rope_embed.reset_parameters()
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.zeros_(self.mask_token)
        if self.storage_tokens is not None:
            nn.init.normal_(self.storage_tokens, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, _LayerScale)):
                module.reset_parameters()

    def embed_video(self, video: torch.Tensor) -> torch.Tensor:
        """Embed a ``[B, C, T, H, W]`` clip into ``[B, T, Hp*Wp, D]`` tokens.

        Frames are embedded independently with the shared 2D patch projection;
        patch tokens are row-major (row index outer, column index inner).
        """
        if video.ndim != 5:
            raise ValueError(f"video must be [B, C, T, H, W], got {tuple(video.shape)}.")
        batch, channels, frames, height, width = video.shape
        patch = self.config.patch_size
        if height % patch or width % patch:
            raise ValueError(
                f"video spatial size ({height}, {width}) must be divisible by patch size {patch}."
            )
        if channels != self.config.in_chans:
            raise ValueError(f"video must have {self.config.in_chans} channels, got {channels}.")
        frames_flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        tokens = self.patch_embed(frames_flat)  # [B*T, Hp*Wp, D]
        grid_h, grid_w = height // patch, width // patch
        return tokens.reshape(batch, frames, grid_h * grid_w, -1)

    def cast_for_inference(
        self, device: torch.device | str | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """Cast weights to ``dtype`` while keeping the RoPE buffers fp32.

        The published teacher runs bf16 weights with fp32 RoPE buffers; a plain
        ``.to(dtype=...)`` would silently downgrade the position tables and
        destabilize the temporal angles, so the buffers are restored after the
        cast (hard constraint from the module docs).
        """
        rope_buffers = {name: buffer.detach().clone() for name, buffer in self.rope_embed.named_buffers()}
        self.to(device=device, dtype=dtype)
        target = torch.device(device) if device is not None else self.rope_embed.periods.device
        for name, buffer in rope_buffers.items():
            self.rope_embed.get_buffer(name).data = buffer.to(device=target)

    def forward(
        self, packed: PackedVideoTokens, *, fps: float | torch.Tensor | None = None
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run the blocks; return (intermediate outputs, final tokens).

        ``intermediate outputs`` holds the (un-normed) output of every block in
        order; the teacher applies the final norm itself so CLS and patch
        tokens can be normed with the checkpoint's exact pooling order.
        """
        x = packed.flat
        if x.shape[-1] != self.config.embed_dim:
            raise ValueError(
                f"packed tokens have dim {x.shape[-1]} but the backbone expects {self.config.embed_dim}."
            )
        outputs: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x, packed, self.rope_embed, fps=fps, backend=self.attention_backend)
            outputs.append(x)
        return outputs, x

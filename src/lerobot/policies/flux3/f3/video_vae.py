# Copyright 2026 Black Forest Labs. All rights reserved.
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
# Vendored from black-forest-labs/flux-action (src/flux_action/models/video_vae.py).
"""Video VAE (ViTNormInference). Swin3D + neighborhood attention with built-in DistributedRunningStats normalization."""

import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file
from torch import Tensor

from lerobot.utils.import_utils import _natten_available, require_package

from ..utils import resolve_weights
from .positional import prc_vid, scatter_ids, times_to_ids

if TYPE_CHECKING or _natten_available:
    from natten import backends as nb
    from natten.functional import na2d, na3d

logger = logging.getLogger(__name__)

_norm_layer = partial(nn.LayerNorm, eps=1e-5)

# NATTEN neighborhood-attention backend, resolved at first use per tensor
# layout (2D/3D). Prebuilt NATTEN wheels only carry fused CUTLASS kernels for
# some GPU architectures, so the fastest supported backend is probed at
# runtime: blackwell (SM100) > hopper (SM90) > generic CUTLASS > flex-attention
# (pure torch, runs everywhere at a speed/memory cost). Set $F3_NATTEN_BACKEND
# (blackwell-fna | hopper-fna | cutlass-fna | flex-fna) to override.
_NATTEN_BACKEND_ENV = "F3_NATTEN_BACKEND"
_PERSISTENT_KERNEL_BACKENDS = ("blackwell-fna", "hopper-fna")
_natten_backends: dict[int, str] = {}


def _natten_attention_kwargs(q: Tensor, k: Tensor, v: Tensor) -> dict:
    backend = _natten_backends.get(q.ndim)
    if backend is None:
        backend = os.environ.get(_NATTEN_BACKEND_ENV)
        if not backend:
            if nb.can_run_cutlass_blackwell_fna(q, k, v):
                backend = "blackwell-fna"
            elif nb.can_run_cutlass_hopper_fna(q, k, v):
                backend = "hopper-fna"
            elif nb.can_run_cutlass_fna(q, k, v):
                backend = "cutlass-fna"
            else:
                backend = "flex-fna"  # universal fallback, no compiled kernels needed
        logger.info("natten backend (%dD tokens): %s", q.ndim - 3, backend)
        _natten_backends[q.ndim] = backend
    kwargs = {"backend": backend}
    if backend in _PERSISTENT_KERNEL_BACKENDS:
        kwargs["run_persistent_kernel"] = True
    return kwargs


@dataclass
class ViTNormInferenceParams:
    z_dim: int = 96
    embed_dim: int = 256
    patch_size: list[int] = field(default_factory=lambda: [1, 4, 4])
    window_size: list[int] = field(default_factory=lambda: [5, 5, 5])
    alternate_window_size: list[int] | None = None
    enc_depths: list[int] = field(default_factory=lambda: [1, 4, 8, 8])
    dec_depths: list[int] = field(default_factory=lambda: [1, 4, 8, 8])
    num_heads: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    temporal: list[bool] = field(default_factory=lambda: [False, False, True, True])
    enc_causal: bool = True
    dec_causal: bool = False
    qk_norm: bool = True
    patch_norm: bool = False
    smooth: bool = True
    dtype: str = "bfloat16"
    use_compile: bool = True
    compile_config: dict | None = field(default_factory=lambda: {"dynamic_compile": True})
    compile_decoder: bool = False
    chunked_encode: bool = False
    chunk_size_frames: int = 45
    chunked_decode: bool = False
    chunk_size_latent_frames: int = 8
    chunk_overlap_latent_frames: int = 4


# Full-decode peak ~= ELEM * latent.numel() + FIXED, profiled with the
# compiled decoder (calibrated at 720p, T_lat 8..64). Linear in latent elements,
# so it holds across resolutions. Used to pick full vs chunked decode at runtime.
DECODE_BYTES_PER_LATENT_ELEM = 1.35e6
DECODE_FIXED_BYTES = 2 * 2**30
DECODE_SAFETY_FACTOR = 1.25


class DistributedRunningStats(nn.Module):
    def __init__(self, num_channels: int, momentum: float = 0.01, device: str | torch.device | None = None):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(num_channels, device=device))
        self.register_buffer("running_var", torch.ones(num_channels, device=device))
        self.register_buffer("initialized", torch.tensor(False, device=device))

    def _shape(self, x: Tensor) -> tuple:
        return (1, -1) + (1,) * (x.dim() - 2)

    @torch.no_grad()
    @torch.compiler.disable()
    def update(self, x: Tensor, group=None) -> None:
        c = x.shape[1]
        x_flat = x.transpose(0, 1).reshape(c, -1).double()
        n = torch.tensor([x_flat.shape[1]], device=x.device, dtype=torch.float64)
        s = x_flat.sum(dim=1)
        sq = x_flat.pow(2).sum(dim=1)
        stats = torch.stack([n.expand(c), s, sq], dim=0)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, group=group)
        n, s, sq = stats[0, 0], stats[1], stats[2]
        mean = (s / n).float()
        var = (sq / n - mean.pow(2)).clamp_min(0.0).float()
        if not self.initialized:
            self.running_mean.data.copy_(mean)
            self.running_var.data.copy_(var)
            self.initialized.data.fill_(True)
        else:
            self.running_mean.data.lerp_(mean, self.momentum)
            self.running_var.data.lerp_(var, self.momentum)

    def normalize(self, x: Tensor, mode: str = "channel") -> Tensor:
        mean = self.running_mean.detach().clone()
        var = self.running_var.detach().clone()
        s = self._shape(x)
        centered = x - mean.view(s)
        if mode == "channel":
            return centered / var.sqrt().view(s)
        if mode == "vector":
            return centered / var.sum().sqrt()
        raise ValueError(f"Unknown mode: {mode}")

    def denormalize(self, x: Tensor, mode: str = "channel") -> Tensor:
        mean = self.running_mean.detach().clone()
        var = self.running_var.detach().clone()
        s = self._shape(x)
        if mode == "channel":
            return x * var.sqrt().view(s) + mean.view(s)
        if mode == "vector":
            return x * var.sum().sqrt() + mean.view(s)
        raise ValueError(f"Unknown mode: {mode}")


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int | None = None, avg_skip: bool = False):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = _norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.avg_skip = avg_skip

    def forward(self, x: Tensor) -> Tensor:
        B, D, H, W, C = x.shape  # noqa: N806
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        B, D, H, W, C = x.shape  # noqa: N806
        x = x.reshape(B, D, H // 2, 2, W // 2, 2, C)
        if self.avg_skip:
            x_ = x.mean(dim=(3, 5))
            x_ = torch.concat([x_, x_], -1)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).flatten(4)
        x = self.reduction(self.norm(x))
        if self.avg_skip:
            x = x + x_
        return x


class TemporalMerging(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int | None = None,
        zero_pad: bool = False,
        avg_skip: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = _norm_layer(2 * dim)
        self.reduction = nn.Linear(2 * dim, self.out_dim, bias=False)
        self.zero_pad = zero_pad
        self.avg_skip = avg_skip

    def forward(self, x: Tensor) -> Tensor:
        B, D, H, W, C = x.shape  # noqa: N806
        if D % 2 == 1 and self.zero_pad:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, D % 2))
        elif D % 2 == 1:
            x = torch.concat([x[:, :1], x], dim=1)
        B, D, H, W, C = x.shape  # noqa: N806
        x = x.reshape(B, D // 2, 2, H, W, C)
        if self.avg_skip:
            x_ = x.mean(2)
        x = x.permute(0, 1, 3, 4, 2, 5).reshape(B, D // 2, H, W, 2 * C)
        x = self.reduction(self.norm(x))
        if self.avg_skip:
            x = x + x_
        return x


class PatchExpansion(nn.Module):
    def __init__(self, dim: int, out_dim: int | None = None, avg_skip: bool = False):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else (dim // 2)
        self.norm = _norm_layer(dim)
        self.expansion = nn.Linear(dim, 4 * self.out_dim, bias=False)
        self.avg_skip = avg_skip

    def forward(self, x: Tensor) -> Tensor:
        B, D, H, W, C = x.shape  # noqa: N806
        assert self.dim == C
        if self.avg_skip:
            exp = torch.concat([x, x], -1)
        x = self.expansion(self.norm(x))
        if self.avg_skip:
            x = x + exp
        x = x.view(B, D, H, W, 2, 2, self.out_dim)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        return x.view(B, D, H * 2, W * 2, self.out_dim)


class TemporalExpansion(nn.Module):
    def __init__(self, dim: int, out_dim: int | None = None, avg_skip: bool = True):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = _norm_layer(dim)
        self.expansion = nn.Linear(dim, 2 * self.out_dim, bias=False)
        self.avg_skip = avg_skip

    def forward(self, x: Tensor) -> Tensor:
        B, D, H, W, C = x.shape  # noqa: N806
        assert self.dim == C
        if self.avg_skip:
            exp = torch.concat([x, x], -1)
        x = self.expansion(self.norm(x))
        if self.avg_skip:
            x = x + exp
        B, D, H, W, C = x.shape  # noqa: N806
        x = x.view(B, D, H, W, 2, self.out_dim)
        x = x.permute(0, 1, 4, 2, 3, 5).contiguous()
        x = x.view(B, D * 2, H, W, self.out_dim)
        return x[:, 1:]


def _compute_pad_size_3d(
    size_dhw: tuple[int, int, int], patch_size: tuple[int, int, int]
) -> tuple[int, int, int]:
    pad = [(patch_size[i] - size_dhw[i] % patch_size[i]) % patch_size[i] for i in range(3)]
    return pad[0], pad[1], pad[2]


torch.fx.wrap("_compute_pad_size_3d")


class RotaryPositionEmbedding3D(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base_t: float = 256.0,
        base_h: float = 256.0,
        base_w: float = 256.0,
    ):
        super().__init__()
        assert head_dim % 8 == 0
        self.head_dim = head_dim
        self.chunk_dim = head_dim // 4

        inv_freq = torch.stack(
            [
                1.0 / (base_t ** (torch.arange(0, self.chunk_dim, 2).float() / self.chunk_dim)),
                1.0 / (base_h ** (torch.arange(0, self.chunk_dim, 2).float() / self.chunk_dim)),
                1.0 / (base_w ** (torch.arange(0, self.chunk_dim, 2).float() / self.chunk_dim)),
                torch.zeros(self.chunk_dim // 2),
            ]
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        _, t, h, w, _, _ = q.shape
        device = q.device
        dtype = q.dtype
        grids = torch.meshgrid(
            torch.arange(t, device=device, dtype=torch.float32),
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        pos = torch.stack(grids + (torch.zeros_like(grids[0]),), dim=-1)
        freqs = torch.einsum("...a,af->...af", pos, self.inv_freq.float())
        freqs = freqs.reshape(1, t, h, w, 1, -1)
        freqs = torch.cat([freqs, freqs], dim=-1)
        # Rotate in bf16 rather than upcasting q/k to fp32, which avoids large
        # fp32 transients during decode. cos/sin keep full precision from fp32
        # freqs and cast down only before the products.
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


class Natten3D(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: list[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        causal: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert len(window_size) == 3
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.qk_norm = qk_norm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = RotaryPositionEmbedding3D(self.head_dim)
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
            self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)

    def forward(self, x: Tensor) -> Tensor:
        b, t, h, w, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(b, t, h, w, 3, self.num_heads, self.head_dim).unbind(4)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rope(q, k)

        if t == 1:
            q2, k2, v2 = q.squeeze(1), k.squeeze(1), v.squeeze(1)
            out = na2d(
                q2,
                k2,
                v2,
                kernel_size=self.window_size[1:],
                attention_kwargs=_natten_attention_kwargs(q2, k2, v2),
            ).unsqueeze(1)
        else:
            out = na3d(
                q,
                k,
                v,
                is_causal=[self.causal, False, False],
                kernel_size=self.window_size,
                attention_kwargs=_natten_attention_kwargs(q, k, v),
            )
        out = out.reshape(b, t, h, w, c)
        return self.proj(out)


class GLU_MLP(nn.Module):  # noqa: N801 (vendored name)
    def __init__(self, dim: int, align_to: int = 64, dropout: float = 0.0):
        super().__init__()
        hidden_dim = align_to * ((int(dim * 8 / 3) + align_to - 1) // align_to)
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.drop(self.down_proj(F.silu(gate) * up))


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: list[int],
        dropout: float = 0.0,
        causal: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.norm1 = _norm_layer(dim)
        self.attn = Natten3D(dim, window_size, num_heads, causal=causal, qk_norm=qk_norm)
        self.norm2 = _norm_layer(dim)
        self.mlp = GLU_MLP(dim, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed3d(nn.Module):
    def __init__(
        self,
        patch_size: list[int],
        in_channels: int = 3,
        embed_dim: int = 96,
        norm: bool = True,
    ):
        super().__init__()
        self.tuple_patch_size = (patch_size[0], patch_size[1], patch_size[2])
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.tuple_patch_size,
            stride=self.tuple_patch_size,
        )
        self.norm = _norm_layer(embed_dim) if norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, t, h, w = x.size()
        pad = _compute_pad_size_3d((t, h, w), self.tuple_patch_size)
        x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
        x = self.proj(x)
        x = x.permute(0, 2, 3, 4, 1)
        return self.norm(x)


class DecoderSwin3D(nn.Module):
    def __init__(
        self,
        z_ch: int,
        patch_size: list[int],
        embed_dim: int,
        depths: list[int],
        temporal: list[bool],
        num_heads: list[int],
        window_size: list[int],
        dropout: float = 0.0,
        upsample_layer: Callable[..., nn.Module] = PatchExpansion,
        causal: bool = False,
        qk_norm: bool = False,
        alternate_window_size: list[int] | None = None,
    ):
        super().__init__()
        assert len(temporal) == len(depths)
        self.ps = patch_size
        self.pos_drop = nn.Dropout(p=dropout)
        self.proj_in = nn.Linear(z_ch, embed_dim * 2 ** (len(depths) - 1))
        self.proj_out = nn.Linear(embed_dim, math.prod(patch_size) * 3)

        layers: list[nn.Module] = []
        for i_stage in reversed(range(len(depths))):
            stage: list[nn.Module] = []
            dim = embed_dim * 2**i_stage
            stage_depth = depths[i_stage]
            for i_layer in range(stage_depth):
                ws = (
                    alternate_window_size
                    if (alternate_window_size is not None and i_layer % 2 == 1)
                    else window_size
                )
                stage.append(
                    SwinTransformerBlock(
                        dim,
                        num_heads[i_stage],
                        ws,
                        dropout=dropout,
                        causal=causal,
                        qk_norm=qk_norm,
                    )
                )
            layers.append(nn.Sequential(*stage))
            if temporal[i_stage]:
                layers.append(TemporalExpansion(dim, dim))
                ws = (
                    alternate_window_size
                    if (alternate_window_size is not None and stage_depth % 2 == 1)
                    else window_size
                )
                layers.append(
                    SwinTransformerBlock(
                        dim,
                        num_heads[i_stage],
                        ws,
                        dropout=dropout,
                        causal=causal,
                        qk_norm=qk_norm,
                    )
                )
            if i_stage > 0:
                layers.append(upsample_layer(dim, dim // 2))
        self.features = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.proj_in(x)
        x = self.pos_drop(x)
        x = self.features(x)
        x = self.proj_out(x)
        b, t, h, w, _ = x.shape
        x = x.view(b, t, h, w, self.ps[0], self.ps[1], self.ps[2], 3)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(b, t * self.ps[0], h * self.ps[1], w * self.ps[2], 3)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class EncoderSwin3D(nn.Module):
    def __init__(
        self,
        z_ch: int,
        patch_size: list[int],
        embed_dim: int,
        depths: list[int],
        temporal: list[bool],
        num_heads: list[int],
        window_size: list[int],
        dropout: float = 0.0,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        causal: bool = False,
        qk_norm: bool = False,
        alternate_window_size: list[int] | None = None,
        patch_norm: bool = True,
    ):
        super().__init__()
        assert len(temporal) == len(depths)
        self.proj = nn.Linear(embed_dim * 2 ** (len(depths) - 1), z_ch)
        self.patch_embed = PatchEmbed3d(patch_size=patch_size, embed_dim=embed_dim, norm=patch_norm)
        self.pos_drop = nn.Dropout(p=dropout)

        layers: list[nn.Module] = []
        for i_stage in range(len(depths)):
            stage: list[nn.Module] = []
            dim = embed_dim * 2**i_stage
            stage_depth = depths[i_stage]
            for i_layer in range(stage_depth):
                ws = (
                    alternate_window_size
                    if (alternate_window_size is not None and i_layer % 2 == 1)
                    else window_size
                )
                stage.append(
                    SwinTransformerBlock(
                        dim,
                        num_heads[i_stage],
                        ws,
                        dropout=dropout,
                        causal=causal,
                        qk_norm=qk_norm,
                    )
                )
            layers.append(nn.Sequential(*stage))
            downsampled = False
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, 2 * dim))
                downsampled = True
            if temporal[i_stage]:
                if downsampled:
                    i_stage += 1
                    dim = 2 * dim
                ws = (
                    alternate_window_size
                    if (alternate_window_size is not None and stage_depth % 2 == 1)
                    else window_size
                )
                layers.append(
                    SwinTransformerBlock(
                        dim,
                        num_heads[i_stage],
                        ws,
                        dropout=dropout,
                        causal=causal,
                        qk_norm=qk_norm,
                    )
                )
                layers.append(TemporalMerging(dim, dim))
        self.features = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        _, c, _, _, _ = x.shape
        assert c == 3
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        # Inductor tiling assertion under dynamic=True without this break.
        torch._dynamo.graph_break()
        x = self.features(x)
        x = self.proj(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class ViTNorm(nn.Module):
    def __init__(
        self,
        z_dim: int,
        embed_dim: int,
        patch_size: list[int],
        window_size: list[int],
        enc_depths: list[int],
        dec_depths: list[int],
        num_heads: list[int],
        temporal: list[bool],
        all_average: bool = False,
        enc_causal: bool = False,
        dec_causal: bool = False,
        temporal_causal: bool = False,
        qk_norm: bool = False,
        alternate_window_size: list[int] | None = None,
        patch_norm: bool = False,
        smooth: bool = False,
    ):
        super().__init__()
        if temporal_causal:
            enc_causal = True
            dec_causal = True
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        self.z_dim = z_dim
        downsample_layer = partial(PatchMerging, avg_skip=True) if all_average else PatchMerging
        upsample_layer = partial(PatchExpansion, avg_skip=True) if all_average else PatchExpansion
        self.encoder = EncoderSwin3D(
            z_ch=2 * z_dim,
            patch_size=patch_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=enc_depths,
            num_heads=num_heads,
            temporal=temporal,
            downsample_layer=downsample_layer,
            causal=enc_causal,
            qk_norm=qk_norm,
            patch_norm=patch_norm,
            alternate_window_size=alternate_window_size,
        )
        self.decoder = DecoderSwin3D(
            z_ch=z_dim,
            patch_size=patch_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=dec_depths,
            num_heads=num_heads,
            temporal=temporal,
            upsample_layer=upsample_layer,
            causal=dec_causal,
            qk_norm=qk_norm,
            alternate_window_size=alternate_window_size,
        )
        self.smooth = smooth
        self.z_normalizer = DistributedRunningStats(z_dim, momentum=0.01)

    def encode(self, x: Tensor, normalize: bool = True) -> Tensor:
        mu, _ = self.encoder(x).chunk(2, dim=-4)
        return self.z_normalizer.normalize(mu) if normalize else mu

    def decode(self, z: Tensor, normalize: bool = True) -> Tensor:
        if normalize:
            z = self.z_normalizer.denormalize(z)
        return self.decoder(z)


class ViTNormInference(nn.Module):
    TEMPORAL_DOWNSAMPLE = 4
    SPATIAL_DOWNSAMPLE = 32

    @classmethod
    def get_latent_count(cls, num_frames: int, height: int, width: int) -> int:
        t_latent = 1 + (num_frames - 1) // cls.TEMPORAL_DOWNSAMPLE
        h_latent = height // cls.SPATIAL_DOWNSAMPLE
        w_latent = width // cls.SPATIAL_DOWNSAMPLE
        return t_latent * h_latent * w_latent

    def __init__(self, params: ViTNormInferenceParams = ViTNormInferenceParams()):
        super().__init__()
        self.params = params
        if list(params.temporal) != [False, False, True, True]:
            raise ValueError(
                f"TEMPORAL_DOWNSAMPLE={self.TEMPORAL_DOWNSAMPLE} depends on temporal, got {params.temporal}"
            )
        if list(params.patch_size) != [1, 4, 4] or len(params.enc_depths) != 4:
            raise ValueError(
                f"SPATIAL_DOWNSAMPLE={self.SPATIAL_DOWNSAMPLE} depends on "
                f"patch_size and enc_depths, got {params.patch_size}, "
                f"{params.enc_depths}"
            )
        self.chunked_encode = params.chunked_encode
        self.chunk_size_frames = params.chunk_size_frames
        self.chunked_decode = params.chunked_decode
        self.chunk_size_latent_frames = params.chunk_size_latent_frames
        self.chunk_overlap_latent_frames = params.chunk_overlap_latent_frames
        self.model = ViTNorm(
            z_dim=params.z_dim,
            embed_dim=params.embed_dim,
            patch_size=params.patch_size,
            window_size=params.window_size,
            alternate_window_size=params.alternate_window_size,
            enc_depths=params.enc_depths,
            dec_depths=params.dec_depths,
            num_heads=params.num_heads,
            temporal=params.temporal,
            enc_causal=params.enc_causal,
            dec_causal=params.dec_causal,
            qk_norm=params.qk_norm,
            patch_norm=params.patch_norm,
            smooth=params.smooth,
        )
        if params.dtype == "bfloat16":
            self.model = self.model.bfloat16()
        elif params.dtype == "float16":
            self.model = self.model.half()
        self.model.eval()

    def apply_compile(self):
        params = self.params
        if not (params.use_compile or params.compile_decoder):
            return
        cfg = params.compile_config or {}
        mode = cfg.get("compile_mode", "default")
        fullgraph = cfg.get("fullgraph", False)
        dynamic = cfg.get("dynamic_compile", False)
        backend = cfg.get("backend", "inductor")
        if params.use_compile:
            for layer_id, layer in self.model.encoder.features.named_children():
                self.model.encoder.features.register_module(
                    layer_id,
                    torch.compile(
                        layer,
                        mode=mode,
                        fullgraph=fullgraph,
                        dynamic=dynamic,
                        backend=backend,
                    ),
                )
        if params.compile_decoder:
            for layer_id, layer in self.model.decoder.features.named_children():
                self.model.decoder.features.register_module(
                    layer_id,
                    torch.compile(
                        layer,
                        mode=mode,
                        fullgraph=fullgraph,
                        dynamic=dynamic,
                        backend=backend,
                    ),
                )

    def min_chunk_enc(
        self,
        video: Tensor,
        chunk_size_frames: int | None = None,
        overlap: int = 1,
        target_num_frames: int | None = None,
    ) -> Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected (B, C, T, H, W), got {video.shape}")
        chunk_size_frames = chunk_size_frames or self.chunk_size_frames
        stride = chunk_size_frames - overlap
        if stride <= 0:
            raise ValueError(f"chunk_size_frames must be > {overlap}, got {chunk_size_frames}")
        num_frames = video.shape[2]
        if target_num_frames is None:
            target_num_frames = num_frames
        if target_num_frames > num_frames:
            raise ValueError(
                f"target_num_frames ({target_num_frames}) cannot exceed encoded frames ({num_frames})"
            )
        if num_frames < chunk_size_frames:
            raise ValueError(
                f"Chunked encoding expects at least {chunk_size_frames} frames; got T={num_frames}"
            )
        if (num_frames - chunk_size_frames) % stride != 0:
            raise ValueError(
                f"Chunked encoding expects T = {chunk_size_frames} + n * {stride}; got T={num_frames}"
            )

        latent_pieces = []
        for start in range(0, num_frames - chunk_size_frames + 1, stride):
            chunk = video[:, :, start : start + chunk_size_frames]
            z_chunk = self.model.encode(chunk, normalize=True)
            if start == 0:
                latent_pieces.append(z_chunk)
            else:
                skip_latent_frames = 1 + (overlap - 1) // self.TEMPORAL_DOWNSAMPLE
                latent_pieces.append(z_chunk[:, :, skip_latent_frames:])

        latent = torch.cat(latent_pieces, dim=2)
        target_latent_frames = 1 + (target_num_frames - 1) // self.TEMPORAL_DOWNSAMPLE
        if target_latent_frames > latent.shape[2]:
            raise ValueError(
                f"Chunked encode produced {latent.shape[2]} latent frames, cannot "
                f"trim to {target_latent_frames}"
            )
        return latent[:, :, :target_latent_frames]

    def encode(
        self,
        x: Tensor,
        *,
        chunked_encode: bool | None = None,
        target_num_frames: int | None = None,
    ) -> Tensor:
        if chunked_encode is None:
            chunked_encode = self.chunked_encode
        if chunked_encode:
            return self.min_chunk_enc(
                x,
                chunk_size_frames=self.chunk_size_frames,
                target_num_frames=target_num_frames,
            )
        return self.model.encode(x, normalize=True)

    def min_chunk_dec(self, z: Tensor) -> Tensor:
        # Decode the (already-denormalized) latent in overlapping temporal
        # windows, keep each window's core pixel frames, concatenate. The
        # overlap is decoded purely as temporal context and discarded (no
        # blending), larger overlap brings seam frames closer to full decode.
        if z.ndim != 5:
            raise ValueError(f"Expected (B, C, T, H, W), got {z.shape}")
        core = self.chunk_size_latent_frames
        overlap = self.chunk_overlap_latent_frames
        if core < 1 or overlap < 1:
            raise ValueError(
                f"chunk_size/overlap latent frames must be >= 1, got core={core}, overlap={overlap}"
            )
        # Shrink core if free VRAM can't hold a full window (core + 2*overlap).
        free, _ = torch.cuda.mem_get_info(z.device)
        budget = free / DECODE_SAFETY_FACTOR - DECODE_FIXED_BYTES
        window_fit = int(budget / (DECODE_BYTES_PER_LATENT_ELEM * z[0, 0, 0].numel()))
        core = max(1, min(core, window_fit - 2 * overlap))
        t_lat = z.shape[2]
        logger.info(
            f"tiled video decode: t_lat={t_lat} core={core} overlap={overlap} free={free / 2**30:.1f}GiB"
        )

        pieces = []
        for core_lo in range(0, t_lat, core):
            core_hi = min(core_lo + core, t_lat)
            lo = max(0, core_lo - overlap)
            hi = min(t_lat, core_hi + overlap)
            decoded = self.model.decoder(z[:, :, lo:hi])
            # Slice the core latent range to its pixel range within this window.
            # core_lo == 0 keeps the leading pixel frame, interior cores start
            # at 4*(core_lo - lo) - 3. overlap >= 1 keeps local_start >= 0.
            local_start = 0 if core_lo == 0 else 4 * (core_lo - lo) - 3
            local_end = 4 * (core_hi - lo) - 3
            pieces.append(decoded[:, :, local_start:local_end])

        out = torch.cat(pieces, dim=2)
        assert out.shape[2] == 4 * t_lat - 3, (
            f"tiling misaligned: {out.shape[2]} frames, expected {4 * t_lat - 3}"
        )
        return out

    def _full_decode_fits(self, z: Tensor) -> bool:
        free, _ = torch.cuda.mem_get_info(z.device)
        need = DECODE_BYTES_PER_LATENT_ELEM * z[0, 0].numel() + DECODE_FIXED_BYTES
        return need * DECODE_SAFETY_FACTOR < free

    def decode(self, z: Tensor, *, chunked_decode: bool | None = None) -> Tensor:
        auto = chunked_decode is None
        if auto:
            chunked_decode = self.chunked_decode and not self._full_decode_fits(z)
        if not chunked_decode:
            try:
                return self.model.decode(z, normalize=True).clamp(-1, 1)
            except torch.OutOfMemoryError:
                # Estimate was too optimistic, fall through to the chunked path below.
                if not (auto and self.chunked_decode):
                    raise
                logger.warning(f"full video decode OOM (t_latent={z.shape[2]}), falling back to chunked")
                torch.cuda.empty_cache()
        z = self.model.z_normalizer.denormalize(z)
        return self.min_chunk_dec(z).clamp(-1, 1)

    def encode_to_latent_with_ids(
        self,
        video: Tensor,
        fps: float = 24.0,
        target_num_frames: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        if video.ndim != 4:
            raise ValueError(f"Expected (C, T, H, W), got {video.shape}")
        latent = self.encode(
            video.unsqueeze(0),
            target_num_frames=target_num_frames,
        )[0]
        t = latent.shape[1]
        t_coord = times_to_ids(torch.arange(t, device=latent.device).float() * self.TEMPORAL_DOWNSAMPLE / fps)
        return prc_vid(latent, t_coord=t_coord)

    def decode_from_latent_with_ids(self, tokens: Tensor, ids: Tensor) -> Tensor:
        latent = scatter_ids(tokens.unsqueeze(0), ids.unsqueeze(0))[0]
        return self.decode(latent)[0]

    def latent_shape_with_ids(
        self, *, num_frames: int, height: int, width: int
    ) -> tuple[tuple[int, int], Tensor]:
        seq_len = self.get_latent_count(num_frames, height, width)
        channels = self.model.z_dim
        t_latent = 1 + (num_frames - 1) // self.TEMPORAL_DOWNSAMPLE
        h_latent = height // self.SPATIAL_DOWNSAMPLE
        w_latent = width // self.SPATIAL_DOWNSAMPLE
        coords = {
            "t": torch.arange(t_latent),
            "h": torch.arange(h_latent),
            "w": torch.arange(w_latent),
            "l": torch.arange(1),
        }
        ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
        return (seq_len, channels), ids


# ---------------------------------------------------------------------------------------------
# Policy-side wrapper and loader. The Video VAE is frozen,
# loaded from a local file / directory or a Hub repo, and never part of the policy's own
# ``model.safetensors``.
# ---------------------------------------------------------------------------------------------
from .packing import padded_chunk_length  # noqa: E402
from .runtime import random_init_  # noqa: E402

VIDEO_VAE_WEIGHTS_FILENAME = "video_vae.safetensors"


class VideoVAE:
    """Frozen video VAE behind the encode/decode contract the packing code relies on.

    ``encode`` takes ``(B, 3, T, H, W)`` in ``[-1, 1]`` with ``T == 1 (mod 4)`` and returns normalized
    latents ``(B, 96, 1 + (T - 1) // 4, H // 32, W // 32)``. Encoding is chunked (45-frame chunks,
    1-frame overlap, first latent of each later chunk dropped); shorter clips are padded by repeating
    the last frame and trimmed back. ``packing.encode_video`` pads with black to the chunk length
    *before* calling this, so training and deployment see the same latent distribution.
    """

    def __init__(self, model: "ViTNormInference"):
        self._model = model

    @property
    def module(self) -> "ViTNormInference":
        return self._model

    @torch.inference_mode()
    def encode(self, video: Tensor) -> Tensor:
        num_frames = video.shape[2]
        if (num_frames - 1) % 4 != 0:
            raise ValueError(
                f"video VAE encode expects T == 1 (mod 4) frames, got T={num_frames}; "
                "trim or pad the clip to the 4k+1 grid (e.g. 45, 49, 121)"
            )
        chunk = self._model.chunk_size_frames
        padded = padded_chunk_length(num_frames, chunk)
        if padded > num_frames:
            tail = video[:, :, -1:].expand(-1, -1, padded - num_frames, -1, -1)
            video = torch.cat([video, tail], dim=2)
        return self._model.min_chunk_enc(video, target_num_frames=num_frames)

    @torch.inference_mode()
    def encode_task(self, video: Tensor) -> Tensor:
        """Native task-LoRA encoding: independent snapshots/future clip, without DROID black padding."""
        return self._model.encode(video, chunked_encode=False)

    @torch.inference_mode()
    def decode(self, latents: Tensor) -> Tensor:
        return self._model.decode(latents)


def load_video_vae(
    weights: str | None,
    device: str | torch.device = "cpu",
    *,
    compile_model: bool = False,
) -> VideoVAE:
    """Build the frozen Video VAE.

    ``weights``: ``None`` -> random init (shapes only, for wiring tests), a ``.safetensors`` file, a
    distributed-checkpoint directory (``.metadata`` + ``__*.distcp``), or a Hub ``repo_id[:filename]``
    (default filename ``video_vae.safetensors``). Requires NATTEN to construct the model.
    """
    try:
        require_package("natten", "flux3")
    except ImportError as e:
        raise ImportError(
            "The FLUX3 video VAE requires NATTEN. Install a torch/CUDA-matched wheel from "
            "https://whl.natten.org; see docs/source/flux3.mdx."
        ) from e
    params = ViTNormInferenceParams(
        use_compile=compile_model,
        compile_decoder=compile_model,
        chunked_decode=True,
        chunked_encode=True,
    )
    if weights is None:
        model = ViTNormInference(params)
        random_init_(model)
    else:
        path = resolve_weights(weights, VIDEO_VAE_WEIGHTS_FILENAME)
        if os.path.isdir(path) or (Path(path) / ".metadata").is_file():
            model = ViTNormInference(replace(params, dtype="float32"))
            state_dict = {"model": model.model.state_dict()}
            dcp.load(state_dict, checkpoint_id=str(path))  # nosec B614 - distributed checkpoint reader, no pickle
            model.model.load_state_dict(state_dict["model"])
            model.model.bfloat16()
        else:
            model = ViTNormInference(params)
            model.load_state_dict(load_file(path, device="cpu"))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if compile_model:
        model.apply_compile()
    return VideoVAE(model)

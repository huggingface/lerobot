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

"""First-party, weight-compatible runtime for the MoGe v2 native-depth teacher.

This module is a clean-room LeRobot implementation. It is written against the
layout of the published ``model.pt`` checkpoints (e.g.
``Ruicheng/moge-2-vitb-normal``) using stock PyTorch primitives only. It never
imports an upstream repository or a vendored source tree, and it does not need
the third-party helpers the original runtime used (``utils3d``, ``scipy``,
``cv2``): the focal/shift recovery, intrinsics, and point-map geometry are
re-derived in plain torch.

Structural fidelity: every attribute below is named to mirror the published
checkpoint so the whole weight file restores with a single strict
``load_state_dict`` call:

- ``encoder.backbone.*``          DINOv2-style ViT-B/14: cls/pos/mask tokens,
                                  12 blocks with fused-qkv attention and layer
                                  scale, bicubic positional interpolation for
                                  arbitrary token grids.
- ``encoder.output_projections``  one 1x1 conv per selected intermediate layer.
- ``encoder.image_mean/std``      ImageNet normalization buffers.
- ``neck / points_head / normal_head / mask_head``
                                  DPT-style ConvStack decoders: 1x1 input
                                  blocks, x2 resamplers (conv-transpose then
                                  bilinear), residual conv blocks, 1x1 outputs.
- ``scale_head.*``                class-token MLP predicting a metric scale.

Public entry points::

    teacher = load_moge_v2_teacher("path/to/model.pt", device)
    depth = teacher.infer(images, resolution_level=3, num_tokens=256, apply_mask=False)["depth"]

``infer`` matches the upstream call shape used by ``DepthTeacherBundle`` and
returns a dict whose ``depth`` entry is a float32 ``(B, H, W)`` tensor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 - conventional torch alias

__all__ = ["MoGeV2Model", "load_moge_v2_teacher"]


# ---------------------------------------------------------------------------
# Geometry helpers (plain torch replacements for upstream utils3d/scipy calls)
# ---------------------------------------------------------------------------


def _normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Pixel-center uv grid on the normalized view plane, shape (H, W, 2).

    The left-top corner maps to (-width/diagonal, -height/diagonal) and the
    right-bottom corner to (width/diagonal, height/diagonal), in units of the
    image diagonal, so pixel centers span (width-1)/width of each edge.
    """
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5
    u = torch.linspace(
        -span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device
    )
    v = torch.linspace(
        -span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def _solve_focal_shift(
    uv: torch.Tensor,
    xy: torch.Tensor,
    z: torch.Tensor,
    known_focal: float | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-3,
) -> tuple[float, float]:
    """Solve ``min |focal * xy / (z + shift) - uv|`` for shift (and focal).

    One-parameter Levenberg-Marquardt starting from ``shift = 0``. When
    ``known_focal`` is given, the focal is fixed and only the shift is
    optimized. Runs in float64 internally; returns ``(shift, focal)``.
    """
    uv = uv.detach().to(torch.float64)
    xy = xy.detach().to(torch.float64)
    z = z.detach().to(torch.float64)

    def _evaluate(shift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        denom = z + shift
        proj = xy / denom.unsqueeze(-1)
        square = (proj * proj).sum()
        if not torch.isfinite(square) or square <= 0:
            return None
        if known_focal is None:
            cross = (proj * uv).sum()
            focal = cross / square
        else:
            focal = torch.as_tensor(known_focal, dtype=torch.float64, device=uv.device)
        residual = focal * proj - uv
        cost = (residual * residual).sum()
        if not torch.isfinite(cost):
            return None
        # Analytic Jacobian w.r.t. shift.
        proj_deriv = -proj / denom.unsqueeze(-1)
        if known_focal is None:
            cross_deriv = (proj_deriv * uv).sum()
            square_deriv = 2 * (proj * proj_deriv).sum()
            cross = (proj * uv).sum()
            focal_deriv = (cross_deriv * square - cross * square_deriv) / (square * square)
            jacobian = focal_deriv * proj + focal * proj_deriv
        else:
            jacobian = focal * proj_deriv
        return residual, jacobian, cost

    shift = torch.zeros((), dtype=torch.float64, device=uv.device)
    evaluation = _evaluate(shift)
    if evaluation is None:
        return 0.0, 1.0
    residual, jacobian, cost = evaluation
    lam = 1e-3
    for _ in range(max_iterations):
        jtj = (jacobian * jacobian).sum()
        jtr = (jacobian * residual).sum()
        accepted = False
        improvement = 0.0
        for _ in range(12):
            delta = -jtr / (jtj + lam * jtj.clamp_min(1e-12))
            if not torch.isfinite(delta):
                lam *= 10.0
                continue
            candidate = _evaluate(shift + delta)
            if candidate is not None and candidate[2] < cost:
                improvement = float((cost - candidate[2]) / cost.clamp_min(1e-30))
                shift = shift + delta
                residual, jacobian, cost = candidate
                lam = max(lam / 3.0, 1e-9)
                accepted = True
                break
            lam *= 10.0
            if lam > 1e12:
                break
        if not accepted or improvement < tolerance:
            break
    shift = float(shift)
    if known_focal is not None:
        return shift, float(known_focal)
    return shift, float(_closed_form_focal(uv, xy, z, shift))


def _closed_form_focal(uv: torch.Tensor, xy: torch.Tensor, z: torch.Tensor, shift: float) -> torch.Tensor:
    proj = xy / (z + shift).unsqueeze(-1)
    return (proj * uv).sum() / (proj * proj).sum()


def _recover_focal_shift(
    points: torch.Tensor,
    mask: torch.Tensor | None = None,
    focal: torch.Tensor | None = None,
    downsample_size: tuple[int, int] = (64, 64),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover focal length (relative to half the image diagonal) and z-shift.

    Assumes a centered optical axis, an undistorted and isometric image, and a
    point map known up to affine z-shift/scale. Inputs are downsampled with
    nearest interpolation before the per-sample least-squares solve.
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    points = points.reshape(-1, height, width, 3)
    mask = None if mask is None else mask.reshape(-1, height, width)
    focal_flat = None if focal is None else focal.reshape(-1)

    uv = _normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)
    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode="nearest").permute(0, 2, 3, 1)
    uv_lr = (
        F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest")
        .squeeze(0)
        .permute(1, 2, 0)
    )
    mask_lr = (
        None
        if mask is None
        else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode="nearest").squeeze(1)
        > 0
    )

    shifts: list[float] = []
    focals: list[float] = []
    for i in range(points.shape[0]):
        if mask_lr is None:
            points_i, uv_i = points_lr[i], uv_lr
        else:
            selection = mask_lr[i]
            points_i, uv_i = points_lr[i][selection], uv_lr[selection]
        if uv_i.shape[0] < 2:
            # Degenerate sample: fall back to neutral affine parameters.
            shifts.append(0.0)
            focals.append(1.0)
            continue
        known = None if focal_flat is None else float(focal_flat[i])
        shift_i, focal_i = _solve_focal_shift(uv_i, points_i[..., :2], points_i[..., 2], known_focal=known)
        shifts.append(shift_i)
        focals.append(focal_i)

    shift = torch.tensor(shifts, device=points.device, dtype=torch.float32).reshape(shape[:-3])
    if focal_flat is None:
        focal_out = torch.tensor(focals, device=points.device, dtype=torch.float32).reshape(shape[:-3])
    else:
        focal_out = focal.reshape(shape[:-3])
    return focal_out, shift


def _intrinsics_from_focal_center(
    fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor
) -> torch.Tensor:
    """Build (B, 3, 3) pinhole intrinsics from focal lengths and the principal point."""
    batch_shape = fx.shape
    intrinsics = fx.new_zeros(*batch_shape, 3, 3)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = cx
    intrinsics[..., 1, 2] = cy
    intrinsics[..., 2, 2] = 1
    return intrinsics


def _depth_map_to_point_map(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Back-project a (..., H, W) depth map to (..., H, W, 3) camera-space points.

    Uses normalized pixel centers ``(x + 0.5) / W``; ``intrinsics`` entries are
    expressed in the same normalized units.
    """
    height, width = depth.shape[-2], depth.shape[-1]
    u = (torch.arange(width, device=depth.device, dtype=depth.dtype) + 0.5) / width
    v = (torch.arange(height, device=depth.device, dtype=depth.dtype) + 0.5) / height
    batch_view = (1,) * (depth.dim() - 2)
    fx = intrinsics[..., 0, 0][..., None, None]
    fy = intrinsics[..., 1, 1][..., None, None]
    cx = intrinsics[..., 0, 2][..., None, None]
    cy = intrinsics[..., 1, 2][..., None, None]
    points_x = (u.view(*batch_view, 1, width) - cx) * depth / fx
    points_y = (v.view(*batch_view, height, 1) - cy) * depth / fy
    return torch.stack([points_x, points_y, depth], dim=-1)


# ---------------------------------------------------------------------------
# ViT-B/14 backbone (DINOv2 layout, stock PyTorch)
# ---------------------------------------------------------------------------


class _LayerScale(nn.Module):
    """Per-channel multiplicative residual scaling (``ls1`` / ``ls2``)."""

    def __init__(self, dim: int, init_values: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class _Attention(nn.Module):
    """Fused-qkv multi-head attention computed through ``scaled_dot_product_attention``."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, proj_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.permute(0, 2, 1, 3).reshape(batch, tokens, channels)
        return self.proj(x)


class _Mlp(nn.Module):
    """Standard GELU MLP with ``fc1`` / ``fc2`` linear layers."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _ViTBlock(nn.Module):
    """Pre-norm transformer block with layer-scaled attention and MLP residuals."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, init_values: float = 1.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads)
        self.ls1 = _LayerScale(dim, init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = _LayerScale(dim, init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(nn.Module):
    """Patch embedding as a strided convolution (``proj``), flattened to (B, N, C)."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C, H, W)
        height, width = x.shape[-2:]
        return x.flatten(2).transpose(1, 2)  # (B, H*W, C)


class _DINOv2Backbone(nn.Module):
    """DINOv2-style ViT with bicubic positional interpolation for dynamic grids.

    The published MoGe v2 checkpoint fine-tunes the ``dinov2_vitb14`` layout
    (518 px / 14 px = 37x37 patch grid + class token); other widths share the
    same block structure and only differ in width/depth/heads.
    """

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        init_values: float = 1.0,
        interpolate_offset: float = 0.1,
        interpolate_antialias: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.interpolate_offset = interpolate_offset
        self.interpolate_antialias = interpolate_antialias

        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.patch_embed = _PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.blocks = nn.ModuleList(
            _ViTBlock(embed_dim, num_heads, mlp_ratio, init_values) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def _interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Bicubic positional interpolation onto the (h/14, w/14) token grid.

        Follows the DINOv2 scheme: float32 interpolation with the historical
        ``+0.1`` scale-factor offset, and a short-circuit when the input grid
        already matches the stored 37x37 embedding.
        """
        previous_dtype = x.dtype
        num_patches = x.shape[1] - 1
        stored_patches = self.pos_embed.shape[1] - 1
        if num_patches == stored_patches and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0, :]
        patch_pos_embed = pos_embed[:, 1:, :]
        dim = x.shape[-1]
        grid_h, grid_w = h // self.patch_size, w // self.patch_size
        grid = int(math.sqrt(stored_patches))  # stored grid is square
        if stored_patches != grid * grid:
            raise ValueError(f"Non-square positional embedding with {stored_patches} patches.")
        scale_y = float(grid_h + self.interpolate_offset) / grid
        scale_x = float(grid_w + self.interpolate_offset) / grid
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, grid, grid, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            scale_factor=(scale_y, scale_x),
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        class_pos_embed = class_pos_embed[:, None, :].expand(patch_pos_embed.shape[0], -1, -1)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(previous_dtype)

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return x + self._interpolate_pos_encoding(x, h, w)

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> tuple:
        """Run the transformer and collect block outputs after the final norm.

        ``n`` is either a count (last ``n`` blocks) or explicit block indices.
        With ``return_class_token`` the result pairs each layer's
        ``(patch_tokens, class_token)``.
        """
        img_h, img_w = x.shape[-2:]
        x = self._prepare_tokens(x)
        indices = n if isinstance(n, Sequence) else range(len(self.blocks) - n, len(self.blocks))
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in indices:
                outputs.append(x)
        if len(outputs) != len(indices):
            raise ValueError(f"Only {len(outputs)} of the requested {len(indices)} blocks exist.")
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1:] for out in outputs]
        if reshape:
            batch = outputs[0].shape[0]
            grid_h, grid_w = img_h // self.patch_size, img_w // self.patch_size
            outputs = [
                out.reshape(batch, grid_h, grid_w, -1).permute(0, 3, 1, 2).contiguous() for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens, strict=True))
        return tuple(outputs)


# dinov2 backbone name -> (embed_dim, depth, num_heads) at patch size 14 / 518 px.
_DINOV2_BACKBONES: dict[str, tuple[int, int, int]] = {
    "dinov2_vits14": (384, 12, 6),
    "dinov2_vitb14": (768, 12, 12),
    "dinov2_vitl14": (1024, 24, 16),
}


class _DinoV2Encoder(nn.Module):
    """ViT encoder front end: resize, normalize, sum projected intermediate layers.

    The input RGB image in [0, 1] is bilinearly resized (antialiased) to
    ``(token_rows * 14, token_cols * 14)``, ImageNet-normalized, and encoded;
    each selected intermediate layer is projected by a 1x1 conv and the
    projections are summed into one (B, dim_out, rows, cols) feature map.
    """

    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        intermediate_layers: int | Sequence[int] = 11,
        dim_out: int = 768,
    ) -> None:
        super().__init__()
        if backbone not in _DINOV2_BACKBONES:
            raise ValueError(f"Unsupported backbone {backbone!r}; available: {sorted(_DINOV2_BACKBONES)}.")
        embed_dim, depth, num_heads = _DINOV2_BACKBONES[backbone]
        self.intermediate_layers = intermediate_layers
        self.backbone = _DINOv2Backbone(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
        num_projections = (
            len(intermediate_layers)
            if isinstance(intermediate_layers, Sequence)
            else int(intermediate_layers)
        )
        self.output_projections = nn.ModuleList(
            nn.Conv2d(embed_dim, dim_out, kernel_size=1) for _ in range(num_projections)
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _layer_indices(self, depth: int) -> Sequence[int]:
        if isinstance(self.intermediate_layers, Sequence):
            return list(self.intermediate_layers)
        return list(range(depth - self.intermediate_layers, depth))

    def forward(
        self,
        image: torch.Tensor,
        token_rows: int,
        token_cols: int,
        return_class_token: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        image = F.interpolate(
            image, (token_rows * 14, token_cols * 14), mode="bilinear", align_corners=False, antialias=True
        )
        image = (image - self.image_mean) / self.image_std
        features = self.backbone.get_intermediate_layers(
            image, n=self._layer_indices(len(self.backbone.blocks)), return_class_token=True
        )
        x = torch.stack(
            [
                proj(feat.permute(0, 2, 1).unflatten(2, (token_rows, token_cols)).contiguous())
                for proj, (feat, _cls) in zip(self.output_projections, features, strict=True)
            ],
            dim=1,
        ).sum(dim=1)
        if return_class_token:
            return x, features[-1][1]
        return x


# ---------------------------------------------------------------------------
# DPT-style ConvStack decoder (neck and prediction heads)
# ---------------------------------------------------------------------------

_NORM_MODES = ("none", "group_norm", "layer_norm", "instance_norm")
_ACTIVATIONS = ("relu", "leaky_relu", "silu", "elu")


def _norm_layer(mode: str, channels: int) -> nn.Module:
    if mode == "none":
        return nn.Identity()
    if mode == "group_norm":
        return nn.GroupNorm(channels // 32, channels)
    if mode == "layer_norm":
        return nn.GroupNorm(1, channels)
    if mode == "instance_norm":
        return nn.InstanceNorm2d(channels)
    raise ValueError(f"Unsupported norm {mode!r}; available: {_NORM_MODES}.")


def _activation_layer(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)
    if name == "silu":
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation {name!r}; available: {_ACTIVATIONS}.")


class _ResidualConvBlock(nn.Module):
    """Conv residual block with a fixed six-slot layer sequence.

    Slot layout matters for weight compatibility: the two 3x3 convolutions live
    at ``layers.2`` / ``layers.5``, after (norm, activation) pairs that are
    identities for the published ``res_block_*_norm='none'`` configuration.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_channels: int | None = None,
        kernel_size: int = 3,
        padding_mode: str = "replicate",
        activation: str = "relu",
        in_norm: str = "none",
        hidden_norm: str = "none",
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.layers = nn.Sequential(
            _norm_layer(in_norm, in_channels),
            _activation_layer(activation),
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
            _norm_layer(hidden_norm, hidden_channels),
            _activation_layer(activation),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
        )
        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + self.skip_connection(x)


def _make_resampler(in_channels: int, out_channels: int, type_: str, scale_factor: int = 2) -> nn.Sequential:
    """x2 upsampling resampler used between ConvStack levels."""
    if type_ == "conv_transpose":
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate"
            ),
        )
    if type_ in ("nearest", "bilinear"):
        return nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor,
                mode=type_,
                align_corners=False if type_ == "bilinear" else None,
            ),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="replicate"
            ),
        )
    raise ValueError(f"Unsupported resampler {type_!r}; available: conv_transpose, nearest, bilinear.")


def _make_mlp(dims: Sequence[int]) -> nn.Sequential:
    """Linear/ReLU MLP whose parameter indices match the published ``scale_head``."""
    layers: list[nn.Module] = []
    for dim_in, dim_out in zip(dims[:-2], dims[1:-1], strict=True):
        layers.extend([nn.Linear(dim_in, dim_out), nn.ReLU()])
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def _as_sequence(value: Any, length: int) -> Sequence[Any]:
    """Normalize a per-level config value that may be scalar or a list."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return [value] * length


class _ConvStack(nn.Module):
    """DPT-style multi-level conv stack.

    Five levels: a 1x1 ``input_block`` fuses each incoming feature map into the
    running pyramid (levels 1..4 typically receive only uv coordinate maps),
    ``res_blocks`` refine each level, ``output_blocks`` produce per-level
    outputs, and ``resamplers`` upsample x2 between levels.
    """

    def __init__(
        self,
        dim_in: Sequence[int | None] | int | None,
        dim_res_blocks: Sequence[int],
        dim_out: Sequence[int | None] | int | None = None,
        resamplers: str | Sequence[str] = "conv_transpose",
        num_res_blocks: int | Sequence[int] = 1,
        dim_times_res_block_hidden: int = 1,
        res_block_in_norm: str = "layer_norm",
        res_block_hidden_norm: str = "group_norm",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        levels = len(dim_res_blocks)
        dim_in_seq = _as_sequence(dim_in, levels)
        dim_out_seq = _as_sequence(dim_out, levels)
        num_res_blocks_seq = _as_sequence(num_res_blocks, levels)
        resamplers_seq = _as_sequence(resamplers, max(levels - 1, 1))[: max(levels - 1, 1)]

        self.input_blocks = nn.ModuleList(
            nn.Conv2d(dim_in_, dim_res, kernel_size=1) if dim_in_ is not None else nn.Identity()
            for dim_in_, dim_res in zip(dim_in_seq, dim_res_blocks, strict=True)
        )
        self.resamplers = nn.ModuleList(
            _make_resampler(dim_prev, dim_succ, type_)
            for dim_prev, dim_succ, type_ in zip(
                dim_res_blocks[:-1], dim_res_blocks[1:], resamplers_seq, strict=True
            )
        )
        self.res_blocks = nn.ModuleList(
            nn.Sequential(
                *(
                    _ResidualConvBlock(
                        dim_res,
                        dim_res,
                        dim_times_res_block_hidden * dim_res,
                        activation=activation,
                        in_norm=res_block_in_norm,
                        hidden_norm=res_block_hidden_norm,
                    )
                    for _ in range(count)
                )
            )
            for count, dim_res in zip(num_res_blocks_seq, dim_res_blocks, strict=True)
        )
        self.output_blocks = nn.ModuleList(
            nn.Conv2d(dim_res, dim_out_, kernel_size=1) if dim_out_ is not None else nn.Identity()
            for dim_out_, dim_res in zip(dim_out_seq, dim_res_blocks, strict=True)
        )

    def forward(self, in_features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        out_features = []
        x = None
        for i in range(len(self.res_blocks)):
            feature = self.input_blocks[i](in_features[i])
            x = feature if i == 0 else x + feature
            x = self.res_blocks[i](x)
            out_features.append(self.output_blocks[i](x))
            if i < len(self.res_blocks) - 1:
                x = self.resamplers[i](x)
        return out_features


# ---------------------------------------------------------------------------
# MoGe v2 model
# ---------------------------------------------------------------------------


class MoGeV2Model(nn.Module):
    """First-party MoGe v2 monocular geometry model.

    Architecture: a DINOv2 ViT-B/14 encoder, a shared conv ``neck`` over the
    token grid (with uv-coordinate conditioning at every level), ConvStack
    ``points_head`` / ``normal_head`` / ``mask_head``, and a class-token
    ``scale_head`` for the metric scale. Depth is recovered from the predicted
    affine point map by solving for the camera focal (relative to half the
    image diagonal) and the z shift, then applying the metric scale.
    """

    def __init__(
        self,
        *,
        encoder: dict[str, Any],
        neck: dict[str, Any],
        points_head: dict[str, Any] | None = None,
        mask_head: dict[str, Any] | None = None,
        normal_head: dict[str, Any] | None = None,
        scale_head: dict[str, Any] | None = None,
        remap_output: str = "linear",
        num_tokens_range: Sequence[int] = (1200, 3600),
    ) -> None:
        super().__init__()
        if remap_output not in ("linear", "sinh", "exp", "sinh_exp"):
            raise ValueError(f"Unsupported remap_output {remap_output!r}.")
        self.remap_output = remap_output
        self.num_tokens_range = list(num_tokens_range)
        self.encoder = _DinoV2Encoder(**encoder)
        self.neck = _ConvStack(**neck)
        if points_head is not None:
            self.points_head = _ConvStack(**points_head)
        if mask_head is not None:
            self.mask_head = _ConvStack(**mask_head)
        if normal_head is not None:
            self.normal_head = _ConvStack(**normal_head)
        if scale_head is not None:
            self.scale_head = _make_mlp(scale_head["dims"])

    # ------------------------------ loading ------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MoGeV2Model:
        """Build a model from a checkpoint ``model_config`` dict."""
        allowed = {
            "encoder",
            "neck",
            "points_head",
            "mask_head",
            "normal_head",
            "scale_head",
            "remap_output",
            "num_tokens_range",
        }
        missing = {"encoder", "neck"} - set(config)
        if missing:
            raise ValueError(f"model_config is missing required keys: {sorted(missing)}.")
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"model_config has unsupported keys: {sorted(unknown)}.")
        return cls(**config)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> MoGeV2Model:
        """Load a published ``model.pt`` with ``weights_only`` and strict state restore.

        The checkpoint must carry ``model_config`` and ``model`` entries; every
        tensor is restored by a strict ``load_state_dict`` so any layout drift
        fails loudly instead of silently running partial weights.
        """
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MoGe v2 checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model_config" not in checkpoint or "model" not in checkpoint:
            raise ValueError(f"{path} is not a published MoGe v2 checkpoint (needs model_config/model).")
        model = cls.from_config(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"], strict=True)
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=torch.device(device))
        return model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    # ------------------------------ forward ------------------------------

    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        """Undo the output parameterization of the points head."""
        if self.remap_output == "linear":
            return points
        if self.remap_output == "sinh":
            return torch.sinh(points)
        if self.remap_output == "exp":
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            return torch.cat([xy * z, z], dim=-1)
        if self.remap_output == "sinh_exp":
            xy, z = points.split([2, 1], dim=-1)
            return torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        raise ValueError(f"Invalid remap output type: {self.remap_output}")

    def forward(self, image: torch.Tensor, num_tokens: int) -> dict[str, torch.Tensor]:
        """Run the network at a target base token count.

        ``num_tokens`` picks the ViT token grid: the image is resized so it
        holds about ``num_tokens`` patches (rounded per side, aspect
        preserved). Returns raw ``points`` / ``normal`` / ``mask`` /
        ``metric_scale`` at the input image resolution.
        """
        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype

        aspect_ratio = img_w / img_h
        base_h = round((num_tokens / aspect_ratio) ** 0.5)
        base_w = round((num_tokens * aspect_ratio) ** 0.5)

        features, cls_token = self.encoder(image, base_h, base_w, return_class_token=True)
        pyramid: list[torch.Tensor | None] = [features, None, None, None, None]
        for level in range(5):
            uv = _normalized_view_plane_uv(
                width=base_w * 2**level,
                height=base_h * 2**level,
                aspect_ratio=aspect_ratio,
                dtype=dtype,
                device=device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            pyramid[level] = uv if pyramid[level] is None else torch.cat([pyramid[level], uv], dim=1)

        features = self.neck(pyramid)
        points = self.points_head(features)[-1] if hasattr(self, "points_head") else None
        normal = self.normal_head(features)[-1] if hasattr(self, "normal_head") else None
        mask = self.mask_head(features)[-1] if hasattr(self, "mask_head") else None
        metric_scale = self.scale_head(cls_token) if hasattr(self, "scale_head") else None

        def _resize(value: torch.Tensor | None) -> torch.Tensor | None:
            return (
                F.interpolate(value, (img_h, img_w), mode="bilinear", align_corners=False, antialias=False)
                if value is not None
                else None
            )

        points, normal, mask = (_resize(value) for value in (points, normal, mask))
        if points is not None:
            points = self._remap_points(points.permute(0, 2, 3, 1))
        if normal is not None:
            normal = F.normalize(normal.permute(0, 2, 3, 1), dim=-1)
        if mask is not None:
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        output = {"points": points, "normal": normal, "mask": mask, "metric_scale": metric_scale}
        return {key: value for key, value in output.items() if value is not None}

    # ------------------------------ inference ------------------------------

    @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        num_tokens: int | None = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: bool = True,
        fov_x: float | None = None,
        use_fp16: bool = True,
    ) -> dict[str, torch.Tensor]:
        """User-friendly inference matching the upstream call shape.

        Args:
            image: (B, 3, H, W) or (3, H, W) RGB tensor in [0, 1].
            num_tokens: base ViT token budget; ``None`` derives it from
                ``resolution_level`` (level 0 -> min, level 9 -> max of
                ``num_tokens_range``).
            resolution_level: 0-9 resolution knob, ignored when ``num_tokens``
                is given.
            force_projection: recompute the point map from the recovered depth
                and intrinsics.
            apply_mask: set masked-out pixels to inf (points/depth) / zeros
                (normal).
            fov_x: optional horizontal field of view in degrees; recovered from
                the point map when ``None``.
            use_fp16: run the network forward under float16 autocast (CUDA).

        Returns:
            Dict with ``depth`` (B, H, W) float32 and, when computed,
            ``points`` (B, H, W, 3), ``intrinsics`` (B, 3, 3), ``mask``
            (B, H, W) bool, and ``normal`` (B, H, W, 3). Batch dimension is
            dropped if the input had none.
        """
        omit_batch_dim = image.dim() == 3
        if omit_batch_dim:
            image = image.unsqueeze(0)
        image = image.to(dtype=self.dtype, device=self.device)

        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=use_fp16 and self.dtype != torch.float16,
        ):
            output = self.forward(image, num_tokens=num_tokens)
        points, normal, mask, metric_scale = (
            output.get(key, None) for key in ("points", "normal", "mask", "metric_scale")
        )
        points, normal, mask, metric_scale = (
            value.float() if isinstance(value, torch.Tensor) else value
            for value in (points, normal, mask, metric_scale)
        )

        mask_binary = mask > 0.5 if mask is not None else None
        depth = None
        intrinsics = None
        if points is not None:
            if fov_x is None:
                focal, shift = _recover_focal_shift(points, mask_binary)
            else:
                focal = (
                    aspect_ratio
                    / (1 + aspect_ratio**2) ** 0.5
                    / torch.tan(
                        torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2)
                    )
                )
                if focal.ndim == 0:
                    focal = focal[None].expand(points.shape[0])
                _, shift = _recover_focal_shift(points, mask_binary, focal=focal)
            fx = focal / 2 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio
            fy = focal / 2 * (1 + aspect_ratio**2) ** 0.5
            intrinsics = _intrinsics_from_focal_center(
                fx, fy, torch.tensor(0.5, device=points.device), torch.tensor(0.5, device=points.device)
            )
            points[..., 2] += shift[..., None, None]
            if mask_binary is not None:
                mask_binary = mask_binary & (points[..., 2] > 0)
            depth = points[..., 2].clone()

        if force_projection and depth is not None:
            points = _depth_map_to_point_map(depth, intrinsics)

        if metric_scale is not None:
            if points is not None:
                points = points * metric_scale[:, None, None, None]
            if depth is not None:
                depth = depth * metric_scale[:, None, None]

        if apply_mask and mask_binary is not None:
            if points is not None:
                points = torch.where(mask_binary[..., None], points, torch.inf)
            if depth is not None:
                depth = torch.where(mask_binary, depth, torch.inf)
            if normal is not None:
                normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal))

        return_dict = {
            "points": points,
            "intrinsics": intrinsics,
            "depth": depth,
            "mask": mask_binary,
            "normal": normal,
        }
        return_dict = {key: value for key, value in return_dict.items() if value is not None}
        if omit_batch_dim:
            return_dict = {key: value.squeeze(0) for key, value in return_dict.items()}
        return return_dict


def load_moge_v2_teacher(
    checkpoint_path: str | Path,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> MoGeV2Model:
    """Load and freeze the first-party MoGe v2 depth teacher.

    Teachers are inference-only: gradients are disabled and the module is put
    in eval mode, mirroring how ``DepthTeacherBundle`` freezes its modules.
    """
    model = MoGeV2Model.from_checkpoint(checkpoint_path, dtype=dtype)
    model.requires_grad_(False)
    if device is not None:
        model = model.to(device=torch.device(device))
    return model.eval()

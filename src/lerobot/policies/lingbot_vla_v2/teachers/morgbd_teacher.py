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

"""First-party LingBot-Depth (MoRGBD) teacher: RGBD-DINOv2-L target extraction.

This module is the LeRobot-maintained, weight-compatible runtime for the
published LingBot-Depth ``depth/model.pt`` checkpoint (RGBD-DINOv2-L: a
DINOv2 ViT-L/14 backbone extended with a depth patch embedding, fused by
token concatenation). It was written against the behavioral contract of the
published checkpoint alone: it imports no upstream source tree, mirrors no
upstream code path, and depends only on ``torch``.

What it provides
----------------
``MoRGBDTeacher.from_pretrained(path)`` loads the published checkpoint with
``torch.load(..., weights_only=True)`` and ``load_state_dict(strict=False)``
and ``MoRGBDTeacher.infer_feat`` reproduces the upstream teacher semantics
used by the alignment recipe::

    feat, cls = teacher.infer_feat(
        image,                   # (B, 3, H, W) in [0, 1]
        depth,                   # (B, H, W) or (B, 1, H, W)
        num_tokens=256,
        resolution_level=3,      # only used when num_tokens is None
        depth_down_scale=1,      # accepted for signature parity; inert (see below)
        enable_depth_mask=False,
    )
    # feat: (B, 1024, 16, 16) patch features for 256 tokens on a square image
    # cls:  (B, 1024) class token of the last extracted layer

Semantics reproduced exactly (including op order / dtype flow under bf16
autocast): image resized to a 14-pixel-per-token grid (bilinear, antialias)
and ImageNet-normalized; depth resized nearest to the same grid, invalids
clamped to zero, validity mask ``> 0.01``; image and depth patch embeddings
each add a bicubic-interpolated positional encoding offset by a distinct
data-type constant (+1 image, +2 depth); a class token (with its positional
encoding) is prepended; the concatenated sequence runs through 24 pre-norm
ViT blocks; the four extracted layers (config: [5, 11, 17, 23]) are
final-normed, image tokens are 1x1-projected and summed into the output
feature map; the class token comes from the last extracted layer.

The ``strict=False`` load quirk
-------------------------------
The published checkpoint stores its depth patch embedding under
``encoder.backbone.depth_mask_patch_embed.*`` while the model it configures
(``depth_emb_mode: "conv_1c"``) constructs ``...depth_patch_embed.*``. With
``strict=False`` the checkpoint keys are silently dropped and the module's
depth patch embedding therefore stays at its construction initialization
(``nn.Conv2d`` default init; the ViT init routines touch only linears).
Second, the upstream model accepts the ``normal_head`` config entry but
never constructs that stack, so the checkpoint's ``normal_head.*`` weights
are dropped as well. Upstream ships and loads the checkpoint exactly this
way, so this runtime reproduces both quirks rather than papering over
them: the only tolerated missing keys are ``depth_patch_embed.*`` and the
only tolerated unexpected keys are ``depth_mask_patch_embed.*`` /
``normal_head.*`` — anything else raises. Any other incompatibility means
the checkpoint is not the published one and must fail loudly, unlike the
upstream silent drop.

Scope notes
-----------
* ``depth_down_scale`` is inert: in the upstream API it is forwarded through
  ``**kwargs`` into token preparation and never read there. It is accepted
  (and ignored) so call sites keep working unmodified.
* ``enable_depth_mask=True`` (upstream: drop depth-invalid patches and run
  per-sample block-diagonal attention via xformers) is not implemented; the
  released alignment recipe always runs ``enable_depth_mask=False`` and the
  teacher raises ``NotImplementedError`` for anything else.
* The neck / depth / mask decoder stacks are constructed so the published
  checkpoint's decoder weights are absorbed exactly as the upstream load
  absorbs them, but no decode forward is ported: this runtime exposes
  feature extraction only.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 - conventional torch alias
from torch import Tensor, nn

__all__ = ["MoRGBDTeacher"]

_LAYERNORM_EPS = 1e-6

# Architecture table for the DINOv2 backbone names accepted in the published
# ``model_config``. Shared construction constants follow the published
# configuration: patch 14, img_size 518 (37x37 patch grid), LayerScale
# init 1.0, MLP FFN, no block chunking, no register tokens, bicubic
# positional-encoding interpolation with the +0.1 offset kludge.
_DINOV2_ARCHS: dict[str, dict[str, int]] = {
    "dinov2_vits14": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "dinov2_vitb14": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "dinov2_vitl14": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
}
_DEFAULT_PATCH_SIZE = 14
_DEFAULT_IMG_SIZE = 518
_DEFAULT_INIT_VALUES = 1.0
_DEFAULT_MLP_RATIO = 4.0
_DEFAULT_INTERPOLATE_OFFSET = 0.1

# The only key groups tolerated by the strict=False load (see module
# docstring). Everything else must match the constructed module exactly.
# ``normal_head.*`` is unexpected too because the upstream model accepts the
# ``normal_head`` config entry but never constructs that stack, so those
# checkpoint weights are dropped by the upstream strict=False load as well.
_QUIRK_MISSING_PREFIX = "encoder.backbone.depth_patch_embed."
_QUIRK_UNEXPECTED_PREFIXES = (
    "encoder.backbone.depth_mask_patch_embed.",
    "normal_head.",
)


class _LayerScale(nn.Module):
    """Per-channel residual scaling (``ls1.gamma`` / ``ls2.gamma`` keys)."""

    def __init__(self, dim: int, init_values: float | Tensor) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class _Mlp(nn.Module):
    """GELU FFN with ``mlp.fc1`` / ``mlp.fc2`` keys."""

    def __init__(self, in_features: int, hidden_features: int, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Attention(nn.Module):
    """Multi-head self-attention with fused ``attn.qkv`` / ``attn.proj`` keys.

    Uses ``F.scaled_dot_product_attention`` — the same kernel the upstream
    memory-efficient path falls back to when xformers is absent, and the only
    backend this first-party runtime needs.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, proj_bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, dim = x.shape
        head_dim = dim // self.num_heads
        qkv = (
            self.qkv(x)
            .reshape(batch_size, num_tokens, 3, self.num_heads, head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_tokens, dim)
        return self.proj(out)


class _ViTBlock(nn.Module):
    """Pre-norm transformer block with LayerScale residuals (eval math only)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        proj_bias: bool,
        ffn_bias: bool,
        init_values: float,
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=_LAYERNORM_EPS)
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(dim, num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.ls1 = _LayerScale(dim, init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), bias=ffn_bias)
        self.ls2 = _LayerScale(dim, init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(nn.Module):
    """Patchify via strided conv with ``patch_embed.proj.*`` keys, no norm."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Input spatial size {(height, width)} must be a multiple of "
                f"patch size {self.patch_size}."
            )
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class _RGBDDinoVisionTransformer(nn.Module):
    """DINOv2 ViT extended with a 1-channel depth patch embedding.

    Depth fusion is token concatenation: image and depth patches are embedded
    separately, each add a (data-type-offset) interpolated positional
    encoding, and the two token streams are concatenated behind the class
    token before the blocks. Only the batched no-masking path is implemented
    (``enable_depth_mask=False``), which is the exact path the released
    LingBot-Depth alignment recipe runs.
    """

    def __init__(
        self,
        img_size: int = _DEFAULT_IMG_SIZE,
        patch_size: int = _DEFAULT_PATCH_SIZE,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = _DEFAULT_MLP_RATIO,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: float | None = _DEFAULT_INIT_VALUES,
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = _DEFAULT_INTERPOLATE_OFFSET,
    ) -> None:
        super().__init__()
        if num_register_tokens != 0:
            raise NotImplementedError(
                "First-party MoRGBD teacher supports only num_register_tokens=0 "
                "(the published RGBD-DINOv2-L configuration)."
            )
        self.patch_size = patch_size
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.num_tokens = 1  # class token

        grid = img_size // patch_size
        num_patches = grid * grid
        self.patch_embed = _PatchEmbed(3, embed_dim, patch_size)
        # Constructed (and, per the strict=False quirk, left at construction
        # init by the published checkpoint) 1-channel depth patch embedding.
        self.depth_patch_embed = _PatchEmbed(1, embed_dim, patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.register_tokens = None
        self.blocks = nn.ModuleList(
            _ViTBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=_LAYERNORM_EPS)
        # Present in the published checkpoint; unused by feature extraction.
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

    def _interpolate_patch_pos_embed(self, grid_h: int, grid_w: int, target_dtype: torch.dtype) -> Tensor:
        """Bicubic-resize the patch positional grid to ``(grid_h, grid_w)``.

        Computed in float32 and cast back to ``target_dtype`` (the token
        dtype), preserving the historical ``+0.1`` scale-factor offset used
        to dodge floating-point truncation in the resize. The unresized
        shortcut returns the raw parameter (no cast), matching the upstream
        dtype flow.
        """
        patch_pos_embed = self.pos_embed[:, 1:]  # (1, N, D), class entry excluded
        num_patches = patch_pos_embed.shape[1]
        grid_origin = int(math.sqrt(num_patches))
        if num_patches != grid_origin * grid_origin:
            raise ValueError(
                f"Positional embedding has {num_patches} patches, not a square grid."
            )
        if grid_h * grid_w == num_patches and grid_h == grid_w:
            return patch_pos_embed
        kwargs: dict[str, Any] = (
            {"scale_factor": (float(grid_h + self.interpolate_offset) / grid_origin,
                              float(grid_w + self.interpolate_offset) / grid_origin)}
            if self.interpolate_offset > 0
            else {"size": (grid_h, grid_w)}
        )
        resized = F.interpolate(
            patch_pos_embed.float().reshape(1, grid_origin, grid_origin, -1).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        return resized.permute(0, 2, 3, 1).flatten(1, 2).to(target_dtype)

    def prepare_rgbd_tokens(self, x_img: Tensor, x_depth: Tensor) -> Tensor:
        """Embed image + depth, add offset positional encodings, prepend cls."""
        batch_size = x_img.shape[0]
        img_tokens = self.patch_embed(x_img)
        depth_tokens = self.depth_patch_embed(x_depth)
        pos = self._interpolate_patch_pos_embed(
            x_img.shape[-2] // self.patch_size, x_img.shape[-1] // self.patch_size, img_tokens.dtype
        )
        # Data-type encodings: image tokens shift the positional grid by +1,
        # depth tokens by +2, so the transformer can tell the modalities apart.
        img_tokens = img_tokens + (1 + pos).repeat(batch_size, 1, 1)
        depth_tokens = depth_tokens + (2 + pos).repeat(batch_size, 1, 1)
        cls = self.cls_token + self.pos_embed[:, :1]
        return torch.cat([cls.expand(batch_size, -1, -1), img_tokens, depth_tokens], dim=1)

    def intermediate_layer_features(
        self, x_img: Tensor, x_depth: Tensor, layers: Sequence[int] | int
    ) -> list[Tensor]:
        """Run the blocks; return final-normed token maps for the given layers.

        ``layers`` as int selects the last n blocks (upstream convention);
        as a sequence it selects block indices directly. Image tokens occupy
        positions ``[1 : 1 + num_img_tokens]`` of each output; depth tokens
        follow and are dropped by the caller.
        """
        if isinstance(layers, int):
            blocks_to_take = range(len(self.blocks) - layers, len(self.blocks))
        else:
            blocks_to_take = sorted(layers)
        x = self.prepare_rgbd_tokens(x_img, x_depth)
        outputs: list[Tensor] = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in blocks_to_take:
                normed = self.norm(x)
                outputs.append(normed)
        if len(outputs) != len(list(blocks_to_take)):
            raise ValueError(
                f"Requested layers {layers} exceed the {len(self.blocks)}-block backbone."
            )
        return outputs


class _RGBDEncoder(nn.Module):
    """RGBD encoding pipeline around the ViT: resize, normalize, project, sum.

    Mirrors the upstream encoder contract: inputs are resized to a
    14-pixel-per-token grid (image bilinear+antialias then ImageNet
    normalization; depth nearest), invalid depth is zeroed with a
    ``> 0.01`` validity mask and an optional input remap, and the extracted
    layer features are 1x1-projected and summed into one map.
    """

    def __init__(
        self,
        backbone: str,
        intermediate_layers: Sequence[int] | int,
        dim_out: int,
        in_chans: int = 3,
        img_depth_fuse_mode: str = "cat_token",
        depth_emb_mode: str = "conv_1c",
        img_mask_ratio: float = 0.0,
        **legacy_kwargs: Any,
    ) -> None:
        super().__init__()
        if backbone not in _DINOV2_ARCHS:
            supported = ", ".join(sorted(_DINOV2_ARCHS))
            raise NotImplementedError(
                f"First-party MoRGBD teacher supports backbones {supported}; got {backbone!r}."
            )
        if img_depth_fuse_mode != "cat_token":
            raise NotImplementedError(
                f"img_depth_fuse_mode={img_depth_fuse_mode!r} is not supported; "
                "the published configuration uses 'cat_token'."
            )
        if depth_emb_mode != "conv_1c":
            raise NotImplementedError(
                f"depth_emb_mode={depth_emb_mode!r} is not supported; "
                "the published configuration uses 'conv_1c'."
            )
        if in_chans != 3:
            raise NotImplementedError(f"in_chans={in_chans} is not supported; the RGB image path is fixed.")
        if img_mask_ratio != 0.0:
            raise NotImplementedError("img_mask_ratio > 0 is not supported by this runtime.")
        # `strict` and `ignore_layers` gate an unused upstream pretrained-weight
        # refresh path; accepted for config compatibility, inert here.
        del legacy_kwargs

        self.intermediate_layers = intermediate_layers
        self.backbone = _RGBDDinoVisionTransformer(**_DINOV2_ARCHS[backbone])
        dim_features = self.backbone.blocks[0].attn.qkv.in_features
        num_projections = intermediate_layers if isinstance(intermediate_layers, int) else len(intermediate_layers)
        self.output_projections = nn.ModuleList(
            nn.Conv2d(dim_features, dim_out, kernel_size=1, stride=1, padding=0)
            for _ in range(num_projections)
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self,
        image: Tensor,
        depth: Tensor,
        token_rows: int,
        token_cols: int,
        remap_depth_in: str = "linear",
    ) -> tuple[Tensor, Tensor]:
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        grid_hw = (token_rows * self.backbone.patch_size, token_cols * self.backbone.patch_size)
        image = F.interpolate(image, grid_hw, mode="bilinear", align_corners=False, antialias=True)
        image = (image - self.image_mean) / self.image_std
        depth = F.interpolate(depth, grid_hw, mode="nearest")

        # Invalid depth (inf / nan / non-positive) is clamped to zero and a
        # validity mask keeps only values above 1 cm.
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mask = depth > 0.01
        depth = depth * depth_mask.to(depth.dtype)
        if remap_depth_in == "linear":
            pass
        elif remap_depth_in == "log":
            depth = torch.log(depth)
            depth = depth * depth_mask.to(depth.dtype)
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise NotImplementedError(f"remap_depth_in={remap_depth_in!r} is not supported.")

        outputs = self.backbone.intermediate_layer_features(
            image, depth, self.intermediate_layers
        )
        num_img_tokens = token_rows * token_cols
        projected = [
            projection(
                normed[:, 1 : 1 + num_img_tokens]
                .permute(0, 2, 1)
                .unflatten(2, (token_rows, token_cols))
                .contiguous()
            )
            for projection, normed in zip(self.output_projections, outputs, strict=True)
        ]
        features = torch.stack(projected, dim=1).sum(dim=1)  # (B, dim_out, rows, cols)
        cls_token = outputs[-1][:, 0]  # class token of the last extracted layer
        return features, cls_token


class _ResidualConvBlock(nn.Module):
    """Weight-compatible container for the decoder residual blocks.

    Parameter structure only (``layers.{0..5}``, ``skip_connection``): these
    absorb the published checkpoint's decoder weights but no decode forward
    is provided by this runtime.
    """

    _ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU, "elu": nn.ELU}

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        activation: str = "relu",
        in_norm: str = "none",
        hidden_norm: str = "none",
    ) -> None:
        super().__init__()
        if activation not in self._ACTIVATIONS:
            raise NotImplementedError(f"activation={activation!r} is not supported.")
        act_cls = self._ACTIVATIONS[activation]

        def norm(kind: str, channels: int) -> nn.Module:
            if kind in ("none", None):
                return nn.Identity()
            if kind == "group_norm":
                return nn.GroupNorm(channels // 32, channels)
            if kind == "layer_norm":
                return nn.GroupNorm(1, channels)
            if kind == "instance_norm":
                return nn.InstanceNorm2d(channels)
            raise NotImplementedError(f"norm={kind!r} is not supported.")

        self.layers = nn.Sequential(
            norm(in_norm, in_channels),
            act_cls(),
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, padding_mode="replicate"),
            norm(hidden_norm, hidden_channels),
            act_cls(),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1, padding_mode="replicate"),
        )
        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )


class _Resampler(nn.Sequential):
    """Weight-compatible container for the decoder 2x upsamplers.

    Subclasses ``nn.Sequential`` directly so parameter keys stay flat
    (``resamplers.<i>.<j>.*``), matching the published checkpoint layout.
    """

    def __init__(self, in_channels: int, out_channels: int, type_: str) -> None:
        if type_ == "conv_transpose":
            super().__init__(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="replicate"),
            )
        elif type_ == "bilinear":
            super().__init__(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="replicate"),
            )
        else:
            raise NotImplementedError(f"resampler={type_!r} is not supported.")


class _ConvStack(nn.Module):
    """Weight-compatible container for the published decoder stacks.

    Mirrors the checkpoint's decoder parameter layout (``input_blocks``,
    ``resamplers``, ``res_blocks``, ``output_blocks``) so every decoder
    weight in the published checkpoint is absorbed and the strict=False
    load report matches the upstream load exactly. No forward is provided:
    this teacher extracts features only.
    """

    def __init__(
        self,
        dim_in: Sequence[int | None],
        dim_res_blocks: Sequence[int],
        dim_out: Sequence[int | None] | None,
        resamplers: Sequence[str],
        num_res_blocks: Sequence[int] | int = 1,
        res_block_in_norm: str = "layer_norm",
        res_block_hidden_norm: str = "group_norm",
        activation: str = "relu",
        dim_times_res_block_hidden: int = 1,
    ) -> None:
        super().__init__()
        self.input_blocks = nn.ModuleList(
            nn.Conv2d(dim_in_, dim_res, kernel_size=1, stride=1, padding=0)
            if dim_in_ is not None
            else nn.Identity()
            for dim_in_, dim_res in zip(dim_in, dim_res_blocks, strict=True)
        )
        self.resamplers = nn.ModuleList(
            _Resampler(dim_prev, dim_succ, type_=resampler)
            for dim_prev, dim_succ, resampler in zip(
                dim_res_blocks[:-1], dim_res_blocks[1:], resamplers, strict=True
            )
        )
        counts = num_res_blocks if isinstance(num_res_blocks, Sequence) else [num_res_blocks] * len(dim_res_blocks)
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
            for dim_res, count in zip(dim_res_blocks, counts, strict=True)
        )
        self.output_blocks = nn.ModuleList(
            nn.Conv2d(dim_res, dim_out_, kernel_size=1, stride=1, padding=0)
            if dim_out_ is not None
            else nn.Identity()
            for dim_out_, dim_res in zip(dim_out or [None] * len(dim_res_blocks), dim_res_blocks, strict=True)
        )


class MoRGBDTeacher(nn.Module):
    """Frozen LingBot-Depth (MoRGBD) teacher for RGBD-DINOv2-L distillation.

    Built from (and only from) the published checkpoint's ``model_config``;
    weights come from ``from_pretrained``. Hold instances as a plain policy
    attribute (the pattern ``DepthTeacherBundle`` uses) so teacher weights
    never enter optimizers, FSDP/DDP state dicts, or saved checkpoints.
    """

    def __init__(
        self,
        encoder: dict[str, Any],
        neck: dict[str, Any] | None = None,
        depth_head: dict[str, Any] | None = None,
        normal_head: dict[str, Any] | None = None,
        mask_head: dict[str, Any] | None = None,
        remap_depth_in: str = "linear",
        remap_output: str = "linear",
        num_tokens_range: Sequence[int] = (1200, 3600),
    ) -> None:
        super().__init__()
        # `normal_head` is accepted (the published model_config carries it and
        # the checkpoint stores its weights) but deliberately not constructed,
        # mirroring the upstream model: its constructor never assigns the
        # normal stack, so the strict=False load drops those keys. The
        # published RGBD-DINOv2-L recipe never decodes normals.
        del normal_head
        self.remap_depth_in = remap_depth_in
        # `remap_output` shapes the (unported) depth-decode outputs; recorded
        # for config fidelity.
        self.remap_output = remap_output
        self.num_tokens_range = list(num_tokens_range)
        self.encoder = _RGBDEncoder(**encoder)
        self.neck = _ConvStack(**neck) if neck is not None else None
        self.depth_head = _ConvStack(**depth_head) if depth_head is not None else None
        self.mask_head = _ConvStack(**mask_head) if mask_head is not None else None
        # Populated by from_pretrained for the strict=False load report.
        self.load_missing_keys: list[str] = []
        self.load_unexpected_keys: list[str] = []

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, *, device: torch.device | str | None = None
    ) -> MoRGBDTeacher:
        """Load a published ``depth/model.pt`` checkpoint (weights_only)."""
        checkpoint_path = Path(pretrained_model_name_or_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"MoRGBD teacher checkpoint not found: {checkpoint_path}. Supply the "
                "published LingBot-Depth depth/model.pt weight file."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        try:
            model_config = checkpoint["model_config"]
            model_state = checkpoint["model"]
        except KeyError as error:
            raise ValueError(
                f"{checkpoint_path} is not a published LingBot-Depth checkpoint "
                f"(missing {error} entry)."
            ) from error
        model = cls(**model_config)
        # strict=False reproduces the upstream load, including its quirk: the
        # checkpoint's depth_mask_patch_embed.* keys are dropped and the
        # constructed depth_patch_embed stays at its construction init.
        report = model.load_state_dict(model_state, strict=False)
        model._record_load_report(sorted(report.missing_keys), sorted(report.unexpected_keys))
        model.requires_grad_(False)
        model.eval()
        if device is not None:
            model.to(device=torch.device(device))
        return model

    def _record_load_report(self, missing: list[str], unexpected: list[str]) -> None:
        self.load_missing_keys = missing
        self.load_unexpected_keys = unexpected
        bad_missing = [key for key in missing if not key.startswith(_QUIRK_MISSING_PREFIX)]
        bad_unexpected = [
            key
            for key in unexpected
            if not key.startswith(_QUIRK_UNEXPECTED_PREFIXES)
        ]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "Checkpoint does not match the first-party MoRGBD teacher beyond the "
                "known strict=False quirk. Unexpected keys: "
                f"{bad_unexpected}. Missing keys: {bad_missing}."
            )
        if missing or unexpected:
            warnings.warn(
                "Published-checkpoint strict=False quirk reproduced: "
                f"{missing} stay at construction initialization while "
                f"{unexpected} from the checkpoint are dropped (matching the "
                "upstream load).",
                stacklevel=3,
            )

    def _num_tokens_for(self, num_tokens: int | None, resolution_level: int) -> int:
        if num_tokens is not None:
            return int(num_tokens)
        min_tokens, max_tokens = self.num_tokens_range
        return int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

    @torch.inference_mode()
    def infer_feat(
        self,
        image: Tensor,
        depth: Tensor,
        num_tokens: int | None = 256,
        resolution_level: int = 3,
        depth_down_scale: int = 1,
        enable_depth_mask: bool = False,
        use_fp16: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Extract frozen RGBD-DINOv2-L patch features and the class token.

        Args:
            image: ``(B, 3, H, W)`` (or ``(3, H, W)``) RGB in [0, 1].
            depth: ``(B, H, W)`` / ``(B, 1, H, W)`` (or unbatched) depth map;
                invalid (non-positive / inf / nan) entries are masked to zero.
            num_tokens: base token budget for the token grid; ``None`` derives
                it from ``resolution_level`` over ``num_tokens_range``.
            resolution_level: 0-9 resolution hint, used only when
                ``num_tokens is None``.
            depth_down_scale: accepted for upstream signature parity; inert
                (the upstream implementation forwards but never reads it).
            enable_depth_mask: must be False; the depth-token masking path is
                not implemented in this first-party runtime.
            use_fp16: run the forward under bf16 autocast (upstream default).

        Returns:
            Tuple ``(feat, cls)`` with ``feat`` ``(B, 1024, rows, cols)``
            ( ``(256, 256)``-token square inputs yield ``rows = cols = 16``)
            and ``cls`` ``(B, 1024)``.
        """
        if enable_depth_mask:
            raise NotImplementedError(
                "enable_depth_mask=True is not implemented in the first-party MoRGBD "
                "teacher; the released LingBot-Depth alignment recipe always runs "
                "enable_depth_mask=False."
            )
        if depth is None:
            raise ValueError("infer_feat requires a depth map (got None).")
        del depth_down_scale  # Inert in the upstream implementation as well.

        omit_batch_dim = image.dim() == 3
        if omit_batch_dim:
            image = image.unsqueeze(0)
        image = image.to(dtype=self.dtype, device=self.device)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        if depth.device != image.device or depth.dtype != image.dtype:
            depth = depth.to(device=image.device, dtype=image.dtype)

        height, width = image.shape[-2:]
        num_tokens = self._num_tokens_for(num_tokens, resolution_level)
        aspect_ratio = width / height
        base_h = round((num_tokens / aspect_ratio) ** 0.5)
        base_w = round((num_tokens * aspect_ratio) ** 0.5)

        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=use_fp16 and self.dtype != torch.bfloat16
        ):
            features, cls_token = self.encoder(
                image, depth, base_h, base_w, remap_depth_in=self.remap_depth_in
            )
        return features, cls_token

"""Frozen depth teachers for the dual-query distillation: MoGe-2 (relative depth) feeding LingBot-Depth (MoRGBD),
whose RGB-D encoder features are the current / future depth targets. Training only.

Adapted from Robbyant/lingbot-vla-v2 ``lingbotvla/models/vla/vision_models/``:
``MoGe/moge/model/{v2,modules}.py`` and ``MoGe/moge/utils/geometry_{torch,numpy}.py`` (MoGe, MIT License,
Copyright (c) Microsoft Corporation), ``lingbot-depth/mdm/model/{v2,modules_rgbd_encoder}.py`` and the vendored
DINOv2 backbones (Apache License 2.0). The ViT blocks are transformers' ``Dinov2Layer``. Only what the depth
targets need is kept: no normal head, no point-map projection, no MoRGBD decoder.
"""

import itertools
import math
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
from transformers.models.dinov2.modeling_dinov2 import Dinov2Layer

# arch: (hidden size, depth, heads), DINOv2 /14 with a 518 px (37 x 37) position grid
DINOV2_ARCHS = {"dinov2_vitb14": (768, 12, 12), "dinov2_vitl14": (1024, 24, 16)}
DINOV2_BLOCK_KEYS = {
    "attn.qkv": ("attention.attention.query", "attention.attention.key", "attention.attention.value"),
    "attn.proj": "attention.output.dense",
    "ls1.gamma": "layer_scale1.lambda1",
    "ls2.gamma": "layer_scale2.lambda1",
}


def convert_block_keys(state_dict: dict, mapping: dict) -> dict:
    """Rename upstream ViT block weights (fused ``qkv``, ``ls*.gamma``) to the transformers layer names."""
    out = {}
    for key, value in state_dict.items():
        old = next((name for name in mapping if f".{name}" in key), None)
        new = mapping.get(old)
        if isinstance(new, tuple):  # fused qkv rows -> q, k, v
            out.update({key.replace(old, n): chunk for n, chunk in zip(new, value.chunk(3), strict=True)})
        else:
            out[key.replace(old, new) if old else key] = value
    return out


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, dim: int, patch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Dinov2Backbone(nn.Module):
    """DINOv2 ViT/14 trunk (upstream ``DinoVisionTransformer``), optionally with MoRGBD's depth patch embedding."""

    def __init__(self, arch: str, depth_embed: bool = False):
        super().__init__()
        dim, depth, heads = DINOV2_ARCHS[arch]
        cfg = Dinov2Config(
            hidden_size=dim, num_hidden_layers=depth, num_attention_heads=heads, layerscale_value=1.0
        )
        cfg._attn_implementation = "sdpa"
        self.patch_size = 14
        self.patch_embed = PatchEmbed(3, dim, 14)
        self.depth_mask_patch_embed = PatchEmbed(1, dim, 14) if depth_embed else None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 37 * 37 + 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.blocks = nn.ModuleList(Dinov2Layer(cfg) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def patch_pos_embed(self, x, h, w):
        """Upstream ``interpolate_pos_encoding`` (bicubic, ``interpolate_offset=0.1``) without the cls row."""
        pos = self.pos_embed[:, 1:]
        n = pos.shape[1]
        if x.shape[1] == n and w == h:
            return pos
        m, h0, w0 = int(math.sqrt(n)), h // self.patch_size, w // self.patch_size
        pos = F.interpolate(
            pos.float().reshape(1, m, m, -1).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=False,
            scale_factor=((h0 + 0.1) / m, (w0 + 0.1) / m),
        )
        return pos.permute(0, 2, 3, 1).flatten(1, 2).to(x.dtype)

    def intermediate_layers(self, x, layers):
        """Final-norm outputs of the given blocks (upstream ``get_intermediate_layers(norm=True)``)."""
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in layers:
                outputs.append(self.norm(x))
        return outputs


class Dinov2Encoder(nn.Module):
    """Upstream ``DINOv2Encoder`` / ``DINOv2_RGBD_Encoder``: projected sum of intermediate patch features."""

    def __init__(self, backbone, intermediate_layers, dim_out, depth_emb_mode="", **_):
        super().__init__()
        self.intermediate_layers = intermediate_layers
        self.backbone = Dinov2Backbone(backbone, depth_embed=depth_emb_mode == "conv_1c")
        dim = DINOV2_ARCHS[backbone][0]
        self.output_projections = nn.ModuleList(nn.Conv2d(dim, dim_out, 1) for _ in intermediate_layers)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image, rows, cols, depth=None):
        image = F.interpolate(
            image, (rows * 14, cols * 14), mode="bilinear", align_corners=False, antialias=True
        )
        image = (image - self.image_mean) / self.image_std
        bb = self.backbone
        if depth is None:  # MoGe: [cls, patches] + interpolated position embedding
            x = bb.patch_embed(image)
            x = torch.cat((bb.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            pos = torch.cat(
                (bb.pos_embed[:, :1].float(), bb.patch_pos_embed(x[:, 1:], *image.shape[-2:]).float()), 1
            )
            x = x + pos.to(x.dtype)
        else:  # MoRGBD "cat_token": [cls, image patches + 1 + pos, depth patches + 2 + pos], no depth masking
            depth = F.interpolate(depth, (rows * 14, cols * 14), mode="nearest")
            depth[torch.isinf(depth)] = 0.0
            depth[torch.isnan(depth)] = 0.0
            depth = depth * (depth > 0.01).float()
            x_img, x_depth = bb.patch_embed(image), bb.depth_mask_patch_embed(depth)
            batch = x_img.shape[0]
            x_img = x_img + (1 + bb.patch_pos_embed(x_img, *image.shape[-2:]).repeat(batch, 1, 1))
            x_depth = x_depth + (2 + bb.patch_pos_embed(x_depth, *depth.shape[-2:]).repeat(batch, 1, 1))
            cls = bb.cls_token.squeeze(0) + bb.pos_embed.squeeze(0)[:1]
            x = torch.cat([torch.cat([cls, x_img[i], x_depth[i]]).unsqueeze(0) for i in range(batch)])
        outputs = bb.intermediate_layers(x, self.intermediate_layers)
        n = rows * cols
        feats = torch.stack(
            [
                proj(out[:, 1 : 1 + n].permute(0, 2, 1).unflatten(2, (rows, cols)).contiguous())
                for proj, out in zip(self.output_projections, outputs, strict=True)
            ],
            dim=1,
        ).sum(dim=1)
        return feats, outputs[-1][:, 0]


class ResidualConvBlock(nn.Module):
    """Upstream block with ``in_norm = hidden_norm = 'none'`` and ReLU (the released MoGe-2 config)."""

    def __init__(self, channels):
        super().__init__()
        conv = partial(nn.Conv2d, channels, channels, kernel_size=3, padding=1, padding_mode="replicate")
        self.layers = nn.Sequential(nn.Identity(), nn.ReLU(), conv(), nn.Identity(), nn.ReLU(), conv())

    def forward(self, x):
        return self.layers(x) + x


def resampler(dim_in, dim_out, kind):
    conv = nn.Conv2d(
        dim_out if kind == "conv_transpose" else dim_in, dim_out, 3, padding=1, padding_mode="replicate"
    )
    if kind == "conv_transpose":
        return nn.Sequential(nn.ConvTranspose2d(dim_in, dim_out, kernel_size=2, stride=2), conv)
    if kind == "bilinear":
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), conv)
    raise ValueError(f"Unsupported MoGe resampler {kind!r}.")


class ConvStack(nn.Module):
    def __init__(self, dim_in, dim_res_blocks, dim_out, resamplers, num_res_blocks=1, **_):
        super().__init__()

        def per_level(value):  # upstream accepts one value for every level
            return value if isinstance(value, (list, tuple)) else [value] * len(dim_res_blocks)

        dim_in, dim_out, resamplers, num_res_blocks = map(
            per_level, (dim_in, dim_out, resamplers, num_res_blocks)
        )
        self.input_blocks = nn.ModuleList(
            nn.Conv2d(i, r, 1) if i is not None else nn.Identity()
            for i, r in zip(dim_in, dim_res_blocks, strict=True)
        )
        self.resamplers = nn.ModuleList(
            resampler(a, b, k)
            for a, b, k in zip(dim_res_blocks[:-1], dim_res_blocks[1:], resamplers, strict=False)
        )
        self.res_blocks = nn.ModuleList(
            nn.Sequential(*(ResidualConvBlock(r) for _ in range(n)))
            for r, n in zip(dim_res_blocks, num_res_blocks, strict=True)
        )
        self.output_blocks = nn.ModuleList(
            nn.Conv2d(r, o, 1) if o is not None else nn.Identity()
            for o, r in zip(dim_out, dim_res_blocks, strict=True)
        )

    def forward(self, features):
        outputs, x = [], 0
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x + self.input_blocks[i](features[i]))
            outputs.append(self.output_blocks[i](x))
            if i < len(self.res_blocks) - 1:
                x = self.resamplers[i](x)
        return outputs


def view_plane_uv(width, height, aspect_ratio, dtype, device):
    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5
    u = torch.linspace(
        -span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device
    )
    v = torch.linspace(
        -span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device
    )
    return torch.stack(torch.meshgrid(u, v, indexing="xy"), dim=-1)


def recover_shift(points, mask, size=(64, 64)):
    """Upstream ``recover_focal_shift``: z-shift of the affine point map (Levenberg-Marquardt, scipy)."""
    from scipy.optimize import least_squares

    height, width = points.shape[-3:-1]
    uv = view_plane_uv(width, height, width / height, points.dtype, points.device)
    points_lr = (
        F.interpolate(points.permute(0, 3, 1, 2), size, mode="nearest").permute(0, 2, 3, 1).cpu().numpy()
    )
    uv_lr = (
        F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), size, mode="nearest").squeeze(0).permute(1, 2, 0)
    )
    uv_lr = uv_lr.cpu().numpy()
    mask_lr = (F.interpolate(mask.float().unsqueeze(1), size, mode="nearest").squeeze(1) > 0).cpu().numpy()

    def residual(uv, xy, z, shift):
        xy_proj = xy / (z + shift)[:, None]
        focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        return (focal * xy_proj - uv).ravel()

    shifts = []
    for pts, m in zip(points_lr, mask_lr, strict=True):
        pts, uv_i = pts[m], uv_lr[m]
        if uv_i.shape[0] < 2:
            shifts.append(0.0)
            continue
        uv_i, xy, z = uv_i.reshape(-1, 2), pts[..., :2].reshape(-1, 2), pts[..., 2].reshape(-1)
        solution = least_squares(partial(residual, uv_i, xy, z), x0=0, ftol=1e-3, method="lm")
        shifts.append(float(solution["x"].squeeze().astype(np.float32)))
    return torch.tensor(shifts, device=points.device, dtype=points.dtype)


class MoGe2(nn.Module):
    """MoGe-2 metric depth (``MoGeModel.infer`` with ``apply_mask=False``)."""

    def __init__(self, encoder, neck, points_head, mask_head, scale_head, remap_output="exp", **_):
        super().__init__()
        if remap_output != "exp":
            raise ValueError(f"Only the MoGe-2 'exp' point remap is supported, got {remap_output!r}.")
        self.encoder = Dinov2Encoder(**encoder)
        self.neck = ConvStack(**neck)
        self.points_head = ConvStack(**points_head)
        self.mask_head = ConvStack(**mask_head)
        dims = scale_head["dims"]
        layers = [
            [nn.Linear(a, b), nn.ReLU(inplace=True)] for a, b in zip(dims[:-2], dims[1:-1], strict=True)
        ]
        self.scale_head = nn.Sequential(*itertools.chain(*layers), nn.Linear(dims[-2], dims[-1]))

    @torch.no_grad()
    def depth(self, image, num_tokens=256, use_fp16=True):
        batch, _, img_h, img_w = image.shape
        aspect = img_w / img_h
        rows, cols = round((num_tokens / aspect) ** 0.5), round((num_tokens * aspect) ** 0.5)
        with torch.autocast(image.device.type, dtype=torch.float16, enabled=use_fp16):
            features, cls_token = self.encoder(image, rows, cols)
            features = [features, None, None, None, None]
            for level in range(5):
                uv = view_plane_uv(cols * 2**level, rows * 2**level, aspect, image.dtype, image.device)
                uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch, -1, -1, -1)
                features[level] = (
                    uv if features[level] is None else torch.concat([features[level], uv], dim=1)
                )
            features = self.neck(features)
            points, mask = (head(features)[-1] for head in (self.points_head, self.mask_head))
            metric_scale = self.scale_head(cls_token)
            points, mask = (
                F.interpolate(v, (img_h, img_w), mode="bilinear", align_corners=False, antialias=False)
                for v in (points, mask)
            )
            points = points.permute(0, 2, 3, 1)
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
            mask = mask.squeeze(1).sigmoid()
            metric_scale = metric_scale.squeeze(1).exp()
        points, mask, metric_scale = points.float(), mask.float(), metric_scale.float()
        with torch.autocast(image.device.type, enabled=False):
            shift = recover_shift(points, mask > 0.5)
            return (points[..., 2] + shift[..., None, None]) * metric_scale[:, None, None]


class MoRGBDEncoder(nn.Module):
    """LingBot-Depth (MoRGBD) ``infer_feat``: RGB-D encoder features, [B, rows * cols, dim_out]."""

    def __init__(self, encoder, remap_depth_in="linear", **_):
        super().__init__()
        if remap_depth_in != "linear" or encoder.get("img_depth_fuse_mode") != "cat_token":
            raise ValueError(
                "Only the released LingBot-Depth encoder (linear depth, cat_token) is supported."
            )
        self.encoder = Dinov2Encoder(**encoder)

    @torch.no_grad()
    def features(self, image, depth, num_tokens=256, use_fp16=True):
        aspect = image.shape[-1] / image.shape[-2]
        rows, cols = round((num_tokens / aspect) ** 0.5), round((num_tokens * aspect) ** 0.5)
        with torch.autocast(image.device.type, dtype=torch.bfloat16, enabled=use_fp16):
            feats, _ = self.encoder(image, rows, cols, depth=depth.unsqueeze(1))
        return feats.permute(0, 2, 3, 1).flatten(1, 2)


def load_depth_teacher(cls, path: str):
    """Build from an upstream ``{"model_config", "model"}`` checkpoint (only the modules ``cls`` needs)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = cls(**checkpoint["model_config"])
    state = convert_block_keys(checkpoint["model"], DINOV2_BLOCK_KEYS)
    own = model.state_dict()
    missing = set(own) - set(state)
    if missing:
        raise ValueError(f"{path} is missing teacher weights: {sorted(missing)[:5]}")
    model.load_state_dict({k: v for k, v in state.items() if k in own})
    return model

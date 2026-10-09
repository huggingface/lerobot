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

"""Frozen DINO-Video teacher for the dual-query distillation (current / future frame patch targets). Training only.

Reimplements the teacher of Robbyant/lingbot-vla-v2 ``lingbotvla/models/vla/vision_models/dino_video/`` on top of
transformers' ``DINOv3ViTLayer`` (Apache License 2.0). Only the per-frame [cls, storage, patches] token layout, the
3D (time-interleaved) RoPE and the frame-causal attention mask are written here, as new code that reproduces
upstream's outputs; no source file of upstream's ``lumos_dinov3`` package (Meta, DINOv3 License) is copied.

The teacher weights (``robbyant/lingbot-vla-v2-6b``, ``dino_video/``) are derived from DINOv3 and are subject to
the DINOv3 License (https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). They are downloaded on the
first distillation training step and are not distributed with LeRobot.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from torch import nn
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTLayer

from .depth_teachers import convert_block_keys

ARCHS = {"vit_large": (1024, 24, 16)}  # hidden size, depth, heads
BLOCK_KEYS = {
    "attn.qkv": ("attention.q_proj", "attention.k_proj", "attention.v_proj"),
    "attn.proj": "attention.o_proj",
    "ls1.gamma": "layer_scale1.lambda1",
    "ls2.gamma": "layer_scale2.lambda1",
    "mlp.fc1": "mlp.up_proj",
    "mlp.fc2": "mlp.down_proj",
}
# The released teacher's config.yaml values this port implements.
EXPECTED = {
    "ffn_layer": "mlp",
    "norm_layer": "layernormbf16",
    "mask_k_bias": False,
    "untie_cls_and_patch_norms": False,
    "pos_embed_rope_normalize_coords": "separate",
    "pos_embed_rope_min_period": None,
    "pos_embed_rope_max_period": None,
    "pos_embed_rope_dtype": "fp32",
    "pos_embed_rope_3d": True,
    "pos_embed_rope_prefix_temporal": True,
}


class DinoVideoTeacher(nn.Module):
    """Block-causal video ViT (upstream ``NaviTVideoViT`` + ``CausalEvalAdapter``) on one crop group."""

    def __init__(self, student: dict, cls_pool: str = "mean"):
        super().__init__()
        bad = {k: student.get(k) for k, v in EXPECTED.items() if student.get(k) != v}
        if bad or student["arch"] not in ARCHS:
            raise ValueError(f"Unsupported DINO-Video teacher config: {bad or student['arch']}")
        dim, depth, heads = ARCHS[student["arch"]]
        cfg = DINOv3ViTConfig(
            hidden_size=dim,
            intermediate_size=4 * dim,
            num_attention_heads=heads,
            layer_norm_eps=1e-5,  # layernormbf16
            query_bias=student["qkv_bias"],
            key_bias=student["qkv_bias"],
            value_bias=student["qkv_bias"],
            proj_bias=student["proj_bias"],
            mlp_bias=student["ffn_bias"],
        )
        cfg._attn_implementation = "sdpa"
        self.patch_size, self.head_dim = student["patch_size"], dim // heads
        self.n_storage = student["n_storage_tokens"]
        self.base_fps = float(student.get("pos_embed_rope_base_fps", 24.0))
        self.cls_pool = cls_pool
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, self.n_storage, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.register_buffer("periods", torch.zeros(self.head_dim // 4))
        self.register_buffer("periods_t", torch.zeros(self.head_dim // 8))
        self.blocks = nn.ModuleList(DINOv3ViTLayer(cfg) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-5)

    def rope(self, frames, rows, cols, fps):
        """(sin, cos) per token: 2D spatial RoPE with every 4th channel carrying the frame time
        (``base_fps / fps`` per frame); the prefix tokens only get their frame's time channels."""
        dd = {"device": self.periods.device, "dtype": torch.float32}
        coords_h = 2.0 * (torch.arange(0.5, rows, **dd) / rows) - 1.0
        coords_w = 2.0 * (torch.arange(0.5, cols, **dd) / cols) - 1.0
        coords_t = torch.arange(frames, **dd) * (self.base_fps / fps if fps is not None else 1.0)
        grid_t, grid_h, grid_w = (
            g.reshape(-1, 1) for g in torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij")
        )
        angles = torch.cat(
            [
                (2 * math.pi * grid_h / self.periods[None, :]).repeat(1, 2),
                (2 * math.pi * grid_w / self.periods[None, :]).repeat(1, 2),
            ],
            dim=1,
        )
        t_mask = torch.arange(self.head_dim, device=angles.device) % 4 == 3
        angles[:, t_mask] = (grid_t / self.periods_t[None, :]).repeat(1, 2)
        sin, cos = (f(angles).view(frames, rows * cols, -1) for f in (torch.sin, torch.cos))
        n_prefix = 1 + self.n_storage
        sin_pre = sin.new_zeros(frames, n_prefix, sin.shape[-1])
        cos_pre = cos.new_ones(frames, n_prefix, cos.shape[-1])
        sin_pre[:, :, t_mask] = sin[:, :1, t_mask]
        cos_pre[:, :, t_mask] = cos[:, :1, t_mask]
        return torch.cat([sin_pre, sin], 1).flatten(0, 1), torch.cat([cos_pre, cos], 1).flatten(0, 1)

    @torch.no_grad()
    def forward(self, video, fps=None):
        """video [B, 3, T, H, W] (ImageNet-normalized) -> last-block normed patches [B, T, P, D],
        per-frame cls [B, T, D] and the pooled cls [B, D]."""
        batch, _, frames = video.shape[:3]
        x = self.patch_embed(video.permute(0, 2, 1, 3, 4).flatten(0, 1))
        rows, cols = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2).reshape(batch, frames, rows * cols, -1)
        cls = (self.cls_token + 0 * self.mask_token).unsqueeze(0).expand(batch, frames, 1, -1)
        storage = self.storage_tokens.unsqueeze(0).expand(batch, frames, -1, -1)
        x = torch.cat([cls, storage, x], dim=2)
        frame_len = x.shape[2]
        x = x.reshape(batch, frames * frame_len, -1)
        sin, cos = self.rope(frames, rows, cols, fps)
        frame = torch.arange(frames * frame_len, device=x.device) // frame_len
        causal = (frame[None, :] <= frame[:, None])[None, None]  # a frame sees itself and the past
        for block in self.blocks:
            x = block(x, attention_mask=causal, position_embeddings=(cos, sin))
        x = x.view(batch, frames, frame_len, -1)
        patches = self.norm(x[:, :, 1 + self.n_storage :])
        raw_cls = x[:, :, 0]
        pooled = self.norm(raw_cls.mean(dim=1) if self.cls_pool == "mean" else raw_cls[:, -1])
        return patches, self.norm(raw_cls), pooled


def load_video_teacher(ckpt_path: str, config_path: str, cls_pool: str = "mean"):
    with open(config_path) as f:
        student = yaml.safe_load(f)["dinov3"]["student"]
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = raw.get("teacher", raw.get("model", raw))
    state = {k.removeprefix("backbone."): v for k, v in state.items()}
    state["periods"], state["periods_t"] = state.pop("rope_embed.periods"), state.pop("rope_embed.periods_t")
    state["patch_embed.weight"], state["patch_embed.bias"] = (
        state.pop("patch_embed.proj.weight"),
        state.pop("patch_embed.proj.bias"),
    )
    model = DinoVideoTeacher(student, cls_pool=cls_pool)
    state = convert_block_keys(state, BLOCK_KEYS)
    own = model.state_dict()
    missing = set(own) - set(state)
    if missing:
        raise ValueError(f"{ckpt_path} is missing teacher weights: {sorted(missing)[:5]}")
    model.load_state_dict({k: v for k, v in state.items() if k in own})
    return model


def video_input(frames, size):
    """Upstream ``get_video_target`` preprocessing: [B, 3, H, W] frames in [0, 1] -> ImageNet-normalized, resized."""
    if frames.shape[-2:] != (size, size):
        frames = F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
    return (frames - mean) / std

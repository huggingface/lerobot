# Copyright (C) 2026 Tencent. All rights reserved.
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

# ruff: noqa: B007, B023, E731, E741, N806, SIM102, SIM108

"""LeRobot policy and native model implementation for Hy-Embodied-0.5-VLA."""

from __future__ import annotations

import copy
import math
import sys
import types
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .configuration_hy_vla import HyVLAConfig
from .modeling.hunyuan_vl_mot import HunYuanVLMoTConfig, HunYuanVLMoTForConditionalGeneration
from .modeling.hunyuan_vl_mot.modeling_hunyuan_vl_mot import _HunYuanVLMoTTextForCausalLM

# -----------------------------------------------------------------------------
# Space-time video attention
# -----------------------------------------------------------------------------


class SpaceTimeBlock(nn.Module):
    """MEM space-time separable attention block.

    Wraps a base ViT block (the bundled ``_ViTBlock``) by reference: all
    submodules (norm1, attn, ls1, drop_path1, norm2, mlp, ls2,
    drop_path2) are *adopted*, not copied, so the wrapped block has the
    SAME state_dict keys as the unwrapped block and zero new parameters.
    The causal time embedding e(t) is fixed sinusoidal with e(0)=0 and
    is rebuilt on-device inside ``forward`` (no buffer / no parameter,
    to survive DeepSpeed ZeRO-3 sharding).

    forward signature: ``(x, cu_slens=None, num_frames=1)`` where
    ``x.shape = (B*K, N, D)``. When ``num_frames == 1`` the block
    short-circuits to the base block's behaviour exactly.
    """

    def __init__(
        self,
        base_block: nn.Module,
        max_num_frames: int,
        learnable_time_embed: bool = False,
        time_embed_base: float = 100.0,
    ) -> None:
        super().__init__()
        assert max_num_frames >= 1
        self.max_num_frames = max_num_frames
        self.learnable_time_embed = learnable_time_embed
        self.time_embed_base = time_embed_base

        # Adopt submodules by reference: preserves vanilla state_dict keys.
        self.norm1 = base_block.norm1
        self.attn = base_block.attn
        self.ls1 = base_block.ls1
        self.drop_path1 = base_block.drop_path1
        self.norm2 = base_block.norm2
        self.mlp = base_block.mlp
        self.ls2 = base_block.ls2
        self.drop_path2 = base_block.drop_path2

        dim = self.attn.num_heads * self.attn.head_dim
        self._time_embed_dim = dim

        if learnable_time_embed:
            # Trainable table; row 0 pinned to 0 via grad hook to keep e(0)=0
            # structurally.
            t = torch.arange(max_num_frames, dtype=torch.float32).unsqueeze(1)
            inv_freq = torch.exp(
                torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(time_embed_base) / dim)
            )
            pe = torch.zeros(max_num_frames, dim, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(t * inv_freq)
            pe[:, 1::2] = torch.cos(t * inv_freq) - 1.0
            self.time_embed = nn.Embedding(max_num_frames, dim)
            with torch.no_grad():
                self.time_embed.weight.copy_(pe)
            self.time_embed.weight.register_hook(lambda g: torch.cat([torch.zeros_like(g[:1]), g[1:]], dim=0))
        # Fixed-sinusoidal branch: rebuilt on-device in forward (see _build_time_pe).

    def _build_time_pe(self, kf: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Fixed sinusoidal e(t) with e(0)=0, shape (kf, D), on ``device``."""
        dim = self._time_embed_dim
        t = torch.arange(kf, dtype=torch.float32, device=device).unsqueeze(1)
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32, device=device)
            * (-math.log(self.time_embed_base) / dim)
        )
        pe = torch.empty(kf, dim, dtype=torch.float32, device=device)
        pe[:, 0::2] = torch.sin(t * inv_freq)
        pe[:, 1::2] = torch.cos(t * inv_freq) - 1.0
        return pe.to(dtype=dtype)

    def _qkv(self, h: torch.Tensor):
        """Shared QKV + q/k-norm on ``h``. Returns q, k, v in (M, H, L, d)."""
        a = self.attn
        m, l, _ = h.shape
        if a.q_bias is not None:
            bias = torch.cat((a.q_bias, torch.zeros_like(a.v_bias), a.v_bias))
            qkv = F.linear(h, a.qkv.weight, bias)
        else:
            qkv = a.qkv(h)
        q, k, v = qkv.reshape(m, l, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        return a.q_norm(q), a.k_norm(k), v

    def _time_softmax_on_v(self, q, k, v, b: int, kf: int) -> torch.Tensor:
        """Causal time softmax contracted onto V: v_mixed = A_time @ v."""
        bk, heads, n, d = v.shape
        # (B*K, H, N, d) -> (B*N, H, K, d): fold K into per-position sequence.
        reshape_to_time = lambda t: (
            t.view(b, kf, heads, n, d).permute(0, 3, 2, 1, 4).reshape(b * n, heads, kf, d)
        )
        q_t, k_t, v_t = reshape_to_time(q), reshape_to_time(k), reshape_to_time(v)

        scores = (q_t @ k_t.transpose(-2, -1)) * self.attn.scale  # (B*N, H, K, K)
        mask = torch.triu(torch.ones(kf, kf, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        v_mixed_t = scores.softmax(dim=-1).to(v_t.dtype) @ v_t  # (B*N, H, K, d)

        return v_mixed_t.view(b, n, heads, kf, d).permute(0, 3, 2, 1, 4).reshape(bk, heads, n, d)

    def _space_attn(self, q, k, v, cu_slens=None) -> torch.Tensor:
        """A_space softmax, then the single W_O. (B*K, H, N, d) in."""
        a = self.attn
        bk, _, n, _ = q.shape

        def attend(q_part, k_part, v_part):
            scores = (q_part.float() @ k_part.float().transpose(-2, -1)) * a.scale
            probabilities = scores.softmax(dim=-1).to(v_part.dtype)
            probabilities = a.attn_drop(probabilities)
            return probabilities @ v_part

        if cu_slens is not None:
            if bk != 1:
                raise ValueError("Packed eager space attention expects batch size 1.")
            out = torch.zeros_like(q)
            for start, end in zip(cu_slens[:-1].tolist(), cu_slens[1:].tolist(), strict=True):
                out[:, :, start:end] = attend(
                    q[:, :, start:end],
                    k[:, :, start:end],
                    v[:, :, start:end],
                )
        else:
            out = attend(q, k, v)
        return a.proj_drop(a.proj(out.transpose(1, 2).reshape(bk, n, -1)))

    def forward(self, x: torch.Tensor, cu_slens=None, num_frames: int = 1) -> torch.Tensor:
        """``x``: (B*K, N, D); ``num_frames`` = K."""
        bk, n, d = x.shape
        assert bk % num_frames == 0, f"B*K={bk} not divisible by num_frames={num_frames}"
        b, kf = bk // num_frames, num_frames

        if self.learnable_time_embed:
            pe = self.time_embed.weight[:kf].to(x.dtype)  # (K, D)
        else:
            pe = self._build_time_pe(kf, x.device, x.dtype)  # (K, D)
        h = self.norm1(x.view(b, kf, n, d) + pe.view(1, kf, 1, d)).view(bk, n, d)
        q, k, v = self._qkv(h)

        v = self._time_softmax_on_v(q, k, v, b, kf)
        attn_out = self._space_attn(q, k, v, cu_slens=cu_slens)

        x = x + self.drop_path1(self.ls1(attn_out))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Entry point: inject temporal-spatial attention into a bundled visual instance.
# ---------------------------------------------------------------------------
def _patched_forward_video_features(self, x: torch.Tensor):
    """Replacement for ``forward_video_features`` on the bundled
    ``_HYViT2VisionTransformer``. Invoked only when the caller passes a
    5-D ``(B, K, C, H, W)`` tensor; the existing 4-D / list paths are
    unchanged.
    """
    assert x.ndim == 5, f"video encoder expects (B, K, C, H, W); got {tuple(x.shape)}"
    b, k, c, h_in, w_in = x.shape
    x = x.reshape(b * k, c, h_in, w_in)

    h = h_in // self.patch_embed.patch_size[0]
    w = w_in // self.patch_embed.patch_size[1]

    x = self.patch_embed(x)
    x = x + self.rescale_positional_embedding(out_size=(h, w))
    x = self.patch_drop(x)
    x = self.norm_pre(x)

    drop_at = self.past_drop_layer
    if drop_at is not None:
        assert 0 <= drop_at <= len(self.blocks), (
            f"past_drop_layer must be in [0, {len(self.blocks)}]; got {drop_at}"
        )

    for i, blk in enumerate(self.blocks):
        if drop_at is not None and i == drop_at and k > 1:
            # Collapse time dim: keep only the current-frame tokens.
            x = x.view(b, k, -1, x.shape[-1])[:, -1]
            k = 1
        if isinstance(blk, SpaceTimeBlock):
            x = blk(x, cu_slens=None, num_frames=k)
        else:
            x = blk(x)

    # Drop past-timestep tokens; keep only the current frame (t = K-1).
    x = x.view(b, k, -1, x.shape[-1])[:, -1]  # (B, N, D)
    return x, (h, w)


def _make_vision_tower_forward(orig_forward):
    """Wrap the bundled ``_HYViT2VisionTransformer.forward`` with a 5-D
    detection that routes video tensors through ``forward_video_features``.
    All other input shapes (list, 4-D tensor) fall through unchanged.
    """

    def _forward(self, x, cal_attn_pool=False):
        if getattr(self, "use_video_encoder", False) and torch.is_tensor(x) and x.ndim == 5:
            feats, image_sizes = _patched_forward_video_features(self, x)
            if not cal_attn_pool:
                return feats, image_sizes, None
            cls_token = self.forward_head(feats)
            return feats, image_sizes, cls_token
        return orig_forward(x, cal_attn_pool=cal_attn_pool)

    return _forward


def _make_wrapper_forward_func(orig_forward_func):
    """Wrap ``HYViT2_400MAnyRes._forward_func`` so a 5-D input is dispatched
    to the inner ViT's video path (which our patched _forward picks up).
    """

    def _forward_func(self, images, cal_attn_pool=False):
        if torch.is_tensor(images) and images.ndim == 5:
            # Route 5-D directly into the patched vision_tower.forward
            # (which detects ndim==5 and routes to forward_video_features).
            image_features, img_size, cls_token = self.vision_tower(
                images.to(self.dtype), cal_attn_pool=cal_attn_pool
            )
            image_features = image_features.to(images.dtype)
            return image_features, img_size, cls_token
        return orig_forward_func(images, cal_attn_pool=cal_attn_pool)

    return _forward_func


def _make_wrapper_forward(orig_forward):
    """Wrap ``HYViT2_400MAnyRes.forward`` so 5-D input emits a single
    flattened (B*N, C) feature list (matching dual_tower.embed_image's
    5-D contract). 4-D / list inputs preserve original behaviour.
    """

    def _forward(self, images, cal_attn_pool=False):
        if torch.is_tensor(images) and images.ndim == 5:
            image_features, img_size, _ = self._forward_func(images, cal_attn_pool=cal_attn_pool)
            # image_features is (B, N, 1152) coming from forward_video_features.
            # Apply the merger projection -> (B, N', 2048), then flatten the
            # batch into the (B*N', 2048) layout dual_tower.embed_image expects.
            image_features = self.merger(image_features, img_size)
            C = image_features.shape[-1]
            # Single-element list to match the existing 4-D return contract
            # ``[(B*N, C)]`` consumed by ``dual_tower.embed_image``.
            return [image_features.reshape(-1, C)]
        return orig_forward(images, cal_attn_pool=cal_attn_pool)

    return _forward


def apply_video_encoder_patch(
    visual: nn.Module,
    spacetime_layer_stride: int = 4,
    past_drop_layer: int | None = None,
    max_num_frames: int = 18,
    learnable_time_embed: bool = False,
    time_embed_base: float = 100.0,
) -> None:
    """Enable the MEM space-time path on a bundled ``HYViT2_400MAnyRes``.

    Idempotent: calling it twice is a no-op (the second call sees
    ``use_video_encoder=True`` and returns immediately). State_dict
    layout is preserved exactly: SpaceTimeBlock adopts the wrapped
    block's submodules by reference, and the fixed-sinusoidal time
    embedding is rebuilt on-device at every forward.

    Args:
        visual: a ``HYViT2_400MAnyRes`` instance, typically
            ``policy.model.dual_tower.vlm.model.visual``.
        spacetime_layer_stride: every Nth block (1-indexed: block at
            stride-1, 2*stride-1, ...) is wrapped with SpaceTimeBlock.
            Defaults to 4; matches in-repo default.
        past_drop_layer: 0-based index of the first block past which the
            (B*K) batch is collapsed to (B) by keeping only the current
            frame. ``None`` disables this MEM-paper optimisation.
        max_num_frames: upper bound on K (history length). Affects
            ``learnable_time_embed`` table size only; ignored for the
            fixed-sinusoidal branch (default).
        learnable_time_embed: if True, the time embedding table becomes a
            trainable ``nn.Embedding(max_num_frames, dim)`` (with row 0
            pinned to 0 via grad hook). This adds parameters to the model
            and is incompatible with checkpoints trained with the
            fixed-sinusoidal branch.
        time_embed_base: base period of the sinusoidal e(t).
    """
    if getattr(visual, "use_video_encoder", False):
        return

    if not hasattr(visual, "vision_tower"):
        raise TypeError(
            f"apply_video_encoder_patch: expected a HYViT2_400MAnyRes-like "
            f"wrapper with .vision_tower; got {type(visual).__name__}"
        )

    vision_tower = visual.vision_tower
    if not hasattr(vision_tower, "blocks"):
        raise TypeError(
            f"apply_video_encoder_patch: expected vision_tower with .blocks; "
            f"got {type(vision_tower).__name__}"
        )

    blocks = vision_tower.blocks
    depth = len(blocks)
    assert spacetime_layer_stride >= 1
    for i in range(spacetime_layer_stride - 1, depth, spacetime_layer_stride):
        if not isinstance(blocks[i], SpaceTimeBlock):
            blocks[i] = SpaceTimeBlock(
                blocks[i],
                max_num_frames=max_num_frames,
                learnable_time_embed=learnable_time_embed,
                time_embed_base=time_embed_base,
            )

    vision_tower.use_video_encoder = True
    vision_tower.spacetime_layer_stride = spacetime_layer_stride
    vision_tower.past_drop_layer = past_drop_layer
    vision_tower.max_num_frames = max_num_frames

    if not getattr(vision_tower, "_video_forward_patched", False):
        orig = vision_tower.forward
        vision_tower.forward = _make_vision_tower_forward(orig).__get__(vision_tower, type(vision_tower))
        vision_tower._video_forward_patched = True

    if not getattr(visual, "_video_forward_patched", False):
        orig_ff = visual._forward_func
        visual._forward_func = _make_wrapper_forward_func(orig_ff).__get__(visual, type(visual))
        orig_f = visual.forward
        visual.forward = _make_wrapper_forward(orig_f).__get__(visual, type(visual))
        visual._video_forward_patched = True

    visual.use_video_encoder = True
    visual.spacetime_layer_stride = spacetime_layer_stride
    visual.past_drop_layer = past_drop_layer
    visual.max_num_frames = max_num_frames


# -----------------------------------------------------------------------------
# Dual-tower model
# -----------------------------------------------------------------------------


def mask_apply(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    text_funcs,
    vision_funcs,
    out_dims=None,
):
    """Batch-flattened modality routing for the MoT dual-tower forward.

    Args:
        hidden_states: ``(B, S, D)`` token features.
        mask: ``(B, S)`` bool / int. ``True`` (or ``1``) -> vision token,
            ``False`` (or ``0``) -> text token.
        text_funcs: callables applied to text tokens (one per output).
        vision_funcs: callables applied to vision tokens (one per output).
        out_dims: optional list of per-output last-dim sizes. ``None``
            means each output keeps the input ``D``.

    Returns:
        ``list[Tensor]`` with shape ``(B, S, out_dim_i)``; entries the
        functions did not write are zeros (``torch.empty`` slots that
        were never indexed are explicitly zero-initialised when neither
        modality covers the full sequence, see below).
    """
    B, S, D = hidden_states.size()
    flat = hidden_states.reshape(B * S, D)
    mask_flat = mask.reshape(B * S).bool()

    if out_dims is None:
        out_flat = [torch.zeros(B * S, D, device=flat.device, dtype=flat.dtype) for _ in text_funcs]
    else:
        out_flat = [torch.zeros(B * S, od, device=flat.device, dtype=flat.dtype) for od in out_dims]

    text_idx = ~mask_flat
    if text_idx.any():
        hs_t = flat[text_idx]
        for i, fn in enumerate(text_funcs):
            out_flat[i][text_idx] = fn(hs_t)

    vis_idx = mask_flat
    if vis_idx.any():
        hs_v = flat[vis_idx]
        for i, fn in enumerate(vision_funcs):
            out_flat[i][vis_idx] = fn(hs_v)

    return [o.view(B, S, -1) for o in out_flat]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply Rotary Position Embedding to the query and key tensors.

    Args:
        q: query tensor.
        k: key tensor.
        cos: cosine part of the rotary embedding.
        sin: sine part of the rotary embedding.
        position_ids: unused, kept for signature compatibility.
        unsqueeze_dim: which axis to unsqueeze on for broadcasting.

    Returns:
        ``(q_rotated, k_rotated)``.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class HyDualTowerConfig(PretrainedConfig):
    """Config for :class:`HyDualTower`.

    Both ``vlm_config`` and ``expert_config`` are full ``PretrainedConfig``
    instances and are passed in directly.
    """

    model_type = "hy_dual_tower"
    sub_configs = {
        "vlm_config": HunYuanVLMoTConfig,
        "expert_config": HunYuanVLMoTConfig,
    }

    def __init__(
        self,
        vlm_config: PretrainedConfig | None = None,
        expert_config: PretrainedConfig | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        **kwargs,
    ):
        if isinstance(vlm_config, dict):
            vlm_config = HunYuanVLMoTConfig.from_dict(vlm_config)
        if isinstance(expert_config, dict):
            expert_config = HunYuanVLMoTConfig.from_dict(expert_config)
        self.vlm_config = vlm_config
        self.expert_config = expert_config

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation

        # Optional reference to the outer ``HyVLAConfig`` (used by
        # HyVLAFlowMatching to keep its proj_width in sync with the
        # expert tower's hidden_size).
        self.config = kwargs.pop("config", None)

        super().__init__(**kwargs)

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )
        if self.attention_implementation != "eager":
            raise ValueError(
                f"Wrong value provided for `attention_implementation` "
                f"({self.attention_implementation}). Expected 'eager'."
            )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class HyDualTower(PreTrainedModel):
    """Plug-in dual-tower container: VLM + action expert with shared attention.

    The two slot attributes ``self.vlm`` and ``self.expert`` are
    architecture-neutral. The default factory uses
    :class:`HunYuanVLMoTForConditionalGeneration` and
    :class:`HunYuanDenseV1MoTForCausalLM`; any HuggingFace-style decoder
    that satisfies the modality-aware MoT layer contract can be plugged
    in via :meth:`from_components`.

    State-dict layout:
        ``vlm.<rest>``     -- VLM weights
        ``expert.<rest>``  -- expert weights
    """

    config_class = HyDualTowerConfig

    def __init__(self, config: HyDualTowerConfig):
        super().__init__(config=config)
        self.config = config
        self.vlm = HunYuanVLMoTForConditionalGeneration(config=config.vlm_config)
        # Action-expert: inner LM-head wrapper of the same HunYuanVLMoT
        # family. Picking *ForCausalLM* preserves the attribute layout
        # ``expert.model.{layers, norm, rotary_emb, embed_tokens}`` and
        # the external ``expert.lm_head``.
        self.expert = _HunYuanVLMoTTextForCausalLM(config=config.expert_config)
        # The expert reuses the VLM tokenizer's text embeddings (set
        # externally by ``HyVLAFlowMatching``); drop the unused
        # expert-side embed_tokens to save memory.
        self.expert.model.embed_tokens = None

        self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

    # ------------------------------------------------------------------
    # Component-level factory (the "plug-in" contract).
    # ------------------------------------------------------------------
    @classmethod
    def from_components(
        cls,
        *,
        vlm: PreTrainedModel,
        expert: PreTrainedModel,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        outer_config: PretrainedConfig | None = None,
    ) -> HyDualTower:
        """Build a ``HyDualTower`` from pre-instantiated VLM / expert modules.

        The returned tower assumes ownership of both modules; the caller
        should not keep separate references. Useful for swapping in
        custom backbones without subclassing.

        Example::

            from lerobot.policies.hy_vla.modeling_hy_vla import HyDualTower

            tower = HyDualTower.from_components(
                vlm=MyCustomVLM.from_pretrained("..."),
                expert=MyCustomExpert.from_pretrained("..."),
            )
        """
        cfg = HyDualTowerConfig(
            vlm_config=vlm.config,
            expert_config=expert.config,
            freeze_vision_encoder=freeze_vision_encoder,
            train_expert_only=train_expert_only,
            attention_implementation=attention_implementation,
            config=outer_config,
        )
        instance = cls.__new__(cls)
        PreTrainedModel.__init__(instance, config=cfg)
        instance.config = cfg
        instance.vlm = vlm
        instance.expert = expert
        if instance.expert.model.embed_tokens is not None:
            instance.expert.model.embed_tokens = None
        instance.to_bfloat16_like_physical_intelligence()
        instance.set_requires_grad()
        return instance

    # ------------------------------------------------------------------
    # Training-mode toggles
    # ------------------------------------------------------------------
    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.vlm.model.visual.eval()
            for params in self.vlm.model.visual.parameters():
                params.requires_grad = False
        else:
            self._unfreeze_vision_tower_inplace(self.vlm.model.visual)

        if self.config.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self.vlm.model.visual.eval()

        if self.config.train_expert_only:
            self.vlm.eval()

    @staticmethod
    def _unfreeze_vision_tower_inplace(visual_module: nn.Module) -> None:
        # Step 1: parameters
        for p in visual_module.parameters():
            p.requires_grad = True

        # Step 2: forward patch. Bind once -- subsequent calls overwrite
        # with the same closure so re-entry is harmless.
        def _forward_with_grad(self, images, cal_attn_pool=False):
            # Mirrors HYViT2_400MAnyRes.forward but without ``no_grad``.
            image_features, img_size, cls_token = self._forward_func(images, cal_attn_pool=cal_attn_pool)
            if isinstance(images, list):
                image_features = [
                    self.merger(x, s).squeeze(0) for x, s in zip(image_features, img_size, strict=False)
                ]
            else:
                image_features = self.merger(image_features, img_size)
                C = image_features.shape[-1]
                image_features = [image_features.reshape(-1, C)]
            return image_features

        visual_module.forward = types.MethodType(_forward_with_grad, visual_module)

    def to_bfloat16_like_physical_intelligence(self):
        """Mirror the openpi-style precision policy.

        Cast the entire VLM tower + the layer parameters of both towers
        to bfloat16; everything outside ``layers`` / ``visual`` keeps its
        original dtype (typically fp32 for embeddings and projections).
        """
        self.vlm = self.vlm.to(dtype=torch.bfloat16)

        params_to_change_dtype = [
            "language_model.model.layers",
            "expert.model.layers",
            "visual",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    # Visual + language token embedders (called by the outer policy)
    # ------------------------------------------------------------------
    def embed_image(self, image: torch.Tensor):
        """Encode RGB inputs through the VLM vision tower.

        Args:
            image: ``(C, H, W)`` (single frame), ``(B, C, H, W)`` (batch
                of single frames), or ``(B, K, C, H, W)`` (MEM video
                stack -- routed to the SpaceTime-augmented ViT).

        Returns:
            ``(B, N, D)`` patch features, where ``N`` is the per-frame
            (or per-stack) token count and ``D`` is the visual hidden
            dim.
        """
        if image.dim() == 5:
            # Wrapper returns [(B*N, C)] (batch flattened); restore to (B, N, C).
            B = image.shape[0]
            feat = self.vlm.visual(image)[0]
            return feat.view(B, -1, feat.shape[-1]).contiguous()

        image_list = list(image.unsqueeze(1) if image.dim() == 3 else image.split(1, dim=0))
        # image_list: list of (1, 3, h, w)
        image_features = self.vlm.visual(image_list)  # list of (num_tokens, 2048)
        image_features = torch.stack(image_features, dim=0)
        return image_features

    def embed_language_tokens(self, tokens: torch.Tensor):
        """Look up text-token embeddings via the VLM language tower."""
        return self.vlm.language_model.model.embed_tokens(tokens)

    # ------------------------------------------------------------------
    # Shared-attention dual-tower forward
    # ------------------------------------------------------------------
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        modality_masks: list[torch.FloatTensor] = None,
    ):
        models = [self.vlm.language_model.model, self.expert.model]
        att_vis_output = []
        prefix_emb_layer_outputs = []
        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        num_layers = self.vlm.config.num_hidden_layers

        # ``position_embeddings`` are constant across layers; compute once.
        # The first arg picks output dtype/device, so we pass a float
        # tensor (not the int64 ``position_ids``).
        _dtype_ref = next(h for h in inputs_embeds if h is not None)
        position_embeddings = models[0].rotary_emb(_dtype_ref.float(), position_ids)

        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []

            # Per-tower sequence length (used to slice the concatenated
            # q/k tensors before the per-tower q/k layernorm).
            seq_len_list = []

            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue

                layer = models[i].layers[layer_idx]
                modality_mask = modality_masks[i]

                hidden_states = mask_apply(
                    hidden_states,
                    modality_mask,
                    [lambda x: layer.input_layernorm(x)],
                    [lambda x: layer.input_layernorm_v(x)],
                )[0]

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                # Batch-flattened modality routing (see ``mask_apply``
                # docstring). The dual-tower forward always supplies a
                # non-None ``modality_mask`` so we bypass the bundled model's
                # per-sample fallback path entirely.
                query_state, key_state, value_state = mask_apply(
                    hidden_states,
                    modality_mask,
                    [
                        lambda x: layer.self_attn.q_proj(x),
                        lambda x: layer.self_attn.k_proj(x),
                        lambda x: layer.self_attn.v_proj(x),
                    ],
                    [
                        lambda x: layer.self_attn.q_proj_v(x),
                        lambda x: layer.self_attn.k_proj_v(x),
                        lambda x: layer.self_attn.v_proj_v(x),
                    ],
                    out_dims=[
                        self.config.vlm_config.num_attention_heads * layer.self_attn.head_dim,
                        self.config.vlm_config.num_key_value_heads * layer.self_attn.head_dim,
                        self.config.vlm_config.num_key_value_heads * layer.self_attn.head_dim,
                    ],
                )

                # (batch_size, num_heads, seq_len, head_dim)
                query_state = query_state.view(hidden_shape).transpose(1, 2)
                key_state = key_state.view(hidden_shape).transpose(1, 2)
                value_state = value_state.view(hidden_shape).transpose(1, 2)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)
                seq_len_list.append(hidden_states.shape[1])

            query_states = torch.cat(query_states, dim=2)
            key_states = torch.cat(key_states, dim=2)
            value_states = torch.cat(value_states, dim=2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            q_parts = query_states.split(seq_len_list, dim=2)
            k_parts = key_states.split(seq_len_list, dim=2)
            q_normed = []
            k_normed = []

            vlm_layer = models[0].layers[layer_idx]
            for q_part, k_part in zip(q_parts, k_parts, strict=False):
                q_normed.append(vlm_layer.self_attn.query_layernorm(q_part))
                k_normed.append(vlm_layer.self_attn.key_layernorm(k_part))
            query_states = torch.cat(q_normed, dim=2)
            key_states = torch.cat(k_normed, dim=2)

            # (batch_size, seq_len, num_heads, head_dim)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
                    key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                    value_states = torch.cat(
                        [past_key_values[layer_idx]["value_states"], value_states], dim=1
                    )
                    past_key_values[layer_idx]["key_states"] = key_states
                    past_key_values[layer_idx]["value_states"] = value_states

            attention_interface = self.get_attention_interface()
            att_output, probs = attention_interface(
                attention_mask,
                batch_size,
                layer.self_attn.head_dim,
                query_states,
                key_states,
                value_states,
            )

            att_output = att_output.to(dtype=torch.bfloat16)  # (b, seq_vlm, ...)
            att_vis_output.append(probs)  # probs (b, 8, seq, seq)

            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                modality_mask = modality_masks[i]
                layer = models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)

                    out_emb = mask_apply(
                        att_output[:, start:end],
                        modality_mask,
                        [lambda x: layer.self_attn.o_proj(x)],
                        [lambda x: layer.self_attn.o_proj_v(x)],
                        out_dims=[models[i].config.hidden_size],
                    )[0]

                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()
                    out_emb = mask_apply(
                        out_emb,
                        modality_mask,
                        [lambda x: layer.mlp(layer.post_attention_layernorm(x))],
                        [lambda x: layer.mlp_v(layer.post_attention_layernorm_v(x))],
                    )[0]

                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)
                    start = end
                else:
                    outputs_embeds.append(None)

            prefix_emb_layer_outputs.append(outputs_embeds[0])
            inputs_embeds = outputs_embeds

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        return outputs_embeds, past_key_values, att_vis_output, prefix_emb_layer_outputs

    # ------------------------------------------------------------------
    # Attention backends
    # ------------------------------------------------------------------
    def get_attention_interface(self):
        return self.eager_attention_forward

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.config.vlm_config.num_attention_heads
        num_key_value_heads = self.config.vlm_config.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        big_neg = -2.3819763e38  # bf16 -inf approximation

        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output, probs


# -----------------------------------------------------------------------------
# Flow-matching model
# -----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# VLM config loading: returns a concrete ``HunYuanVLMoTConfig``
# (text_config + vision_config nested form) so downstream dual_tower
# construction has a single config schema to worry about. There are two
# checkpoint flavours the loader handles:
#
#   (a) Self-contained VLA ckpt (the released layout):
#       embeds ``vlm_config_dict`` with ``model_type=hunyuan_vl_mot`` and a
#       populated ``text_config`` block. We instantiate ``HunYuanVLMoTConfig``
#       directly from the embedded dict -- no disk / network access.
#
#   (b) Bare VLM directory or HF Hub repo id (e.g. ``tencent/HY-Embodied-0.5``):
#       loaded directly through ``HunYuanVLMoTConfig.from_pretrained``.
#
# Returns: a ``HunYuanVLMoTConfig`` (always).
# ---------------------------------------------------------------------------


def _load_vlm_config(config_or_path):
    """Load a concrete ``HunYuanVLMoTConfig``.

    Accepts either:
      * a ``HyVLAConfig`` instance -- in which case ``config.vlm_config_dict``
        is the authoritative source: no disk / network access is needed.
      * a string ``model_path`` (local dir or HF repo id) -- in which case
        the concrete config class loads ``config.json`` without executing
        checkpoint-provided code.
    """
    # ---- Path 1: embedded vlm_config_dict (self-contained VLA ckpt) ----
    if isinstance(config_or_path, HyVLAConfig) or hasattr(config_or_path, "vlm_config_dict"):
        cfg = config_or_path
        embedded = getattr(cfg, "vlm_config_dict", None)
        if embedded:
            data = dict(embedded)
            mt = data.get("model_type")
            if mt != "hunyuan_vl_mot" or "text_config" not in data:
                raise ValueError(
                    "vlm_config_dict embedded in HyVLAConfig is not in the "
                    "expected nested schema (model_type='hunyuan_vl_mot' "
                    f"with a 'text_config' "
                    f"block, got model_type={mt!r}, "
                    f"has_text_config={'text_config' in data})."
                )
            print(
                "[modeling_hy_vla] VLM config loaded from embedded "
                "vlm_config_dict (nested hunyuan_vl_mot schema).",
                file=sys.stderr,
                flush=True,
            )
            data.pop("model_type", None)
            return HunYuanVLMoTConfig(**data)
        # Fall through: raw-VLM bootstrap (``pretrain_source`` in
        # {``vlm``, ``scratch``}); resolve from ``cfg.vlm_model_path``.
        model_path = cfg.vlm_model_path
        if not model_path:
            raise ValueError(
                "_load_vlm_config: HyVLAConfig has no "
                "``vlm_config_dict`` AND no ``vlm_model_path``. "
                "Self-contained ckpts must embed ``vlm_config_dict``; "
                "raw-VLM bootstrap flows must set ``vlm_model_path``."
            )
    else:
        model_path = config_or_path
    return HunYuanVLMoTConfig.from_pretrained(model_path, trust_remote_code=False)


def _get_safe_dtype(dtype: torch.dtype, device: str | torch.device) -> torch.dtype:
    """Return ``dtype`` clamped to one supported on ``device``.

    MPS does not support float64; everything else does.
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    return dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = _get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma_alpha_dist = torch.distributions.Gamma(alpha, 1)
    gamma_beta_dist = torch.distributions.Gamma(beta, 1)

    x = gamma_alpha_dist.sample((bsize,)).to(device)
    y = gamma_beta_dist.sample((bsize,)).to(device)
    return x / (x + y)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


class HyVLAFlowMatching(nn.Module):
    """Hy-VLA flow-matching action expert.

    Owns the dual-tower (VLM + action expert) and the flow-matching
    training and sampling logic used by :class:`HyVLAPolicy`.

    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │action│        │
    │  ┌──────────►│expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x N   │        │
    │ │         │  └▲──▲──┘        │
    │ │   VLM   │   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
    """

    def __init__(self, config, language_tokenizer):
        super().__init__()
        self.config = config
        self.language_tokenizer = language_tokenizer

        # Self-contained checkpoints read the VLM config from
        # ``self.config.vlm_config_dict``; fresh models resolve it from
        # ``self.config.vlm_model_path``.
        vlm_inner_config = _load_vlm_config(self.config)

        # Expert config = VLM config with ``hidden_size`` overridden by
        # ``proj_width``. Released ckpt: ``hidden_size=1024`` (vs the VLM's
        # 2048) and ``intermediate_size=2048``; everything else (layers,
        # heads, vocab, rope) is shared with the VLM.
        expert_inner_config = copy.deepcopy(vlm_inner_config)
        expert_inner_config.hidden_size = self.config.proj_width
        expert_inner_config.intermediate_size = 2048
        if hasattr(expert_inner_config, "dense_list"):
            expert_inner_config.dense_list = [self.config.proj_width, 0]

        dual_tower_config = HyDualTowerConfig(
            vlm_config=vlm_inner_config,
            expert_config=expert_inner_config,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
            config=self.config,  # outer HyVLAConfig (kept for proj_width etc.)
        )
        self.dual_tower = HyDualTower(dual_tower_config)

        # Projections are float32
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.set_requires_grad()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for the dual-tower transformer processing.

        Layout (per sample):
            <bos><hy_User>
            for each image:
                <vision_start>
                image_patch_grid interleaved with <vision_split> at the end of every row
                <vision_end>
            language_tokens
        """
        embs = []
        pad_masks = []
        att_masks = []
        modality_mask = []

        # Special tokens (BOS / role / vision boundaries / split / assistant)
        img = images[0]
        # add <｜hy_begin▁of▁sentence｜><｜hy_User｜>
        bos_token = torch.full(
            (img.shape[0], 1), self.language_tokenizer.convert_tokens_to_ids("<｜hy_begin▁of▁sentence｜>")
        )
        bos_token = bos_token.to(img.device)
        bos_emb = self.dual_tower.embed_language_tokens(bos_token)
        hy_user_token = torch.full(
            (img.shape[0], 1), self.language_tokenizer.convert_tokens_to_ids("<｜hy_User｜>")
        )
        hy_user_token = hy_user_token.to(img.device)
        hy_user_emb = self.dual_tower.embed_language_tokens(hy_user_token)

        # add <｜hy_place▁holder▁no▁666｜> vision_start_token
        vision_start_token = torch.full(
            (img.shape[0], 1), self.language_tokenizer.convert_tokens_to_ids("<｜hy_place▁holder▁no▁666｜>")
        )
        vision_start_token = vision_start_token.to(img.device)
        vision_start_emb = self.dual_tower.embed_language_tokens(vision_start_token)

        # add <｜hy_place▁holder▁no▁666｜> vision_end_token
        vision_end_token = torch.full(
            (img.shape[0], 1), self.language_tokenizer.convert_tokens_to_ids("<｜hy_place▁holder▁no▁667｜>")
        )
        vision_end_token = vision_end_token.to(img.device)
        vision_end_emb = self.dual_tower.embed_language_tokens(vision_end_token)

        # add <｜hy_place▁holder▁no▁666｜> vision_split_token
        vision_split_token = torch.full(
            (img.shape[0], 1), self.language_tokenizer.convert_tokens_to_ids("<｜hy_place▁holder▁no▁671｜>")
        )
        vision_split_token = vision_split_token.to(img.device)
        vision_split_emb = self.dual_tower.embed_language_tokens(vision_split_token)

        # 1. Add [bos_token, hy_user_token]
        embs.extend([bos_emb, hy_user_emb])
        pad_masks.append(torch.ones((images[0].shape[0], 2), dtype=torch.bool, device=images[0].device))
        att_masks.extend([1, 1])
        modality_mask.extend([False, False])

        # Track image-token index ranges so the visual-segment attention mask
        # tweak (see ``_apply_visual_segment_mask``) can address them later.
        image_idx_ranges = []  # per-row patch ranges (excludes split tokens)
        image_full_ranges = []  # full per-image span (patches + split rows)

        # 2. Add vision_start + image patches with row-wise split tokens + vision_end
        for i, (img, img_mask) in enumerate(zip(images, img_masks, strict=True)):
            bs = img.shape[0]

            # vision_start
            embs.append(vision_start_emb)
            pad_masks.append(torch.ones((bs, 1), dtype=torch.bool, device=img.device))
            att_masks.append(1)
            modality_mask.append(False)

            # embed image (bs, num_patches, emb_dim)
            img_emb = self.dual_tower.embed_image(img).to(dtype=torch.bfloat16)
            num_patches, emb_dim = img_emb.shape[1], img_emb.shape[2]
            grid_size = int(num_patches**0.5)
            assert grid_size * grid_size == num_patches, "num_patches must be square"

            img_emb_grid = img_emb.view(bs, grid_size, grid_size, emb_dim)
            split_expanded = vision_split_emb.unsqueeze(1).expand(bs, grid_size, 1, emb_dim)
            img_emb_with_split = torch.cat([img_emb_grid, split_expanded], dim=2)
            img_emb_with_split = img_emb_with_split.view(bs, -1, emb_dim)
            embs.append(img_emb_with_split)

            row_len = grid_size + 1
            total_img_tokens = grid_size * row_len
            start_idx = len(att_masks)

            # Per-row patch ranges (exclude the trailing split token of each row).
            row_ranges = [
                (start_idx + r * row_len, start_idx + r * row_len + grid_size) for r in range(grid_size)
            ]
            image_idx_ranges.extend(row_ranges)

            # Full span of this image's visual segment (patches + split tokens).
            image_full_ranges.append((start_idx, start_idx + total_img_tokens))

            att_masks.extend([1] * total_img_tokens)
            # Each grid row: ``grid_size`` patch tokens (modality=True) + 1 split token (False).
            modality_mask.extend(([True] * grid_size + [False] * 1) * grid_size)

            img_mask_expanded = img_mask[:, None].expand(bs, total_img_tokens)
            pad_masks.append(img_mask_expanded)

            # vision_end
            embs.append(vision_end_emb)
            pad_masks.append(torch.ones((bs, 1), dtype=torch.bool, device=img.device))
            att_masks.append(1)
            modality_mask.append(False)

        # 3. Language tokens
        lang_emb = self.dual_tower.embed_language_tokens(lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks.extend([1] * num_lang_embs)
        modality_mask.extend([False] * num_lang_embs)

        # 4. Stack into tensors
        bsize = images[0].shape[0]
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1).to(torch.bool)

        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, -1)

        modality_mask = torch.tensor(modality_mask, dtype=torch.bool, device=pad_masks.device)
        modality_mask = modality_mask[None, :].expand(bsize, -1)

        return embs, pad_masks, att_masks, modality_mask, image_idx_ranges, image_full_ranges

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions and timestep for the action expert.

        Emits a single absolute state token from ``state`` (passed through
        ``state_proj`` and cast to bf16), then the action / time embedding
        block. The state token shares one attention block with the action
        chunk: leading ``att_masks=1`` followed by ``0`` for the action
        tokens.
        """
        embs = []
        pad_masks = []
        att_masks = []
        modality_mask = []

        # --- State token ----------------------------------------------------
        assert state is not None, "embed_suffix: ``state`` is required."
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        # (B, D) -> (B, 1, D)
        state_block = state_emb[:, None, :]
        embs.append(state_block)

        bsize = state_block.shape[0]
        T_state = state_block.shape[1]
        device = state_block.device

        state_mask = torch.ones(bsize, T_state, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # All state tokens share one attention block: leading 1, rest 0.
        # Mirrors the action-chunk wiring further down.
        att_masks += [1] + [0] * (T_state - 1)
        modality_mask += [True] * T_state

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=torch.bfloat16)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions.to(torch.bfloat16))  # torch.float32 -> bf16

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)  # torch.float32

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))
        modality_mask += [True] * (self.config.n_action_steps)

        embs = torch.cat(embs, dim=1)  # torch.bfloat16
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        modality_mask = torch.tensor(modality_mask, dtype=torch.bool, device=pad_masks.device)
        modality_mask = modality_mask[None, :].expand(bsize, len(modality_mask))

        return embs, pad_masks, att_masks, modality_mask

    def _apply_visual_segment_mask(
        self,
        att_2d_masks,
        image_idx_ranges,
        image_full_ranges,
    ):
        """In-place rewrite the visual-segment portion of ``att_2d_masks``.

        Two scopes are selectable via ``self.config.visual_segment_isolation``:

        * ``False`` -- *patch-only* (default, backward-compatible):
          1. collect every image's ``image_idx_ranges`` (image-patch tokens,
             excluding the per-row split tokens) and zero out their pairwise
             visibility;
          2. inside each image's ``image_full_range``, set the image-patch
             tokens to be bidirectionally visible.
          Image-patch / split-row tokens still see segment-external tokens
          via the causal mask, which differs slightly from the VLM-time
          eager MoT attention behaviour.

        * ``True`` -- *full-segment isolation* (matches
          eager MoT attention): for each image's
          ``image_full_range`` (image patches + split / newline rows,
          excluding ``vision_start`` / ``vision_end``):
          1. clear all visibility on the rows of those tokens;
          2. enable bidirectional visibility within the segment.
          The released RoboTwin post-train ckpt was trained under this mode,
          so reproducing it requires ``visual_segment_isolation=True`` in
          ``config.json``.

        Args:
            att_2d_masks: ``(B, S, S)`` bool tensor; modified in place.
            image_idx_ranges: per-row image-patch ``[start, end)`` ranges
                (excluding split tokens).
            image_full_ranges: per-image ``[start, end)`` ranges covering
                image patches plus split / newline rows.
        """
        if getattr(self.config, "visual_segment_isolation", False):
            # Full-segment isolation: rewrite each image_full_range as a
            # self-contained bidirectional block.
            for img_full_start, img_full_end in image_full_ranges:
                full_range_idx = torch.arange(img_full_start, img_full_end, device=att_2d_masks.device)
                # Clear outward visibility for image-patch + split rows.
                att_2d_masks[:, full_range_idx, :] = False
                # Re-enable visibility within the segment.
                att_2d_masks[:, full_range_idx[:, None], full_range_idx[None, :]] = True
            return

        # Patch-only (default): only adjust image-patch tokens; split rows
        # stay on the causal pathway.
        # Step 1: clear pairwise visibility between every image-patch token
        # (this also drops the causal-pathway visibility between them).
        all_img_indices = []
        for s, e in image_idx_ranges:
            all_img_indices.extend(range(s, e))
        if all_img_indices:
            idx = torch.tensor(all_img_indices, device=att_2d_masks.device)
            att_2d_masks[:, idx[:, None], idx[None, :]] = False

        # Step 2: re-enable bidirectional visibility among image-patch
        # tokens that belong to the same image.
        for img_full_start, img_full_end in image_full_ranges:
            img_indices = []
            for s, e in image_idx_ranges:
                if s >= img_full_start and e <= img_full_end:
                    img_indices.extend(range(s, e))
            if img_indices:
                idx = torch.tensor(img_indices, device=att_2d_masks.device)
                att_2d_masks[:, idx[:, None], idx[None, :]] = True

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state=None,
        actions=None,
        noise=None,
        time=None,
        lang_token_labels=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        losses_flow = None
        losses_ntp = None

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            modality_mask_prefix,
            image_idx_ranges,
            image_full_ranges,
        ) = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # action, text + action
        if actions is not None:
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)

            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)

            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            suffix_embs, suffix_pad_masks, suffix_att_masks, modality_mask_suffix = self.embed_suffix(
                state,
                x_t,
                time,
            )

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # text only
        else:
            suffix_embs = None
            pad_masks = torch.cat([prefix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Adjust visual-segment attention according to the configured scope.
        self._apply_visual_segment_mask(att_2d_masks, image_idx_ranges, image_full_ranges)

        (prefix_out, suffix_out), _, att_vis_output, _ = self.dual_tower.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            modality_masks=[modality_mask_prefix, modality_mask_suffix],
        )

        # Flow matching prediction
        if actions is not None:
            suffix_out = suffix_out[:, -self.config.n_action_steps :]
            v_t = self.action_out_proj(suffix_out)  # torch.float32 -> bf16
            losses_flow = F.mse_loss(u_t.float(), v_t.float(), reduction="none")  # bf16 -> torch.float32

        # Next-token prediction
        if lang_token_labels is not None:
            attention_mask = None
            logits = self.dual_tower.vlm.language_model.lm_head(prefix_out)

            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., -self.config.tokenizer_max_length : -1, :]
            shift_labels = lang_token_labels[..., 1:]

            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()

            # Flatten the tokens
            losses_ce = nn.CrossEntropyLoss(
                reduction="none",
                ignore_index=self.dual_tower.vlm.config.ignore_index,
            )

            flat_logits = shift_logits.view(-1, self.dual_tower.vlm.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            losses_ntp = losses_ce(flat_logits, flat_labels)

        return losses_flow, losses_ntp

    # @torch.compile(mode="reduce-overhead")
    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None, vis_attn=False
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            modality_mask_prefix,
            image_idx_ranges,
            image_full_ranges,
        ) = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Adjust visual-segment attention according to the configured scope.
        self._apply_visual_segment_mask(prefix_att_2d_masks, image_idx_ranges, image_full_ranges)

        # Compute image and language key value cache
        (prefix_out, _), past_key_values, _, _ = self.dual_tower.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            modality_masks=[modality_mask_prefix, None],
        )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t, att_vis_output = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt

        if vis_attn:
            # Strip non-patch tokens from att_vis_output, leaving the
            # contiguous (B, H, suffix_len, num_patches * num_views)
            # tensor that downstream visualisation tooling expects.
            all_img_indices = []
            for s, e in image_idx_ranges:
                all_img_indices.extend(range(s, e))
            img_idx_tensor = torch.tensor(all_img_indices, dtype=torch.long, device=device)

            cleaned_att = []
            for layer_att in att_vis_output:
                cleaned_att.append(layer_att[:, :, :, img_idx_tensor])
            return x_t, cleaned_att

        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        # IMPORTANT: copy the past_key_values, or its size will increase during n-step denoise.
        past_key_values_vlm = copy.deepcopy(past_key_values)

        suffix_embs, suffix_pad_masks, suffix_att_masks, modality_mask_suffix = self.embed_suffix(
            state,
            x_t,
            timestep,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _, att_vis_output, _ = self.dual_tower.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values_vlm,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            modality_masks=[None, modality_mask_suffix],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        v_t = self.action_out_proj(suffix_out)  # bf16 -> torch.float32
        return v_t, att_vis_output


# -----------------------------------------------------------------------------
# LeRobot policy wrapper
# -----------------------------------------------------------------------------


def _resize_with_pad(image: Tensor, height: int, width: int, pad_value: float = 0) -> Tensor:
    if image.shape[-2:] == (height, width):
        return image
    ratio = max(image.shape[-2] / height, image.shape[-1] / width)
    resized_height = max(1, int(image.shape[-2] / ratio))
    resized_width = max(1, int(image.shape[-1] / ratio))
    resized = F.interpolate(
        image,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    pad_height, pad_width = height - resized_height, width - resized_width
    top, left = pad_height // 2, pad_width // 2
    return F.pad(
        resized,
        (left, pad_width - left, top, pad_height - top),
        value=pad_value,
    )


def _pad_last(value: Tensor, dimension: int) -> Tensor:
    if value.shape[-1] > dimension:
        raise ValueError(f"Cannot pad dimension {value.shape[-1]} to {dimension}.")
    if value.shape[-1] == dimension:
        return value
    output = value.new_zeros(*value.shape[:-1], dimension)
    output[..., : value.shape[-1]] = value
    return output


class HyVLAPolicy(PreTrainedPolicy):
    """First-class LeRobot policy for the released Hy-VLA checkpoints."""

    config_class = HyVLAConfig
    name = "hy_vla"

    def __init__(self, config: HyVLAConfig, **_: Any):
        super().__init__(config)
        config.validate_features()

        tokenizer_source = getattr(config, "_tokenizer_source", None) or config.vlm_model_path
        self.language_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=False,
            revision=getattr(config, "_tokenizer_revision", None),
            fix_mistral_regex=True,
        )
        self.model = HyVLAFlowMatching(config, self.language_tokenizer)
        # The action expert consumes injected action/state embeddings and never
        # produces vocabulary logits. The author checkpoints therefore omit
        # this otherwise randomly initialized, unused CausalLM output head.
        self.model.dual_tower.expert.lm_head = None
        # The released runtime trains and evaluates the complete policy in
        # BF16. LeRobot's generic factory moves policies to a device but does
        # not change dtype, so establish the checkpoint dtype here instead of
        # requiring a private training runner to call ``policy.to(bfloat16)``.
        self.model.to(dtype=torch.bfloat16)
        self.enable_video_encoder_if_needed()
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        revision: str | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> HyVLAPolicy:
        if config is None:
            config = HyVLAConfig.from_pretrained(
                pretrained_name_or_path,
                revision=revision,
                force_download=kwargs.get("force_download", False),
                cache_dir=kwargs.get("cache_dir"),
                local_files_only=kwargs.get("local_files_only", False),
                token=kwargs.get("token"),
            )
        if not isinstance(config, HyVLAConfig):
            raise TypeError(f"Expected HyVLAConfig, got {type(config)!r}.")
        # Transient fields are intentionally not dataclass fields, so absolute
        # local paths are never baked into config.json on the next save.
        config._tokenizer_source = str(pretrained_name_or_path)
        config._tokenizer_revision = revision
        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            revision=revision,
            strict=strict,
            **kwargs,
        )

    def _save_pretrained(self, save_directory: Path, state_dict: dict[str, Tensor] | None = None) -> None:
        if not self.config.vlm_config_dict:
            self.config.vlm_config_dict = self.model.dual_tower.vlm.config.to_dict()
        super()._save_pretrained(save_directory, state_dict=state_dict)
        self.language_tokenizer.save_pretrained(str(save_directory))

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.execution_horizon)
        history_span = (self.config.img_history_size - 1) * self.config.img_history_interval + 1
        self._image_history: dict[str, deque[Tensor]] = {
            key: deque(maxlen=history_span) for key in self.config.image_features
        }

    def _append_inference_history(self, batch: dict[str, Any]) -> None:
        """Record every eval frame, including frames served from the action queue."""

        for key, history in self._image_history.items():
            image = batch.get(key)
            if not isinstance(image, Tensor) or image.ndim != 4:
                raise ValueError(f"MEM inference requires {key} as BCHW, got {getattr(image, 'shape', None)}")
            history.append(image.detach().to("cpu"))

    def _with_inference_history(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Build the author's deterministic K-frame, interval-S MEM stacks."""

        output = dict(batch)
        history_size = self.config.img_history_size
        interval = self.config.img_history_interval
        for key, history_queue in self._image_history.items():
            history = list(history_queue)
            step = len(history) - 1
            current = history[-1]
            frames: list[Tensor] = []
            for slot in range(history_size):
                unclipped = step - (history_size - 1 - slot) * interval
                frames.append(torch.zeros_like(current) if unclipped < 0 else history[unclipped])
            output[key] = torch.stack(frames, dim=1).to(batch[key])
        return output

    def enable_video_encoder_if_needed(self) -> None:
        if not self.config.use_video_encoder:
            return
        apply_video_encoder_patch(
            self.model.dual_tower.vlm.model.visual,
            spacetime_layer_stride=self.config.spacetime_layer_stride,
            past_drop_layer=self.config.past_drop_layer,
            max_num_frames=self.config.max_num_frames,
        )

    def get_optim_params(self):
        return self.parameters()

    def _model_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.model.parameters())
        return parameter.device, parameter.dtype

    def prepare_images(self, batch: dict[str, Any]) -> tuple[list[Tensor], list[Tensor]]:
        model_device, model_dtype = self._model_device_dtype()
        expected_keys = list(self.config.image_features)
        present_keys = [key for key in expected_keys if key in batch]
        missing_keys = [key for key in expected_keys if key not in batch]
        if not present_keys:
            raise ValueError(f"All expected image features are missing: {expected_keys}")

        processed: dict[str, tuple[Tensor, Tensor]] = {}
        template: Tensor | None = None
        template_mask: Tensor | None = None
        for key in present_keys:
            image = batch[key]
            if not isinstance(image, Tensor) or image.ndim not in {4, 5}:
                raise ValueError(
                    f"{key} must be BCHW or BKCHW tensor, got {type(image)} {getattr(image, 'shape', None)}"
                )
            image = image.to(device=model_device, dtype=model_dtype)
            if image.ndim == 5 and not self.config.use_video_encoder:
                image = image[:, -1]
            elif image.ndim == 4 and self.config.use_video_encoder:
                image = image.unsqueeze(1)
            if self.config.resize_imgs_with_padding is not None:
                height, width = self.config.resize_imgs_with_padding
                if image.ndim == 5:
                    batch_size, history, channels, old_height, old_width = image.shape
                    image = _resize_with_pad(
                        image.reshape(batch_size * history, channels, old_height, old_width),
                        height,
                        width,
                    ).reshape(batch_size, history, channels, height, width)
                else:
                    image = _resize_with_pad(image, height, width)
            image = image * 2 - 1
            mask = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)
            processed[key] = (image, mask)
            template, template_mask = image, mask

        if len(missing_keys) > self.config.empty_cameras:
            raise ValueError(
                f"Missing {len(missing_keys)} camera(s), but empty_cameras={self.config.empty_cameras}: "
                f"{missing_keys}"
            )
        assert template is not None and template_mask is not None
        images: list[Tensor] = []
        masks: list[Tensor] = []
        for key in expected_keys:
            if key in processed:
                image, mask = processed[key]
            else:
                # Preserve the configured camera-slot order. Appending all
                # empty cameras at the end would silently move a later real
                # camera into the wrong visual segment.
                image = torch.full_like(template, -1)
                mask = torch.zeros_like(template_mask)
            images.append(image)
            masks.append(mask)
        return images, masks

    def _format_tasks(self, tasks: list[str]) -> list[str]:
        """Apply only model chat formatting; preserve every raw task byte."""

        self._last_raw_tasks = tuple(tasks)
        return [
            task if task.endswith(self.config.task_suffix) else task + self.config.task_suffix
            for task in tasks
        ]

    def prepare_language(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        device = next(
            value.device
            for key, value in batch.items()
            if key.startswith(OBS_IMAGES) and isinstance(value, Tensor)
        )
        raw_tasks = batch.get("task")
        if isinstance(raw_tasks, str):
            raw_tasks = [raw_tasks]
        if not isinstance(raw_tasks, list | tuple) or not all(isinstance(task, str) for task in raw_tasks):
            raise ValueError("Hy-VLA requires an already-selected raw LeRobot task string per sample.")
        tasks = self._format_tasks(list(raw_tasks))
        labels = batch.get("text_label")
        if labels is not None:
            labels = [label + self.language_tokenizer.eos_token for label in labels]
        tokenized = self.language_tokenizer(
            tasks,
            text_pair=labels,
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
            add_special_tokens=False,
            return_token_type_ids=True,
        )
        tokens = tokenized["input_ids"].to(device)
        masks = tokenized["attention_mask"].to(device=device, dtype=torch.bool)
        token_types = tokenized.get("token_type_ids", torch.zeros_like(tokens)).to(device)
        return tokens, masks, token_types

    def prepare_state(self, batch: dict[str, Any]) -> Tensor:
        model_device, model_dtype = self._model_device_dtype()
        return _pad_last(batch[OBS_STATE], self.config.max_state_dim).to(
            device=model_device, dtype=model_dtype
        )

    def prepare_action(self, batch: dict[str, Any]) -> Tensor:
        model_device, model_dtype = self._model_device_dtype()
        return _pad_last(batch[ACTION], self.config.max_action_dim).to(device=model_device, dtype=model_dtype)

    def _prepare_model_inputs(self, batch: dict[str, Any]):
        images, image_masks = self.prepare_images(batch)
        tokens, language_masks, token_types = self.prepare_language(batch)
        return images, image_masks, tokens, language_masks, token_types

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        images, image_masks, tokens, language_masks, token_types = self._prepare_model_inputs(batch)
        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)
        labels = None
        if batch.get("text_label") is not None:
            labels = tokens.masked_fill(
                token_types == self.model.dual_tower.vlm.config.pad_token_id,
                self.model.dual_tower.vlm.config.ignore_index,
            )
        flow_losses, language_losses = self.model(
            images,
            image_masks,
            tokens,
            language_masks,
            state,
            actions,
            noise,
            time,
            labels,
        )
        if flow_losses is None:
            raise RuntimeError("Hy-VLA training requires an action target.")
        flow_losses = flow_losses[..., : self.config.model_action_dim]
        action_mask = batch.get(f"{ACTION}.mask")
        if action_mask is not None:
            if not isinstance(action_mask, Tensor):
                raise ValueError(f"{ACTION}.mask must be a tensor.")
            action_mask = action_mask[..., : self.config.model_action_dim].to(
                device=flow_losses.device, dtype=flow_losses.dtype
            )
            if action_mask.shape != flow_losses.shape:
                action_mask = torch.broadcast_to(action_mask, flow_losses.shape)
            valid_per_sample = action_mask.sum(dim=(-2, -1)).clamp_min(1)
            flow_loss_per_sample = (flow_losses * action_mask).sum(dim=(-2, -1)) / valid_per_sample
        else:
            flow_loss_per_sample = flow_losses.mean(dim=(-2, -1))
        if reduction == "none":
            flow_loss = flow_loss_per_sample
        elif reduction == "mean":
            flow_loss = flow_loss_per_sample.mean()
        else:
            raise ValueError(f"Unsupported reduction {reduction!r}.")
        language_loss = language_losses.mean() if language_losses is not None else flow_loss.new_zeros(())
        loss = flow_loss + language_loss
        return loss, {
            "loss": loss,
            "flow_loss": flow_loss.detach(),
            "language_loss": language_loss.detach(),
        }

    def _pair_relative_absolute(self, actions: Tensor) -> Tensor:
        if self.config.action_representation == "relative":
            return actions
        horizon = self.config.physical_action_horizon
        if actions.shape[-2] != 2 * horizon:
            raise RuntimeError(f"Expected {2 * horizon} rel/abs tokens, got {actions.shape[-2]}.")
        return torch.cat((actions[:, :horizon], actions[:, horizon:]), dim=-1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        if self.config.use_video_encoder:
            self._append_inference_history(batch)
        if not self._action_queue:
            model_batch = self._with_inference_history(batch) if self.config.use_video_encoder else batch
            images, image_masks, tokens, language_masks, _ = self._prepare_model_inputs(model_batch)
            actions = self.model.sample_actions(
                images,
                image_masks,
                tokens,
                language_masks,
                self.prepare_state(model_batch),
                noise=noise,
                vis_attn=False,
            )
            actions = actions[..., : self.config.model_action_dim]
            actions = self._pair_relative_absolute(actions)
            self._action_queue.extend(actions[:, : self.config.execution_horizon].transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Return the complete normalized chunk without mutating the action queue."""

        self.eval()
        images, image_masks, tokens, language_masks, _ = self._prepare_model_inputs(batch)
        actions = self.model.sample_actions(
            images,
            image_masks,
            tokens,
            language_masks,
            self.prepare_state(batch),
            noise=noise,
            vis_attn=False,
        )
        return self._pair_relative_absolute(actions[..., : self.config.model_action_dim])


__all__ = ["HyVLAPolicy"]

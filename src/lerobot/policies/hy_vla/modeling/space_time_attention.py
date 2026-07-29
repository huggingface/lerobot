# Copyright (C) 2026 Tencent.  All rights reserved.
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

# ruff: noqa: E731, E741, N806, SIM108

"""Space-time separable attention for the HYViT2 vision tower.

MEM (Multi-scale Embodied Memory) wraps a vanilla ViT block with a
"space-time separable attention" patch that reduces to identity when
K=1 and adds a single causal time-softmax pass when K>1. The patch adds
zero new parameters (the time embedding is a fixed sinusoidal e(t)
with e(0)=0) and preserves the underlying ViT block's state_dict
layout.

The MEM behaviour is installed by monkey-patching a bundled
``HYViT2_400MAnyRes`` instance at runtime, leaving the model source
untouched::

    apply_video_encoder_patch(
        visual,  # the HYViT2_400MAnyRes wrapper
        spacetime_layer_stride=4,
        past_drop_layer=None,
        max_num_frames=18,
    )

After the patch:

  * ``visual.use_video_encoder = True``
  * ``visual.vision_tower.blocks[stride-1::stride]`` are wrapped in
    ``SpaceTimeBlock`` (in-place, by reference).
  * ``visual(images)`` accepts ``(B, K, C, H, W)`` 5-D tensors and
    routes them through ``forward_video_features``. The list / 4-D
    paths are unchanged.
  * Calling ``apply_video_encoder_patch`` again is a no-op (idempotent).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


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


__all__ = ["SpaceTimeBlock", "apply_video_encoder_patch"]

#!/usr/bin/env python

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

"""SigLIP2 dense patch features (MaskCLIP-style) + text query encoding.

Ported from the dyna360 research stack. Default checkpoint
``google/siglip2-so400m-patch16-384``. For per-patch dense matching
against text, raw ``last_hidden_state`` is the wrong space: SigLIP2's
image-text matching lives in the MAP (Multihead Attention Pooling) head
output. We use the MaskCLIP recipe — apply the MAP head's value
projection + output projection + LayerNorm + MLP residual to each patch
token, skipping the attention reduction — so each patch lands in
(approximately) the shared text/vision space. Outputs are L2-normalized
fp16.

For dry-run and tests, :class:`BasisVectorFeatureExtractor` provides a
deterministic name→vector stand-in with the same interface, no models
required.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Protocol, runtime_checkable

import numpy as np

LOG = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "google/siglip2-so400m-patch16-384"


@runtime_checkable
class FeatureExtractor(Protocol):
    """What the navigation stack needs from a vision-language encoder.

    ``encode_text`` is required (used by ``locate``/``explore`` queries);
    ``feature_dim`` reports the embedding size. Dense image encoding
    (``encode_views``) is only needed by the live mapping pipeline.
    """

    @property
    def feature_dim(self) -> int: ...

    def encode_text(self, text: str) -> np.ndarray: ...


def _select_autocast(device: str) -> tuple[Any, str]:
    """Pick an autocast context + label for the given device."""
    import torch

    if device != "cuda":
        return nullcontext(), "no-autocast"
    if not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
    cap = torch.cuda.get_device_capability()[0]
    dtype = torch.bfloat16 if cap >= 8 else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype), f"cuda/{str(dtype).split('.')[-1]}"


class SiglipFeatureExtractor:
    """Lazy-loaded SigLIP2 wrapper for dense patch features + text query."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str = "cuda",
        max_batch: int = 8,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.max_batch = int(max_batch)
        self._model: Any | None = None
        self._processor: Any | None = None
        self._patch_grid: tuple[int, int] | None = None
        self._feature_dim: int | None = None

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            raise RuntimeError("SigLIP2 not loaded yet; call encode_views first")
        return self._feature_dim

    @property
    def patch_grid(self) -> tuple[int, int]:
        if self._patch_grid is None:
            raise RuntimeError("SigLIP2 not loaded yet; call encode_views first")
        return self._patch_grid

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        LOG.info("loading SigLIP2 (%s) on %s ...", self.checkpoint, self.device)
        self._processor = AutoProcessor.from_pretrained(self.checkpoint)
        self._model = AutoModel.from_pretrained(self.checkpoint).to(self.device).eval()
        LOG.info("SigLIP2 loaded")

    def _maskclip_project(self, patches):
        """Push raw patch tokens through the MAP head with the attention
        reduction removed — value-projects + post-processes each patch so it
        lives in the shared text/vision space. ``patches``: (B, P, D)."""
        import torch

        assert self._model is not None
        head = self._model.vision_model.head
        mha = head.attention  # nn.MultiheadAttention
        embed_dim = patches.shape[-1]

        # in_proj_weight is concatenated [Q | K | V], (3*D, D). Slice out V.
        v_weight = mha.in_proj_weight[2 * embed_dim : 3 * embed_dim]
        v_bias = mha.in_proj_bias[2 * embed_dim : 3 * embed_dim] if mha.in_proj_bias is not None else None
        v = torch.nn.functional.linear(patches, v_weight, v_bias)  # (B, P, D)
        v = mha.out_proj(v)

        residual = v
        v = head.layernorm(v)
        v = residual + head.mlp(v)
        return v

    def encode_views(self, views_rgb_uint8: np.ndarray) -> np.ndarray:
        """Encode ``(N, H, W, 3)`` RGB uint8 views to ``(N, Hp, Wp, D)`` fp16
        dense patch features in the shared text/vision space, L2-normalized."""
        import torch

        if views_rgb_uint8.ndim != 4 or views_rgb_uint8.shape[-1] != 3:  # noqa: N806
            raise ValueError(f"expected (N, H, W, 3), got {views_rgb_uint8.shape}")
        if views_rgb_uint8.dtype != np.uint8:
            raise ValueError(f"expected uint8, got {views_rgb_uint8.dtype}")
        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        autocast_ctx, autocast_label = _select_autocast(self.device)
        LOG.info(
            "SigLIP2 forward (MaskCLIP-projected patches): N=%d (batched up to %d), %s",
            views_rgb_uint8.shape[0],
            self.max_batch,
            autocast_label,
        )

        out_list: list[np.ndarray] = []
        for s in range(0, views_rgb_uint8.shape[0], self.max_batch):
            e = s + self.max_batch
            chunk = [views_rgb_uint8[i] for i in range(s, min(e, views_rgb_uint8.shape[0]))]
            inputs = self._processor(images=chunk, return_tensors="pt").to(self.device)
            with torch.no_grad(), autocast_ctx:
                vision = self._model.vision_model(**inputs)
                patches = vision.last_hidden_state  # (B, P, D)
                patches = self._maskclip_project(patches)  # (B, P, D) shared-space
            patches = torch.nn.functional.normalize(patches.float(), dim=-1)
            out_list.append(patches.to(torch.float16).cpu().numpy())

        feats = np.concatenate(out_list, axis=0)  # (N, P, D)
        n, p, d = feats.shape
        side = int(round(p**0.5))
        if side * side != p:
            raise RuntimeError(
                f"SigLIP2 returned a non-square patch grid (P={p}); non-square inputs aren't supported yet"
            )
        self._patch_grid = (side, side)
        self._feature_dim = d
        return feats.reshape(n, side, side, d)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text query to a single (D,) fp16 unit vector.

        SigLIP2 uses last-token ([EOS]) pooling for text. We extract it
        explicitly because ``get_text_features`` behaves differently across
        ``transformers`` versions.
        """
        import torch

        self._ensure_loaded()
        assert self._model is not None and self._processor is not None
        autocast_ctx, _ = _select_autocast(self.device)
        inputs = self._processor(text=[text], return_tensors="pt", padding="max_length").to(self.device)
        with torch.no_grad(), autocast_ctx:
            text_outputs = self._model.text_model(**inputs)

        pooled = getattr(text_outputs, "pooler_output", None)
        if pooled is not None and pooled.dim() == 2:
            feat = pooled[0]
        else:
            feat = text_outputs.last_hidden_state[0, -1]

        feat = feat.float()
        feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.to(torch.float16).cpu().numpy()


class BasisVectorFeatureExtractor:
    """Deterministic name→vector stand-in for :class:`SiglipFeatureExtractor`.

    Maps known names to their stored feature vectors; unknown queries get a
    deterministic per-text pseudo-random unit vector (same string → same
    vector), so a locate threshold reliably rejects absent objects. Used by
    the synthetic-scene dry-run and by tests — no models required.
    """

    def __init__(self, name_to_vec: dict[str, np.ndarray], feature_dim: int) -> None:
        self.name_to_vec = name_to_vec
        self._feature_dim = int(feature_dim)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def encode_text(self, text: str) -> np.ndarray:
        v = self.name_to_vec.get(text)
        if v is None:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.normal(size=self._feature_dim).astype(np.float32)
        v = v.astype(np.float32)
        v = v / max(float(np.linalg.norm(v)), 1e-6)
        return v

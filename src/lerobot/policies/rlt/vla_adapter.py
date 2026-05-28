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

"""VLA adapters for RLT.

The RLT policy expects VLA-derived observations:
  - ``observation.vla_embeddings``
  - ``observation.reference_action``
  - ``observation.rlt_state`` (optional, compact online replay state)

This module provides a minimal PI0.5 adapter that uses
``PI05Pytorch.embed_prefix`` as the VLA embedding source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

if TYPE_CHECKING:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

OBS_VLA_EMBEDDINGS = "observation.vla_embeddings"
OBS_REFERENCE_ACTION = "observation.reference_action"
OBS_RLT_STATE = "observation.rlt_state"


class PI05PrefixRLTAdapter:
    """Populate RLT observation fields from a frozen PI0.5 policy.

    This adapter is intentionally shallow: it extracts the concatenated image
    and language prefix embeddings before the PaliGemma transformer. That makes
    it a small integration point for the RLT data path.
    """

    def __init__(self, pi05_policy: PI05Policy, rlt_chunk_size: int):
        self.pi05 = pi05_policy.eval()
        self.rlt_chunk_size = rlt_chunk_size

        for param in self.pi05.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return a copy of ``batch`` enriched with RLT VLA fields.

        Args:
            batch: A batch already prepared for PI0.5 inference. It must contain
                image features plus ``observation.language.tokens`` and
                ``observation.language.attention_mask``.

        Returns:
            Batch copy with ``observation.vla_embeddings`` and flattened
            ``observation.reference_action`` fields added.
        """
        batch = dict(batch)

        images, img_masks = self.pi05._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, _, _ = self.pi05.model.embed_prefix(images, img_masks, tokens, masks)
        ref_actions = self.pi05.predict_action_chunk(batch)

        if ref_actions.shape[1] < self.rlt_chunk_size:
            raise ValueError(
                f"PI0.5 reference action chunk has length {ref_actions.shape[1]}, "
                f"but RLT requires {self.rlt_chunk_size}."
            )

        ref_actions = ref_actions[:, : self.rlt_chunk_size]
        ref_actions = ref_actions.reshape(ref_actions.shape[0], -1)

        batch[OBS_VLA_EMBEDDINGS] = prefix_embs
        batch[OBS_REFERENCE_ACTION] = ref_actions
        return batch

    @staticmethod
    def add_reference_action_to_complementary_info(batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Build complementary info expected by the current RLT algorithm.

        ``RLTPolicy.select_action`` reads ``observation.reference_action`` while
        ``RLTAlgorithm`` currently reads ``complementary_info["reference_action"]``.
        This helper bridges that mismatch until the interface is unified.
        """
        if OBS_REFERENCE_ACTION not in batch:
            raise KeyError(f"Missing {OBS_REFERENCE_ACTION}; run PI05PrefixRLTAdapter first.")
        return {"reference_action": batch[OBS_REFERENCE_ACTION]}

    @staticmethod
    def unflatten_action_chunk(action_chunk: Tensor, action_dim: int) -> Tensor:
        """Convert RLT's flattened action chunk to ``(B, C, action_dim)``."""
        if action_chunk.dim() == 1:
            action_chunk = action_chunk.unsqueeze(0)
        return action_chunk.reshape(action_chunk.shape[0], -1, action_dim)

    @staticmethod
    def flatten_action_chunk(action_chunk: Tensor) -> Tensor:
        """Convert ``(B, C, action_dim)`` action chunks to RLT's flat format."""
        if action_chunk.dim() == 1:
            return action_chunk.unsqueeze(0)
        if action_chunk.dim() == 2:
            action_chunk = action_chunk.unsqueeze(0)
        return action_chunk.reshape(action_chunk.shape[0], -1)

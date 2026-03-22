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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, nn

from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE
from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import CONFIG_MAPPING
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
else:
    CONFIG_MAPPING = None
    PaliGemmaForConditionalGeneration = None

from lerobot.policies.pi05_full.modeling_pi05 import get_gemma_config


@dataclass
class RECAPValueNetworkConfig:
    """Configuration for the standalone RECAP value network."""

    paligemma_variant: str = "gemma_300m"
    precision: Literal["bfloat16", "float32"] = "float32"
    image_size: int = 224
    freeze_vision_encoder: bool = False
    num_value_bins: int = 201
    dropout: float = 0.1


class RECAPValueNetwork(nn.Module):
    """
    Standalone distributional value network for RECAP.

    The network predicts a categorical value distribution over bins in [-1, 0]
    and recovers a scalar expected value from this distribution.
    """

    def __init__(self, config: RECAPValueNetworkConfig):
        super().__init__()
        if PaliGemmaForConditionalGeneration is None or CONFIG_MAPPING is None:
            raise ImportError("transformers is required to instantiate RECAPValueNetwork.")

        self.config = config
        gemma_config = get_gemma_config(config.paligemma_variant)

        paligemma_config_hf = CONFIG_MAPPING["paligemma"]()
        paligemma_config_hf._vocab_size = 257152  # noqa: SLF001
        paligemma_config_hf.image_token_index = 257152
        paligemma_config_hf.text_config.hidden_size = gemma_config.width
        paligemma_config_hf.text_config.intermediate_size = gemma_config.mlp_dim
        paligemma_config_hf.text_config.num_attention_heads = gemma_config.num_heads
        paligemma_config_hf.text_config.head_dim = gemma_config.head_dim
        paligemma_config_hf.text_config.num_hidden_layers = gemma_config.depth
        paligemma_config_hf.text_config.num_key_value_heads = gemma_config.num_kv_heads
        paligemma_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        paligemma_config_hf.text_config.torch_dtype = "float32"
        paligemma_config_hf.text_config.vocab_size = 257152
        paligemma_config_hf.vision_config.image_size = config.image_size
        paligemma_config_hf.vision_config.intermediate_size = 4304
        paligemma_config_hf.vision_config.projection_dim = 2048
        paligemma_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        paligemma_config_hf.vision_config.torch_dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGeneration(config=paligemma_config_hf)

        if config.precision == "bfloat16":
            self.paligemma = self.paligemma.to(dtype=torch.bfloat16)
        elif config.precision == "float32":
            self.paligemma = self.paligemma.to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid precision: {config.precision}")

        if config.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for param in self.paligemma.vision_tower.parameters():
                param.requires_grad = False

        self.fusion_head = nn.Sequential(
            nn.Linear(gemma_config.width, gemma_config.width),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(gemma_config.width, gemma_config.width),
            nn.SiLU(),
        )
        self.value_head = nn.Linear(gemma_config.width, config.num_value_bins)

        self.register_buffer(
            "value_bin_support",
            torch.linspace(-1.0, 0.0, config.num_value_bins, dtype=torch.float32),
            persistent=True,
        )

    def _prepare_attention_masks_4d(self, att_2d_masks: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        result = torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        if dtype is not None:
            result = result.to(dtype=dtype)
        return result

    def _build_prefix_embeddings(self, images: Tensor, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor, int]:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape [B, N_cam, 3, H, W] or [B, 3, H, W], got {tuple(images.shape)}")

        batch_size, n_cams = images.shape[:2]
        prefix_embs = []
        prefix_masks = []
        image_token_len = 0

        for cam_idx in range(n_cams):
            cam_images = images[:, cam_idx]
            img_emb = self.paligemma.model.get_image_features(cam_images)
            prefix_embs.append(img_emb)
            cam_mask = torch.ones(
                batch_size,
                img_emb.shape[1],
                dtype=torch.bool,
                device=img_emb.device,
            )
            prefix_masks.append(cam_mask)
            image_token_len += img_emb.shape[1]

        lang_emb = self.paligemma.language_model.embed_tokens(input_ids)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        prefix_embs.append(lang_emb)
        prefix_masks.append(attention_mask.bool())

        full_embs = torch.cat(prefix_embs, dim=1)
        full_mask = torch.cat(prefix_masks, dim=1)
        return full_embs, full_mask, image_token_len

    def forward(self, images: Tensor, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        """
        Args:
            images: [B, N_cam, 3, H, W] or [B, 3, H, W]
            input_ids: [B, T]
            attention_mask: [B, T]
        Returns:
            Dictionary with value logits, probabilities and expected value.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids with shape [B, T], got {tuple(input_ids.shape)}")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

        input_ids = input_ids.long()
        attention_mask = attention_mask.bool()

        prefix_embs, prefix_pad_mask, image_token_len = self._build_prefix_embeddings(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        att_2d_masks = prefix_pad_mask[:, None, :] * prefix_pad_mask[:, :, None]
        att_4d_masks = self._prepare_attention_masks_4d(att_2d_masks, dtype=prefix_embs.dtype)
        position_ids = torch.cumsum(prefix_pad_mask, dim=1) - 1

        prefix_output = self.paligemma.language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=att_4d_masks,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state

        lang_hidden = prefix_output[:, image_token_len:, :]
        text_mask = attention_mask.unsqueeze(-1).to(lang_hidden.dtype)
        text_denom = text_mask.sum(dim=1).clamp(min=1.0)
        text_feat = (lang_hidden * text_mask).sum(dim=1) / text_denom

        fused = self.fusion_head(text_feat)
        value_logits = self.value_head(fused.float())

        value_probs = torch.softmax(value_logits.float(), dim=-1)
        expected_value = (value_probs * self.value_bin_support).sum(dim=-1, keepdim=True)

        return {
            "value_logits": value_logits,
            "value_probs": value_probs,
            "expected_value": expected_value,
        }

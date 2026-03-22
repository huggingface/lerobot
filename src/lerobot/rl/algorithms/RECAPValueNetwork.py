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
from importlib import import_module
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

def _load_get_gemma_config():
    try:
        module = import_module("lerobot.policies.pi05.modeling_pi05")
    except ModuleNotFoundError:
        module = import_module("lerobot.policies.pi05_full.modeling_pi05")
    return module.get_gemma_config


get_gemma_config = _load_get_gemma_config()


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
        paligemma_config_hf.vision_config.projection_dim = gemma_config.width
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
            vision_tower = self._get_vision_tower()
            vision_tower.eval()
            for param in vision_tower.parameters():
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

    def _get_paligemma_backbone(self):
        if hasattr(self.paligemma, "model"):
            return self.paligemma.model
        return self.paligemma

    def _get_language_model(self):
        backbone = self._get_paligemma_backbone()
        if hasattr(backbone, "language_model"):
            return backbone.language_model
        if hasattr(self.paligemma, "language_model"):
            return self.paligemma.language_model
        raise AttributeError("Unable to locate PaliGemma language model module.")

    def _get_vision_tower(self):
        backbone = self._get_paligemma_backbone()
        if hasattr(backbone, "vision_tower"):
            return backbone.vision_tower
        if hasattr(self.paligemma, "vision_tower"):
            return self.paligemma.vision_tower
        raise AttributeError("Unable to locate PaliGemma vision tower module.")

    def _extract_image_embeddings(self, image_features_output) -> Tensor:
        """Normalize image-feature outputs across transformers versions."""
        if isinstance(image_features_output, Tensor):
            return image_features_output

        projected = getattr(image_features_output, "pooler_output", None)
        if isinstance(projected, Tensor):
            return projected

        fallback = getattr(image_features_output, "last_hidden_state", None)
        if isinstance(fallback, Tensor):
            return fallback

        raise TypeError(
            "Unsupported image feature output type from PaliGemma get_image_features: "
            f"{type(image_features_output)}"
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
        flat_images = images.reshape(batch_size * n_cams, *images.shape[2:])
        image_features_output = self._get_paligemma_backbone().get_image_features(
            flat_images,
            return_dict=True,
        )
        flat_img_emb = self._extract_image_embeddings(image_features_output)
        if flat_img_emb.ndim == 2:
            flat_img_emb = flat_img_emb.unsqueeze(1)
        if flat_img_emb.ndim != 3:
            raise ValueError(
                "Expected image embeddings with shape [B*N_cam, T_img, D] "
                f"or [B*N_cam, D], got {tuple(flat_img_emb.shape)}"
            )

        img_token_len = flat_img_emb.shape[1]
        img_emb = flat_img_emb.reshape(batch_size, n_cams * img_token_len, flat_img_emb.shape[-1])
        image_token_len = img_emb.shape[1]
        image_mask = torch.ones(
            batch_size,
            image_token_len,
            dtype=torch.bool,
            device=img_emb.device,
        )

        lang_emb = self.paligemma.get_input_embeddings()(input_ids)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        text_mask = attention_mask.bool()

        full_embs = torch.cat((img_emb, lang_emb), dim=1)
        full_mask = torch.cat((image_mask, text_mask), dim=1)
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

        att_2d_masks = prefix_pad_mask[:, None, :] & prefix_pad_mask[:, :, None]
        att_4d_masks = self._prepare_attention_masks_4d(att_2d_masks, dtype=prefix_embs.dtype)
        position_ids = torch.cumsum(prefix_pad_mask, dim=1) - 1

        prefix_output = self._get_language_model().forward(
            inputs_embeds=prefix_embs,
            attention_mask=att_4d_masks,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state

        lang_hidden = prefix_output[:, image_token_len:, :]
        text_mask = attention_mask.unsqueeze(-1).to(lang_hidden.dtype)
        text_denom = text_mask.sum(dim=1).clamp(min=1.0)
        text_feat = (lang_hidden * text_mask).sum(dim=1) / text_denom

        fusion_dtype = self.fusion_head[0].weight.dtype
        value_head_dtype = self.value_head.weight.dtype
        fused = self.fusion_head(text_feat.to(dtype=fusion_dtype))
        value_logits = self.value_head(fused.to(dtype=value_head_dtype))

        value_probs = torch.softmax(value_logits.float(), dim=-1)
        expected_value = (value_probs * self.value_bin_support).sum(dim=-1, keepdim=True)

        return {
            "value_logits": value_logits,
            "value_probs": value_probs,
            "expected_value": expected_value,
        }

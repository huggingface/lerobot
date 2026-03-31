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

import logging
import math
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.rl.algorithms.recap_utils import load_pretrained_vlm_weights
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import _transformers_available

if _transformers_available:
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
    tokenizer_name: str = "google/paligemma-3b-pt-224"
    hidden_dim: int = 768
    num_value_bins: int = 50
    v_min: float = -1.0
    v_max: float = 0.0
    freeze_vision_encoder: bool = False
    freeze_backbone: bool = False
    num_unfrozen_backbone_layers: int = 0
    num_vlm_layers: int = 18
    value_head_depth: int = 1
    dropout: float = 0.1
    pretrained_path: str | None = None


class RECAPValueNetwork(nn.Module):
    value_bin_support: Tensor

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

        self.paligemma = PaliGemmaForConditionalGeneration(config=paligemma_config_hf)  # ty: ignore[invalid-argument-type]

        if config.precision == "bfloat16":
            self.paligemma = self.paligemma.to(dtype=torch.bfloat16)  # ty: ignore[missing-argument]
        elif config.precision == "float32":
            self.paligemma = self.paligemma.to(dtype=torch.float32)  # ty: ignore[missing-argument]
        else:
            raise ValueError(f"Invalid precision: {config.precision}")

        lm_inner = self.paligemma.model.language_model
        if hasattr(lm_inner, "model"):
            lm_inner = lm_inner.model
        if config.num_vlm_layers > 0:
            total_layers = len(lm_inner.layers)
            if config.num_vlm_layers > total_layers:
                raise ValueError(
                    f"num_vlm_layers={config.num_vlm_layers} exceeds model depth {total_layers}"
                )
            lm_inner.layers = lm_inner.layers[: config.num_vlm_layers]
            logging.info(f"Using first {len(lm_inner.layers)} PaliGemma text layers for value network")

        if config.freeze_backbone:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False
            if config.num_unfrozen_backbone_layers > 0:
                num_layers = len(lm_inner.layers)
                if config.num_unfrozen_backbone_layers > num_layers:
                    raise ValueError(
                        f"num_unfrozen_backbone_layers={config.num_unfrozen_backbone_layers} "
                        f"exceeds available layers {num_layers}"
                    )
                unfrozen_layers = lm_inner.layers[-config.num_unfrozen_backbone_layers:]
                for layer in unfrozen_layers:
                    layer.train()
                    for param in layer.parameters():
                        param.requires_grad = True
                logging.info(
                    f"Unfreezing last {config.num_unfrozen_backbone_layers}/{num_layers} "
                    f"backbone transformer layers"
                )
        elif config.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False

        self.register_buffer(
            "value_bin_support",
            torch.linspace(config.v_min, config.v_max, config.num_value_bins, dtype=torch.float32),
            persistent=True,
        )

        value_head_layers: list[nn.Module] = []
        for i in range(config.value_head_depth):
            value_head_layers.append(nn.Linear(gemma_config.width if i == 0 else config.hidden_dim, config.hidden_dim))
            value_head_layers.append(nn.GELU())
        value_head_layers.append(nn.Linear(config.hidden_dim, config.num_value_bins))
        self.value_head = nn.Sequential(*value_head_layers)

        if config.pretrained_path:
            load_pretrained_vlm_weights(self, config.pretrained_path)

    def forward(self, batch: dict[str, Any], images: Tensor) -> dict[str, Tensor]:
        """Forward pass returning logits over value bins.

        Args:
            batch: Preprocessed batch with language tokens and attention mask.
            images: Camera images of shape ``[B, n_cams, C, H, W]``.

        Returns:
            Dictionary with value_logits, value_probs and expected_value.
        """
        device = next(self.parameters()).device
        batch_size, n_cams = images.shape[:2]

        # Paligemma SigLIP image encoder
        flat_images = images.reshape(batch_size * n_cams, *images.shape[2:]).to(device)
        image_outputs = self.paligemma.model.get_image_features(flat_images)
        flat_img_emb = image_outputs.pooler_output
        if flat_img_emb.ndim == 2:
            flat_img_emb = flat_img_emb.unsqueeze(1)
        img_token_len = flat_img_emb.shape[1]
        img_emb = flat_img_emb.reshape(batch_size, n_cams * img_token_len, flat_img_emb.shape[-1])
        img_mask = torch.ones(batch_size, img_emb.shape[1], dtype=torch.bool, device=device)

        # Language instruction embedding
        input_ids = batch[OBS_LANGUAGE_TOKENS].to(device)
        attention_mask = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device)
        lang_emb = self.paligemma.model.language_model.embed_tokens(input_ids)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        text_mask = attention_mask.bool()

        # Concat image + language embeddings
        full_embs = torch.cat((img_emb, lang_emb), dim=1)
        full_mask = torch.cat((img_mask, text_mask), dim=1)

        position_ids = torch.cumsum(full_mask, dim=1) - 1
        position_ids = position_ids.masked_fill(~full_mask, 0).long()

        # Forward pass. The property is called language_model but it is in fact a vlm
        vlm = self.paligemma.model.language_model
        text_dtype = next(vlm.parameters()).dtype
        text_model_inputs = {
            "inputs_embeds": full_embs.to(dtype=text_dtype),
            "attention_mask": full_mask,
            "use_cache": False,
        }
        try:
            text_model_inputs["position_ids"] = position_ids
            hidden_states = vlm.forward(**text_model_inputs).last_hidden_state
        except TypeError:
            text_model_inputs.pop("position_ids", None)
            hidden_states = vlm.forward(**text_model_inputs).last_hidden_state

        seq_lengths = full_mask.sum(dim=1) - 1
        last_token_hidden_state = hidden_states[torch.arange(batch_size, device=device), seq_lengths.long()]

        # Feed last hidden state into the value head
        value_logits = self.value_head(last_token_hidden_state.float())
        value_probs = torch.softmax(value_logits, dim=-1)
        expected_value = (value_probs * self.value_bin_support).sum(dim=-1, keepdim=True)

        return {
            "value_logits": value_logits,
            "value_probs": value_probs,
            "expected_value": expected_value,
        }

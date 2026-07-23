"""Distributional value head on the pretrained nanoVLM-460M checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.configs.types import FeatureType
from lerobot.rewards.distributional_value_function.common import DistributionalValueMixin
from lerobot.rewards.distributional_value_function.modeling_distributional_value_function import ValueHead
from lerobot.rewards.distributional_value_function.processor_distributional_value_function import (
    IMAGE_MASK_SUFFIX,
)
from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .configuration_nanovlm_value_function import NanoVLMVFConfig


class NanoVLMVFRewardModel(DistributionalValueMixin, PreTrainedRewardModel):
    """Use nanoVLM's aligned decoder readout for RECAP return classification."""

    name = "nanovlm_value_function"
    config_class = NanoVLMVFConfig

    def __init__(self, config: NanoVLMVFConfig, **kwargs):
        super().__init__(config)
        self.config = config
        config.validate_features()
        self.image_keys = [
            key for key, feature in config.input_features.items() if feature.type == FeatureType.VISUAL
        ]
        code_path = Path(config.nanovlm_code_path)
        if not code_path.is_absolute():
            code_path = Path(__file__).resolve().parents[4] / code_path
        if not code_path.exists():
            raise FileNotFoundError(f"nanoVLM code not found at {code_path}")
        if str(code_path) not in sys.path:
            sys.path.insert(0, str(code_path))
        from models.vision_language_model import VisionLanguageModel

        self.nanovlm = VisionLanguageModel.from_pretrained(config.nanovlm_pretrained_path)
        hidden_size = self.nanovlm.cfg.lm_hidden_dim
        self.value_query = nn.Embedding(1, hidden_size)
        nn.init.normal_(self.value_query.weight, std=0.02)
        self.value_head = ValueHead(
            hidden_size,
            config.num_value_bins,
            config.value_support_min,
            config.value_support_max,
            config.value_dropout,
        )
        bin_width = (config.value_support_max - config.value_support_min) / (config.num_value_bins - 1)
        self.hl_gauss_sigma = config.hl_gauss_sigma_ratio * bin_width
        self._set_requires_grad()

    def _set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            self.nanovlm.vision_encoder.requires_grad_(False).eval()
        if self.config.freeze_multimodal_projector:
            self.nanovlm.MP.requires_grad_(False).eval()
        if self.config.freeze_language_model:
            self.nanovlm.decoder.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.nanovlm.vision_encoder.eval()
        if self.config.freeze_multimodal_projector:
            self.nanovlm.MP.eval()
        if self.config.freeze_language_model:
            self.nanovlm.decoder.eval()
        return self

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        return self._distributional_forward(batch)

    def _get_value_readout(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size = batch[OBS_LANGUAGE_TOKENS].shape[0]
        image_tokens = []
        image_masks = []
        for key in self.image_keys:
            image = batch[key]
            mask = batch[key + IMAGE_MASK_SUFFIX].bool()
            features = self.nanovlm.MP(self.nanovlm.vision_encoder(image))
            image_tokens.append(features * mask[:, None, None].to(features.dtype))
            image_masks.append(mask[:, None].expand(batch_size, features.shape[1]))

        text_tokens = self.nanovlm.decoder.token_embedding(batch[OBS_LANGUAGE_TOKENS])
        query = self.value_query(torch.zeros(batch_size, 1, dtype=torch.long, device=text_tokens.device)).to(
            text_tokens.dtype
        )
        inputs = torch.cat([*image_tokens, text_tokens, query], dim=1)
        attention_mask = torch.cat(
            [
                *image_masks,
                batch[OBS_LANGUAGE_ATTENTION_MASK].bool(),
                torch.ones(batch_size, 1, dtype=torch.bool, device=text_tokens.device),
            ],
            dim=1,
        )
        hidden, _ = self.nanovlm.decoder(inputs, attention_mask=attention_mask)
        return hidden[:, -1]

    def get_optim_params(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

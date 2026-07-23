"""Distributional value head on the pretrained nanoVLM-460M checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.rewards.distributional_value_function.common import DistributionalValueMixin
from lerobot.rewards.distributional_value_function.modeling_distributional_value_function import ValueHead
from lerobot.rewards.nanovlm_value_function.processor_nanovlm_value_function import (
    NANOVLM_ATTENTION_MASK,
    NANOVLM_IMAGES,
    NANOVLM_INPUT_IDS,
)
from lerobot.rewards.pretrained import PreTrainedRewardModel

from .configuration_nanovlm_value_function import NanoVLMVFConfig


class NanoVLMVFRewardModel(DistributionalValueMixin, PreTrainedRewardModel):
    """Use nanoVLM's aligned decoder readout for RECAP return classification."""

    name = "nanovlm_value_function"
    config_class = NanoVLMVFConfig

    def __init__(self, config: NanoVLMVFConfig, **kwargs):
        super().__init__(config)
        self.config = config
        config.validate_features()
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
        input_ids = batch[NANOVLM_INPUT_IDS]
        attention_mask = batch[NANOVLM_ATTENTION_MASK].bool()
        batch_size = input_ids.shape[0]
        images = self.nanovlm._process_images(batch[NANOVLM_IMAGES], input_ids.device)
        text_tokens = self.nanovlm.decoder.token_embedding(input_ids)
        if images is not None:
            image_tokens = self.nanovlm.MP(self.nanovlm.vision_encoder(images))
            placeholder_count = (input_ids == self.nanovlm.tokenizer.image_token_id).sum().item()
            image_token_count = image_tokens.shape[0] * image_tokens.shape[1]
            if placeholder_count != image_token_count:
                raise ValueError(
                    "nanoVLM image placeholders do not match projected image tokens: "
                    f"{placeholder_count} placeholders versus {image_token_count} tokens. "
                    "The prompt may have been truncated; increase tokenizer_max_length."
                )
            text_tokens = self.nanovlm._replace_img_tokens_with_embd(
                input_ids,
                text_tokens,
                image_tokens,
            )
        query = self.value_query(torch.zeros(batch_size, 1, dtype=torch.long, device=text_tokens.device)).to(
            text_tokens.dtype
        )
        inputs = torch.cat([text_tokens, query], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(batch_size, 1, dtype=torch.bool, device=text_tokens.device),
            ],
            dim=1,
        )
        hidden, _ = self.nanovlm.decoder(inputs, attention_mask=attention_mask)
        return hidden[:, -1]

    def get_optim_params(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

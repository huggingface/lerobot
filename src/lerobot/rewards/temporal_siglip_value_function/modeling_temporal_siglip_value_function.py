"""Past-only temporal SigLIP2 distributional value function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs.types import FeatureType
from lerobot.rewards.distributional_value_function.common import DistributionalValueMixin
from lerobot.rewards.distributional_value_function.modeling_distributional_value_function import ValueHead
from lerobot.rewards.distributional_value_function.processor_distributional_value_function import (
    IMAGE_MASK_SUFFIX,
)
from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_temporal_siglip_value_function import TemporalSiglipVFConfig

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModel
else:
    AutoModel = None  # type: ignore[assignment]


class TemporalSiglipVFRewardModel(DistributionalValueMixin, PreTrainedRewardModel):
    """Fuse three-camera history, task, and state before causal temporal attention."""

    name = "temporal_siglip_value_function"
    config_class = TemporalSiglipVFConfig

    def __init__(self, config: TemporalSiglipVFConfig, **kwargs):
        require_package("transformers", extra="recap")
        super().__init__(config)
        self.config = config
        config.validate_features()
        self.image_keys = [
            key for key, feature in config.input_features.items() if feature.type == FeatureType.VISUAL
        ]
        self.siglip = AutoModel.from_pretrained(config.siglip_path)
        self.siglip.requires_grad_(False).eval()

        vision_dim = self.siglip.config.vision_config.hidden_size
        text_dim = self.siglip.config.text_config.hidden_size
        hidden_size = config.hidden_size
        self.camera_proj = nn.Linear(vision_dim, hidden_size)
        self.camera_embedding = nn.Embedding(len(self.image_keys), hidden_size)
        self.task_proj = nn.Linear(text_dim, hidden_size)
        self.state_proj = nn.Linear(config.state_dim, hidden_size)
        self.frame_fusion = nn.Sequential(
            nn.LayerNorm((len(self.image_keys) + 2) * hidden_size),
            nn.Linear((len(self.image_keys) + 2) * hidden_size, hidden_size),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=config.num_heads,
            dim_feedforward=4 * hidden_size,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(hidden_size),
        )
        self.time_embedding = nn.Embedding(config.history_steps, hidden_size)
        self.value_head = ValueHead(
            hidden_size,
            config.num_value_bins,
            config.value_support_min,
            config.value_support_max,
            config.dropout,
        )
        bin_width = (config.value_support_max - config.value_support_min) / (config.num_value_bins - 1)
        self.hl_gauss_sigma = config.hl_gauss_sigma_ratio * bin_width

    def train(self, mode: bool = True):
        super().train(mode)
        self.siglip.eval()
        return self

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        return self._distributional_forward(batch)

    def _get_value_readout(self, batch: dict[str, Tensor]) -> Tensor:
        images = [batch[key] for key in self.image_keys]
        masks = [batch[key + IMAGE_MASK_SUFFIX].bool() for key in self.image_keys]
        state = batch[self.config.state_key]
        if state.ndim == 2:
            state = state[:, None]
        batch_size, history_steps = images[0].shape[:2]
        if history_steps != self.config.history_steps:
            raise ValueError(f"Expected {self.config.history_steps} frames, got {history_steps}")

        camera_tokens = []
        vision_dtype = next(self.siglip.vision_model.parameters()).dtype
        for camera_index, (image, mask) in enumerate(zip(images, masks, strict=True)):
            with torch.no_grad():
                features = self.siglip.vision_model(
                    pixel_values=image.flatten(0, 1).to(vision_dtype),
                    interpolate_pos_encoding=True,
                    return_dict=True,
                ).pooler_output
            features = self.camera_proj(features).unflatten(0, (batch_size, history_steps))
            camera_ids = torch.full(
                (batch_size, history_steps),
                camera_index,
                dtype=torch.long,
                device=features.device,
            )
            camera_tokens.append(
                (features + self.camera_embedding(camera_ids)) * mask[..., None].to(features.dtype)
            )

        with torch.no_grad():
            task_features = self.siglip.text_model(
                input_ids=batch[OBS_LANGUAGE_TOKENS],
                attention_mask=batch[OBS_LANGUAGE_ATTENTION_MASK],
                return_dict=True,
            ).pooler_output
        task_token = self.task_proj(task_features)[:, None].expand(-1, history_steps, -1)
        state = self._fit_state_dim(state).to(task_token.dtype)
        state_token = self.state_proj(state)
        frame_tokens = self.frame_fusion(torch.cat([*camera_tokens, task_token, state_token], -1))
        frame_tokens = (
            frame_tokens + self.time_embedding(torch.arange(history_steps, device=frame_tokens.device))[None]
        )

        causal_mask = torch.triu(
            torch.ones(history_steps, history_steps, dtype=torch.bool, device=frame_tokens.device),
            diagonal=1,
        )
        frame_valid = torch.stack(masks).any(0)
        hidden = self.temporal_transformer(
            frame_tokens,
            mask=causal_mask,
            src_key_padding_mask=~frame_valid,
            is_causal=True,
        )
        last_valid = frame_valid.long().sum(-1).sub(1).clamp_min(0)
        return hidden[torch.arange(batch_size, device=hidden.device), last_valid]

    def _fit_state_dim(self, state: Tensor) -> Tensor:
        if state.shape[-1] > self.config.state_dim:
            return state[..., : self.config.state_dim]
        return F.pad(state, (0, self.config.state_dim - state.shape[-1]))

    def get_optim_params(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

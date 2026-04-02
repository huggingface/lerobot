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

from typing import Protocol

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.rewards.pretrained import PreTrainedRewardModel
from lerobot.rewards.vita.adaptation import VitaAdaptationModule, VitaAdaptationState
from lerobot.rewards.vita.configuration_vita import VitaConfig
from lerobot.utils.constants import OBS_STR


class VitaBackbone(Protocol):
    """Backbone interface returning image and text embeddings."""

    def encode(self, batch: dict[str, Tensor], config: VitaConfig) -> tuple[Tensor, Tensor]:
        """Encode batch inputs into image and text features."""
        ...


class IdentityVitaBackbone(nn.Module):
    """Lightweight backbone for pre-encoded image/text features."""

    def encode(self, batch: dict[str, Tensor], config: VitaConfig) -> tuple[Tensor, Tensor]:
        image_features = batch[config.image_feature_key]
        text_features = batch[config.text_feature_key]
        return image_features.float(), text_features.float()


class DummyVitaBackbone(nn.Module):
    """Deterministic test backbone with fixed linear encoders."""

    def __init__(self, image_input_dim: int, text_input_dim: int, image_output_dim: int, text_output_dim: int):
        super().__init__()
        self.image_encoder = nn.Linear(image_input_dim, image_output_dim, bias=False)
        self.text_encoder = nn.Linear(text_input_dim, text_output_dim, bias=False)
        with torch.no_grad():
            self.image_encoder.weight.copy_(torch.eye(image_output_dim, image_input_dim))
            self.text_encoder.weight.copy_(torch.eye(text_output_dim, text_input_dim))

    def encode(self, batch: dict[str, Tensor], config: VitaConfig) -> tuple[Tensor, Tensor]:
        image = batch[config.image_feature_key].float()
        text = batch[config.text_feature_key].float()
        return self.image_encoder(image), self.text_encoder(text)


class VitaRewardModel(PreTrainedRewardModel):
    """VITA reward model with sequential test-time adaptation."""

    name = "vita"
    config_class = VitaConfig

    def __init__(self, config: VitaConfig, backbone: VitaBackbone | None = None):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.backbone = backbone if backbone is not None else IdentityVitaBackbone()

        if isinstance(self.backbone, nn.Module):
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.key_projection = nn.Linear(config.latent_dim, config.adaptation_dim)
        self.value_projection = nn.Linear(config.latent_dim, config.adaptation_dim)
        self.query_projection = nn.Linear(config.latent_dim, config.adaptation_dim)
        self.adaptation_module = VitaAdaptationModule(config.adaptation_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(config.adaptation_dim, config.reward_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.reward_hidden_dim, 1),
        )

        self._adaptation_state: VitaAdaptationState | None = None
        self.last_adaptation_losses: Tensor | None = None

    @property
    def adaptation_state(self) -> VitaAdaptationState | None:
        return self._adaptation_state

    def reset_adaptation_state(self, batch_size: int | None = None, device: torch.device | None = None) -> None:
        if batch_size is None:
            self._adaptation_state = None
            return
        state_device = device if device is not None else self.adaptation_module.base_weight.device
        self._adaptation_state = VitaAdaptationState.initialize(
            base_weight=self.adaptation_module.base_weight,
            batch_size=batch_size,
            device=state_device,
        )

    def reset(self) -> None:
        self.reset_adaptation_state()

    def _ensure_sequence(self, tensor: Tensor) -> Tensor:
        if tensor.ndim == 2:
            return tensor.unsqueeze(1)
        if tensor.ndim == 3:
            return tensor
        raise ValueError(f"Expected tensor rank 2 or 3, got rank {tensor.ndim}")

    def _prepare_episode_starts(self, episode_start: Tensor | None, batch_size: int, seq_len: int, device) -> Tensor:
        if episode_start is None:
            return torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        starts = episode_start.to(device=device, dtype=torch.bool)
        if starts.ndim == 1:
            if starts.shape[0] != batch_size:
                raise ValueError("episode_start must match batch size.")
            starts = starts.unsqueeze(1).expand(batch_size, seq_len)
        elif starts.ndim == 2:
            if starts.shape != (batch_size, seq_len):
                raise ValueError("episode_start must have shape (B, T).")
        else:
            raise ValueError("episode_start must be rank 1 or 2.")
        return starts

    def _split_support_query(self, tensor: Tensor) -> tuple[Tensor, Tensor]:
        support_len = self.config.support_len
        query_len = self.config.query_len
        total_len = support_len + query_len
        if tensor.shape[1] < total_len:
            raise ValueError(
                f"Expected at least {total_len} timesteps for support/query split, got {tensor.shape[1]}."
            )
        support = tensor[:, :support_len]
        query = tensor[:, support_len : support_len + query_len]
        return support, query

    def _adapt_fast_weights(
        self,
        image_sequence: Tensor,
        text_sequence: Tensor,
        fast_weights: Tensor,
        adaptation_lr: float,
    ) -> tuple[Tensor, Tensor]:
        adaptation_losses: list[Tensor] = []
        for timestep in range(image_sequence.shape[1]):
            latent = torch.cat([image_sequence[:, timestep], text_sequence[:, timestep]], dim=-1)
            keys = self.key_projection(latent)
            values = self.value_projection(latent)

            predictions = self.adaptation_module(keys, fast_weights=fast_weights)
            errors = predictions - values
            adaptation_dim = keys.shape[-1]
            grad = (2.0 / adaptation_dim) * torch.einsum("bi,bj->bij", errors, keys)
            if self.config.first_order:
                grad = grad.detach()
            fast_weights = fast_weights - adaptation_lr * grad
            adaptation_losses.append(torch.mean((predictions - values) ** 2, dim=-1))

        return fast_weights, torch.stack(adaptation_losses, dim=1)

    def _predict_reward_sequence(self, image_sequence: Tensor, text_sequence: Tensor, fast_weights: Tensor) -> Tensor:
        rewards: list[Tensor] = []
        for timestep in range(image_sequence.shape[1]):
            latent = torch.cat([image_sequence[:, timestep], text_sequence[:, timestep]], dim=-1)
            queries = self.query_projection(latent)
            adapted_queries = self.adaptation_module(queries, fast_weights=fast_weights)
            rewards.append(self.reward_head(adapted_queries).squeeze(-1))
        return torch.stack(rewards, dim=1)

    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        observation = batch.get(OBS_STR, batch)
        image_features, text_features = self.backbone.encode(observation, self.config)
        image_features = self._ensure_sequence(image_features)
        text_features = self._ensure_sequence(text_features)
        if image_features.shape[:2] != text_features.shape[:2]:
            raise ValueError("Image and text features must have compatible (B, T) dimensions.")

        batch_size, seq_len, _ = image_features.shape
        device = image_features.device

        if self._adaptation_state is None or self._adaptation_state.fast_weights.shape[0] != batch_size:
            self.reset_adaptation_state(batch_size=batch_size, device=device)
        assert self._adaptation_state is not None

        episode_starts = self._prepare_episode_starts(
            observation.get("episode_start"), batch_size=batch_size, seq_len=seq_len, device=device
        )

        rewards: list[Tensor] = []
        adaptation_losses: list[Tensor] = []
        for timestep in range(seq_len):
            step_starts = episode_starts[:, timestep]
            self._adaptation_state.reset_indices(base_weight=self.adaptation_module.base_weight, mask=step_starts)

            latent = torch.cat([image_features[:, timestep], text_features[:, timestep]], dim=-1)
            keys = self.key_projection(latent)
            values = self.value_projection(latent)
            queries = self.query_projection(latent)

            updated_fast_weights, losses = self.adaptation_module.adaptation_step(
                keys=keys,
                values=values,
                fast_weights=self._adaptation_state.fast_weights,
                adaptation_lr=self.config.adaptation_lr,
            )
            self._adaptation_state.fast_weights = updated_fast_weights
            self._adaptation_state.num_updates = self._adaptation_state.num_updates + 1
            adaptation_losses.append(losses)

            adapted_queries = self.adaptation_module(queries, fast_weights=self._adaptation_state.fast_weights)
            rewards.append(self.reward_head(adapted_queries).squeeze(-1))

        reward_tensor = torch.stack(rewards, dim=1)
        self.last_adaptation_losses = torch.stack(adaptation_losses, dim=1)
        if seq_len == 1:
            return reward_tensor[:, 0]
        return reward_tensor

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        if not self.config.meta_enabled:
            raise NotImplementedError("Meta-learning is disabled in config; use compute_reward() for inference.")

        observation = batch.get(OBS_STR, batch)
        image_features, text_features = self.backbone.encode(observation, self.config)
        image_features = self._ensure_sequence(image_features)
        text_features = self._ensure_sequence(text_features)

        target = observation.get(self.config.target_reward_key)
        if target is None:
            raise ValueError(f"Missing target reward key '{self.config.target_reward_key}' in batch.")
        target = target.float()
        if target.ndim == 1:
            target = target.unsqueeze(1)
        if target.ndim == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        if target.ndim != 2:
            raise ValueError(f"Target reward must be (B,T) or (B,T,1), got shape {tuple(target.shape)}.")

        support_image, query_image = self._split_support_query(image_features)
        support_text, query_text = self._split_support_query(text_features)
        support_target, query_target = self._split_support_query(target)
        del support_target  # Targets for support are not required by the adaptation loss.

        batch_size = image_features.shape[0]
        device = image_features.device
        state = VitaAdaptationState.initialize(
            base_weight=self.adaptation_module.base_weight,
            batch_size=batch_size,
            device=device,
        )
        fast_weights = state.fast_weights

        inner_losses: list[Tensor] = []
        for _ in range(self.config.inner_steps):
            fast_weights, adaptation_losses = self._adapt_fast_weights(
                image_sequence=support_image,
                text_sequence=support_text,
                fast_weights=fast_weights,
                adaptation_lr=self.config.inner_lr,
            )
            inner_losses.append(adaptation_losses.mean())

        query_predictions = self._predict_reward_sequence(
            image_sequence=query_image,
            text_sequence=query_text,
            fast_weights=fast_weights,
        )
        outer_loss = F.mse_loss(query_predictions, query_target)
        loss = self.config.outer_loss_weight * outer_loss

        metrics = {
            "loss": loss.detach(),
            "outer_loss": outer_loss.detach(),
            "inner_loss": torch.stack(inner_losses).mean().detach(),
        }
        return loss, metrics

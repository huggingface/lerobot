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

"""Conditional Flow Matching for action-chunk generation.

The objective follows "Flow Matching for Generative Modeling"
(https://huggingface.co/papers/2210.02747). It regresses the velocity of a
straight probability path from Gaussian noise at ``t=0`` to normalized action
chunks at ``t=1`` and samples actions with Euler integration.
"""

import math
from collections import deque
from typing import TYPE_CHECKING, Literal, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import (
    ACTION,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_flow_matching import FlowMatchingConfig

if TYPE_CHECKING or _transformers_available:
    from transformers import CLIPTextModel
else:
    CLIPTextModel = None


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for continuous integration time."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        half_dim = self.dim // 2
        exponent = (
            -math.log(10_000) * torch.arange(half_dim, device=time.device, dtype=time.dtype) / (half_dim - 1)
        )
        frequencies = exponent.exp()
        angles = time.unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class CLIPTaskEncoder(nn.Module):
    """Encode a natural-language task into one trainable condition token."""

    def __init__(self, model_name: str, hidden_dim: int, freeze: bool) -> None:
        super().__init__()
        require_package("transformers", extra="flow_matching")
        if CLIPTextModel is None:
            raise ImportError(
                "CLIPTextModel is unavailable despite the transformers package being installed."
            )

        self.freeze = freeze
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        if freeze:
            self.text_encoder.requires_grad_(False)
        self.projection = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError(
                "Language tokens and attention mask must both have shape (B, sequence_length), "
                f"got {tuple(input_ids.shape)} and {tuple(attention_mask.shape)}."
            )

        if self.freeze:
            with torch.no_grad():
                pooled = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).pooler_output
        else:
            pooled = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).pooler_output
        return self.projection(pooled)


class ObservationEncoder(nn.Module):
    """Encode and concatenate the configured observation history."""

    def __init__(self, config: FlowMatchingConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_cameras = len(config.image_features)

        modality_count = 0
        if config.robot_state_feature is not None:
            self.state_projection = nn.Linear(config.robot_state_feature.shape[0], config.hidden_dim)
            modality_count += 1
        else:
            self.state_projection = None

        if config.env_state_feature is not None:
            self.env_projection = nn.Linear(config.env_state_feature.shape[0], config.hidden_dim)
            modality_count += 1
        else:
            self.env_projection = None

        if self.num_cameras:
            try:
                backbone_kwargs = {"weights": config.pretrained_backbone_weights}
                if config.use_group_norm:
                    backbone_kwargs["norm_layer"] = lambda channels: nn.GroupNorm(
                        num_groups=max(1, channels // 16),
                        num_channels=channels,
                    )
                backbone = getattr(torchvision.models, config.vision_backbone)(**backbone_kwargs)
            except AttributeError as exc:
                raise ValueError(f"Unknown torchvision vision backbone {config.vision_backbone!r}.") from exc
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.vision_backbone = backbone
            self.image_projection = nn.Linear(feature_dim, config.hidden_dim)
            modality_count += self.num_cameras
        else:
            self.vision_backbone = None
            self.image_projection = None

        if config.text_encoder_name is not None:
            self.task_encoder = CLIPTaskEncoder(
                model_name=config.text_encoder_name,
                hidden_dim=config.hidden_dim,
                freeze=config.freeze_text_encoder,
            )
        else:
            self.task_encoder = None

        condition_tokens = config.n_obs_steps * modality_count + int(self.task_encoder is not None)
        self.conditioning_dim = condition_tokens * config.hidden_dim

    def _ensure_vector_history(self, value: Tensor, key: str) -> Tensor:
        if value.ndim == 2:
            value = value.unsqueeze(1).expand(-1, self.config.n_obs_steps, -1)
        if value.ndim != 3 or value.shape[1] != self.config.n_obs_steps:
            raise ValueError(
                f"`{key}` must have shape (B, {self.config.n_obs_steps}, D) or (B, D), "
                f"got {tuple(value.shape)}."
            )
        return value

    def _ensure_image_history(self, images: Tensor) -> Tensor:
        # A single environment observation is (B, cameras, C, H, W).
        if images.ndim == 5:
            images = images.unsqueeze(1).expand(-1, self.config.n_obs_steps, -1, -1, -1, -1)
        expected_prefix = (self.config.n_obs_steps, self.num_cameras, 3)
        if images.ndim != 6 or tuple(images.shape[1:4]) != expected_prefix:
            raise ValueError(
                "`observation.images` must have shape "
                f"(B, {self.config.n_obs_steps}, {self.num_cameras}, 3, H, W) or "
                f"(B, {self.num_cameras}, 3, H, W), got {tuple(images.shape)}."
            )
        return images

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        tokens = []

        if self.state_projection is not None:
            state = self._ensure_vector_history(batch[OBS_STATE], OBS_STATE)
            tokens.append(self.state_projection(state))

        if self.env_projection is not None:
            env_state = self._ensure_vector_history(batch[OBS_ENV_STATE], OBS_ENV_STATE)
            tokens.append(self.env_projection(env_state))

        if self.vision_backbone is not None:
            images = self._ensure_image_history(batch[OBS_IMAGES])
            batch_size, n_obs_steps, num_cameras = images.shape[:3]
            flat_images = images.reshape(-1, *images.shape[-3:])
            image_features = self.vision_backbone(flat_images)
            image_tokens = self.image_projection(image_features)
            image_tokens = image_tokens.reshape(batch_size, n_obs_steps, num_cameras, self.hidden_dim)
            tokens.append(image_tokens.flatten(start_dim=1, end_dim=2))

        if self.task_encoder is not None:
            if OBS_LANGUAGE_TOKENS not in batch or OBS_LANGUAGE_ATTENTION_MASK not in batch:
                raise ValueError(
                    "Task conditioning is enabled, but tokenized language is missing from the batch. "
                    "Build the policy processor with `make_flow_matching_pre_post_processors`."
                )
            task_token = self.task_encoder(
                batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK],
            )
            tokens.append(task_token.unsqueeze(1))

        # Flattening preserves a distinct projection weight for every modality,
        # camera, and observation timestep in the final condition projection.
        return torch.cat(tokens, dim=1).flatten(start_dim=1)


class ActionVelocityTransformer(nn.Module):
    """Transformer vector field conditioned on observation history and time."""

    def __init__(self, config: FlowMatchingConfig, action_dim: int, conditioning_dim: int) -> None:
        super().__init__()
        self.action_projection = nn.Linear(action_dim, config.hidden_dim)
        self.condition_projection = nn.Linear(conditioning_dim, config.hidden_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        )
        self.position_embedding = nn.Embedding(config.horizon, config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feed_forward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(config.hidden_dim, action_dim)

    def forward(self, trajectory: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        horizon = trajectory.shape[1]
        position_ids = torch.arange(horizon, device=trajectory.device)

        condition_token = self.condition_projection(condition).unsqueeze(1)
        action_tokens = self.action_projection(trajectory)
        action_tokens = action_tokens + self.position_embedding(position_ids).unsqueeze(0)
        tokens = torch.cat((condition_token, action_tokens), dim=1)
        tokens = tokens + self.time_embedding(time).unsqueeze(1)

        return self.output_projection(self.transformer(tokens)[:, 1:])


def make_flow_matching_target(
    actions: Tensor,
    noise: Tensor | None = None,
    time: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build a straight noise-to-data path and its constant target velocity."""

    if noise is None:
        noise = torch.randn_like(actions)
    elif noise.shape != actions.shape:
        raise ValueError(f"`noise` must have shape {tuple(actions.shape)}, got {tuple(noise.shape)}.")
    else:
        noise = noise.to(device=actions.device, dtype=actions.dtype)

    batch_size = actions.shape[0]
    if time is None:
        time = torch.rand(batch_size, device=actions.device, dtype=actions.dtype)
    elif time.shape != (batch_size,):
        raise ValueError(f"`time` must have shape ({batch_size},), got {tuple(time.shape)}.")
    else:
        time = time.to(device=actions.device, dtype=actions.dtype)

    broadcast_time = time.reshape(batch_size, *([1] * (actions.ndim - 1)))
    interpolated = (1.0 - broadcast_time) * noise + broadcast_time * actions
    target_velocity = actions - noise
    return interpolated, target_velocity, time


class FlowMatchingModel(nn.Module):
    """Observation encoder, velocity field, training objective, and ODE sampler."""

    def __init__(self, config: FlowMatchingConfig) -> None:
        super().__init__()
        self.config = config
        self.action_dim = config.action_feature.shape[0]
        self.observation_encoder = ObservationEncoder(config)
        self.velocity_field = ActionVelocityTransformer(
            config,
            action_dim=self.action_dim,
            conditioning_dim=self.observation_encoder.conditioning_dim,
        )

    def compute_loss(
        self,
        batch: dict[str, Tensor],
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: Literal["mean", "none"] = "mean",
    ) -> tuple[Tensor, dict[str, float]]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"`reduction` must be 'mean' or 'none', got {reduction!r}.")

        actions = batch[ACTION]
        if actions.ndim != 3 or actions.shape[1:] != (self.config.horizon, self.action_dim):
            raise ValueError(
                f"`action` must have shape (B, {self.config.horizon}, {self.action_dim}), "
                f"got {tuple(actions.shape)}."
            )

        condition = self.observation_encoder(batch)
        if self.training and self.config.conditioning_dropout_prob > 0:
            keep_condition = (
                torch.rand(actions.shape[0], 1, device=condition.device)
                >= self.config.conditioning_dropout_prob
            )
            condition = condition * keep_condition.to(condition.dtype)

        trajectory, target_velocity, time = make_flow_matching_target(actions, noise=noise, time=time)
        predicted_velocity = self.velocity_field(trajectory, time, condition)
        squared_error = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
        absolute_error = F.l1_loss(predicted_velocity, target_velocity, reduction="none")

        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError("`action_is_pad` is required when `do_mask_loss_for_padding=True`.")
            if batch["action_is_pad"].shape != actions.shape[:2]:
                raise ValueError(
                    f"`action_is_pad` must have shape {tuple(actions.shape[:2])}, "
                    f"got {tuple(batch['action_is_pad'].shape)}."
                )
            valid = (~batch["action_is_pad"]).unsqueeze(-1).to(squared_error.dtype)
            denominator = (valid.sum(dim=(1, 2)) * self.action_dim).clamp_min(1)
            mse_per_sample = (squared_error * valid).sum(dim=(1, 2)) / denominator
            l1_per_sample = (absolute_error * valid).sum(dim=(1, 2)) / denominator
        else:
            mse_per_sample = squared_error.mean(dim=(1, 2))
            l1_per_sample = absolute_error.mean(dim=(1, 2))

        loss = mse_per_sample.mean() if reduction == "mean" else mse_per_sample
        metrics = {
            "mse_loss": mse_per_sample.mean().item(),
            "l1_loss": l1_per_sample.mean().item(),
        }
        return loss, metrics

    @torch.no_grad()
    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        condition = self.observation_encoder(batch)
        batch_size = condition.shape[0]
        expected_shape = (batch_size, self.config.horizon, self.action_dim)
        if noise is None:
            trajectory = torch.randn(
                expected_shape,
                device=condition.device,
                dtype=condition.dtype,
            )
        else:
            if noise.shape != expected_shape:
                raise ValueError(f"`noise` must have shape {expected_shape}, got {tuple(noise.shape)}.")
            trajectory = noise.to(device=condition.device, dtype=condition.dtype).clone()

        unconditional = torch.zeros_like(condition)
        step_size = 1.0 / self.config.num_inference_steps
        for step in range(self.config.num_inference_steps):
            # Midpoint time reduces integration bias without another field evaluation.
            time = torch.full(
                (batch_size,),
                (step + 0.5) * step_size,
                device=trajectory.device,
                dtype=trajectory.dtype,
            )
            conditional_velocity = self.velocity_field(trajectory, time, condition)
            if self.config.guidance_scale == 1.0:
                velocity = conditional_velocity
            else:
                unconditional_velocity = self.velocity_field(trajectory, time, unconditional)
                velocity = unconditional_velocity + self.config.guidance_scale * (
                    conditional_velocity - unconditional_velocity
                )
            trajectory = trajectory + step_size * velocity

        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return trajectory[:, start:end]


class FlowMatchingPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for conditional Flow Matching."""

    config_class = FlowMatchingConfig
    name = "flow_matching"

    def __init__(self, config: FlowMatchingConfig, **_kwargs) -> None:
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = FlowMatchingModel(config)
        self._queues = None
        self.reset()

    def get_optim_params(self):
        if self.model.observation_encoder.vision_backbone is None:
            return self.parameters()

        backbone_parameters = []
        other_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("model.observation_encoder.vision_backbone"):
                backbone_parameters.append(parameter)
            else:
                other_parameters.append(parameter)
        return [
            {"params": other_parameters},
            {"params": backbone_parameters, "lr": self.config.optimizer_lr_backbone},
        ]

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        if self.config.robot_state_feature is not None:
            self._queues[OBS_STATE] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature is not None:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        prepared = dict(batch)
        if self.config.image_features and OBS_IMAGES not in prepared:
            prepared[OBS_IMAGES] = torch.stack(
                [prepared[key] for key in self.config.image_features],
                dim=-4,
            )
        return prepared

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: Literal["mean", "none"] = "mean",
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the Flow Matching velocity-regression loss."""

        return self.model.compute_loss(self._prepare_batch(batch), reduction=reduction)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Predict the next ``n_action_steps`` actions."""

        prepared = self._prepare_batch(batch)
        return self.model.generate_actions(prepared, noise=kwargs.get("noise"))

    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Return one action while caching observation history and action chunks."""

        prepared = self._prepare_batch(batch)
        prepared.pop(ACTION, None)
        self._queues = populate_queues(self._queues, prepared)

        if len(self._queues[ACTION]) == 0:
            history = {
                key: torch.stack(list(queue), dim=1) for key, queue in self._queues.items() if key != ACTION
            }
            if self.config.text_encoder_name is not None:
                history[OBS_LANGUAGE_TOKENS] = prepared[OBS_LANGUAGE_TOKENS]
                history[OBS_LANGUAGE_ATTENTION_MASK] = prepared[OBS_LANGUAGE_ATTENTION_MASK]
            actions = self.predict_action_chunk(history, **kwargs)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

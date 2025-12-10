#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
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

"""Multi-Task Diffusion Transformer (DiT) Policy

Transformer-based diffusion policy for multi-task robot learning with text and vision conditioning.
Supports both diffusion and flow matching objectives for action generation.
"""

from collections import deque

import torch
from torch import Tensor

from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.multi_task_dit.modules.objectives import DiffusionObjective, FlowMatchingObjective
from lerobot.policies.multi_task_dit.modules.observation_encoder import ObservationEncoder
from lerobot.policies.multi_task_dit.modules.transformer import DiffusionTransformer
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES


class MultiTaskDiTPolicy(PreTrainedPolicy):
    config_class = MultiTaskDiTConfig
    name = "multi_task_dit"

    def __init__(self, config: MultiTaskDiTConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None

        self.observation_encoder = ObservationEncoder(config)
        conditioning_dim = self.observation_encoder.conditioning_dim
        self.noise_predictor = DiffusionTransformer(config, conditioning_dim=conditioning_dim)

        action_dim = config.action_feature.shape[0]
        horizon = config.horizon

        if config.is_diffusion:
            self.objective = DiffusionObjective(
                config,
                action_dim=action_dim,
                horizon=horizon,
                do_mask_loss_for_padding=config.do_mask_loss_for_padding,
            )
        elif config.is_flow_matching:
            self.objective = FlowMatchingObjective(
                config,
                action_dim=action_dim,
                horizon=horizon,
                do_mask_loss_for_padding=config.do_mask_loss_for_padding,
            )
        else:
            raise ValueError(f"Unsupported objective: {config.objective}")

        self.reset()

    def get_optim_params(self) -> list:
        """Returns parameter groups with different learning rates for vision vs non-vision parameters."""
        non_vision_params = []
        vision_encoder_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "observation_encoder.vision_encoder" in name:
                vision_encoder_params.append(param)
            else:
                non_vision_params.append(param)

        return [
            {"params": non_vision_params},
            {
                "params": vision_encoder_params,
                "lr": self.config.optimizer_lr * self.config.vision_encoder_lr_multiplier,
            },
        ]

    def _generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        conditioning_vec = self.observation_encoder.encode(batch)
        actions = self.objective.conditional_sample(self.noise_predictor, batch_size, conditioning_vec)

        start_idx = n_obs_steps - 1
        end_idx = start_idx + self.config.n_action_steps
        return actions[:, start_idx:end_idx]

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`."""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }

        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)

        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

        # Always include task queue for text conditioning
        self._queues["task"] = deque(maxlen=self.config.n_obs_steps)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        conditioning_vec = self.observation_encoder.encode(batch)
        loss = self.objective.compute_loss(self.noise_predictor, batch, conditioning_vec)

        return loss, None

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        original_batch_keys = set(batch.keys())
        new_batch = {}
        for k in self._queues:
            if k in original_batch_keys:
                if self._queues[k] and isinstance(self._queues[k][-1][0], str):
                    # for task description which is a list of strings
                    new_batch[k] = self._queues[k][-1]
                else:
                    queue_values = list(self._queues[k])
                    new_batch[k] = torch.stack(queue_values, dim=1)
        batch = new_batch

        actions = self._generate_actions(batch)
        return actions

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method manages caching of observations and actions by generating an action chunk
        and returning actions from the cache until it's depleted.
        """
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. 
# All rights reserved.
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

from collections import deque

import einops

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from huggingface_hub import PyTorchModelHubMixin
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.sac.configuration_sac import SACConfig

class SACPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "RL", "SAC"],
):
    
    def __init__(
        self, config: SACConfig | None = None, dataset_stats: dict[str, dict[str, Tensor]] | None = None
    ):
        
        super().__init__()

        if config is None:
            config = SACConfig()
        self.config = config

        if config.input_normalization_modes is not None:
            self.normalize_inputs = Normalize(
                config.input_shapes, config.input_normalization_modes, dataset_stats
            )
        else:
            self.normalize_inputs = nn.Identity()
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        
        self.critic_ensemble = ...
        self.critic_target = ...
        self.actor_network = ...

        self.temperature = ...

    def reset(self):
        """
        Clear observation and action queues. Should be called on `env.reset()`
        queues are populated during rollout of the policy, they contain the n latest observations and actions
        """

        self._queues = {
            "observation.state": deque(maxlen=1),
            "action": deque(maxlen=1),
        }
        if self._use_image:
            self._queues["observation.image"] = deque(maxlen=1)
        if self._use_env_state:
            self._queues["observation.environment_state"] = deque(maxlen=1)
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        actions, _ = self.actor_network(batch['observations'])###

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | float]:
        """Run the batch through the model and compute the loss.

        Returns a dictionary with loss as a tensor, and other information as native floats.
        """
        batch = self.normalize_inputs(batch)
        # batch shape is (b, 2, ...) where index 1 returns the current observation and 
        # the next observation for caluculating the right td index. 
        actions = batch["action"][:, 0]
        rewards = batch["next.reward"][:, 0]
        observations = {}
        next_observations = {}
        for k in batch:
            if k.startswith("observation."):
                observations[k] = batch[k][:, 0]
                next_observations[k] = batch[k][:, 1]
       
        # perform image augmentation

        # reward bias
        # from HIL-SERL code base 
        # add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]} in reward_batch
        

        # calculate critics loss
        # 1- compute actions from policy
        action_preds, log_probs = self.actor_network(observations)
        # 2- compute q targets
        q_targets = self.target_qs(next_observations, action_preds)

        # critics subsample size
        min_q = q_targets.min(dim=0)

        # backup entropy    
        td_target = rewards + self.discount * min_q

        # 3- compute predicted qs
        q_preds = self.critic_ensemble(observations, actions)

        # 4- Calculate loss
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        critics_loss = (   
            F.mse_loss(
                    q_preds,
                    einops.repeat(td_target, "t b -> e t b", e=q_preds.shape[0]),
                    reduction="none",
                ).sum(0)  # sum over ensemble
                # `q_preds_ensemble` depends on the first observation and the actions.
                * ~batch["observation.state_is_pad"][0]
                * ~batch["action_is_pad"]
                # q_targets depends on the reward and the next observations.
                * ~batch["next.reward_is_pad"]
                * ~batch["observation.state_is_pad"][1:]
            ).sum(0).mean()
        
        # calculate actors loss
        # 1- temperature
        temperature = self.temperature()

        # 2- get actions (batch_size, action_dim) and log probs (batch_size,)
        actions, log_probs = self.actor_network(observations) \

        # 3- get q-value predictions
        with torch.no_grad():
            q_preds = self.critic_ensemble(observations, actions, return_type="mean")
        actor_loss = (
            -(q_preds - temperature * log_probs).mean()
            * ~batch["observation.state_is_pad"][0]
            * ~batch["action_is_pad"]
        ).mean()


        # calculate temperature loss
        # 1- calculate entropy
        entropy = -log_probs.mean()
        temperature_loss = temperature * (entropy - self.target_entropy).mean()

        loss = critics_loss + actor_loss + temperature_loss

        return {
                "critics_loss": critics_loss.item(),
                "actor_loss": actor_loss.item(),
                "temperature_loss": temperature_loss.item(),
                "temperature": temperature.item(),
                "entropy": entropy.item(),
                "loss": loss,

            }
    
    def update(self):
        self.critic_target.lerp_(self.critic_ensemble, self.config.critic_target_update_weight)
        #for target_param, param in zip(self.critic_target.parameters(), self.critic_ensemble.parameters()):
        #    target_param.data.copy_(target_param.data * (1.0 - self.config.critic_target_update_weight) + param.data * self.critic_target_update_weight)

class SACObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: SACConfig):

        super().__init__()
        self.config = config

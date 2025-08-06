#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PolicyFeature
from lerobot.processor.pipeline import EnvTransition, TransitionKey
from lerobot.utils.utils import get_safe_torch_device


@dataclass
class DeviceProcessor:
    """Processes transitions by moving tensors to the specified device.

    This processor ensures that all tensors in the transition are moved to the
    specified device (CPU or GPU) before they are returned.
    """

    device: torch.device = "cpu"

    def __post_init__(self):
        self.device = get_safe_torch_device(self.device)
        self.non_blocking = "cuda" in str(self.device)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Create a copy of the transition
        new_transition = transition.copy()

        # Process observation tensors
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_observation = {
                k: v.to(self.device, non_blocking=self.non_blocking) if isinstance(v, torch.Tensor) else v
                for k, v in observation.items()
            }
            new_transition[TransitionKey.OBSERVATION] = new_observation

        # Process action tensor
        action = transition.get(TransitionKey.ACTION)
        if action is not None and isinstance(action, torch.Tensor):
            new_transition[TransitionKey.ACTION] = action.to(self.device, non_blocking=self.non_blocking)

        # Process reward tensor
        reward = transition.get(TransitionKey.REWARD)
        if reward is not None and isinstance(reward, torch.Tensor):
            new_transition[TransitionKey.REWARD] = reward.to(self.device, non_blocking=self.non_blocking)

        # Process done tensor
        done = transition.get(TransitionKey.DONE)
        if done is not None and isinstance(done, torch.Tensor):
            new_transition[TransitionKey.DONE] = done.to(self.device, non_blocking=self.non_blocking)

        # Process truncated tensor
        truncated = transition.get(TransitionKey.TRUNCATED)
        if truncated is not None and isinstance(truncated, torch.Tensor):
            new_transition[TransitionKey.TRUNCATED] = truncated.to(
                self.device, non_blocking=self.non_blocking
            )

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
        return {"device": self.device}

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features

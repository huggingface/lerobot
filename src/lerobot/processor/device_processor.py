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
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.processor.pipeline import EnvTransition, TransitionIndex


@dataclass
class DeviceProcessor:
    """Processes transitions by moving tensors to the specified device.

    This processor ensures that all tensors in the transition are moved to the
    specified device (CPU or GPU) before they are returned.
    """

    device: str = "cpu"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation: dict[str, torch.Tensor] = transition[TransitionIndex.OBSERVATION]
        action = transition[TransitionIndex.ACTION]
        reward = transition[TransitionIndex.REWARD]
        done = transition[TransitionIndex.DONE]
        truncated = transition[TransitionIndex.TRUNCATED]
        info = transition[TransitionIndex.INFO]
        complementary_data = transition[TransitionIndex.COMPLEMENTARY_DATA]

        if observation is not None:
            observation = {k: v.to(self.device) for k, v in observation.items()}
        if action is not None:
            action = action.to(self.device)

        return (
            observation,
            action,
            reward,
            done,
            truncated,
            info,
            complementary_data,
        )

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
        return {"device": self.device}

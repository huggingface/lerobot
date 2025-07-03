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

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, TransitionIndex


@dataclass
@ProcessorStepRegistry.register(name="rename_processor")
class RenameProcessor:
    """Rename processor that renames keys in the observation."""

    rename_map: dict[str, str] = field(default_factory=dict)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition[TransitionIndex.OBSERVATION]
        if observation is None:
            return transition

        processed_obs = {}
        for key, value in observation.items():
            if key in self.rename_map:
                processed_obs[self.rename_map[key]] = value
            else:
                processed_obs[key] = value
        return (
            processed_obs,
            transition[TransitionIndex.ACTION],
            transition[TransitionIndex.REWARD],
            transition[TransitionIndex.DONE],
            transition[TransitionIndex.TRUNCATED],
            transition[TransitionIndex.INFO],
            transition[TransitionIndex.COMPLEMENTARY_DATA],
        )

    def get_config(self) -> dict[str, Any]:
        return {"rename_map": self.rename_map}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        pass

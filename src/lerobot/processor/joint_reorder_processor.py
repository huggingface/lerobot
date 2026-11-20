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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION

from .pipeline import ProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="joint_reorder_processor")
class JointReorderProcessorStep(ProcessorStep):
    """Reorder or select joints: ``x[..., indices]`` on listed observation keys and on the action.

    Attributes:
        observation_indices: For each observation key, the input index of each output entry.
        action_indices: The input index of each output action entry, or ``None`` to leave it unchanged.
            Only tensor or array actions are indexed.
    """

    observation_indices: dict[str, list[int]] = field(default_factory=dict)
    action_indices: list[int] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            transition[TransitionKey.OBSERVATION] = {
                key: value[..., self.observation_indices[key]] if key in self.observation_indices else value
                for key, value in observation.items()
            }
        action = transition.get(TransitionKey.ACTION)
        if self.action_indices is not None and isinstance(action, torch.Tensor | np.ndarray):
            transition[TransitionKey.ACTION] = action[..., self.action_indices]
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"observation_indices": self.observation_indices, "action_indices": self.action_indices}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        indices = {
            PipelineFeatureType.OBSERVATION: self.observation_indices,
            PipelineFeatureType.ACTION: {} if self.action_indices is None else {ACTION: self.action_indices},
        }
        return {
            feature_type: {
                key: PolicyFeature(type=ft.type, shape=(*ft.shape[:-1], len(indices[feature_type][key])))
                if key in indices.get(feature_type, {})
                else ft
                for key, ft in group.items()
            }
            for feature_type, group in features.items()
        }

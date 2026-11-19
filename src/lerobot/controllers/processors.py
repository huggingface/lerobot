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

"""Processor pipelines between a robot's state and a controller's network.

A controller's preprocessor normalizes the robot state it reads (joint positions around the default pose,
scaled velocities, ...) and reorders joints from robot order to the order its network was trained in. Its
postprocessor maps the network's action back: reorder to robot order, then unnormalize into absolute joint
targets. Statistics are always given in robot joint order.

Normalization is the policies' ``NormalizerProcessorStep`` / ``UnnormalizerProcessorStep`` in ``MEAN_STD``
mode: a residual ``default + scale * a`` is ``mean=default, std=scale``, and an observation scaled as
``(x - offset) * scale`` is ``mean=offset, std=1 / scale``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    NormalizerProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    observation_to_transition,
    policy_action_to_transition,
    transition_to_observation,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import DataProcessorPipeline
from lerobot.utils.constants import ACTION

ArrayLike = Sequence[float] | np.ndarray


@dataclass
@ProcessorStepRegistry.register(name="joint_reorder_processor")
class JointReorderProcessorStep(ProcessorStep):
    """Index the joint axis of observation features and of the action: ``x[..., indices]``.

    Reorders joints between the robot's order and a network's, or selects the ones a network reads or drives.

    Args:
        observation_indices: For each observation key to reorder, the robot index of each output entry.
        action_indices: For the action, the input index of each output entry.
    """

    observation_indices: dict[str, list[int]] = field(default_factory=dict)
    action_indices: list[int] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None and self.observation_indices:
            observation = dict(observation)
            for key, indices in self.observation_indices.items():
                if key in observation:
                    observation[key] = observation[key][..., indices]
            new_transition[TransitionKey.OBSERVATION] = observation
        action = new_transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor | np.ndarray) and self.action_indices is not None:
            new_transition[TransitionKey.ACTION] = action[..., self.action_indices]
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"observation_indices": self.observation_indices, "action_indices": self.action_indices}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = {feature_type: dict(group) for feature_type, group in features.items()}
        observation = features.get(PipelineFeatureType.OBSERVATION, {})
        for key, indices in self.observation_indices.items():
            if key in observation:
                observation[key] = PolicyFeature(type=observation[key].type, shape=(len(indices),))
        action = features.get(PipelineFeatureType.ACTION, {})
        if self.action_indices is not None and ACTION in action:
            action[ACTION] = PolicyFeature(type=action[ACTION].type, shape=(len(self.action_indices),))
        return features


def make_controller_processors(
    observation_stats: Mapping[str, tuple[ArrayLike, ArrayLike]],
    action_stats: tuple[ArrayLike, ArrayLike],
    observation_order: Mapping[str, ArrayLike] | None = None,
    action_order: ArrayLike | None = None,
) -> tuple[
    DataProcessorPipeline[dict[str, Any], dict[str, Any]], DataProcessorPipeline[torch.Tensor, torch.Tensor]
]:
    """Build a controller's preprocessor and postprocessor.

    Args:
        observation_stats: For each observation key to normalize, its ``(mean, std)`` in robot joint order.
        action_stats: The action's ``(mean, std)`` in robot joint order: the default pose and the action scale.
        observation_order: For each observation key to reorder after normalization, the robot index of
            each entry the network reads.
        action_order: The network-action index of each robot joint the postprocessor returns.

    Returns:
        The preprocessor, from a dict of robot-state arrays to normalized tensors in network order, and the
        postprocessor, from a network action tensor to absolute joint targets in robot order.
    """

    def stats(mean: ArrayLike, std: ArrayLike) -> dict[str, np.ndarray]:
        return {"mean": np.asarray(mean, np.float32), "std": np.asarray(std, np.float32)}

    def feature(feature_type: FeatureType, mean: ArrayLike) -> PolicyFeature:
        return PolicyFeature(type=feature_type, shape=(len(np.atleast_1d(mean)),))

    preprocessor: DataProcessorPipeline[dict[str, Any], dict[str, Any]] = DataProcessorPipeline(
        steps=[
            NormalizerProcessorStep(
                features={
                    key: feature(FeatureType.STATE, mean) for key, (mean, _) in observation_stats.items()
                },
                norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
                stats={key: stats(*value) for key, value in observation_stats.items()},
            ),
            JointReorderProcessorStep(
                observation_indices={
                    key: [int(i) for i in order] for key, order in (observation_order or {}).items()
                }
            ),
        ],
        name="controller_preprocessor",
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    postprocessor: DataProcessorPipeline[torch.Tensor, torch.Tensor] = DataProcessorPipeline(
        steps=[
            JointReorderProcessorStep(
                action_indices=None if action_order is None else [int(i) for i in action_order]
            ),
            UnnormalizerProcessorStep(
                features={ACTION: feature(FeatureType.ACTION, action_stats[0])},
                norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
                stats={ACTION: stats(*action_stats)},
            ),
        ],
        name="controller_postprocessor",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor

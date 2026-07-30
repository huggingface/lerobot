#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

from typing import Any

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .configuration_wall_oss_05 import WallOSS05Config


@ProcessorStepRegistry.register(name="wall_oss_05_task_passthrough")
class WallOSS05TaskPassthrough(ComplementaryDataProcessorStep):
    """Validate language input without selecting, punctuating, or rewriting it."""

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        if "task" not in complementary_data:
            raise KeyError("Wall-OSS-0.5 requires the already-selected LeRobot task string.")
        task = complementary_data["task"]
        if not isinstance(task, str) and not (
            isinstance(task, list) and all(isinstance(value, str) for value in task)
        ):
            raise TypeError("task must be a string or a list of strings.")
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="wall_oss_05_clamp_normalized")
class WallOSS05ClampNormalizedProcessorStep(ProcessorStep):
    """Match Wall's clipped q01/q99 normalization after the standard LeRobot normalizer."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        output = transition.copy()
        observation = output.get(TransitionKey.OBSERVATION)
        if observation is not None and OBS_STATE in observation:
            observation = dict(observation)
            observation[OBS_STATE] = observation[OBS_STATE].clamp(-1, 1)
            output[TransitionKey.OBSERVATION] = observation
        action = output.get(TransitionKey.ACTION)
        if action is not None:
            output[TransitionKey.ACTION] = action.clamp(-1, 1)
        return output

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_wall_oss_05_pre_post_processors(
    config: WallOSS05Config,
    dataset_stats: dict | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the serializable Wall input/output pipelines."""

    steps = make_default_policy_processor_steps(config, dataset_stats)
    return make_policy_processor_pipelines(
        input_steps=[
            steps.rename_observations,
            steps.add_batch_dim,
            WallOSS05TaskPassthrough(),
            steps.normalize,
            WallOSS05ClampNormalizedProcessorStep(),
            steps.to_device,
        ],
        output_steps=[steps.unnormalize, steps.to_cpu],
    )

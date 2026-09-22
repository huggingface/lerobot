#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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

from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    make_default_policy_processor_steps,
    make_default_pre_post_processors,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import OBS_STATE

from .configuration_diffusion import DiffusionConfig


@ProcessorStepRegistry.register("diffusion_relative_actions_processor")
@dataclass
class DiffusionRelativeActionsProcessorStep(RelativeActionsProcessorStep):
    """Anchors the relative conversion on the newest observed frame.

    Diffusion's ``observation_delta_indices`` is ``[1 - n_obs_steps, ..., 0]``, so the dataset
    hands training a ``(B, n_obs_steps, state_dim)`` state whose index 0 is
    ``t - n_obs_steps + 1``. At inference the environment yields a single frame and the state is
    ``(B, state_dim)`` -- the stacking happens inside the policy, after this pipeline. The base
    step collapses a 3-D state with ``state[:, 0]``, which would anchor training on an older frame
    than inference uses. Hand it the newest frame instead so the two agree, then put the stacked
    state back for the model.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if state is None or state.ndim != 3:
            return super().__call__(transition)

        collapsed = transition.copy()
        collapsed[TransitionKey.OBSERVATION] = {**observation, OBS_STATE: state[:, -1]}
        new_transition = super().__call__(collapsed)
        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition


def make_diffusion_pre_post_processors(
    config: DiffusionConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for a diffusion policy.

    The pre-processing pipeline prepares the input data for the model by:
    1. Renaming features.
    2. Normalizing the input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Moving the data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving the data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the diffusion policy,
            containing feature definitions, normalization mappings, and device information.
        dataset_stats: A dictionary of statistics used for normalization.
            Defaults to None.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """
    if not config.use_relative_actions:
        return make_default_pre_post_processors(config, dataset_stats)

    relative_step = DiffusionRelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=config.relative_exclude_joints,
        action_names=config.action_feature_names,
    )
    steps = make_default_policy_processor_steps(config, dataset_stats)
    # raw -> relative -> normalize -> model -> unnormalize -> absolute
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
        relative_step,
        steps.normalize,
    ]
    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step),
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)

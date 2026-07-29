#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import torch

from lerobot.processor import (
    AbsoluteEEActionsStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RelativeEEActionsStep,
    RelativeEEDeriveStateStep,
    RelativeEEStateStep,
    make_default_policy_processor_steps,
    make_default_pre_post_processors,
    make_policy_processor_pipelines,
)

from .configuration_act import ACTConfig


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """
    if config.use_relative_ee:
        steps = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=config.device)
        relative_step = RelativeEEActionsStep()
        return make_policy_processor_pipelines(
            input_steps=[
                steps.rename_observations,
                steps.add_batch_dim,
                steps.to_device,
                RelativeEEDeriveStateStep(),
                relative_step,
                RelativeEEStateStep(),
                steps.normalize,
            ],
            output_steps=[steps.unnormalize, AbsoluteEEActionsStep(relative_step), steps.to_cpu],
        )
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)

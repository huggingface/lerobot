#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_wall_x import WallXConfig


def make_wall_x_pre_post_processors(
    config: WallXConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the Wall-X policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations
    2. Adding a batch dimension
    4. Normalizing input and output features based on dataset statistics
    5. Moving all data to the specified device

    The post-processing pipeline handles the model's output by:
    1. Unnormalizing the output actions to their original scale
    2. Moving data to the CPU

    Args:
        config: The configuration object for the Wall-X policy
        dataset_stats: A dictionary of statistics for normalization

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines
    """

    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        WallXTaskProcessor(),  # Process task description
        steps.normalize,
        steps.to_device,
    ]

    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


@ProcessorStepRegistry.register(name="wall_x_task_processor")
class WallXTaskProcessor(ComplementaryDataProcessorStep):
    """
    A processor step that ensures the task description is properly formatted for Wall-X.

    This step handles task preprocessing similar to Qwen-VL requirements.
    """

    def complementary_data(self, complementary_data):
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            # Provide default task if none specified
            complementary_data["task"] = "Execute the robot action."
            return complementary_data

        new_complementary_data = dict(complementary_data)

        # Handle both string and list of strings
        if isinstance(task, str):
            # Single string: ensure proper formatting
            if not task.endswith("."):
                new_complementary_data["task"] = f"{task}."
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # List of strings: format each
            new_complementary_data["task"] = [t if t.endswith(".") else f"{t}." for t in task]

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

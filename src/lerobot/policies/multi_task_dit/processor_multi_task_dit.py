#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
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
    PolicyAction,
    PolicyProcessorPipeline,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_multi_task_dit import MultiTaskDiTConfig


def make_multi_task_dit_pre_post_processors(
    config: MultiTaskDiTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for a Multi-Task DiT policy.

    The pre-processing pipeline prepares the input data for the model by:
    1. Renaming features.
    2. Adding a batch dimension.
    3. Tokenizing the language task description (if present).
    4. Moving the data to the specified device.
    5. Normalizing the input and output features based on dataset statistics.

    The post-processing pipeline handles the model's output by:
    1. Unnormalizing the output features to their original scale.
    2. Moving the data to the CPU.

    Args:
        config: The configuration object for the Multi-Task DiT policy,
            containing feature definitions, normalization mappings, and device information.
        dataset_stats: A dictionary of statistics used for normalization.
            Defaults to None.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    steps = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=config.device)

    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        TokenizerProcessorStep(
            tokenizer_name=config.text_encoder_name,
            padding=config.tokenizer_padding,
            padding_side=config.tokenizer_padding_side,
            max_length=config.tokenizer_max_length,
            truncation=config.tokenizer_truncation,
        ),
        steps.to_device,
        steps.normalize,
    ]
    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)

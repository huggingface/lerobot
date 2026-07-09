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

"""Pre/post-processor pipelines for the LingBot-VA policy.

The preprocessor passes inputs through (IDENTITY) and the postprocessor maps the policy's
``[-1, 1]`` actions back to physical units with the built-in ``UnnormalizerProcessorStep``
(QUANTILES) using per-channel q01/q99 restored from the checkpoint.
"""

from typing import Any

import torch

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RelativeActionsProcessorStep,
    UnnormalizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)

from .configuration_lingbot_va import LingBotVAConfig


def make_lingbot_va_pre_post_processors(
    config: LingBotVAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the pre/post processor pipelines for LingBot-VA."""

    steps = make_default_policy_processor_steps(config, dataset_stats)

    # Shared relative-action step (OpenPI order: raw -> relative -> normalize -> model ->
    # unnormalize -> absolute). The SAME instance is passed to AbsoluteActionsProcessorStep
    # below so its cached raw state (set during preprocessing) flows to postprocessing.
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        relative_step,
        steps.normalize,
        steps.to_device,
    ]

    # Unnormalize actions back to physical units. Config-driven norm_map (was hardcoded QUANTILES)
    # so it stays symmetric with the preprocessor's NormalizerProcessorStep — required for
    # use_relative_actions with ACTION=IDENTITY (and unchanged for QUANTILES runs).
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)

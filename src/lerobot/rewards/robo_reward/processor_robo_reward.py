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

from typing import Any

import torch

from lerobot.processor import (
    DeviceProcessorStep,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.rewards.robo_reward.configuration_robo_reward import RoboRewardConfig


def make_robo_reward_pre_post_processors(
    config: RoboRewardConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Construct pre- and post-processor pipelines for RoboReward.

    The VLM processor (Qwen3-VL) handles image normalisation internally, so the
    pre-processing pipeline only moves tensors to the correct device. The
    post-processor is an identity pass.

    Args:
        config: RoboRewardConfig instance.
        dataset_stats: Optional statistics for normalisation steps.

    Returns:
        Tuple of (pre-processor, post-processor) pipelines.
    """
    input_steps = [
        NormalizerProcessorStep(
            features=config.input_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps = [DeviceProcessorStep(device="cpu"), IdentityProcessorStep()]

    return (
        PolicyProcessorPipeline(
            steps=input_steps,
            name="robo_reward_preprocessor",
        ),
        PolicyProcessorPipeline(
            steps=output_steps,
            name="robo_reward_postprocessor",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )

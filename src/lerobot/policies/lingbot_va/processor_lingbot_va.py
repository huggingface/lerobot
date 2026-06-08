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

The policy itself handles image resizing, scaling to [-1, 1] and VAE encoding (the VAE
lives inside the policy), so the preprocessor only renames, batches, normalizes (IDENTITY)
and moves to device. The policy emits actions in the normalized ``[-1, 1]`` space; the
postprocessor maps them back to physical units with the standard ``UnnormalizerProcessorStep``
in QUANTILES mode (``(action + 1) / 2 * (q99 - q01) + q01``). The per-channel q01/q99 are NOT
hardcoded: they are saved in each checkpoint's post-processor state and restored on load. A
fresh (unconverted) policy has no action stats, so the step is a no-op (identity passthrough).
"""

from typing import Any

import torch

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
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

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    # Unnormalize predicted actions from [-1, 1] back to physical units via per-channel q01/q99
    # (QUANTILES mode), overriding the policy's IDENTITY action mapping. The q01/q99 stats are
    # restored from the checkpoint on load; a fresh build has no action stats and is a passthrough.
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES},
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )

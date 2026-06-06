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
and moves to device. The postprocessor reverses the *fixed* action quantile normalization
(``(action + 1) / 2 * (q99 - q01 + 1e-6) + q01``) baked into the released checkpoints — this
is a fixed transform, not a dataset-stats one, so it cannot use the standard
``UnnormalizerProcessorStep`` and is implemented as a dedicated step below.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_lingbot_va import LingBotVAConfig

# Upstream LIBERO action-normalization quantiles (single 7-DoF arm + gripper).
# Verbatim from wan_va/configs/va_libero_cfg.py (channels 0-6 of a 30-dim action space).
# These are the fixed (un)normalization stats baked into the released LIBERO checkpoint; they
# live here (in the processor) and are serialized into the saved post-processor config.
LIBERO_ACTION_Q01 = [
    -0.6589285731315613,
    -0.84375,
    -0.9375,
    -0.12107142806053162,
    -0.15964286029338837,
    -0.26571428775787354,
    -1.0,
]
LIBERO_ACTION_Q99 = [
    0.8999999761581421,
    0.8544642925262451,
    0.9375,
    0.17142857611179352,
    0.1842857152223587,
    0.34392857551574707,
    1.0,
]


# Upstream RoboTwin action quantiles, reordered to the model's used-channel layout
# [left xyz+quat (0-6), left gripper (28), right xyz+quat (7-13), right gripper (29)] = 16 channels.
# Verbatim from wan_va/configs/va_robotwin_cfg.py ``norm_stat`` (quaternion + gripper channels use the
# neutral [-1, 1] / [0, 1] mapping). Positions are quantile-scaled; rotations pass through.
ROBOTWIN_ACTION_Q01 = [
    -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456, -1.0, -1.0, -1.0, -1.0,
    0.0,
    -0.3547105032205582, -1.3113021850585938e-06, -0.11975435614585876, -1.0, -1.0, -1.0, -1.0,
    0.0,
]  # fmt: skip
ROBOTWIN_ACTION_Q99 = [
    0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1.0, 1.0, 1.0, 1.0,
    1.0,
    0.034201726913452024, 0.39142737388610793, 0.1792279863357542, 1.0, 1.0, 1.0, 1.0,
    1.0,
]  # fmt: skip


def _default_action_quantiles(n_used: int) -> tuple[list[float], list[float]]:
    """Return the fixed (q01, q99) for the used action channels, by benchmark channel count.

    LIBERO = 7 (single 7-DoF arm), RoboTwin = 16 (dual-arm eef pose + grippers). Falls back to a
    neutral ``[-1, 1]`` mapping (no rescale) for any other channel count.
    """
    if n_used == len(LIBERO_ACTION_Q01):
        return list(LIBERO_ACTION_Q01), list(LIBERO_ACTION_Q99)
    if n_used == len(ROBOTWIN_ACTION_Q01):
        return list(ROBOTWIN_ACTION_Q01), list(ROBOTWIN_ACTION_Q99)
    return [-1.0] * n_used, [1.0] * n_used


@dataclass
@ProcessorStepRegistry.register(name="lingbot_va_action_unnormalize")
class LingBotVAActionUnnormalizeStep(PolicyActionProcessorStep):
    """Reverse LingBot-VA's fixed per-channel quantile normalization on predicted actions.

    The policy emits actions in the normalized ``[-1, 1]`` space of the used action channels.
    This step maps them back to physical units via the fixed quantiles stored in the config.
    """

    action_q01: list[float] = field(default_factory=list)
    action_q99: list[float] = field(default_factory=list)

    def action(self, action: PolicyAction) -> PolicyAction:
        q01 = torch.as_tensor(self.action_q01, dtype=action.dtype, device=action.device)
        q99 = torch.as_tensor(self.action_q99, dtype=action.dtype, device=action.device)
        return (action + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    def get_config(self) -> dict[str, Any]:
        return {"action_q01": list(self.action_q01), "action_q99": list(self.action_q99)}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


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

    action_q01, action_q99 = _default_action_quantiles(len(config.used_action_channel_ids))
    output_steps: list[ProcessorStep] = [
        LingBotVAActionUnnormalizeStep(action_q01=action_q01, action_q99=action_q99),
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

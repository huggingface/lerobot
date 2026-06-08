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

# LingBot-VA applies a *fixed* per-channel action quantile (un)normalization rather than
# dataset-derived stats. The benchmark-specific quantiles (LIBERO 7-DoF, RoboTwin 16-d eef) are
# deliberately NOT hardcoded here: they are serialized into each checkpoint's
# ``policy_postprocessor.json`` (via ``LingBotVAActionUnnormalizeStep.get_config``) and restored on
# load by ``PolicyProcessorPipeline.from_pretrained``. A freshly built (unconverted) policy defaults
# to a neutral ``[-1, 1]`` mapping (identity rescale); the real stats always come from the checkpoint
# (or via ``postprocessor_overrides``). To regenerate a checkpoint from scratch, source the quantiles
# from the upstream ``wan_va/configs/va_{libero,robotwin}_cfg.py`` and pass them through.


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

    # Fresh-build default: neutral [-1, 1] mapping (identity rescale). The real per-benchmark
    # quantiles are restored from the checkpoint's saved post-processor config by from_pretrained.
    n_used = len(config.used_action_channel_ids)
    output_steps: list[ProcessorStep] = [
        LingBotVAActionUnnormalizeStep(action_q01=[-1.0] * n_used, action_q99=[1.0] * n_used),
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

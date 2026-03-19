#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
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
from torch import Tensor

from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import ACTION, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


class BatchNormalizer:
    """Lightweight normalizer that operates on flat batch dicts.

    Wraps a ``NormalizerProcessorStep`` and exposes a callable interface
    compatible with ``preprocess_rl_batch`` in the trainer.  It mirrors the
    old ``NormalizeBuffer`` behaviour: observations are normalised according
    to their feature type (MEAN_STD for images, MIN_MAX for state/env) and
    actions are normalised with MIN_MAX.
    """

    def __init__(self, normalizer: NormalizerProcessorStep) -> None:
        self._normalizer = normalizer

    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        result = self._normalizer._normalize_observation(batch, inverse=False)
        if ACTION in result:
            result[ACTION] = self._normalizer._normalize_action(result[ACTION], inverse=False)
        return result

    def normalize_action(self, action: Tensor) -> Tensor:
        return self._normalizer._normalize_action(action, inverse=False)

    def normalize_observation(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        return self._normalizer._normalize_observation(observation, inverse=False)

    def to(self, device: torch.device | str) -> "BatchNormalizer":
        self._normalizer.to(device=device)
        return self


def make_sac_batch_normalizer(
    config: SACConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
    device: torch.device | str | None = None,
) -> BatchNormalizer:
    """Create a ``BatchNormalizer`` for SAC learner / actor normalisation.

    This replicates the old ``NormalizeBuffer`` setup from commit 7dcf506:
    - Images   (VISUAL) → MEAN_STD with ImageNet stats
    - State    (STATE)  → MIN_MAX
    - Actions  (ACTION) → MIN_MAX

    The returned object can be called on a flat dict of observations +
    actions (as used by ``preprocess_rl_batch``) and also exposes
    ``normalize_action`` / ``normalize_observation`` for use in the
    algorithm's actor-loss computation and in the actor inference loop.
    """
    normalizer = NormalizerProcessorStep(
        features={**config.input_features, **config.output_features},
        norm_map=config.normalization_mapping,
        stats=dataset_stats,
        device=device,
    )
    return BatchNormalizer(normalizer)


def make_sac_pre_post_processors(
    config: SACConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SAC policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the SAC policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    # Add remaining processors
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
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

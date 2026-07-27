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

from __future__ import annotations

from typing import Any

import torch

from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
from lerobot.processor import (
    EnvTransition,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
    UnnormalizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)


@ProcessorStepRegistry.register(name="vla_jepa_clip_actions")
class ClipActionsProcessorStep(ProcessorStep):
    """Clips action tensor to [-1, 1] before unnormalization."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            transition = dict(transition)
            transition[TransitionKey.ACTION] = action.clamp(-1.0, 1.0)
        return transition

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register(name="vla_jepa_pre_snap_gripper")
class PreSnapGripperProcessorStep(ProcessorStep):
    """Snaps a gripper dimension to {0, 1} BEFORE unnormalization.

    Mirrors the original starVLA LIBERO eval:
      normalized[:, gripper_dim] = np.where(normalized[:, gripper_dim] < threshold, 0, 1)
    This ensures the unnormalizer receives an exact binary value, which is
    required when the model was trained with gripper in identity (mask=False)
    space where 0=open and 1=close.
    """

    def __init__(self, gripper_dim: int = 6, threshold: float = 0.5):
        self.gripper_dim = gripper_dim
        self.threshold = threshold

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is not None and action.shape[-1] > self.gripper_dim:
            transition = dict(transition)
            a = action.clone()
            a[..., self.gripper_dim] = (a[..., self.gripper_dim] >= self.threshold).float()
            transition[TransitionKey.ACTION] = a
        return transition

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register(name="vla_jepa_binarize_gripper")
class BinarizeGripperProcessorStep(ProcessorStep):
    """Binarizes a gripper dimension after unnormalization.

    Maps continuous value to {-1, 1}: > threshold → -1, <= threshold → 1 (matches starVLA convention).
    Only applied when action has more dimensions than gripper_dim.
    """

    def __init__(self, gripper_dim: int = 6, threshold: float = 0.5):
        self.gripper_dim = gripper_dim
        self.threshold = threshold

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is not None and action.shape[-1] > self.gripper_dim:
            transition = dict(transition)
            a = action.clone()
            a[..., self.gripper_dim] = 1.0 - 2.0 * (a[..., self.gripper_dim] > self.threshold).float()
            transition[TransitionKey.ACTION] = a
        return transition

    def transform_features(self, features):
        return features


def make_vla_jepa_pre_post_processors(
    config: VLAJEPAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    features = {**config.input_features, **config.output_features}
    steps = make_default_policy_processor_steps(config, dataset_stats)
    input_steps = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.to_device,
        steps.normalize,
    ]
    output_steps: list[ProcessorStep] = []
    if config.clip_normalized_actions:
        output_steps.append(ClipActionsProcessorStep())
    if config.pre_snap_gripper_action:
        output_steps.append(
            PreSnapGripperProcessorStep(gripper_dim=config.gripper_dim, threshold=config.gripper_threshold)
        )
    # NOTE: unlike the default policy unnormalizer (output features only), VLA-JEPA
    # unnormalizes over BOTH input and output features.
    output_steps.append(
        UnnormalizerProcessorStep(
            features=features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )
    )
    if config.binarize_gripper_action:
        output_steps.append(
            BinarizeGripperProcessorStep(gripper_dim=config.gripper_dim, threshold=config.gripper_threshold)
        )
    output_steps.append(steps.to_cpu)
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)

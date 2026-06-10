# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Processor for RECAP's distributional value function.

Paper: "π*0.6: a VLA That Learns From Experience" (Physical Intelligence, 2025)
       https://pi.website/blog/pistar06

Prepares inputs for V^{pi_ref}(o_t, l): single image observation and task text only.
1. Image preprocessing (resize-with-pad + normalize to [-1, 1]) for SigLIP
2. Task prompt formatting ("Task: {task}.") and tokenization via PaliGemma tokenizer

Training targets (mc_return, is_terminal) are NOT routed through the processor.
They are dataset columns read directly from the batch in the model's forward().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
)
from lerobot.processor.converters import to_tensor
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGES,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_distributional_value_function import DistributionalVFConfig

PALIGEMMA_TOKENIZER_NAME = "google/paligemma-3b-pt-224"


@ProcessorStepRegistry.register(name="distributional_vf_prepare_task_prompt")
@dataclass
class DistributionalVFPrepareTaskPromptStep(ProcessorStep):
    """Format the task string for the distributional value function.

    The value function receives only visual observations and task text.
    Builds prompt: "Task: {task}."
    """

    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        tasks = complementary_data.get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        if isinstance(tasks, str):
            tasks = [tasks]

        full_prompts = []
        for task in tasks:
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            full_prompts.append(f"Task: {cleaned_text}.")

        new_complementary_data = dict(complementary_data)
        new_complementary_data[self.task_key] = full_prompts
        transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"task_key": self.task_key}


@ProcessorStepRegistry.register(name="distributional_vf_image_preprocessor")
@dataclass
class DistributionalVFImagePreprocessorStep(ProcessorStep):
    """Resize and normalize images for the value function's SigLIP vision tower.

    Expects float images in [0, 1].
    - Resize-with-pad to ``image_resolution`` (preserves aspect ratio)
    - Scale to [-1, 1] for SigLIP
    """

    image_resolution: tuple[int, int] = (224, 224)
    image_keys: tuple[str, ...] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            raise ValueError("DistributionalVFImagePreprocessorStep requires an observation dict")

        image_keys = self.image_keys or tuple(
            key for key in observation if key == OBS_IMAGES or key.startswith(f"{OBS_IMAGES}.")
        )
        if not image_keys:
            raise KeyError(
                f"Distributional value function expected image keys under {OBS_IMAGES!r} in observation"
            )

        new_observation = dict(observation)
        for image_key in image_keys:
            image = new_observation[image_key]
            if not isinstance(image, Tensor):
                image = to_tensor(image)
            if image.dtype != torch.float32:
                image = image.to(torch.float32)

            is_channels_first = image.ndim == 4 and image.shape[1] == 3
            if is_channels_first:
                image = image.permute(0, 2, 3, 1)

            if image.shape[1:3] != self.image_resolution:
                image = resize_with_pad_torch(image, *self.image_resolution)

            image = image * 2.0 - 1.0

            if is_channels_first:
                image = image.permute(0, 3, 1, 2)

            new_observation[image_key] = image

        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "image_resolution": self.image_resolution,
            "image_keys": list(self.image_keys) if self.image_keys is not None else None,
        }


def _visual_image_keys(config: DistributionalVFConfig) -> tuple[str, ...]:
    return tuple(
        feature_name
        for feature_name, feature in config.input_features.items()
        if feature.type == FeatureType.VISUAL
    )


def make_distributional_vf_pre_post_processors(
    config: DistributionalVFConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create pre/post processors for the distributional value function.

    Preprocessor steps:
        1. Rename observations (no-op by default)
        2. Add a batch dimension
        3. Normalize features (images use identity, so they stay in [0, 1])
        4. Format task prompt: "Task: {task}."
        5. Tokenize with the PaliGemma tokenizer
        6. Resize-with-pad and scale images to [-1, 1] for SigLIP
        7. Move tensors to the configured device

    Training targets (mc_return, is_terminal) are not processed here.
    The model reads them directly from the batch in forward().

    The postprocessor is a no-op because the value function does not need
    action postprocessing.
    """
    image_keys = _visual_image_keys(config)

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
            DistributionalVFPrepareTaskPromptStep(),
            TokenizerProcessorStep(
                tokenizer_name=PALIGEMMA_TOKENIZER_NAME,
                max_length=config.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
            ),
            DistributionalVFImagePreprocessorStep(
                image_resolution=config.image_resolution,
                image_keys=image_keys or None,
            ),
            DeviceProcessorStep(device=config.device or "cpu"),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline(
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
    )
    return preprocessor, postprocessor

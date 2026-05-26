# Copyright 2026 The OpenEAI team and The HuggingFace Inc. team. All rights reserved.
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

"""OpenEAI-VLA pre/post processor pipelines for LeRobot.

OpenEAI-VLA uses Qwen3VLProcessor which handles both images and text jointly.

Processor pipeline order (pre):
    1. RenameObservationsProcessorStep — map dataset keys to standard names
    2. AddBatchDimensionProcessorStep — add batch dim
    3. Qwen3VLProcessorStep — joint image+text tokenization ->
       pixel_values, image_grid_thw, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    4. DeviceProcessorStep — move to device
    5. NormalizerProcessorStep — normalize state/action

Processor pipeline order (post):
    1. UnnormalizerProcessorStep — unnormalize action
    2. DeviceProcessorStep — move to CPU
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, RobotObservation, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available

from .configuration_openeai import OpenEAIVLAConfig

if _transformers_available:
    from transformers import Qwen3VLProcessor


# Mapping from Qwen3VLProcessor output keys to standard LeRobot observation keys.
_QWEN_OUTPUT_KEY_MAP: dict[str, str] = {
    "input_ids": OBS_LANGUAGE_TOKENS,
    "attention_mask": OBS_LANGUAGE_ATTENTION_MASK,
    "pixel_values": "pixel_values",
    "image_grid_thw": "image_grid_thw",
}


@dataclass
class Qwen3VLProcessorStep(ProcessorStep):
    """Process images + language through Qwen3VLProcessor.

    This step loads Qwen3VLProcessor from the config's qwen_path, applies
    the VLA template to images and task text, and writes the following keys
    into the observation:
        - OBS_LANGUAGE_TOKENS (from input_ids)
        - OBS_LANGUAGE_ATTENTION_MASK (from attention_mask)
        - pixel_values
        - image_grid_thw
    """

    processor_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    max_length: int = 128
    padding: str = "longest"
    padding_side: str = "left"
    truncation: bool = True
    _processor: Any = field(default=None, init=False, repr=False)

    VLA_TEMPLATE: ClassVar[str] = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    IMAGE_TOKEN: ClassVar[str] = "<|vision_start|><|image_pad|><|vision_end|>"  # nosec: B105

    def __post_init__(self):
        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is required. "
                "Please install it with `pip install 'lerobot[transformers-dep]'`."
            )
        self._processor = Qwen3VLProcessor.from_pretrained(self.processor_name)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return transition

        new_observation = self._process_observation(observation, transition)
        new_transition = dict(transition)
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def _process_observation(
        self, observation: RobotObservation, transition: EnvTransition
    ) -> RobotObservation:
        cam_keys = sorted([k for k in observation if k.startswith(OBS_IMAGES)])
        if not cam_keys and OBS_IMAGE in observation:
            cam_keys = [OBS_IMAGE]

        task = self._get_task(transition)
        target_device = self._detect_device(transition)

        # Text-only branch
        if not cam_keys:
            if task is None:
                task = [""]
            processed = self._processor(
                text=task,
                return_tensors="pt",
                padding=self.padding,
                max_length=self.max_length if self.padding == "max_length" else None,
                truncation=self.truncation,
                padding_side=self.padding_side,
            )
            return self._merge_outputs(observation, processed, target_device)

        # Image + text branch
        first_img = observation[cam_keys[0]]
        batch_size = first_img.shape[0] if first_img.ndim >= 4 else 1

        if task is None:
            task = [""] * batch_size

        images_per_batch: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        for key in cam_keys:
            img = observation[key]
            if not isinstance(img, torch.Tensor):
                continue
            if img.is_floating_point():
                img = (img.clamp(0, 1) * 255).to(torch.uint8)
            for b in range(batch_size):
                if img.ndim == 5:
                    for c in range(img.shape[1]):
                        images_per_batch[b].append(img[b, c].permute(1, 2, 0))
                elif img.ndim == 4:
                    images_per_batch[b].append(img[b].permute(1, 2, 0))

        all_images: list[torch.Tensor] = []
        texts: list[str] = []
        for b in range(batch_size):
            imgs = images_per_batch[b]
            all_images.extend(imgs)
            image_placeholders = "".join([self.IMAGE_TOKEN] * len(imgs))
            prompt = self.VLA_TEMPLATE.format(image_placeholders + task[b])
            texts.append(prompt)

        processed = self._processor(
            images=all_images if all_images else None,
            text=texts,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length if self.padding == "max_length" else None,
            truncation=self.truncation,
            padding_side=self.padding_side,
        )
        return self._merge_outputs(observation, processed, target_device)

    def _merge_outputs(
        self,
        observation: RobotObservation,
        processed: dict[str, Any],
        target_device: torch.device | None,
    ) -> RobotObservation:
        """Map Qwen3VLProcessor outputs to standard LeRobot keys and move to device."""
        result: dict[str, Any] = {}
        for src_key, dst_key in _QWEN_OUTPUT_KEY_MAP.items():
            if src_key not in processed:
                continue
            val = processed[src_key]
            if isinstance(val, torch.Tensor) and target_device is not None:
                val = val.to(target_device)
            result[dst_key] = val

        new_observation = dict(observation)
        new_observation.update(result)
        return new_observation

    @staticmethod
    def _detect_device(transition: EnvTransition) -> torch.device | None:
        """Detect the torch.device from existing tensors in the transition."""
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation:
            for value in observation.values():
                if isinstance(value, torch.Tensor):
                    return value.device
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            return action.device
        return None

    @staticmethod
    def _get_task(transition: EnvTransition) -> list[str] | None:
        """Extract task/prompt from complementary data.

        Priority: "prompt" (client-sent task description) > "task" (dataset/task ID).
        """
        comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if comp_data is None:
            return None
        task = comp_data.get("prompt")
        if task is None:
            task = comp_data.get("task")
        if task is None:
            return None
        if isinstance(task, str):
            return [task]
        if isinstance(task, list) and task:
            return list(task)
        return None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Declare the new observation features produced by this step."""
        obs_features = features[PipelineFeatureType.OBSERVATION]
        if OBS_LANGUAGE_TOKENS not in obs_features:
            obs_features[OBS_LANGUAGE_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        if OBS_LANGUAGE_ATTENTION_MASK not in obs_features:
            obs_features[OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        return features


def make_openeai_pre_post_processors(
    config: OpenEAIVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build pre-processor and post-processor pipelines for OpenEAI-VLA.

    Pre-processor:
        1. Rename observations
        2. Add batch dimension
        3. Qwen3VLProcessor: images + text -> pixel_values, image_grid_thw,
           OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        4. Move to device
        5. Normalize state/action

    Post-processor:
        1. Unnormalize action
        2. Move to CPU
    """
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        Qwen3VLProcessorStep(
            processor_name=config.qwen_path,
            max_length=config.tokenizer_max_length,
            padding=config.pad_language_to,
            padding_side="left",
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
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

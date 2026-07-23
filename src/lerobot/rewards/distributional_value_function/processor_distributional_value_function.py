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

Prepares inputs for V^{pi_ref}(o_t, l):
1. Resize multi-camera images to 448x448 (with aspect-preserving padding)
2. Normalize images from [0,1] → [-1,1] (SigLIP standard)
3. Handle missing cameras (placeholder + mask)
4. Format task prompt: ``"Task: {task}."``
5. Tokenize with Gemma3 tokenizer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
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
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_distributional_value_function import DistributionalVFConfig

# Keys used by the image processor to store per-camera validity masks.
IMAGE_MASK_SUFFIX = ".mask"


def resize_with_pad_torch(
    images: Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> Tensor:
    """Resize images preserving aspect ratio, padding with black.

    Matches ``resize_with_pad_torch`` in PI0/PI05/PI0-FAST.

    Args:
        images: [*b, h, w, c] or [*b, c, h, w] tensor.
        height: Target height.
        width: Target width.
        mode: Interpolation mode.

    Returns:
        Resized and padded tensor with same shape format as input.
    """
    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    batch_size, channels, cur_height, cur_width = images.shape

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)

    return padded_images


@ProcessorStepRegistry.register(name="distributional_vf_image_preprocessor")
@dataclass
class DistributionalVFImagePreprocessorStep(ProcessorStep):
    """Resize and normalize multi-camera images for the VF.

    Expects LeRobot's standard float image range [0, 1].
    Produces [B, 3, H, W] tensors in [-1, 1] for each camera, plus boolean
    masks indicating which cameras are present. Missing cameras get a black
    placeholder image and mask=False.
    """

    image_resolution: tuple[int, int] = (448, 448)
    image_keys: tuple[str, ...] = ()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION, {}))

        for key in self.image_keys:
            if key in observation:
                img = observation[key]
                if img.dtype != torch.float32:
                    img = img.to(torch.float32)

                is_channels_first = img.shape[1] == 3
                if is_channels_first:
                    img = img.permute(0, 2, 3, 1)  # BCHW → BHWC

                # Gemma3's SigLIP vision tower expects [-1, 1].
                img = img * 2.0 - 1.0

                if img.shape[1:3] != self.image_resolution:
                    img = resize_with_pad_torch(img, *self.image_resolution)

                observation[key] = img.permute(0, 3, 1, 2)  # BHWC → BCHW
                observation[key + IMAGE_MASK_SUFFIX] = torch.ones(
                    img.shape[0], dtype=torch.bool, device=img.device
                )
            else:
                bsize = self._infer_batch_size(observation)
                h, w = self.image_resolution
                observation[key] = torch.full((bsize, 3, h, w), -1.0)
                observation[key + IMAGE_MASK_SUFFIX] = torch.zeros(bsize, dtype=torch.bool)

        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def _infer_batch_size(self, observation: dict) -> int:
        for v in observation.values():
            if isinstance(v, Tensor) and v.ndim >= 2:
                return v.shape[0]
        return 1

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "image_resolution": self.image_resolution,
            "image_keys": self.image_keys,
        }


@ProcessorStepRegistry.register(name="distributional_vf_prepare_task_prompt")
@dataclass
class DistributionalVFPrepareTaskPromptStep(ProcessorStep):
    """Format the task string: ``"Task: {task}."``"""

    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        if isinstance(tasks, str):
            tasks = [tasks]

        full_prompts = []
        for task in tasks:
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            full_prompts.append(f"Task: {cleaned_text}.")

        new_complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        new_complementary_data[self.task_key] = full_prompts
        transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"task_key": self.task_key}


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
        3. Normalize features (identity for images)
        4. Resize + normalize images → [B, 3, 448, 448] in [-1, 1]
        5. Format task prompt: ``"Task: {task}."``
        6. Tokenize with Gemma3 tokenizer
        7. Move tensors to the configured device

    Training targets (mc_return, is_terminal) are not processed here.
    The postprocessor is a no-op (value function does not produce actions).
    """
    image_keys = tuple(k for k, v in config.input_features.items() if v.type == FeatureType.VISUAL)

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
            DistributionalVFImagePreprocessorStep(
                image_resolution=config.image_resolution,
                image_keys=image_keys,
            ),
            DistributionalVFPrepareTaskPromptStep(),
            TokenizerProcessorStep(
                tokenizer_name=config.vlm_pretrained_path or config.gemma3_path,
                max_length=config.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
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

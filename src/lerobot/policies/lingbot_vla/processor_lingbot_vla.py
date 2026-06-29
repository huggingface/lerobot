# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""Pre/post-processing pipeline for the LingBot-VLA policy.

The pre-processor turns a raw LeRobot observation into the model-ready tensors
that ``LingbotVLAPolicy`` consumes:

    - ``images``      (B, num_views, num_patches, patch_dim) Qwen2.5-VL patchified pixels
    - ``img_masks``   (B, num_views) per-view validity mask
    - ``lang_tokens`` (B, L) tokenized instruction
    - ``lang_masks``  (B, L) instruction attention mask

State and action stay at their real (dataset) dimensionality through the
pipeline; the policy pads them to the unified 75-dim slots internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available

from .configuration_lingbot_vla import LingbotVLAConfig
from .utils import resize_with_pad

if _transformers_available:
    from transformers import AutoProcessor, AutoTokenizer
else:
    AutoProcessor = None
    AutoTokenizer = None


@ProcessorStepRegistry.register(name="lingbot_vla_task_processor")
class LingbotVLATaskProcessor(ComplementaryDataProcessorStep):
    """Ensures a task instruction string is always present."""

    def complementary_data(self, complementary_data):
        new_complementary_data = dict(complementary_data)
        task = complementary_data.get("task")
        if task is None:
            new_complementary_data["task"] = "Execute the robot action."
        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="lingbot_vla_images_processor")
class LingbotVLAImagesProcessorStep(ObservationProcessorStep):
    """Resize + Qwen2.5-VL patchify the camera views into ``images`` / ``img_masks``."""

    tokenizer_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_keys: list[str] = field(default_factory=list)
    resize_size: tuple[int, int] = (224, 224)

    image_processor: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not _transformers_available:
            raise ImportError(
                "transformers is required for LingbotVLAImagesProcessorStep. "
                "Install it with `pip install 'lerobot[lingbot]'`."
            )
        processor = AutoProcessor.from_pretrained(
            self.tokenizer_path, padding_side="right", trust_remote_code=True
        )
        self.image_processor = processor.image_processor

    def _patchify_view(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B, C, H, W) in [0, 1] -> (B, num_patches, patch_dim)
        img = img.to(torch.float32) * 255.0
        img = resize_with_pad(img, self.resize_size[0], self.resize_size[1], pad_value=0)
        patches = []
        for b in range(img.shape[0]):
            pixel_values = self.image_processor(img[b])["pixel_values"]
            if not isinstance(pixel_values, torch.Tensor):
                pixel_values = torch.as_tensor(np.asarray(pixel_values))
            patches.append(pixel_values)
        return torch.stack(patches, dim=0)

    def observation(self, observation):
        new_observation = dict(observation)
        present_keys = [k for k in self.image_keys if k in observation]
        if not present_keys:
            raise ValueError(f"None of the configured image keys {self.image_keys} found in the observation.")

        ref = observation[present_keys[0]]
        batch_size = ref.shape[0]

        view_tensors, masks = [], []
        for key in self.image_keys:
            if key in observation:
                view_tensors.append(self._patchify_view(observation[key]))
                masks.append(torch.ones(batch_size, dtype=torch.bool))
            else:
                view_tensors.append(torch.zeros_like(view_tensors[0]))
                masks.append(torch.zeros(batch_size, dtype=torch.bool))
            new_observation.pop(key, None)

        new_observation["images"] = torch.stack(view_tensors, dim=1)  # (B, n_views, L, patch_dim)
        new_observation["img_masks"] = torch.stack(masks, dim=1)  # (B, n_views)
        return new_observation

    def get_config(self) -> dict[str, Any]:
        return {
            "tokenizer_path": self.tokenizer_path,
            "image_keys": list(self.image_keys),
            "resize_size": list(self.resize_size),
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="lingbot_vla_language_processor")
class LingbotVLALanguageProcessorStep(ObservationProcessorStep):
    """Tokenize the instruction into ``lang_tokens`` / ``lang_masks`` (Qwen2.5-VL tokenizer)."""

    tokenizer_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_length: int = 72

    tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not _transformers_available:
            raise ImportError(
                "transformers is required for LingbotVLALanguageProcessorStep. "
                "Install it with `pip install 'lerobot[lingbot]'`."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, padding_side="right", trust_remote_code=True
        )

    def observation(self, observation):
        complementary_data = self.transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        task = complementary_data.get("task")
        if task is None:
            raise ValueError("A 'task' instruction is required but was not found in complementary data.")
        if isinstance(task, str):
            task = [task]

        # Match the LingBot-VLA training prompt format: "<bos>{task}\n".
        prompts = [p if p.startswith("<bos>") else f"<bos>{p}" for p in task]
        prompts = [p if p.endswith("\n") else f"{p}\n" for p in prompts]

        tokenized = self.tokenizer(
            prompts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            padding_side="right",
            return_tensors="pt",
        )

        new_observation = dict(observation)
        new_observation["lang_tokens"] = tokenized["input_ids"]
        new_observation["lang_masks"] = tokenized["attention_mask"].to(dtype=torch.bool)
        return new_observation

    def get_config(self) -> dict[str, Any]:
        return {"tokenizer_path": self.tokenizer_path, "max_length": self.max_length}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if "lang_tokens" not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION]["lang_tokens"] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        if "lang_masks" not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION]["lang_masks"] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        return features


def make_lingbot_vla_pre_post_processors(
    config: LingbotVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the LingBot-VLA pre- and post-processing pipelines."""

    image_keys = [key for key, feat in config.input_features.items() if feat.type == FeatureType.VISUAL]

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        LingbotVLATaskProcessor(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        LingbotVLAImagesProcessorStep(
            tokenizer_path=config.tokenizer_path,
            image_keys=image_keys,
            resize_size=tuple(config.resize_imgs_with_padding),
        ),
        LingbotVLALanguageProcessorStep(
            tokenizer_path=config.tokenizer_path,
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
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

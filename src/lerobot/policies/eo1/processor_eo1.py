#!/usr/bin/env python

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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_eo1 import EO1Config

if TYPE_CHECKING or _transformers_available:
    from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
else:
    Qwen2_5_VLProcessor = None

SYSTEM_MESSAGE = "You are a helpful physical assistant."

# EO-1 special tokens
ACTION_START_TOKEN = "<|action_start|>"  # nosec B105
DEFAULT_ACTION_TOKEN = "<|action_pad|>"  # nosec B105
ACTION_END_TOKEN = "<|action_end|>"  # nosec B105
STATE_START_TOKEN = "<|state_start|>"  # nosec B105
DEFAULT_STATE_TOKEN = "<|state_pad|>"  # nosec B105
STATE_END_TOKEN = "<|state_end|>"  # nosec B105
TASK_VLA_TOKEN = "<|vla|>"  # nosec B105

EO1_SPECIAL_TOKENS = [
    ACTION_START_TOKEN,
    DEFAULT_ACTION_TOKEN,
    ACTION_END_TOKEN,
    STATE_START_TOKEN,
    DEFAULT_STATE_TOKEN,
    STATE_END_TOKEN,
    TASK_VLA_TOKEN,
]


@dataclass
@ProcessorStepRegistry.register(name="eo1_conversation_template_processor")
class EO1ConversationTemplateStep(ComplementaryDataProcessorStep):
    input_features: dict[str, PolicyFeature] | dict[str, dict[str, Any]]
    chunk_size: int

    _image_keys: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        # Robust JSON deserialization handling (guard empty maps).
        if self.input_features:
            first_val = next(iter(self.input_features.values()))
            if isinstance(first_val, dict):
                reconstructed = {}
                for key, ft_dict in self.input_features.items():
                    reconstructed[key] = PolicyFeature(
                        type=FeatureType(ft_dict["type"]), shape=tuple(ft_dict["shape"])
                    )
                self.input_features = reconstructed

        self._image_keys = [
            key for key, value in self.input_features.items() if value.type == FeatureType.VISUAL
        ]

    def complementary_data(self, complementary_data):
        tasks = complementary_data.get("task")
        if tasks is None:
            raise ValueError("Task is required for EO1ConversationTemplateStep.")

        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required for EO1ConversationTemplateStep.")

        if OBS_STATE in observation and observation[OBS_STATE].shape[0] != len(tasks):
            raise ValueError("Batch size mismatch between observation.state and task list.")

        # LeRobot visual observations reach in processor as float32 tensors in [0, 1].
        # Convert to uint8 in [0, 255] to meet the input requirement of Qwen2.5-VL-3B-Instruct.
        images = {
            key: observation[key].clamp(0, 1).mul(255.0).round().to(torch.uint8) for key in self._image_keys
        }
        messages = []
        for i in range(len(tasks)):
            content = [
                *[{"type": "image", "image": images[key][i]} for key in self._image_keys],
                {
                    "type": "text",
                    "text": (
                        f"{STATE_START_TOKEN}{DEFAULT_STATE_TOKEN}{STATE_END_TOKEN}{tasks[i]}{TASK_VLA_TOKEN}"
                    ),
                },
            ]
            messages.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                    {"role": "user", "content": content},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{ACTION_START_TOKEN}{DEFAULT_ACTION_TOKEN * self.chunk_size}{ACTION_END_TOKEN}",
                            }
                        ],
                    },
                ]
            )

        complementary_data["messages"] = messages

        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step only materializes EO1-specific message objects in complementary_data.
        PipelineFeatureType tracks only ACTION and OBSERVATION, so there is no static
        feature contract change to record here.
        """
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "input_features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.input_features.items()
            },
            "chunk_size": self.chunk_size,
        }


@dataclass
@ProcessorStepRegistry.register(name="eo1_qwen_processor")
class EO1QwenProcessorStep(ComplementaryDataProcessorStep):
    processor_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_min_pixels: int | None = 64 * 28 * 28
    image_max_pixels: int | None = 128 * 28 * 28
    use_fast_processor: bool = False

    _processor: Qwen2_5_VLProcessor | None = field(default=None, init=False, repr=False)
    _state_token_id: int | None = field(default=None, init=False, repr=False)
    _action_token_id: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        require_package("transformers", extra="eo1")
        self._processor = Qwen2_5_VLProcessor.from_pretrained(
            self.processor_name,
            use_fast=self.use_fast_processor,
        )
        self._processor.tokenizer.add_tokens(EO1_SPECIAL_TOKENS, special_tokens=True)
        self._state_token_id = self._processor.tokenizer.convert_tokens_to_ids(DEFAULT_STATE_TOKEN)
        self._action_token_id = self._processor.tokenizer.convert_tokens_to_ids(DEFAULT_ACTION_TOKEN)

    def complementary_data(self, complementary_data):
        messages = complementary_data.pop("messages", None)
        if messages is None:
            raise ValueError("Messages are required for EO1QwenProcessorStep.")

        # Rollout batches use left padding so action spans stay aligned across samples.
        # Supervised batches use right padding to match standard training collation.
        padding_side = "right" if self.transition.get(TransitionKey.ACTION) is not None else "left"

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            padding_side=padding_side,
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        complementary_data["input_ids"] = inputs["input_ids"]
        complementary_data["pixel_values"] = inputs["pixel_values"]
        complementary_data["image_grid_thw"] = inputs["image_grid_thw"]
        complementary_data["attention_mask"] = inputs["attention_mask"]
        complementary_data["mm_token_type_ids"] = inputs["mm_token_type_ids"]
        complementary_data["state_token_id"] = self._state_token_id
        complementary_data["action_token_id"] = self._action_token_id

        return complementary_data

    def get_config(self) -> dict[str, Any]:
        return {
            "processor_name": self.processor_name,
            "image_min_pixels": self.image_min_pixels,
            "image_max_pixels": self.image_max_pixels,
            "use_fast_processor": self.use_fast_processor,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step only converts the messages to the model input format.
        """
        return features


def make_eo1_pre_post_processors(
    config: EO1Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build pre/post processor pipelines for EO1."""

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        EO1ConversationTemplateStep(input_features=config.input_features, chunk_size=config.chunk_size),
        EO1QwenProcessorStep(
            processor_name=config.vlm_base,
            image_min_pixels=config.image_min_pixels,
            image_max_pixels=config.image_max_pixels,
            use_fast_processor=config.use_fast_processor,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
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

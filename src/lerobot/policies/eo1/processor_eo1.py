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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs import recipe as recipe_module
from lerobot.configs.recipe import TrainingRecipe
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.processor.render_messages_processor import RenderMessagesStep
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE
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
        recipe_messages = complementary_data.get("messages")
        recipe_streams = complementary_data.get("message_streams")
        recipe_targets = complementary_data.get("target_message_indices")
        messages = []
        adjusted_targets: list[list[int]] = []
        for i in range(len(tasks)):
            if recipe_messages is not None:
                row_messages = recipe_messages[i] if len(tasks) > 1 else recipe_messages[0]
                row_streams = recipe_streams[i] if len(tasks) > 1 else recipe_streams[0]
                row_targets = recipe_targets[i] if len(tasks) > 1 else recipe_targets[0]
                if not isinstance(row_messages, list) or not isinstance(row_streams, list):
                    raise TypeError("EO-1 recipe messages and streams must be batched lists.")

                rendered = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]}]
                image_blocks = [{"type": "image", "image": images[key][i]} for key in self._image_keys]
                injected_images = False
                inserted_observation_turn = not any(
                    str(message.get("role", "user")) == "user" for message in row_messages
                )
                if inserted_observation_turn:
                    rendered.append(
                        {
                            "role": "user",
                            "content": [
                                *image_blocks,
                                {
                                    "type": "text",
                                    "text": f"{STATE_START_TOKEN}{DEFAULT_STATE_TOKEN}{STATE_END_TOKEN}",
                                },
                            ],
                        }
                    )
                    injected_images = True
                for message in row_messages:
                    converted = dict(message)
                    tool_calls = converted.pop("tool_calls", None)
                    content = converted.get("content", "")
                    if isinstance(content, str):
                        blocks = [{"type": "text", "text": content}]
                    elif isinstance(content, list):
                        blocks = []
                        for block in content:
                            block = dict(block)
                            if block.get("type") == "image" and "feature" in block:
                                feature = block.pop("feature")
                                if feature in images:
                                    block["image"] = images[feature][i]
                            blocks.append(block)
                    else:
                        blocks = [{"type": "text", "text": "" if content is None else str(content)}]
                    say_text = "".join(f"<say>{value}</say>" for value in _say_tool_texts(tool_calls))
                    if say_text:
                        blocks.append({"type": "text", "text": say_text})
                    if converted.get("role") == "user" and not injected_images:
                        blocks = [
                            *image_blocks,
                            {
                                "type": "text",
                                "text": f"{STATE_START_TOKEN}{DEFAULT_STATE_TOKEN}{STATE_END_TOKEN}",
                            },
                            *blocks,
                        ]
                        injected_images = True
                    converted["content"] = blocks
                    rendered.append(converted)

                predicts_action = any(stream == "low_level" for stream in row_streams)
                if predicts_action:
                    rendered.extend(
                        [
                            {
                                "role": "user",
                                "content": [
                                    *([] if injected_images else image_blocks),
                                    {
                                        "type": "text",
                                        "text": f"{tasks[i]}{TASK_VLA_TOKEN}",
                                    },
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"{ACTION_START_TOKEN}"
                                            f"{DEFAULT_ACTION_TOKEN * self.chunk_size}"
                                            f"{ACTION_END_TOKEN}"
                                        ),
                                    }
                                ],
                            },
                        ]
                    )
                messages.append(rendered)
                target_offset = 2 if inserted_observation_turn else 1
                adjusted_targets.append([int(index) + target_offset for index in row_targets])
                continue

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
            adjusted_targets.append([])

        complementary_data["messages"] = messages
        complementary_data["target_message_indices"] = adjusted_targets

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
    tokenizer_max_length: int = 1000

    _processor: Qwen2_5_VLProcessor | None = field(default=None, init=False, repr=False)
    _state_token_id: int | None = field(default=None, init=False, repr=False)
    _action_token_id: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        require_package("transformers", extra="eo1")
        self._processor = Qwen2_5_VLProcessor.from_pretrained(
            self.processor_name,
            use_fast=self.use_fast_processor,
            fix_mistral_regex=True,
        )
        self._processor.tokenizer.add_tokens(EO1_SPECIAL_TOKENS, special_tokens=True)
        self._state_token_id = self._processor.tokenizer.convert_tokens_to_ids(DEFAULT_STATE_TOKEN)
        self._action_token_id = self._processor.tokenizer.convert_tokens_to_ids(DEFAULT_ACTION_TOKEN)

    def complementary_data(self, complementary_data):
        messages = complementary_data.pop("messages", None)
        if messages is None:
            raise ValueError("Messages are required for EO1QwenProcessorStep.")
        target_message_indices = complementary_data.pop("target_message_indices", None)
        has_text_targets = bool(
            target_message_indices and any(bool(indices) for indices in target_message_indices)
        )

        # Rollout batches use left padding so action spans stay aligned across samples.
        # Supervised batches use right padding to match standard training collation.
        padding_side = "right" if self.transition.get(TransitionKey.ACTION) is not None else "left"

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "padding_side": padding_side,
                "min_pixels": self.image_min_pixels,
                "max_pixels": self.image_max_pixels,
                "truncation": True,
                "max_length": self.tokenizer_max_length,
                "return_offsets_mapping": has_text_targets,
            },
        )

        complementary_data["input_ids"] = inputs["input_ids"]
        complementary_data["pixel_values"] = inputs["pixel_values"]
        complementary_data["image_grid_thw"] = inputs["image_grid_thw"]
        complementary_data["attention_mask"] = inputs["attention_mask"]
        complementary_data["mm_token_type_ids"] = inputs["mm_token_type_ids"]
        complementary_data["state_token_id"] = self._state_token_id
        complementary_data["action_token_id"] = self._action_token_id
        if has_text_targets:
            complementary_data["text_labels"] = _targeted_assistant_labels(
                self._processor.tokenizer,
                messages,
                target_message_indices,
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs.pop("offset_mapping"),
                self._state_token_id,
                self._action_token_id,
            )

        return complementary_data

    def get_config(self) -> dict[str, Any]:
        return {
            "processor_name": self.processor_name,
            "image_min_pixels": self.image_min_pixels,
            "image_max_pixels": self.image_max_pixels,
            "use_fast_processor": self.use_fast_processor,
            "tokenizer_max_length": self.tokenizer_max_length,
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

    steps = make_default_policy_processor_steps(config, dataset_stats)

    language_steps: list[ProcessorStep] = []
    if config.recipe_path:
        language_steps.append(RenderMessagesStep(recipe=_load_recipe(config.recipe_path)))

    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.normalize,
        *language_steps,
        EO1ConversationTemplateStep(input_features=config.input_features, chunk_size=config.chunk_size),
        EO1QwenProcessorStep(
            processor_name=config.vlm_base,
            image_min_pixels=config.image_min_pixels,
            image_max_pixels=config.image_max_pixels,
            use_fast_processor=config.use_fast_processor,
            tokenizer_max_length=config.tokenizer_max_length,
        ),
        steps.to_device,
    ]

    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


def _load_recipe(path_str: str) -> TrainingRecipe:
    path = Path(path_str)
    if not path.is_absolute() and not path.exists():
        candidate = Path(recipe_module.__file__).resolve().parent / path
        if candidate.exists():
            path = candidate
    return TrainingRecipe.from_yaml(path)


def _say_tool_texts(tool_calls: Any) -> list[str]:
    values = []
    for call in tool_calls or []:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        if function.get("name") != "say":
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                arguments = {}
        if isinstance(arguments, dict) and arguments.get("text"):
            values.append(str(arguments["text"]))
    return values


def _targeted_assistant_labels(
    tokenizer: Any,
    messages: list[list[dict[str, Any]]],
    target_message_indices: list[list[int]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    offsets: torch.Tensor,
    state_token_id: int,
    action_token_id: int,
) -> torch.Tensor:
    """Label only targeted assistant payloads and their closing ``<|im_end|>`` tokens."""
    labels = torch.full_like(input_ids, -100)
    for row, (row_messages, target_indices) in enumerate(zip(messages, target_message_indices, strict=True)):
        rendered = tokenizer.decode(input_ids[row], skip_special_tokens=False)
        cursor = 0
        spans: dict[int, tuple[int, int]] = {}
        for message_index, message in enumerate(row_messages):
            marker = f"<|im_start|>{message.get('role', 'user')}\n"
            marker_start = rendered.find(marker, cursor)
            if marker_start < 0:
                raise ValueError(f"Could not locate EO-1 message {message_index} in rendered prompt.")
            payload_start = marker_start + len(marker)
            payload_end = rendered.find("<|im_end|>", payload_start)
            if payload_end < 0:
                raise ValueError(f"EO-1 message {message_index} has no closing <|im_end|> token.")
            spans[message_index] = (payload_start, payload_end + len("<|im_end|>"))
            cursor = payload_end + len("<|im_end|>")

        for message_index in target_indices:
            if message_index not in spans:
                raise ValueError(f"EO-1 target message index {message_index} is out of range.")
            start, end = spans[message_index]
            overlap = (offsets[row, :, 1] > start) & (offsets[row, :, 0] < end) & attention_mask[row].bool()
            labels[row, overlap] = input_ids[row, overlap]

    labels[labels == state_token_id] = -100
    labels[labels == action_token_id] = -100
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    return labels

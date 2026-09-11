#!/usr/bin/env python

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

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenderRuntimeMessagesStep,
    RenderTrainingMessagesStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION, MESSAGES_RENDERED, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package
from lerobot.utils.language import normalize_semantic_messages, semantic_message_content_text

from .configuration_wall_x import WallXConfig
from .constant import (
    GENERATE_SUBTASK_RATIO,
    MODEL_TYPE,
    PRIORITY_ORDER,
    WALL_X_GENERATION_PROMPT_IDS,
    WALL_X_PROMPT_SEGMENTS,
)
from .utils import (
    get_wallx_normal_text,
    img_key_mapping,
    prepare_wall_x_image_inputs,
    preprocesser_call,
    process_grounding_points,
    replace_action_token,
)

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None


def make_wall_x_pre_post_processors(
    config: WallXConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the Wall-X policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations
    2. Adding a batch dimension
    4. Normalizing input and output features based on dataset statistics
    5. Moving all data to the specified device

    The post-processing pipeline handles the model's output by:
    1. Unnormalizing the output actions to their original scale
    2. Moving data to the CPU

    Args:
        config: The configuration object for the Wall-X policy
        dataset_stats: A dictionary of statistics for normalization

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines
    """

    steps = make_default_policy_processor_steps(config, dataset_stats)

    input_steps = [
        RenderRuntimeMessagesStep(config.recipe),
        RenderTrainingMessagesStep(config.recipe),
        steps.rename_observations,
        WallXTaskProcessor(),  # Process task description
        steps.add_batch_dim,
        steps.normalize,
        WallXPromptProcessorStep(image_keys=list(config.image_features), chunk_size=config.chunk_size),
        WallXTokenizerStep(
            processor_name=config.pretrained_name_or_path,
            processor_revision=config.pretrained_revision,
            action_tokenizer_name=config.action_tokenizer_path,
            image_keys=list(config.image_features),
            chunk_size=config.chunk_size,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            output_action_dim=config.output_features[ACTION].shape[0],
            tokenizer_max_length=config.tokenizer_max_length,
            use_fast_tokenizer=config.use_fast_tokenizer,
        ),
        steps.to_device,
    ]

    output_steps = [
        steps.unnormalize,
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


@ProcessorStepRegistry.register(name="wall_x_task_processor")
class WallXTaskProcessor(ComplementaryDataProcessorStep):
    """
    A processor step that ensures the task description is properly formatted for Wall-X.

    This step handles task preprocessing similar to Qwen-VL requirements.
    """

    def complementary_data(self, complementary_data):
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            # Provide default task if none specified
            complementary_data["task"] = "Execute the robot action."
            return complementary_data

        new_complementary_data = dict(complementary_data)

        # Handle both string and list of strings
        if isinstance(task, str):
            # Single string: ensure proper formatting
            if not task.endswith("."):
                new_complementary_data["task"] = f"{task}."
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # List of strings: format each
            new_complementary_data["task"] = [t if t.endswith(".") else f"{t}." for t in task]

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="wall_x_prompt")
class WallXPromptProcessorStep(ComplementaryDataProcessorStep):
    """Render Wall-X prompts and identify their explicitly supervised text segments."""

    image_keys: list[str]
    chunk_size: int

    def get_config(self) -> dict[str, Any]:
        return {"image_keys": self.image_keys, "chunk_size": self.chunk_size}

    @staticmethod
    def _batched(value: Any, batch_size: int, name: str) -> list[list[Any]]:
        if not isinstance(value, list):
            raise TypeError(f"{name} must be a list.")
        if len(value) == batch_size and all(isinstance(row, list) for row in value):
            return value
        if batch_size == 1:
            return [value]
        raise ValueError(f"Expected {name} for exactly {batch_size} samples.")

    @staticmethod
    def _message_content(message: dict[str, Any]) -> str:
        text = semantic_message_content_text(message.get("content"))
        say_texts = []
        for call in message.get("tool_calls") or []:
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
                say_texts.append(str(arguments["text"]))
        suffix = "".join(f"<say>{value}</say>" for value in say_texts)
        return f"{text}\n{suffix}" if text and suffix else text or suffix

    @staticmethod
    def _observation_prompt(image_labels: list[str]) -> str:
        prompt = "Observation:"
        for label in image_labels:
            prompt += f" {label}: <|vision_start|><|image_pad|><|vision_end|>"
        return prompt

    def _recipe_segments(
        self,
        messages: list[dict[str, Any]],
        streams: list[str | None],
        targets: list[int],
        task: str,
        image_labels: list[str],
    ) -> tuple[list[dict[str, str | bool]], bool]:
        if len(messages) != len(streams):
            raise ValueError("WALL-X recipe messages and streams must have equal length.")
        target_set = set(targets)
        if any(index < 0 or index >= len(messages) for index in target_set):
            raise ValueError("WALL-X recipe target index is out of range.")

        predicts_action = any(stream == "low_level" for stream in streams)
        segments: list[dict[str, str | bool]] = [
            {"text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "target": False}
        ]
        observation_injected = not any(str(message.get("role", "user")) == "user" for message in messages)
        if observation_injected:
            segments.append(
                {
                    "text": (f"<|im_start|>user\n{self._observation_prompt(image_labels)}<|im_end|>\n"),
                    "target": False,
                }
            )

        for index, message in enumerate(messages):
            role = str(message.get("role", "user"))
            content = self._message_content(message)
            if role == "user" and not observation_injected:
                content = f"{self._observation_prompt(image_labels)}\n{content}"
                observation_injected = True
            segments.append({"text": f"<|im_start|>{role}\n", "target": False})
            segments.append({"text": f"{content}<|im_end|>", "target": index in target_set})
            segments.append({"text": "\n", "target": False})

        if predicts_action:
            segments.append(
                {
                    "text": (
                        f"<|im_start|>user\nInstruction: {task}\n"
                        "Predict the next action in robot action.\nProprioception: <|propri|>\n"
                        "<|im_end|>\n<|im_start|>assistant\n<|action_fast|><|im_end|>\n"
                        + "<|action|>"
                        * self.chunk_size
                    ),
                    "target": False,
                }
            )
        return segments, predicts_action

    def _generation_segments(
        self, messages: list[dict[str, Any]], image_labels: list[str]
    ) -> list[dict[str, str | bool]]:
        native = [dict(message) for message in messages]
        for message in native:
            if str(message.get("role", "user")) == "user":
                message["content"] = f"Instruction: {self._message_content(message)}"
                break
        segments, _ = self._recipe_segments(native, [None] * len(native), [], "", image_labels)
        segments.append({"text": "<|im_start|>assistant\n", "target": False})
        return segments

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        tasks = complementary_data.get("task")
        if not isinstance(tasks, list):
            raise ValueError("WALL-X needs a batch of task strings before prompt rendering.")
        if not all(isinstance(task, str) for task in tasks):
            raise TypeError("WALL-X tasks must be strings.")

        image_labels = img_key_mapping(self.image_keys)
        messages = complementary_data.get(MESSAGES_RENDERED)
        if messages is not None:
            message_batch = normalize_semantic_messages(messages, policy_name="WALL-X", batch_size=len(tasks))
            generation = (
                "message_streams" not in complementary_data
                and "target_message_indices" not in complementary_data
            )
            if generation:
                prompts = [self._generation_segments(row, image_labels) for row in message_batch]
            else:
                streams_batch = self._batched(
                    complementary_data.get("message_streams", []), len(tasks), "message_streams"
                )
                targets_batch = self._batched(
                    complementary_data.get("target_message_indices", []),
                    len(tasks),
                    "target_message_indices",
                )
                prompts = [
                    self._recipe_segments(row, streams, targets, task, image_labels)[0]
                    for row, streams, targets, task in zip(
                        message_batch, streams_batch, targets_batch, tasks, strict=True
                    )
                ]
        else:
            frame_indices = complementary_data.get("frame_index", [0] * len(tasks))
            prompts = []
            for index, task in enumerate(tasks):
                frame_index = (
                    frame_indices[index] if isinstance(frame_indices, list | torch.Tensor) else frame_indices
                )
                text, _ = get_wallx_normal_text(
                    {"instruction": task},
                    self.chunk_size,
                    frame_index,
                    PRIORITY_ORDER,
                    self.image_keys,
                    generate_subtask_ratio=GENERATE_SUBTASK_RATIO,
                )
                prompts.append([{"text": text, "target": False}])

        complementary_data[WALL_X_PROMPT_SEGMENTS] = prompts
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="wall_x_tokenizer")
class WallXTokenizerStep(ProcessorStep):
    """Build WALL-X token/image tensors for action, training, and text requests."""

    processor_name: str
    image_keys: list[str]
    chunk_size: int
    max_state_dim: int
    max_action_dim: int
    output_action_dim: int
    tokenizer_max_length: int = 768
    use_fast_tokenizer: bool = False
    action_tokenizer_name: str | None = None
    processor_revision: str | None = None
    _processor: Any = field(default=None, init=False, repr=False)
    _action_tokenizer: Any = field(default=None, init=False, repr=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "processor_name": self.processor_name,
            "image_keys": self.image_keys,
            "chunk_size": self.chunk_size,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "output_action_dim": self.output_action_dim,
            "tokenizer_max_length": self.tokenizer_max_length,
            "use_fast_tokenizer": self.use_fast_tokenizer,
            "action_tokenizer_name": self.action_tokenizer_name,
            "processor_revision": self.processor_revision,
        }

    def _get_processors(self):
        if self._processor is None:
            require_package("transformers", extra="wallx")
            self._processor = AutoProcessor.from_pretrained(
                self.processor_name,
                revision=self.processor_revision,
                use_fast=True,
            )
            if self.use_fast_tokenizer:
                if self.action_tokenizer_name is None:
                    raise ValueError("Fast WALL-X tokenization requires action_tokenizer_name.")
                self._action_tokenizer = AutoProcessor.from_pretrained(
                    self.action_tokenizer_name, trust_remote_code=True
                )
        return self._processor, self._action_tokenizer

    @staticmethod
    def _texts_and_target_spans(
        prompt_segments: list[list[dict[str, str | bool]]],
        dimensions: tuple[int, int, int, int],
    ) -> tuple[list[str], list[list[tuple[int, int]]]]:
        orig_height, orig_width, resized_height, resized_width = dimensions
        texts: list[str] = []
        target_spans: list[list[tuple[int, int]]] = []
        for row in prompt_segments:
            pieces: list[str] = []
            spans: list[tuple[int, int]] = []
            length = 0
            for segment in row:
                text = segment.get("text")
                target = segment.get("target")
                if not isinstance(text, str) or not isinstance(target, bool):
                    raise TypeError(
                        "WALL-X prompt segments must contain string text and boolean target fields."
                    )
                text = process_grounding_points(
                    text, orig_height, orig_width, resized_height, resized_width, MODEL_TYPE
                )
                pieces.append(text)
                if target:
                    spans.append((length, length + len(text)))
                length += len(text)
            texts.append("".join(pieces))
            target_spans.append(spans)
        return texts, target_spans

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        processor, action_tokenizer = self._get_processors()

        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        batch = {**observation, **complementary}
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            batch[ACTION] = action
        state = batch.get(OBS_STATE)
        if not isinstance(state, torch.Tensor):
            raise ValueError("WALL-X requires tensor observation.state before tokenization.")
        batch_size = state.shape[0]
        image_keys = [key for key in self.image_keys if key in batch]
        if not image_keys:
            raise ValueError("WALL-X requires at least one image before tokenization.")
        image_inputs, dimensions = prepare_wall_x_image_inputs(batch, image_keys)
        orig_height, orig_width, resized_height, resized_width = dimensions[image_keys[-1]]
        prompt_segments = complementary.get(WALL_X_PROMPT_SEGMENTS)
        if not isinstance(prompt_segments, list) or len(prompt_segments) != batch_size:
            raise ValueError(
                "WALL-X needs prompt segments from WallXPromptProcessorStep before tokenization."
            )
        texts, target_spans = self._texts_and_target_spans(
            prompt_segments, (orig_height, orig_width, resized_height, resized_width)
        )

        agent_pos = state.unsqueeze(1) if state.dim() == 2 else state
        agent_pos_mask = (~torch.isnan(agent_pos)).float()
        agent_pos = agent_pos.nan_to_num(nan=0.0)
        if agent_pos.shape[-1] < self.max_state_dim:
            pad = self.max_state_dim - agent_pos.shape[-1]
            agent_pos = torch.nn.functional.pad(agent_pos, (0, pad))
            agent_pos_mask = torch.nn.functional.pad(agent_pos_mask, (0, pad))
        elif agent_pos.shape[-1] > self.max_state_dim:
            raise ValueError("WALL-X state exceeds max_state_dim.")

        if action is not None:
            action = action.unsqueeze(1) if action.dim() == 2 else action
            dof_mask = (~torch.isnan(action)).float()
            action = action.nan_to_num(nan=0.0)
            if action.shape[-1] < self.max_action_dim:
                pad = self.max_action_dim - action.shape[-1]
                action = torch.nn.functional.pad(action, (0, pad))
                dof_mask = torch.nn.functional.pad(dof_mask, (0, pad))
            elif action.shape[-1] > self.max_action_dim:
                raise ValueError("WALL-X action exceeds max_action_dim.")
        else:
            dof_mask = torch.cat(
                (
                    torch.ones(batch_size, self.chunk_size, self.output_action_dim),
                    torch.zeros(batch_size, self.chunk_size, self.max_action_dim - self.output_action_dim),
                ),
                dim=-1,
            )
        texts = replace_action_token(
            texts,
            action,
            action_tokenizer if self.use_fast_tokenizer else None,
            dof_mask,
        )
        inputs = preprocesser_call(
            processor=processor,
            text=texts,
            images=image_inputs,
            videos=None,
            device=state.device,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.tokenizer_max_length,
            target_spans=target_spans if MESSAGES_RENDERED in complementary else None,
        )
        generation_prompt_ids = processor.tokenizer.encode(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        if not generation_prompt_ids:
            raise ValueError("WALL-X tokenizer produced no assistant generation-prompt tokens.")
        action_token_id = processor.tokenizer.convert_tokens_to_ids("<|action|>")
        inputs.update(
            {
                "proprioception": agent_pos,
                "agent_pos_mask": agent_pos_mask,
                "action_chunk": action,
                "dof_mask": dof_mask,
                "moe_token_types": inputs.input_ids == action_token_id,
                "frame_index": complementary.get("frame_index", torch.zeros(batch_size, device=state.device)),
                WALL_X_GENERATION_PROMPT_IDS: torch.tensor(
                    generation_prompt_ids,
                    dtype=inputs.input_ids.dtype,
                    device=inputs.input_ids.device,
                ),
            }
        )
        for key, value in inputs.items():
            complementary[key] = value
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

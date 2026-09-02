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
from lerobot.configs.recipe import language_recipe_enabled
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.policies.language import normalize_semantic_messages, semantic_message_content_text
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenderMessagesStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_wall_x import WallXConfig

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor
else:
    AutoProcessor = None

WALL_X_TEXT_TARGET_START = "<|wall_text_target_start|>"  # nosec B105
WALL_X_TEXT_TARGET_END = "<|wall_text_target_end|>"  # nosec B105
WALL_X_GENERATION_PROMPT_IDS = "wall_x.generation_prompt_ids"


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

    render_training = language_recipe_enabled(
        use_language_recipe=config.use_language_recipe,
        recipe_path=config.recipe_path,
    )
    if render_training and config.recipe is None:
        raise ValueError("WALL-X language training requires a recipe in policy config.")

    input_steps = [
        RenderMessagesStep(config.recipe, render_training=render_training),
        steps.rename_observations,
        steps.add_batch_dim,
        WallXTaskProcessor(),  # Process task description
        steps.normalize,
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
    def _observation_prompt(image_keys: list[str], image_labels: list[str]) -> str:
        del image_keys
        prompt = "Observation:"
        for label in image_labels:
            prompt += f" {label}: <|vision_start|><|image_pad|><|vision_end|>"
        return prompt

    def _recipe_text(
        self,
        messages: list[dict[str, Any]],
        streams: list[str | None],
        targets: list[int],
        task: str,
        image_labels: list[str],
    ) -> tuple[str, bool]:
        if len(messages) != len(streams):
            raise ValueError("WALL-X recipe messages and streams must have equal length.")
        target_set = set(targets)
        if any(index < 0 or index >= len(messages) for index in target_set):
            raise ValueError("WALL-X recipe target index is out of range.")
        predicts_action = any(stream == "low_level" for stream in streams)
        text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        observation_injected = not any(str(message.get("role", "user")) == "user" for message in messages)
        if observation_injected:
            text += f"<|im_start|>user\n{self._observation_prompt(self.image_keys, image_labels)}<|im_end|>\n"
        for index, message in enumerate(messages):
            role = str(message.get("role", "user"))
            content = self._message_content(message)
            if WALL_X_TEXT_TARGET_START in content or WALL_X_TEXT_TARGET_END in content:
                raise ValueError("Recipe content contains a reserved WALL-X target marker.")
            if role == "user" and not observation_injected:
                content = f"{self._observation_prompt(self.image_keys, image_labels)}\n{content}"
                observation_injected = True
            payload = f"{content}<|im_end|>"
            if index in target_set:
                payload = f"{WALL_X_TEXT_TARGET_START}{payload}{WALL_X_TEXT_TARGET_END}"
            text += f"<|im_start|>{role}\n{payload}\n"
        if predicts_action:
            text += (
                f"<|im_start|>user\nInstruction: {task}\n"
                "Predict the next action in robot action.\nProprioception: <|propri|>\n"
                "<|im_end|>\n<|im_start|>assistant\n<|action_fast|><|im_end|>\n"
                + "<|action|>"
                * self.chunk_size
            )
        return text, predicts_action

    def _generation_text(self, messages: list[dict[str, Any]], image_labels: list[str]) -> str:
        native = [dict(message) for message in messages]
        for message in native:
            if str(message.get("role", "user")) == "user":
                message["content"] = f"Instruction: {self._message_content(message)}"
                break
        text, _ = self._recipe_text(native, [None] * len(native), [], "", image_labels)
        return f"{text}<|im_start|>assistant\n"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        processor, action_tokenizer = self._get_processors()
        from .constant import GENERATE_SUBTASK_RATIO, MODEL_TYPE, PRIORITY_ORDER
        from .modeling_wall_x import _prepare_wall_x_image_inputs
        from .utils import (
            get_wallx_normal_text,
            img_key_mapping,
            preprocesser_call,
            process_grounding_points,
            replace_action_token,
        )

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
        image_inputs, dimensions = _prepare_wall_x_image_inputs(batch, image_keys)
        orig_height, orig_width, resized_height, resized_width = dimensions[image_keys[-1]]
        image_labels = img_key_mapping(image_keys)
        tasks = complementary.get("task")
        tasks = [tasks] * batch_size if isinstance(tasks, str) else tasks
        if not isinstance(tasks, list) or len(tasks) != batch_size:
            raise ValueError(f"WALL-X expected exactly {batch_size} task strings.")

        messages = complementary.get("messages")
        texts = []
        if messages is not None:
            message_batch = normalize_semantic_messages(messages, policy_name="WALL-X", batch_size=batch_size)
            generation = (
                "message_streams" not in complementary and "target_message_indices" not in complementary
            )
            if generation:
                texts = [self._generation_text(row, image_labels) for row in message_batch]
            else:
                streams_batch = self._batched(
                    complementary.get("message_streams", []), batch_size, "message_streams"
                )
                targets_batch = self._batched(
                    complementary.get("target_message_indices", []), batch_size, "target_message_indices"
                )
                texts = [
                    self._recipe_text(row, streams, targets, str(task), image_labels)[0]
                    for row, streams, targets, task in zip(
                        message_batch, streams_batch, targets_batch, tasks, strict=True
                    )
                ]
        else:
            for index, task in enumerate(tasks):
                frame_index = complementary.get("frame_index", [0] * batch_size)
                frame = frame_index[index] if isinstance(frame_index, list | torch.Tensor) else frame_index
                text, _ = get_wallx_normal_text(
                    {"instruction": task},
                    self.chunk_size,
                    frame,
                    PRIORITY_ORDER,
                    image_keys,
                    generate_subtask_ratio=GENERATE_SUBTASK_RATIO,
                )
                texts.append(text)
        texts = [
            process_grounding_points(text, orig_height, orig_width, resized_height, resized_width, MODEL_TYPE)
            for text in texts
        ]

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
            targeted_text_only=messages is not None and "message_streams" in complementary,
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

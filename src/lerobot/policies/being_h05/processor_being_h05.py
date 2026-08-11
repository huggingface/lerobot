#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as tvf

from lerobot.configs import NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    ProcessorStep,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.processor.render_messages_processor import RenderMessagesStep
from lerobot.processor.text_generation_processor import RenderGenerationPromptStep

from .configuration_being_h05 import BeingH05Config

STATE_SLOTS = {
    "eef_position": (0, 3),
    "eef_rotation": (3, 6),
    "gripper_qpos": (44, 46),
    "base_position": (70, 73),
    "base_rotation": (73, 76),
}
ACTION_SLOTS = {
    "eef_position": (0, 3),
    "eef_rotation": (3, 6),
    "gripper_position": (18, 19),
    "base_motion": (70, 74),
    "control_mode": (74, 75),
}
_BINARY_ACTION_INDICES = (6, 11)
_BINARY_ACTION_STORAGE_KEY = "being_h05_binary_action"


def pack_named(named: dict[str, torch.Tensor], slots: dict[str, tuple[int, int]], dim: int = 200):
    reference = next(iter(named.values()))
    leading = reference.shape[:-1]
    packed = torch.zeros(*leading, dim, dtype=reference.dtype, device=reference.device)
    valid = torch.zeros(*leading, dim, dtype=torch.bool, device=reference.device)
    for key, (start, end) in slots.items():
        value = named.get(key)
        if value is None:
            continue
        if value.shape[-1] != end - start:
            raise ValueError(f"{key} must be {end - start}D, got {value.shape[-1]}")
        packed[..., start:end] = value
        valid[..., start:end] = True
    return packed, valid


def unpack_action(packed: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: packed[..., start:end] for key, (start, end) in ACTION_SLOTS.items()}


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    return "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def _serialize_message(message: dict[str, Any]) -> dict[str, str]:
    role = message.get("role")
    if role not in {"system", "user", "assistant"}:
        raise ValueError(f"Being-H0.5 does not support the message role {role!r}.")
    content = _message_content_to_text(message.get("content"))
    say_texts = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or function.get("name") != "say":
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                arguments = {}
        if isinstance(arguments, dict) and arguments.get("text"):
            say_texts.append(str(arguments["text"]))
    if say_texts:
        markers = "".join(f"<say>{text}</say>" for text in say_texts)
        content = f"{content}\n{markers}" if content else markers
    return {"role": role, "content": content}


@dataclass
@ProcessorStepRegistry.register(name="being_h05_messages")
class BeingH05MessagesStep(ProcessorStep):
    """Validate recipe messages and retain only the text contract consumed by Being-H0.5."""

    def get_config(self) -> dict[str, Any]:
        return {}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        messages = complementary.get("messages")
        if not messages:
            return transition
        is_batched = not isinstance(messages[0], dict)
        if not is_batched:
            messages = [messages]
        streams = complementary.get("message_streams")
        targets = complementary.get("target_message_indices")
        if streams is None:
            streams = [[] for _ in messages]
        elif not is_batched:
            streams = [streams]
        if targets is None:
            targets = [[] for _ in messages]
        elif not is_batched:
            targets = [targets]
        if len(messages) != len(streams) or len(messages) != len(targets):
            raise ValueError("Being-H0.5 messages, streams, and target indices must have equal batches.")

        serialized_batch = []
        predict_actions = []
        for sample_messages, sample_streams, sample_targets in zip(messages, streams, targets, strict=True):
            if len(sample_messages) != len(sample_streams):
                raise ValueError("Being-H0.5 message streams must align with messages.")
            for target in sample_targets:
                if target < 0 or target >= len(sample_messages):
                    raise ValueError(f"Being-H0.5 target message index {target} is out of bounds.")
                if sample_messages[target].get("role") != "assistant":
                    raise ValueError("Being-H0.5 text targets must be assistant messages.")
            serialized_batch.append([_serialize_message(message) for message in sample_messages])
            predict_actions.append(any(stream == "low_level" for stream in sample_streams))

        complementary["being_h05_messages"] = serialized_batch
        complementary["being_h05_target_message_indices"] = [list(value) for value in targets]
        complementary["being_h05_predict_actions"] = torch.tensor(predict_actions, dtype=torch.bool)
        transition = transition.copy()
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="being_h05_binary_action")
class BeingH05BinaryActionStep(ProcessorStep):
    """Preserve Being-H0.5's raw 0/1 action fields around shared normalization steps."""

    restore: bool = False

    def get_config(self) -> dict[str, Any]:
        return {"restore": self.restore}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        if self.restore:
            binary_action = complementary.pop(_BINARY_ACTION_STORAGE_KEY, None)
            if binary_action is None:
                return transition
            action = action.clone()
            action[..., list(_BINARY_ACTION_INDICES)] = binary_action.to(
                device=action.device, dtype=action.dtype
            )
        else:
            if _BINARY_ACTION_STORAGE_KEY in complementary:
                raise ValueError("Being-H0.5 binary action storage is already populated.")
            complementary[_BINARY_ACTION_STORAGE_KEY] = action[..., list(_BINARY_ACTION_INDICES)].clone()

        transition = transition.copy()
        transition[TransitionKey.ACTION] = action
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="being_h05_semantic_pack")
class BeingH05SemanticPackStep(ProcessorStep):
    image_keys: list[str]
    prompt_template: str
    chunk_size: int
    state_slots: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(STATE_SLOTS))
    action_slots: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(ACTION_SLOTS))

    def get_config(self) -> dict[str, Any]:
        return {
            "image_keys": self.image_keys,
            "prompt_template": self.prompt_template,
            "chunk_size": self.chunk_size,
            "state_slots": self.state_slots,
            "action_slots": self.action_slots,
        }

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        task = complementary.get("task")
        if task is None:
            raise ValueError("Being-H0.5 requires the already-selected LeRobot task string.")
        tasks = [task] if isinstance(task, str) else list(task)
        if not all(isinstance(item, str) for item in tasks):
            raise TypeError("Being-H0.5 task values must be strings.")
        # Audit hook: this value is captured before model-specific formatting.
        complementary["being_h05_raw_task"] = list(tasks)
        complementary["being_h05_prompt"] = [
            self.prompt_template.format(task_description=item, k=self.chunk_size) for item in tasks
        ]

        action = transition.get(TransitionKey.ACTION)
        named = {}
        for semantic in self.state_slots:
            key = f"observation.state.{semantic}"
            if key in observation:
                named[semantic] = observation[key]
        if action is not None and action.shape[-1] == 12:
            named.update(
                {
                    "action.eef_position": action[..., 0:3],
                    "action.eef_rotation": action[..., 3:6],
                    "action.gripper_position": action[..., 6:7],
                    "action.base_motion": action[..., 7:11],
                    "action.control_mode": action[..., 11:12],
                }
            )

        state_values = {key: value for key, value in named.items() if not key.startswith("action.")}
        if not state_values:
            raise ValueError("No named Being-H0.5 state modalities were present.")
        state, state_mask = pack_named(state_values, self.state_slots)
        observation["being_h05.state"] = state
        observation["being_h05.state_valid"] = state_mask

        images = []
        image_present = []
        for key in self.image_keys:
            value = observation.get(key)
            if value is None:
                value = torch.zeros(state.shape[0], 3, 224, 224, dtype=state.dtype, device=state.device)
                image_present.append(torch.zeros(state.shape[0], dtype=torch.bool, device=state.device))
            else:
                if value.ndim != 4:
                    raise ValueError(f"{key} must have shape (B,C,H,W), got {tuple(value.shape)}")
                processed = []
                for frame in value:
                    if frame.is_floating_point():
                        frame = (frame.clamp(0, 1) * 255).round().to(torch.uint8)
                    else:
                        frame = frame.to(torch.uint8)
                    pil = Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())
                    pil = tvf.resize(
                        pil,
                        224,
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    )
                    pil = tvf.center_crop(pil, [224, 224])
                    processed.append(tvf.to_tensor(pil))
                value = torch.stack(processed).to(device=state.device)
                mean = value.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
                std = value.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
                value = (value - mean) / std
                image_present.append(torch.ones(state.shape[0], dtype=torch.bool, device=state.device))
            images.append(value)
        observation["being_h05.pixel_values"] = torch.stack(images, dim=1)
        observation["being_h05.image_valid"] = torch.stack(image_present, dim=1)

        if action is not None:
            action_values = {
                key.removeprefix("action."): value
                for key, value in named.items()
                if key.startswith("action.")
            }
            for binary_key in ("gripper_position", "control_mode"):
                if binary_key in action_values:
                    action_values[binary_key] = (action_values[binary_key] > 0.5).to(
                        action_values[binary_key].dtype
                    )
            semantic_action, action_mask = pack_named(action_values, self.action_slots)
            complementary["being_h05.action_valid"] = action_mask
            transition[TransitionKey.ACTION] = semantic_action
        transition[TransitionKey.OBSERVATION] = observation
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="being_h05_semantic_unpack")
class BeingH05SemanticUnpackStep(ProcessorStep):
    def get_config(self) -> dict[str, Any]:
        return {}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        named = unpack_action(action)
        gripper = (named["gripper_position"] > 0.5).to(action.dtype)
        control_mode = (named["control_mode"] > 0.5).to(action.dtype)
        transition[TransitionKey.ACTION] = torch.cat(
            [
                named["eef_position"],
                named["eef_rotation"],
                gripper,
                named["base_motion"],
                control_mode,
            ],
            dim=-1,
        )
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_being_h05_pre_post_processors(
    config: BeingH05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
):
    steps = make_default_policy_processor_steps(config, dataset_stats)
    normalize_actions = (
        config.normalization_mapping.get("ACTION", NormalizationMode.IDENTITY) != NormalizationMode.IDENTITY
    )
    semantic_step = BeingH05SemanticPackStep(
        image_keys=config.image_keys,
        prompt_template=config.prompt_template,
        chunk_size=config.chunk_size,
    )
    input_steps = [RenderGenerationPromptStep(config.recipe), steps.add_batch_dim]
    if normalize_actions:
        input_steps.append(BeingH05BinaryActionStep())
    input_steps.append(steps.normalize)
    if normalize_actions:
        input_steps.append(BeingH05BinaryActionStep(restore=True))
    if config.use_language_recipe or config.recipe_path:
        input_steps.append(RenderMessagesStep(recipe=config.recipe))
    input_steps.extend([semantic_step, BeingH05MessagesStep(), steps.to_device])
    output_steps = [BeingH05SemanticUnpackStep()]
    if normalize_actions:
        output_steps.append(BeingH05BinaryActionStep())
    output_steps.append(steps.unnormalize)
    if normalize_actions:
        output_steps.append(BeingH05BinaryActionStep(restore=True))
    output_steps.append(steps.to_cpu)
    return make_policy_processor_pipelines(
        input_steps=input_steps,
        output_steps=output_steps,
    )

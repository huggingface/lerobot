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
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.configs.recipe import language_recipe_enabled
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.policies.language import normalize_semantic_messages, semantic_message_content_text
from lerobot.processor import (
    ComplementaryDataProcessorStep,
    NormalizerProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenderMessagesStep,
    UnnormalizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_wall_oss_05 import WallOSS05Config

if TYPE_CHECKING or _transformers_available:
    from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
else:
    Qwen2_5_VLProcessor = None

WALL_OSS_ACTION_TOKEN = "<|action|>"  # nosec B105
WALL_OSS_IMAGE_TOKEN = "<|image_pad|>"  # nosec B105
WALL_OSS_PROPRIO_TOKEN = "<|propri|>"  # nosec B105
_TEXT_TARGET_START = "<|wall_text_target_start|>"  # nosec B105
_TEXT_TARGET_END = "<|wall_text_target_end|>"  # nosec B105
WALL_OSS_PREDICT_ACTIONS = "wall_oss_05.predict_actions"
_CAMERA_LABELS = {
    "face_view": "front view",
    "right_wrist_view": "right wrist view",
    "left_wrist_view": "left wrist view",
}


@dataclass
@ProcessorStepRegistry.register(name="wall_oss_05_tokenizer")
class WallOSS05TokenizerStep(ProcessorStep):
    """Create all Qwen text/image token tensors before WALL-OSS policy execution."""

    processor_name: str
    camera_key_mapping: dict[str, str]
    chunk_size: int = 32
    max_state_dim: int = 26
    tokenizer_max_length: int = 1000
    revision: str | None = None
    _processor: Any = field(default=None, init=False, repr=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "processor_name": self.processor_name,
            "camera_key_mapping": self.camera_key_mapping,
            "chunk_size": self.chunk_size,
            "max_state_dim": self.max_state_dim,
            "tokenizer_max_length": self.tokenizer_max_length,
            "revision": self.revision,
        }

    def _get_processor(self):
        if self._processor is not None:
            return self._processor
        require_package("transformers", extra="wall_oss_05")
        self._processor = Qwen2_5_VLProcessor.from_pretrained(
            self.processor_name,
            revision=self.revision,
            fix_mistral_regex=True,
        )
        self._processor.tokenizer.add_tokens([WALL_OSS_PROPRIO_TOKEN, WALL_OSS_ACTION_TOKEN])
        self._processor.tokenizer.padding_side = "left"
        return self._processor

    @staticmethod
    def _image_to_pil(image: torch.Tensor) -> Image.Image:
        if image.ndim == 4:
            image = image[-1]
        if image.ndim != 3:
            raise ValueError(f"Expected CHW image, got shape {tuple(image.shape)}.")
        image = image.detach().cpu()
        if image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)
        if image.dtype.is_floating_point:
            image = image.clamp(0, 1).mul(255).round()
        pil = Image.fromarray(image.to(torch.uint8).numpy())
        width, height = pil.size
        size = (448, int(448 * height / width)) if width > height else (int(448 * width / height), 448)
        return pil.resize(size)

    def _observation_prompt(self) -> str:
        prompt = "Observation:"
        for camera_name in self.camera_key_mapping.values():
            label = _CAMERA_LABELS.get(camera_name, camera_name.replace("_", " "))
            prompt += f" {label}: <|vision_start|>{WALL_OSS_IMAGE_TOKEN}<|vision_end|>"
        return prompt

    def _flow_prompt(self, task: str) -> tuple[str, str]:
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{self._observation_prompt()}\n"
            f"Instruction: {task}\nPredict the next action in robot action.\n"
            f"Proprioception: {WALL_OSS_PROPRIO_TOKEN}\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        return prompt, WALL_OSS_ACTION_TOKEN * self.chunk_size

    @staticmethod
    def _message_content(message: dict[str, Any]) -> str:
        content = semantic_message_content_text(message.get("content"))
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
        return f"{content}\n{suffix}" if content and suffix else content or suffix

    def _recipe_prompt(
        self,
        messages: list[dict[str, Any]],
        streams: list[str | None],
        targets: list[int],
        task: str,
    ) -> tuple[str, str, bool]:
        if len(messages) != len(streams):
            raise ValueError("Recipe messages and message streams must have equal length.")
        if any(index < 0 or index >= len(messages) for index in targets):
            raise ValueError("Recipe target message index is out of range.")
        predict_actions = any(stream == "low_level" for stream in streams)
        if predict_actions and not targets:
            prefix, postfix = self._flow_prompt(task)
            return prefix, postfix, True

        target_set = set(targets)
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        observation_injected = False
        for index, message in enumerate(messages):
            role = str(message.get("role", "user"))
            content = self._message_content(message)
            if _TEXT_TARGET_START in content or _TEXT_TARGET_END in content:
                raise ValueError("Recipe content contains a reserved Wall text-target marker.")
            if role == "user" and not observation_injected:
                content = f"{self._observation_prompt()}\n{content}"
                observation_injected = True
            prompt += f"<|im_start|>{role}\n"
            prompt += (
                f"{_TEXT_TARGET_START}{content}<|im_end|>{_TEXT_TARGET_END}\n"
                if index in target_set
                else f"{content}<|im_end|>\n"
            )
        if not observation_injected:
            system = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            prompt = (
                f"{system}<|im_start|>user\n{self._observation_prompt()}<|im_end|>\n"
                + prompt.removeprefix(system)
            )
        postfix = ""
        if predict_actions:
            prompt += (
                f"<|im_start|>user\nInstruction: {task}\n"
                f"Predict the next action in robot action.\nProprioception: {WALL_OSS_PROPRIO_TOKEN}\n"
                "<|im_end|>\n<|im_start|>assistant\n"
            )
            postfix = WALL_OSS_ACTION_TOKEN * self.chunk_size
        return prompt, postfix, predict_actions

    @staticmethod
    def _target_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
        clean, spans, cursor, clean_length = [], [], 0, 0
        while True:
            start = text.find(_TEXT_TARGET_START, cursor)
            if start < 0:
                clean.append(text[cursor:])
                break
            prefix = text[cursor:start]
            clean.append(prefix)
            clean_length += len(prefix)
            payload_start = start + len(_TEXT_TARGET_START)
            end = text.find(_TEXT_TARGET_END, payload_start)
            if end < 0:
                raise ValueError("Wall text-target start marker has no matching end marker.")
            payload = text[payload_start:end]
            clean.append(payload)
            spans.append((clean_length, clean_length + len(payload)))
            clean_length += len(payload)
            cursor = end + len(_TEXT_TARGET_END)
        return "".join(clean), spans

    @staticmethod
    def _batched(value: Any, batch_size: int, name: str) -> list[list[Any]]:
        if not isinstance(value, list):
            raise TypeError(f"{name} must be a list.")
        if len(value) == batch_size and all(isinstance(row, list) for row in value):
            return value
        if batch_size == 1:
            return [value]
        raise ValueError(f"Expected {name} for exactly {batch_size} samples.")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        processor = self._get_processor()
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        state = observation.get(OBS_STATE)
        if not isinstance(state, torch.Tensor):
            raise ValueError("Wall-OSS-0.5 requires tensor observation.state before tokenization.")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2 or state.shape[-1] != self.max_state_dim:
            raise ValueError(f"Expected canonical state shaped (B,{self.max_state_dim}).")
        batch_size = state.shape[0]
        tasks = complementary.get("task")
        tasks = [tasks] * batch_size if isinstance(tasks, str) else tasks
        if (
            not isinstance(tasks, list)
            or len(tasks) != batch_size
            or not all(isinstance(task, str) for task in tasks)
        ):
            raise ValueError(f"Expected exactly {batch_size} task strings.")

        images = [
            self._image_to_pil(observation[key][row])
            for row in range(batch_size)
            for key in self.camera_key_mapping
        ]
        image_inputs = processor.image_processor(images=images, return_tensors="pt")
        messages = complementary.get("messages")
        prefixes, postfixes, predict_actions = [], [], []
        if messages is None:
            for task in tasks:
                prefix, postfix = self._flow_prompt(task)
                prefixes.append(prefix)
                postfixes.append(postfix)
                predict_actions.append(True)
        else:
            messages_batch = normalize_semantic_messages(
                messages, policy_name="Wall-OSS-0.5", batch_size=batch_size
            )
            streams_value = complementary.get("message_streams")
            targets_value = complementary.get("target_message_indices")
            generation = streams_value is None and targets_value is None
            streams_batch = (
                [[None] * len(row) for row in messages_batch]
                if generation
                else self._batched(streams_value or [], batch_size, "message_streams")
            )
            targets_batch = (
                [[] for _ in messages_batch]
                if generation
                else self._batched(targets_value or [], batch_size, "target_message_indices")
            )
            for row, streams, targets, task in zip(
                messages_batch, streams_batch, targets_batch, tasks, strict=True
            ):
                prefix, postfix, predicts = self._recipe_prompt(row, streams, targets, task)
                if generation:
                    prefix += "<|im_start|>assistant\n"
                prefixes.append(prefix)
                postfixes.append(postfix)
                predict_actions.append(predicts)

        expanded = list(prefixes)
        image_index = 0
        merge_size = int(processor.image_processor.merge_size)
        for text_index, text in enumerate(expanded):
            while WALL_OSS_IMAGE_TOKEN in text:
                count = int(image_inputs["image_grid_thw"][image_index].prod() // merge_size**2)
                text = text.replace(WALL_OSS_IMAGE_TOKEN, "<|wall_image_placeholder|>" * count, 1)
                image_index += 1
            expanded[text_index] = text.replace("<|wall_image_placeholder|>", WALL_OSS_IMAGE_TOKEN)

        bins = np.linspace(-1, 1, 513)[:-1]
        discrete = np.digitize(state.cpu().numpy(), bins=bins) - 1
        for index, text in enumerate(expanded):
            expanded[index] = text.replace(WALL_OSS_PROPRIO_TOKEN, " ".join(map(str, discrete[index])))
        texts, target_spans = [], []
        for prefix, postfix in zip(expanded, postfixes, strict=True):
            clean, spans = self._target_spans(prefix + postfix)
            texts.append(clean)
            target_spans.append(spans)
        has_targets = any(target_spans)
        text_inputs = processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_offsets_mapping=has_targets,
        )
        input_ids = text_inputs["input_ids"]
        action_token_id = processor.tokenizer.convert_tokens_to_ids(WALL_OSS_ACTION_TOKEN)
        complementary.update(
            {
                "input_ids": input_ids,
                "attention_mask": text_inputs["attention_mask"],
                "pixel_values": image_inputs["pixel_values"],
                "image_grid_thw": image_inputs["image_grid_thw"],
                "moe_token_types": input_ids == action_token_id,
                "dof_mask": torch.ones((batch_size, self.chunk_size, self.max_state_dim), dtype=torch.bool),
                WALL_OSS_PREDICT_ACTIONS: torch.tensor(predict_actions, dtype=torch.bool),
            }
        )
        if has_targets:
            offsets = text_inputs["offset_mapping"]
            labels = torch.full_like(input_ids, -100)
            for batch_index, spans in enumerate(target_spans):
                for start, end in spans:
                    overlap = (
                        (offsets[batch_index, :, 1] > start)
                        & (offsets[batch_index, :, 0] < end)
                        & text_inputs["attention_mask"][batch_index].bool()
                    )
                    labels[batch_index, overlap] = input_ids[batch_index, overlap]
            labels[labels == action_token_id] = -100
            if processor.tokenizer.pad_token_id is not None:
                labels[labels == processor.tokenizer.pad_token_id] = -100
            complementary["text_labels"] = labels
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="wall_oss_05_task_passthrough")
class WallOSS05TaskPassthrough(ComplementaryDataProcessorStep):
    """Validate language input without selecting, punctuating, or rewriting it."""

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        if "task" not in complementary_data:
            raise KeyError("Wall-OSS-0.5 requires the already-selected LeRobot task string.")
        task = complementary_data["task"]
        if not isinstance(task, str) and not (
            isinstance(task, list) and all(isinstance(value, str) for value in task)
        ):
            raise TypeError("task must be a string or a list of strings.")
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="wall_oss_05_pad_state")
class WallOSS05PadStateProcessorStep(ObservationProcessorStep):
    """Zero-pad state to the fixed 26D Wall-OSS contract before normalization."""

    max_state_dim: int = 26
    # Shared with the crop step so OMX 6D→26D auto-crops actions back to 6D.
    native_dim_holder: dict[str, int] | None = None

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if OBS_STATE not in observation:
            return observation
        state = observation[OBS_STATE]
        state_dim = state.shape[-1]
        if state_dim > self.max_state_dim:
            raise ValueError(
                f"Wall-OSS-0.5 state has {state_dim} dims, which exceeds max_state_dim={self.max_state_dim}."
            )
        if self.native_dim_holder is not None:
            self.native_dim_holder["dim"] = state_dim
        if state_dim < self.max_state_dim:
            observation = dict(observation)
            observation[OBS_STATE] = torch.nn.functional.pad(state, (0, self.max_state_dim - state_dim))
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features = {ft: feats.copy() for ft, feats in features.items()}
        obs_feats = new_features.setdefault(PipelineFeatureType.OBSERVATION, {})
        if OBS_STATE in obs_feats:
            obs_feats[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(self.max_state_dim,))
        return new_features

    def get_config(self) -> dict[str, Any]:
        return {"max_state_dim": self.max_state_dim}


@dataclass
@ProcessorStepRegistry.register(name="wall_oss_05_pad_action")
class WallOSS05PadActionProcessorStep(ProcessorStep):
    """Zero-pad training actions to the fixed 26D Wall-OSS contract before normalization."""

    max_action_dim: int = 26

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Wall-OSS-0.5 action should be a PolicyAction tensor, got {type(action)}.")
        action_dim = action.shape[-1]
        if action_dim > self.max_action_dim:
            raise ValueError(
                f"Wall-OSS-0.5 action has {action_dim} dims, which exceeds max_action_dim={self.max_action_dim}."
            )
        if action_dim == self.max_action_dim:
            return transition
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = torch.nn.functional.pad(
            action, (0, self.max_action_dim - action_dim)
        )
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features = {ft: feats.copy() for ft, feats in features.items()}
        action_feats = new_features.setdefault(PipelineFeatureType.ACTION, {})
        action_feats[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.max_action_dim,))
        return new_features

    def get_config(self) -> dict[str, Any]:
        return {"max_action_dim": self.max_action_dim}


@dataclass
@ProcessorStepRegistry.register(name="wall_oss_05_crop_action")
class WallOSS05CropActionProcessorStep(PolicyActionProcessorStep):
    """Crop padded actions back to the robot/env action width after unnormalization.

    ``action_dim`` overrides the auto width tracked by the paired pad step.
    """

    action_dim: int | None = None
    native_dim_holder: dict[str, int] | None = None

    def _resolved_dim(self, action: PolicyAction) -> int | None:
        if self.action_dim is not None:
            return self.action_dim
        if self.native_dim_holder is not None:
            return self.native_dim_holder.get("dim")
        return None

    def action(self, action: PolicyAction) -> PolicyAction:
        dim = self._resolved_dim(action)
        if dim is None or action.shape[-1] <= dim:
            return action
        return action[..., :dim]

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        dim = self.action_dim
        if dim is None:
            return features
        new_features = {ft: feats.copy() for ft, feats in features.items()}
        action_feats = new_features.setdefault(PipelineFeatureType.ACTION, {})
        action_feats[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(dim,))
        return new_features

    def get_config(self) -> dict[str, Any]:
        return {"action_dim": self.action_dim}


@ProcessorStepRegistry.register(name="wall_oss_05_clamp_normalized")
class WallOSS05ClampNormalizedProcessorStep(ProcessorStep):
    """Match Wall's clipped q01/q99 normalization after the standard LeRobot normalizer."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        output = transition.copy()
        observation = output.get(TransitionKey.OBSERVATION)
        if observation is not None and OBS_STATE in observation:
            observation = dict(observation)
            observation[OBS_STATE] = observation[OBS_STATE].clamp(-1, 1)
            output[TransitionKey.OBSERVATION] = observation
        action = output.get(TransitionKey.ACTION)
        if action is not None:
            output[TransitionKey.ACTION] = action.clamp(-1, 1)
        return output

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def reconcile_wall_oss_05_processors(
    config: WallOSS05Config,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Insert pad/crop steps into Hub pipelines that predate auto-padding.

    Pad/crop share a holder so a 6D robot state is zero-padded to 26D for the
    model, then actions are cropped back to 6D automatically.
    """
    native_dim_holder: dict[str, int] = {}
    pre_steps = list(preprocessor.steps)
    if not any(isinstance(step, WallOSS05PadStateProcessorStep) for step in pre_steps):
        insert_idx = next(
            (idx for idx, step in enumerate(pre_steps) if isinstance(step, NormalizerProcessorStep)),
            len(pre_steps),
        )
        pre_steps[insert_idx:insert_idx] = [
            WallOSS05PadStateProcessorStep(
                max_state_dim=config.max_state_dim, native_dim_holder=native_dim_holder
            ),
            WallOSS05PadActionProcessorStep(max_action_dim=config.max_action_dim),
        ]
        preprocessor.steps = pre_steps
    else:
        for step in pre_steps:
            if isinstance(step, WallOSS05PadStateProcessorStep):
                step.native_dim_holder = native_dim_holder
                break

    post_steps = list(postprocessor.steps)
    crop_step = WallOSS05CropActionProcessorStep(
        action_dim=config.postprocess_action_dim,
        native_dim_holder=native_dim_holder,
    )
    crop_idx = next(
        (idx for idx, step in enumerate(post_steps) if isinstance(step, WallOSS05CropActionProcessorStep)),
        None,
    )
    if crop_idx is None:
        insert_idx = next(
            (idx + 1 for idx, step in enumerate(post_steps) if isinstance(step, UnnormalizerProcessorStep)),
            0,
        )
        post_steps.insert(insert_idx, crop_step)
    else:
        post_steps[crop_idx] = crop_step
    postprocessor.steps = post_steps

    return preprocessor, postprocessor


def make_wall_oss_05_pre_post_processors(
    config: WallOSS05Config,
    dataset_stats: dict | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the serializable Wall input/output pipelines."""

    steps = make_default_policy_processor_steps(config, dataset_stats)
    render_training = language_recipe_enabled(
        use_language_recipe=config.use_language_recipe,
        recipe_path=config.recipe_path,
    )
    if render_training and config.recipe is None:
        raise ValueError("Wall-OSS-0.5 language training requires a recipe in policy config.")
    native_dim_holder: dict[str, int] = {}
    return make_policy_processor_pipelines(
        input_steps=[
            RenderMessagesStep(config.recipe, render_training=render_training),
            steps.rename_observations,
            steps.add_batch_dim,
            WallOSS05TaskPassthrough(),
            WallOSS05PadStateProcessorStep(
                max_state_dim=config.max_state_dim, native_dim_holder=native_dim_holder
            ),
            WallOSS05PadActionProcessorStep(max_action_dim=config.max_action_dim),
            steps.normalize,
            WallOSS05ClampNormalizedProcessorStep(),
            WallOSS05TokenizerStep(
                processor_name=config.pretrained_name_or_path,
                revision=config.pretrained_revision,
                camera_key_mapping=config.camera_key_mapping,
                chunk_size=config.chunk_size,
                max_state_dim=config.max_state_dim,
                tokenizer_max_length=config.tokenizer_max_length,
            ),
            steps.to_device,
        ],
        output_steps=[
            steps.unnormalize,
            WallOSS05CropActionProcessorStep(
                action_dim=config.postprocess_action_dim,
                native_dim_holder=native_dim_holder,
            ),
            steps.to_cpu,
        ],
    )

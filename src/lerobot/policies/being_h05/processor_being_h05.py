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

from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as tvf

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ProcessorStep,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.types import EnvTransition, TransitionKey

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
    semantic_step = BeingH05SemanticPackStep(
        image_keys=config.image_keys,
        prompt_template=config.prompt_template,
        chunk_size=config.chunk_size,
    )
    return make_policy_processor_pipelines(
        input_steps=[
            steps.add_batch_dim,
            steps.normalize,
            semantic_step,
            steps.to_device,
        ],
        output_steps=[
            BeingH05SemanticUnpackStep(),
            steps.unnormalize,
            steps.to_cpu,
        ],
    )

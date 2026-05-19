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

"""TOPReward pre/post processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    policy_action_to_transition,
)
from lerobot.rewards.topreward.configuration_topreward import TOPRewardConfig
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_PREFIX,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

# Namespace for TOPReward's pre-encoded observation tensors written by the
# processor and consumed by the model. Keys: ``frames`` (one ``(T,H,W,C)``
# uint8 numpy array per sample) and ``task`` (one string per sample).
TOPREWARD_FEATURE_PREFIX = f"{OBS_PREFIX}topreward."


def _video_to_numpy(video: Tensor, *, max_frames: int | None) -> np.ndarray:
    """Convert one trajectory tensor to a ``(T, H, W, C) uint8`` numpy array.

    Mirrors the Robometer helper: accepts ``(T, C, H, W)`` or ``(T, H, W, C)``
    layouts, rescales floats in ``[0, 1]`` to ``[0, 255]``, clips values
    outside the uint8 range and tail-crops to ``max_frames``.
    """
    if max_frames is not None:
        video = video[-max_frames:]
    if video.shape[1] in (1, 3):
        video = video.permute(0, 2, 3, 1)
    elif video.shape[-1] not in (1, 3):
        raise ValueError(f"Expected channel dim of size 1 or 3, got shape {tuple(video.shape)}")

    array = video.detach().cpu().numpy()
    if np.issubdtype(array.dtype, np.floating) and array.size > 0 and array.max() <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _expand_tasks(task: Any, *, batch_size: int, default: str | None) -> list[str]:
    if task is None:
        task = default
    if task is None:
        raise KeyError("TOPReward expected a task description in complementary data")
    if isinstance(task, str):
        return [task] * batch_size
    if isinstance(task, tuple):
        task = list(task)
    if not (isinstance(task, list) and all(isinstance(item, str) for item in task)):
        raise TypeError(f"TOPReward task must be a string or list of strings, got {type(task)}")
    if len(task) == 1 and batch_size > 1:
        return task * batch_size
    if len(task) != batch_size:
        raise ValueError(f"Expected {batch_size} tasks, got {len(task)}")
    return task


@dataclass
@ProcessorStepRegistry.register(name="topreward_encoder")
class TOPRewardEncoderProcessorStep(ProcessorStep):
    """Normalise raw frames + task into TOPReward-namespaced observation entries.

    At call time the step reads:

    - ``observation[image_key]``: ``(B, T, C, H, W)`` or ``(B, C, H, W)`` frames.
    - ``complementary_data[task_key]``: a string or list of strings.

    and writes:

    - ``observation[f"{TOPREWARD_FEATURE_PREFIX}frames"]``: list of
      ``(T, H, W, C) uint8`` numpy arrays, one per sample.
    - ``observation[f"{TOPREWARD_FEATURE_PREFIX}task"]``: list of strings,
      one per sample.

    The actual chat-template / tokenisation happens model-side because
    TOPReward's reward extraction needs the tokenizer to know the
    prompt/suffix split (label masking on suffix tokens only).
    """

    image_key: str = OBS_IMAGES + ".top"
    task_key: str = "task"
    default_task: str | None = None
    max_frames: int | None = 16

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        if not isinstance(observation, dict):
            raise ValueError("TOPRewardEncoderProcessorStep requires an observation dict")

        if self.image_key not in observation:
            raise KeyError(f"TOPReward expected image key {self.image_key!r} in observation")

        frames = observation[self.image_key]
        tensor = frames.detach().cpu() if isinstance(frames, Tensor) else torch.as_tensor(frames)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(1)
        elif tensor.ndim != 5:
            raise ValueError(
                f"Expected TOPReward frames with shape (B,C,H,W) or (B,T,C,H,W); got {tuple(tensor.shape)}"
            )

        batch_size = tensor.shape[0]
        tasks = _expand_tasks(
            complementary.get(self.task_key, self.default_task),
            batch_size=batch_size,
            default=self.default_task,
        )

        frames_per_sample = [
            _video_to_numpy(tensor[i], max_frames=self.max_frames) for i in range(batch_size)
        ]

        new_observation = dict(observation)
        new_observation[f"{TOPREWARD_FEATURE_PREFIX}frames"] = frames_per_sample
        new_observation[f"{TOPREWARD_FEATURE_PREFIX}task"] = list(tasks)

        new_transition = transition.copy()
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "image_key": self.image_key,
            "task_key": self.task_key,
            "default_task": self.default_task,
            "max_frames": self.max_frames,
        }


def make_topreward_pre_post_processors(
    config: TOPRewardConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Pipeline that normalises frames + task for the TOPReward model.

    The preprocessor adds a batch dimension if needed, runs TOPReward's
    encoder, and moves any tensor entries to the configured device. The
    postprocessor is the identity since TOPReward outputs a single reward
    tensor.
    """
    del dataset_stats  # TOPReward's VLM handles its own normalisation.

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            AddBatchDimensionProcessorStep(),
            TOPRewardEncoderProcessorStep(
                image_key=config.image_key,
                task_key=config.task_key,
                default_task=config.default_task,
                max_frames=config.max_frames,
            ),
            DeviceProcessorStep(device=config.device or "cpu"),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline(
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
    )
    return preprocessor, postprocessor

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

"""TOPReward scorer for the shared offline frame-scoring workflow."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

from lerobot.processor import PolicyProcessorPipeline

from ..scoring import FrameSignals, SignalDescriptor
from .modeling_topreward import TOPRewardModel
from .processor_topreward import make_topreward_pre_post_processors

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset

DEFAULT_NUM_SAMPLES = 15
PROGRESS_SIGNAL = "reward.topreward.progress"

TOPREWARD_SIGNAL_DESCRIPTORS = {
    PROGRESS_SIGNAL: SignalDescriptor(
        description=(
            "TOPReward log-probability of the task for the trajectory prefix ending at this frame, "
            "min-max normalized per episode."
        ),
        greater_is_better=True,
        bounds=(0.0, 1.0),
    ),
}


def build_prefix_lengths(num_frames: int, num_samples: int | None) -> NDArray[np.int64]:
    """Return the prefix lengths to score: every prefix, or ``num_samples`` evenly spaced ones."""
    if num_samples is None or num_samples >= num_frames:
        return np.arange(1, num_frames + 1, dtype=np.int64)
    return np.unique(np.linspace(1, num_frames, num_samples).round().astype(np.int64))


def normalize_rewards(rewards: NDArray[np.float64]) -> NDArray[np.float64]:
    """Min-max normalize an episode's prefix rewards to ``[0, 1]``, as upstream TOPReward does."""
    if rewards.size == 1:
        return np.ones(1)
    low, high = rewards.min(), rewards.max()
    if high == low:
        return np.ones_like(rewards)
    return (rewards - low) / (high - low)


class TOPRewardFrameScorer:
    """Score each frame with TOPReward's normalized log-probability for the prefix ending at it."""

    name = "topreward"

    def __init__(
        self,
        model: TOPRewardModel,
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        *,
        num_samples: int | None = DEFAULT_NUM_SAMPLES,
    ) -> None:
        if num_samples is not None and num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        self.model = model
        self.preprocessor = preprocessor
        self.num_samples = num_samples

    @property
    def descriptors(self) -> dict[str, SignalDescriptor]:
        """Describe the per-frame normalized progress signal."""
        return TOPREWARD_SIGNAL_DESCRIPTORS

    @property
    def provenance(self) -> dict[str, Any]:
        """Describe the model and the settings that change the signals."""
        config = self.model.config
        return {
            "model": {
                "type": config.type,
                "id": config.pretrained_path,
                "revision": config.pretrained_revision,
            },
            "scorer": {
                "version": 1,
                "options": {
                    "vlm_name": config.vlm_name,
                    "image_key": config.image_key,
                    "task_key": config.task_key,
                    "default_task": config.default_task,
                    "fps": config.fps,
                    "prompt_prefix": config.prompt_prefix,
                    "prompt_suffix_template": config.prompt_suffix_template,
                    "add_chat_template": config.add_chat_template,
                    "num_samples": self.num_samples,
                },
            },
        }

    def score_episode(self, dataset: LeRobotDataset, episode_index: int) -> FrameSignals:
        config = self.model.config
        episode = dataset.meta.episodes[episode_index]
        episode_start = int(episode["dataset_from_index"])
        episode_end = int(episode["dataset_to_index"])
        num_frames = episode_end - episode_start

        task = None
        if config.task_key == "task":
            task_index = dataset.get_frames(episode_index, ["task_index"], [0])["task_index"][0]
            task = str(dataset.meta.tasks.iloc[int(task_index)].name)
        elif config.task_key in dataset.meta.features:
            mapping = dataset.absolute_to_relative_idx
            position = episode_start if mapping is None else mapping[episode_start]
            task = dataset[position][config.task_key]
        frame_indices = list(range(num_frames))
        episode_frames = dataset.get_frames(episode_index, [config.image_key], frame_indices)[
            config.image_key
        ]

        prefix_lengths = build_prefix_lengths(num_frames, self.num_samples)
        log_probabilities: NDArray[np.float64] = np.empty(len(prefix_lengths), dtype=np.float64)

        self.model.eval()
        with torch.inference_mode():
            for i, length in enumerate(prefix_lengths):
                encoded = self.preprocessor(
                    {config.image_key: episode_frames[:length].unsqueeze(0), config.task_key: task}
                )
                log_probabilities[i] = self.model.compute_log_probability(encoded).item()

        # Frame t ends the prefix of length t + 1; frames between scored prefixes are interpolated.
        progress = np.interp(
            np.arange(1, num_frames + 1), prefix_lengths, normalize_rewards(log_probabilities)
        )
        return FrameSignals(
            frame_indices=np.arange(num_frames, dtype=np.int64),
            signals={PROGRESS_SIGNAL: progress.astype(np.float32)},
        )


def make_topreward_frame_scorer(
    model: TOPRewardModel,
    *,
    num_samples: int | None = DEFAULT_NUM_SAMPLES,
) -> TOPRewardFrameScorer:
    """Construct the standard TOPReward offline frame scorer."""
    # Each prefix is scored whole, so the processor must not crop it to `max_frames`.
    preprocessor, _ = make_topreward_pre_post_processors(replace(model.config, max_frames=None))
    return TOPRewardFrameScorer(model, preprocessor, num_samples=num_samples)

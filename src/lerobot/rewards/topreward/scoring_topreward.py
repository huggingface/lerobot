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

from ..scoring import FrameSignals, SignalDescriptor, interpolate_progress, scoring_options
from .configuration_topreward import TOPREWARD_PROGRESS_SIGNAL, TOPREWARD_SIGNAL_DESCRIPTORS
from .modeling_topreward import TOPRewardModel
from .processor_topreward import make_topreward_pre_post_processors

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset


def _build_prefix_lengths(num_frames: int, num_samples: int | None) -> NDArray[np.int64]:
    """Return the prefix lengths to score: every prefix, or ``num_samples`` evenly spaced ones."""
    if num_samples is None or num_samples >= num_frames:
        return np.arange(1, num_frames + 1, dtype=np.int64)
    return np.unique(np.linspace(1, num_frames, num_samples).round().astype(np.int64))


def _normalize_rewards(rewards: NDArray[np.float64]) -> NDArray[np.float64]:
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
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor

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
                "options": scoring_options(config),
            },
        }

    def score_episode(self, dataset: LeRobotDataset, episode_index: int) -> FrameSignals:
        config = self.model.config
        episode = dataset.meta.episodes[episode_index]
        episode_start = int(episode["dataset_from_index"])
        episode_end = int(episode["dataset_to_index"])
        num_frames = episode_end - episode_start

        mapping = dataset.absolute_to_relative_idx
        positions = [
            index if mapping is None else mapping[index] for index in range(episode_start, episode_end)
        ]
        # `dataset[i]` holds the task under "task" and a custom task feature under its own key.
        task = dataset[positions[0]].get(config.task_key)
        episode_frames = torch.stack([dataset[position][config.image_key] for position in positions])

        prefix_lengths = _build_prefix_lengths(num_frames, config.num_samples)
        log_probabilities: NDArray[np.float64] = np.empty(len(prefix_lengths), dtype=np.float64)

        self.model.eval()
        with torch.inference_mode():
            for i, length in enumerate(prefix_lengths):
                encoded = self.preprocessor(
                    {config.image_key: episode_frames[:length].unsqueeze(0), config.task_key: task}
                )
                log_probabilities[i] = self.model.compute_log_probability(encoded).item()

        # Frame t ends the prefix of length t + 1; frames between scored prefixes are interpolated.
        progress = interpolate_progress(
            prefix_lengths, _normalize_rewards(log_probabilities), np.arange(1, num_frames + 1)
        )
        return FrameSignals(
            frame_indices=np.arange(num_frames, dtype=np.int64),
            signals={TOPREWARD_PROGRESS_SIGNAL: progress},
        )


def make_topreward_frame_scorer(model: TOPRewardModel, *, batch_size: int = 32) -> TOPRewardFrameScorer:
    """Construct the standard TOPReward offline frame scorer.

    `batch_size` is accepted like other scorers' but unused: prefixes have different lengths,
    so each one is scored on its own.
    """
    # Each prefix is scored whole, so the processor must not crop it to `max_frames`.
    preprocessor, _ = make_topreward_pre_post_processors(replace(model.config, max_frames=None))
    return TOPRewardFrameScorer(model, preprocessor)

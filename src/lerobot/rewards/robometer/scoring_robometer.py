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

"""RoboMeter scorer for the shared offline frame-scoring workflow."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

from lerobot.processor import PolicyProcessorPipeline

from ..scoring import FrameSignals, SignalDescriptor, scoring_options
from .configuration_robometer import (
    ROBOMETER_PROGRESS_SIGNAL,
    ROBOMETER_SIGNAL_DESCRIPTORS,
    ROBOMETER_SUCCESS_PROBABILITY_SIGNAL,
)
from .modeling_robometer import RobometerRewardModel
from .processor_robometer import make_robometer_pre_post_processors

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset


def _build_subsample_indices(num_frames: int, num_subsampled_frames: int) -> list[np.ndarray]:
    """For each frame, pick ``num_subsampled_frames`` indices spread over the prefix ending at it.

    Indices are truncated, not rounded, as in RoboMeter's own frame-step evaluation
    (`robometer/evals/eval_server.py`).
    """
    return [
        np.linspace(0, frame_index, num_subsampled_frames, dtype=np.int64)
        for frame_index in range(num_frames)
    ]


class RobometerFrameScorer:
    """Score each frame with RoboMeter's prediction for the prefix ending at it."""

    name = "robometer"

    def __init__(
        self,
        model: RobometerRewardModel,
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        *,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model = model
        self.preprocessor = preprocessor
        self.batch_size = batch_size

    @property
    def descriptors(self) -> dict[str, SignalDescriptor]:
        """Describe the per-frame progress and success signals."""
        return ROBOMETER_SIGNAL_DESCRIPTORS

    @property
    def provenance(self) -> dict[str, Any]:
        """Describe the checkpoint and the settings that change the signals."""
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

        subsample_indices = _build_subsample_indices(num_frames, config.num_subsampled_frames)
        progress: NDArray[np.float32] = np.empty(num_frames, dtype=np.float32)
        success_probability: NDArray[np.float32] = np.empty(num_frames, dtype=np.float32)

        self.model.eval()
        with torch.inference_mode():
            for start in range(0, num_frames, self.batch_size):
                end = min(start + self.batch_size, num_frames)
                # Each batch item is the prefix ending at one target frame.
                prefixes = torch.stack(
                    [episode_frames[subsample_indices[frame]] for frame in range(start, end)]
                )
                encoded = self.preprocessor({config.image_key: prefixes, config.task_key: task})
                prediction = self.model.predict_progress(encoded)
                progress[start:end] = prediction.progress[:, -1].cpu().numpy()
                success_probability[start:end] = prediction.success_probability[:, -1].cpu().numpy()

        return FrameSignals(
            frame_indices=np.arange(num_frames, dtype=np.int64),
            signals={
                ROBOMETER_PROGRESS_SIGNAL: progress,
                ROBOMETER_SUCCESS_PROBABILITY_SIGNAL: success_probability,
            },
        )


def make_robometer_frame_scorer(model: RobometerRewardModel, *, batch_size: int = 32) -> RobometerFrameScorer:
    """Construct the standard RoboMeter offline frame scorer, scoring `batch_size` target frames together."""
    # Each prefix is already sub-sampled, so the processor must not crop it to `max_frames`.
    preprocessor, _ = make_robometer_pre_post_processors(replace(model.config, max_frames=None))
    return RobometerFrameScorer(model, preprocessor, batch_size=batch_size)

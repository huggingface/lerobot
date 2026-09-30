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

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.datasets.scores import FrameSignals, SignalDescriptor
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import PolicyProcessorPipeline

from .modeling_robometer import RobometerRewardModel
from .processor_robometer import make_robometer_pre_post_processors

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset

DEFAULT_NUM_SUBSAMPLED_FRAMES = 4
PROGRESS_SIGNAL = "reward.robometer.progress"
SUCCESS_PROBABILITY_SIGNAL = "reward.robometer.success_probability"

ROBOMETER_SIGNAL_DESCRIPTORS = {
    PROGRESS_SIGNAL: SignalDescriptor(
        description="RoboMeter task progress for the trajectory prefix ending at this frame.",
        direction="higher",
        bounds=(0.0, 1.0),
    ),
    SUCCESS_PROBABILITY_SIGNAL: SignalDescriptor(
        description="RoboMeter success probability for the trajectory prefix ending at this frame.",
        direction="higher",
        bounds=(0.0, 1.0),
    ),
}


def build_subsample_indices(num_frames: int, num_subsampled_frames: int) -> list[np.ndarray]:
    """For each frame, pick ``num_subsampled_frames`` indices spread over the prefix ending at it."""
    return [
        np.linspace(0, frame_index, num_subsampled_frames).round().astype(np.int64)
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
        num_subsampled_frames: int = DEFAULT_NUM_SUBSAMPLED_FRAMES,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if num_subsampled_frames < 1:
            raise ValueError(f"num_subsampled_frames must be >= 1, got {num_subsampled_frames}")
        self.model = model
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.num_subsampled_frames = num_subsampled_frames

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
                "options": {
                    "image_key": config.image_key,
                    "num_subsampled_frames": self.num_subsampled_frames,
                },
            },
        }

    def score_episode(self, dataset: LeRobotDataset, episode_index: int) -> FrameSignals:
        config = self.model.config
        episode = dataset.meta.episodes[episode_index]
        episode_start = int(episode["dataset_from_index"])
        episode_end = int(episode["dataset_to_index"])
        num_frames = episode_end - episode_start

        mapping = dataset.absolute_to_relative_idx
        samples = [
            dataset[index if mapping is None else mapping[index]]
            for index in range(episode_start, episode_end)
        ]
        episode_frames = torch.stack([sample[config.image_key] for sample in samples])
        task = samples[0].get(config.task_key)

        subsample_indices = build_subsample_indices(num_frames, self.num_subsampled_frames)
        progress = np.empty(num_frames, dtype=np.float32)
        success_probability = np.empty(num_frames, dtype=np.float32)

        self.model.eval()
        with torch.inference_mode():
            for start in range(0, num_frames, self.batch_size):
                end = min(start + self.batch_size, num_frames)
                # Each batch item is the prefix ending at one target frame.
                prefixes = torch.stack(
                    [episode_frames[subsample_indices[frame]] for frame in range(start, end)]
                )
                encoded = self.preprocessor(
                    {
                        TransitionKey.OBSERVATION: {config.image_key: prefixes},
                        TransitionKey.COMPLEMENTARY_DATA: {config.task_key: task},
                    }
                )
                prediction = self.model.predict_progress(encoded[TransitionKey.OBSERVATION])
                progress[start:end] = prediction.progress[:, -1].cpu().numpy()
                success_probability[start:end] = prediction.success_probability[:, -1].cpu().numpy()

        return FrameSignals(
            frame_indices=np.arange(num_frames, dtype=np.int64),
            signals={
                PROGRESS_SIGNAL: progress,
                SUCCESS_PROBABILITY_SIGNAL: success_probability,
            },
            descriptors=ROBOMETER_SIGNAL_DESCRIPTORS,
        )


def make_robometer_frame_scorer(
    model: RobometerRewardModel,
    *,
    batch_size: int = 32,
    num_subsampled_frames: int = DEFAULT_NUM_SUBSAMPLED_FRAMES,
) -> RobometerFrameScorer:
    """Construct the standard RoboMeter offline frame scorer."""
    preprocessor, _ = make_robometer_pre_post_processors(model.config)
    return RobometerFrameScorer(
        model,
        preprocessor,
        batch_size=batch_size,
        num_subsampled_frames=num_subsampled_frames,
    )

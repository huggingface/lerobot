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

"""RoboMeter adapter for the shared offline frame-scoring workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.rewards.scoring import FrameSignals, SignalDescriptor

from .configuration_robometer import RobometerConfig
from .modeling_robometer import RobometerPrediction, RobometerRewardModel
from .processor_robometer import RobometerEncoderProcessorStep

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset

DEFAULT_NUM_SUBSAMPLED_FRAMES = 4
PROGRESS_SIGNAL = "reward.robometer.progress"
SUCCESS_PROBABILITY_SIGNAL = "reward.robometer.success_probability"

ROBOMETER_SIGNAL_DESCRIPTORS = {
    PROGRESS_SIGNAL: SignalDescriptor(
        description="RoboMeter task progress for the trajectory prefix ending at this frame.",
        unit=None,
        direction="higher",
        comparison_scope="task",
        missing_values="forbidden",
        bounds=(0.0, 1.0),
    ),
    SUCCESS_PROBABILITY_SIGNAL: SignalDescriptor(
        description="RoboMeter success probability for the trajectory prefix ending at this frame.",
        unit=None,
        direction="higher",
        comparison_scope="task",
        missing_values="forbidden",
        bounds=(0.0, 1.0),
    ),
}


def build_subsample_indices(num_frames: int, num_subsampled_frames: int) -> list[np.ndarray]:
    """Build fixed-size prefix samples using RoboMeter's characterized rule."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if num_subsampled_frames < 1:
        raise ValueError(f"num_subsampled_frames must be >= 1, got {num_subsampled_frames}")
    return [
        np.linspace(0, frame_index, num_subsampled_frames).round().astype(np.int64)
        for frame_index in range(num_frames)
    ]


def make_robometer_scoring_encoder(config: RobometerConfig) -> RobometerEncoderProcessorStep:
    """Build an encoder that preserves prefixes already selected by this adapter."""
    return RobometerEncoderProcessorStep(
        base_model_id=config.base_model_id,
        image_key=config.image_key,
        task_key=config.task_key,
        default_task=config.default_task,
        max_frames=None,
        use_multi_image=config.use_multi_image,
        use_per_frame_progress_token=config.use_per_frame_progress_token,
    )


def _dataset_position(dataset: LeRobotDataset, absolute_index: int) -> int:
    absolute_to_relative = getattr(dataset, "absolute_to_relative_idx", None)
    if absolute_to_relative is None:
        return absolute_index
    try:
        return int(absolute_to_relative[absolute_index])
    except KeyError as exc:
        raise ValueError(
            f"Dataset view does not contain absolute frame index {absolute_index}; "
            "make sure the selected episodes match the scoring selection"
        ) from exc


def _resolve_task(sample: dict[str, Any], *, task_key: str, default: str) -> str:
    task = sample.get(task_key)
    if isinstance(task, str) and task:
        return task
    return default


def _select_last_frame_signals(prediction: RobometerPrediction) -> tuple[torch.Tensor, torch.Tensor]:
    progress = prediction.progress
    success_probability = prediction.success_probability
    if progress.ndim != 2 or progress.shape[1] == 0:
        raise ValueError(
            "RoboMeter progress must have shape (batch, time) with at least one frame, "
            f"got {tuple(progress.shape)}"
        )
    if success_probability.shape != progress.shape:
        raise ValueError(
            "RoboMeter success probability must match progress shape, "
            f"got {tuple(success_probability.shape)} and {tuple(progress.shape)}"
        )
    return progress[:, -1], success_probability[:, -1]


@dataclass
class RobometerFrameScorer:
    """Convert RoboMeter prefix predictions into dense episode frame signals."""

    model: RobometerRewardModel
    encoder: RobometerEncoderProcessorStep
    image_key: str
    task_key: str = "task"
    default_task: str = "perform the task"
    base_model_id: str | None = None
    use_multi_image: bool | None = None
    use_per_frame_progress_token: bool | None = None
    batch_size: int = 32
    num_subsampled_frames: int = DEFAULT_NUM_SUBSAMPLED_FRAMES

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_subsampled_frames < 1:
            raise ValueError(f"num_subsampled_frames must be >= 1, got {self.num_subsampled_frames}")

    @property
    def options(self) -> dict[str, Any]:
        """JSON-serializable adapter settings used in scoring provenance."""
        return {
            "batch_size": self.batch_size,
            "base_model_id": self.base_model_id,
            "default_task": self.default_task,
            "image_key": self.image_key,
            "num_subsampled_frames": self.num_subsampled_frames,
            "sampling": "linspace_round_fixed_prefix",
            "task_key": self.task_key,
            "task_resolution": "first_frame_or_default",
            "use_multi_image": self.use_multi_image,
            "use_per_frame_progress_token": self.use_per_frame_progress_token,
        }

    def __call__(self, dataset: LeRobotDataset, episode_index: int) -> FrameSignals:
        episode = dataset.meta.episodes[episode_index]
        episode_start = int(episode["dataset_from_index"])
        episode_end = int(episode["dataset_to_index"])
        num_frames = episode_end - episode_start
        if num_frames < 1:
            raise ValueError(f"Episode {episode_index} has no frames")

        samples = [
            dataset[_dataset_position(dataset, episode_start + local_index)]
            for local_index in range(num_frames)
        ]
        first_sample = samples[0]
        task = _resolve_task(first_sample, task_key=self.task_key, default=self.default_task)
        try:
            episode_frames = torch.stack([sample[self.image_key] for sample in samples])
        except KeyError as exc:
            raise KeyError(
                f"RoboMeter scoring expected image key {self.image_key!r} in dataset frames"
            ) from exc

        subsample_indices = build_subsample_indices(num_frames, self.num_subsampled_frames)
        progress = np.empty(num_frames, dtype=np.float32)
        success_probability = np.empty(num_frames, dtype=np.float32)

        for start in range(0, num_frames, self.batch_size):
            end = min(start + self.batch_size, num_frames)
            frames_batch = torch.stack(
                [episode_frames[subsample_indices[frame_index]] for frame_index in range(start, end)]
            )
            transition = {
                TransitionKey.OBSERVATION: {self.image_key: frames_batch},
                TransitionKey.COMPLEMENTARY_DATA: {self.task_key: task},
            }
            encoded = self.encoder(transition)
            observation = encoded[TransitionKey.OBSERVATION]
            prediction = self.model.predict_progress(observation)
            batch_progress, batch_success = _select_last_frame_signals(prediction)
            progress[start:end] = batch_progress.detach().cpu().numpy()
            success_probability[start:end] = batch_success.detach().cpu().numpy()

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
    config: RobometerConfig,
    *,
    batch_size: int = 32,
    num_subsampled_frames: int = DEFAULT_NUM_SUBSAMPLED_FRAMES,
) -> RobometerFrameScorer:
    """Construct the standard RoboMeter offline scoring adapter."""
    return RobometerFrameScorer(
        model=model,
        encoder=make_robometer_scoring_encoder(config),
        image_key=config.image_key,
        task_key=config.task_key,
        default_task=config.default_task or "perform the task",
        base_model_id=config.base_model_id,
        use_multi_image=config.use_multi_image,
        use_per_frame_progress_token=config.use_per_frame_progress_token,
        batch_size=batch_size,
        num_subsampled_frames=num_subsampled_frames,
    )

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

"""RynnValue adapter for the shared offline frame-scoring workflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.rewards.scoring import FrameSignals, ScoringSummary, SignalDescriptor, score_dataset

from .configuration_rynnvalue import RYNNVALUE_FEATURE_PREFIX, RynnValueConfig
from .modeling_rynnvalue import RynnValueRewardModel
from .processor_rynnvalue import RynnValueEncoderProcessorStep, _video_to_pil

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset

DEFAULT_INFERENCE_FPS = 1.0
REMAINING_TIME_SIGNAL = "reward.rynnvalue.remaining_time_s"
POTENTIAL_SIGNAL = "reward.rynnvalue.potential"
IS_INFERENCE_FRAME_SIGNAL = "reward.rynnvalue.is_inference_frame"
PROGRESS_SIGNAL = "reward.rynnvalue.progress"

RYNNVALUE_SIGNAL_DESCRIPTORS = {
    REMAINING_TIME_SIGNAL: SignalDescriptor(
        description=(
            "RynnValue predicted seconds remaining for the trajectory prefix, linearly interpolated "
            "between inference frames."
        ),
        unit="s",
        direction="lower",
        missing_values="forbidden",
    ),
    POTENTIAL_SIGNAL: SignalDescriptor(
        description="Negative RynnValue remaining time, suitable as a higher-is-better state potential.",
        unit="s",
        direction="higher",
        missing_values="forbidden",
    ),
    IS_INFERENCE_FRAME_SIGNAL: SignalDescriptor(
        description="Whether RynnValue ran on this frame rather than its value being interpolated.",
        unit=None,
        direction="none",
        missing_values="forbidden",
    ),
}

PROGRESS_DESCRIPTOR = SignalDescriptor(
    description="RynnValue progress derived from an explicitly configured remaining-time horizon.",
    unit=None,
    direction="higher",
    missing_values="forbidden",
    bounds=(0.0, 1.0),
)


def select_anchor_indices(num_frames: int, dataset_fps: float, inference_fps: float) -> np.ndarray:
    """Select approximately evenly timed anchors and include both episode boundaries."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if not np.isfinite(dataset_fps) or dataset_fps <= 0:
        raise ValueError(f"dataset_fps must be positive and finite, got {dataset_fps}")
    if not np.isfinite(inference_fps) or inference_fps <= 0 or inference_fps > dataset_fps:
        raise ValueError(
            f"inference_fps must be in (0, dataset_fps], got {inference_fps} for dataset_fps={dataset_fps}"
        )

    stride = dataset_fps / inference_fps
    anchors = np.rint(np.arange(0, num_frames, stride)).astype(np.int64)
    anchors = np.unique(np.clip(anchors, 0, num_frames - 1))
    if anchors[-1] != num_frames - 1:
        anchors = np.append(anchors, num_frames - 1)
    return anchors


def select_prefix_indices(anchor_index: int, max_frames: int | None) -> np.ndarray:
    """Select a chronological causal prefix ending exactly at ``anchor_index``."""
    if anchor_index < 0:
        raise ValueError(f"anchor_index must be non-negative, got {anchor_index}")
    prefix_length = anchor_index + 1
    if max_frames is None or prefix_length <= max_frames:
        return np.arange(prefix_length, dtype=np.int64)
    if max_frames < 1:
        raise ValueError(f"max_frames must be positive or None, got {max_frames}")
    if max_frames == 1:
        return np.asarray([anchor_index], dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, anchor_index, max_frames)).astype(np.int64))


def interpolate_anchor_values(
    num_frames: int, anchor_indices: np.ndarray, anchor_values: np.ndarray
) -> np.ndarray:
    """Linearly interpolate anchor predictions to every frame in one episode."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if anchor_indices.ndim != 1 or anchor_values.ndim != 1:
        raise ValueError("anchor_indices and anchor_values must be one-dimensional")
    if len(anchor_indices) != len(anchor_values) or not len(anchor_indices):
        raise ValueError("anchor_indices and anchor_values must have the same non-zero length")
    if np.any(np.diff(anchor_indices) <= 0):
        raise ValueError("anchor_indices must be strictly increasing")
    if anchor_indices[0] != 0 or anchor_indices[-1] != num_frames - 1:
        raise ValueError("anchor_indices must include the first and last frame")
    return np.interp(np.arange(num_frames), anchor_indices, anchor_values).astype(np.float32)


def remaining_time_to_progress(remaining_time_s: np.ndarray, horizon_s: float) -> np.ndarray:
    """Convert physical remaining time to bounded progress using an explicit horizon."""
    if not np.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError(f"horizon_s must be positive and finite, got {horizon_s}")
    return np.clip(1.0 - remaining_time_s / horizon_s, 0.0, 1.0).astype(np.float32)


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


def _resolve_task(sample: dict[str, Any], *, task_key: str, default_task: str | None) -> str:
    task = sample.get(task_key)
    if isinstance(task, str) and task:
        return task
    if default_task:
        return default_task
    raise KeyError(f"Dataset sample has no {task_key!r} string; configure default_task explicitly")


@dataclass
class RynnValueFrameScorer:
    """Convert sparse RynnValue prefix inference into dense frame signals."""

    model: RynnValueRewardModel
    encoder: RynnValueEncoderProcessorStep
    image_key: str
    dataset_fps: float
    task_key: str = "task"
    default_task: str | None = None
    batch_size: int = 2
    inference_fps: float = DEFAULT_INFERENCE_FPS
    max_frames: int | None = 8
    horizon_s: float | None = None
    robot_description: str | None = None
    camera_description: str | None = None
    use_meta: bool | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1 or None, got {self.max_frames}")
        select_anchor_indices(1, self.dataset_fps, self.inference_fps)
        if self.horizon_s is not None and (not np.isfinite(self.horizon_s) or self.horizon_s <= 0):
            raise ValueError(f"horizon_s must be positive and finite or None, got {self.horizon_s}")
        if self.use_meta is not False and not (self.robot_description or self.camera_description):
            raise ValueError(
                "RynnValue metadata prompting requires robot_description or camera_description; "
                "set use_meta=False only for an intentional ablation"
            )

    @property
    def options(self) -> dict[str, Any]:
        """JSON-serializable adapter settings used in scoring provenance."""
        return {
            "batch_size": self.batch_size,
            "camera_description": self.camera_description,
            "dataset_fps": self.dataset_fps,
            "default_task": self.default_task,
            "horizon_s": self.horizon_s,
            "image_key": self.image_key,
            "inference_fps": self.inference_fps,
            "interpolation": "linear",
            "max_frames": self.max_frames,
            "prefix_sampling": "linspace_rint_include_endpoints",
            "robot_description": self.robot_description,
            "task_key": self.task_key,
            "task_resolution": "first_frame_or_default",
            "use_meta": self.use_meta,
        }

    def __call__(self, dataset: LeRobotDataset, episode_index: int) -> FrameSignals:
        actual_fps = float(dataset.fps)
        if not np.isclose(actual_fps, self.dataset_fps):
            raise ValueError(
                f"Dataset fps changed after scorer construction: expected {self.dataset_fps}, got {actual_fps}"
            )

        episode = dataset.meta.episodes[episode_index]
        episode_start = int(episode["dataset_from_index"])
        episode_end = int(episode["dataset_to_index"])
        num_frames = episode_end - episode_start
        if num_frames < 1:
            raise ValueError(f"Episode {episode_index} has no frames")

        first_sample = dataset[_dataset_position(dataset, episode_start)]
        task = _resolve_task(first_sample, task_key=self.task_key, default_task=self.default_task)
        anchor_indices = select_anchor_indices(num_frames, self.dataset_fps, self.inference_fps)
        anchor_values: list[float] = []

        for start in range(0, len(anchor_indices), self.batch_size):
            batch_anchors = anchor_indices[start : start + self.batch_size]
            samples = []
            for anchor_index in batch_anchors:
                prefix_indices = select_prefix_indices(int(anchor_index), self.max_frames)
                try:
                    frames = torch.stack(
                        [
                            dataset[_dataset_position(dataset, episode_start + int(index))][self.image_key]
                            for index in prefix_indices
                        ]
                    )
                except KeyError as exc:
                    raise KeyError(
                        f"RynnValue scoring expected image key {self.image_key!r} in dataset frames"
                    ) from exc
                samples.append((_video_to_pil(frames, max_frames=None), task))

            encoded = self.encoder.encode_samples(samples)
            model_batch = {f"{RYNNVALUE_FEATURE_PREFIX}{key}": value for key, value in encoded.items()}
            prediction = self.model.predict_remaining_time(model_batch)
            remaining_time = prediction.remaining_time_s
            if remaining_time.ndim != 1 or remaining_time.shape[0] != len(batch_anchors):
                raise ValueError(
                    "RynnValue remaining time must have shape (batch,), "
                    f"got {tuple(remaining_time.shape)} for batch size {len(batch_anchors)}"
                )
            anchor_values.extend(remaining_time.detach().float().cpu().tolist())

        dense_remaining_time = interpolate_anchor_values(
            num_frames,
            anchor_indices,
            np.asarray(anchor_values, dtype=np.float32),
        )
        is_inference_frame = np.zeros(num_frames, dtype=np.bool_)
        is_inference_frame[anchor_indices] = True
        signals: dict[str, np.ndarray] = {
            REMAINING_TIME_SIGNAL: dense_remaining_time,
            POTENTIAL_SIGNAL: -dense_remaining_time,
            IS_INFERENCE_FRAME_SIGNAL: is_inference_frame,
        }
        descriptors = dict(RYNNVALUE_SIGNAL_DESCRIPTORS)
        if self.horizon_s is not None:
            signals[PROGRESS_SIGNAL] = remaining_time_to_progress(dense_remaining_time, self.horizon_s)
            descriptors[PROGRESS_SIGNAL] = PROGRESS_DESCRIPTOR

        return FrameSignals(
            frame_indices=np.arange(num_frames, dtype=np.int64),
            signals=signals,
            descriptors=descriptors,
        )


def make_rynnvalue_frame_scorer(
    model: RynnValueRewardModel,
    config: RynnValueConfig,
    *,
    dataset_fps: float,
    batch_size: int = 2,
    inference_fps: float = DEFAULT_INFERENCE_FPS,
    max_frames: int | None = 8,
    horizon_s: float | None = None,
) -> RynnValueFrameScorer:
    """Construct the standard RynnValue offline scoring adapter."""
    processor_source = config.pretrained_path or config.model_id
    processor_revision = (
        config.pretrained_revision if config.pretrained_path is not None else config.model_revision
    )
    encoder = RynnValueEncoderProcessorStep(
        model_id=processor_source,
        model_revision=processor_revision,
        image_key=config.image_key,
        task_key=config.task_key,
        default_task=config.default_task,
        # Prefix selection belongs to this adapter and must not be repeated by
        # the processor.
        max_frames=None,
        robot_description=config.robot_description,
        camera_description=config.camera_description,
        use_meta=config.use_meta,
    )
    return RynnValueFrameScorer(
        model=model,
        encoder=encoder,
        image_key=config.image_key,
        dataset_fps=dataset_fps,
        task_key=config.task_key,
        default_task=config.default_task,
        batch_size=batch_size,
        inference_fps=inference_fps,
        max_frames=max_frames,
        horizon_s=horizon_s,
        robot_description=config.robot_description,
        camera_description=config.camera_description,
        use_meta=config.use_meta,
    )


def score_rynnvalue_dataset(
    dataset: LeRobotDataset,
    model: RynnValueRewardModel,
    *,
    output_path: Path,
    model_id: str | None = None,
    model_revision: str | None = None,
    episode_indices: Sequence[int] | None = None,
    resume: bool = True,
    batch_size: int = 2,
    inference_fps: float = DEFAULT_INFERENCE_FPS,
    max_frames: int | None = 8,
    horizon_s: float | None = None,
) -> ScoringSummary:
    """Score a dataset with RynnValue using the shared offline runner."""
    from lerobot import __version__

    config = model.config
    resolved_model_id = model_id or config.pretrained_path
    if resolved_model_id is None:
        raise ValueError("model_id is required when the RynnValue config has no pretrained_path")

    scorer = make_rynnvalue_frame_scorer(
        model,
        config,
        dataset_fps=float(dataset.fps),
        batch_size=batch_size,
        inference_fps=inference_fps,
        max_frames=max_frames,
        horizon_s=horizon_s,
    )
    provenance = {
        "lerobot_version": __version__,
        "dataset": {
            "repo_id": dataset.repo_id,
            "revision": dataset.revision,
        },
        "model": {
            "type": config.type,
            "id": resolved_model_id,
            "revision": model_revision if model_revision is not None else config.pretrained_revision,
        },
        "adapter": {
            "id": "lerobot.rynnvalue.causal_prefix",
            "version": 1,
            "options": scorer.options,
        },
    }
    return score_dataset(
        dataset,
        scorer,
        output_path=output_path,
        provenance=provenance,
        episode_indices=episode_indices,
        resume=resume,
    )

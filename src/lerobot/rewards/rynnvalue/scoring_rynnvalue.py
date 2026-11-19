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

"""RynnValue scorer for the shared offline frame-scoring workflow."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

from lerobot.processor import PolicyProcessorPipeline

from ..scoring import FrameSignals, SignalDescriptor
from .modeling_rynnvalue import RynnValueRewardModel
from .processor_rynnvalue import make_rynnvalue_pre_post_processors

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset

REMAINING_TIME_SIGNAL = "reward.rynnvalue.remaining_time_s"

RYNNVALUE_SIGNAL_DESCRIPTORS = {
    REMAINING_TIME_SIGNAL: SignalDescriptor(
        description=(
            "RynnValue predicted seconds until the task is completed, for the trajectory prefix ending at "
            "this frame, linearly interpolated between the frames RynnValue ran on."
        ),
        greater_is_better=False,
        unit="s",
    ),
}


def select_anchor_indices(num_frames: int, dataset_fps: float, inference_fps: float) -> NDArray[np.int64]:
    """Return the frames to run RynnValue on: about `inference_fps` per second, first and last included."""
    stride = max(dataset_fps / inference_fps, 1.0)
    anchors = np.unique(
        np.clip(np.rint(np.arange(0, num_frames, stride)), 0, num_frames - 1).astype(np.int64)
    )
    if anchors[-1] != num_frames - 1:
        anchors = np.append(anchors, num_frames - 1)
    return anchors


def select_prefix_indices(anchor_index: int, max_frames: int | None) -> NDArray[np.int64]:
    """Return `max_frames` evenly spaced frames, in order, of the prefix ending at `anchor_index`.

    Like the official inference, shorter prefixes repeat frames so RynnValue always sees `max_frames`.
    """
    if max_frames is None:
        return np.arange(anchor_index + 1, dtype=np.int64)
    if max_frames == 1:
        return np.asarray([anchor_index], dtype=np.int64)
    return np.linspace(0, anchor_index, max_frames, dtype=np.int64)


class RynnValueFrameScorer:
    """Score each frame with RynnValue's remaining time for the prefix ending at it."""

    name = "rynnvalue"

    def __init__(
        self,
        model: RynnValueRewardModel,
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor

    @property
    def descriptors(self) -> dict[str, SignalDescriptor]:
        """Describe the per-frame remaining-time signal."""
        return RYNNVALUE_SIGNAL_DESCRIPTORS

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
                    "model_id": config.model_id,
                    "model_revision": config.model_revision,
                    "image_key": config.image_key,
                    "task_key": config.task_key,
                    "default_task": config.default_task,
                    "max_frames": config.max_frames,
                    "max_image_side": config.max_image_side,
                    "robot_description": config.robot_description,
                    "camera_description": config.camera_description,
                    "use_meta": config.use_meta,
                    "inference_fps": config.inference_fps,
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
        positions = [
            index if mapping is None else mapping[index] for index in range(episode_start, episode_end)
        ]
        task = dataset[positions[0]].get(config.task_key)

        anchors = select_anchor_indices(num_frames, float(dataset.fps), config.inference_fps)
        remaining_time: NDArray[np.float64] = np.empty(len(anchors), dtype=np.float64)

        self.model.eval()
        with torch.inference_mode():
            for i, anchor in enumerate(anchors):
                frames = torch.stack(
                    [
                        dataset[positions[index]][config.image_key]
                        for index in select_prefix_indices(int(anchor), config.max_frames)
                    ]
                )
                encoded = self.preprocessor({config.image_key: frames.unsqueeze(0), config.task_key: task})
                remaining_time[i] = self.model.predict_remaining_time(encoded).remaining_time_s.item()

        return FrameSignals(
            frame_indices=np.arange(num_frames, dtype=np.int64),
            signals={
                REMAINING_TIME_SIGNAL: np.interp(np.arange(num_frames), anchors, remaining_time).astype(
                    np.float32
                )
            },
        )


def make_rynnvalue_frame_scorer(model: RynnValueRewardModel) -> RynnValueFrameScorer:
    """Construct the standard RynnValue offline frame scorer."""
    # The scorer picks each prefix's frames, so the processor must not subsample them again.
    preprocessor, _ = make_rynnvalue_pre_post_processors(replace(model.config, max_frames=None))
    return RynnValueFrameScorer(model, preprocessor)

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

DEFAULT_INFERENCE_FPS = 1.0
REMAINING_TIME_SIGNAL = "reward.rynnvalue.remaining_time_s"

RYNNVALUE_SIGNAL_DESCRIPTORS = {
    REMAINING_TIME_SIGNAL: SignalDescriptor(
        description=(
            "RynnValue predicted seconds until the task is completed, for the trajectory prefix ending at "
            "this frame, linearly interpolated between the frames RynnValue ran on."
        ),
        unit="s",
        greater_is_better=False,
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
    """Return up to `max_frames` evenly spaced frames, in order, of the prefix ending at `anchor_index`."""
    if max_frames is None or anchor_index < max_frames:
        return np.arange(anchor_index + 1, dtype=np.int64)
    if max_frames == 1:
        return np.asarray([anchor_index], dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, anchor_index, max_frames)).astype(np.int64))


class RynnValueFrameScorer:
    """Score each frame with RynnValue's remaining time for the prefix ending at it."""

    name = "rynnvalue"

    def __init__(
        self,
        model: RynnValueRewardModel,
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        *,
        inference_fps: float = DEFAULT_INFERENCE_FPS,
    ) -> None:
        if inference_fps <= 0:
            raise ValueError(f"inference_fps must be > 0, got {inference_fps}")
        self.model = model
        self.preprocessor = preprocessor
        self.inference_fps = inference_fps

    @property
    def descriptors(self) -> dict[str, SignalDescriptor]:
        """Describe the remaining-time signal before scoring."""
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
                    "max_frames": config.max_frames,
                    "robot_description": config.robot_description,
                    "camera_description": config.camera_description,
                    "use_meta": config.use_meta,
                    "inference_fps": self.inference_fps,
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

        anchors = select_anchor_indices(num_frames, float(dataset.fps), self.inference_fps)
        remaining_time: NDArray[np.float64] = np.empty(len(anchors), dtype=np.float64)

        self.model.eval()
        with torch.inference_mode():
            for i, anchor in enumerate(anchors):
                frames = dataset.get_frames(
                    episode_index,
                    [config.image_key],
                    select_prefix_indices(int(anchor), config.max_frames).tolist(),
                )[config.image_key]
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


def make_rynnvalue_frame_scorer(
    model: RynnValueRewardModel,
    *,
    inference_fps: float = DEFAULT_INFERENCE_FPS,
) -> RynnValueFrameScorer:
    """Construct the standard RynnValue offline frame scorer."""
    # The scorer picks each prefix's frames, so the processor must not subsample them again.
    preprocessor, _ = make_rynnvalue_pre_post_processors(replace(model.config, max_frames=None))
    return RynnValueFrameScorer(model, preprocessor, inference_fps=inference_fps)

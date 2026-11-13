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

"""SARM scorer for the shared offline frame-scoring workflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from lerobot.datasets.utils import delta_window
from lerobot.processor import PolicyProcessorPipeline

from ..scoring import FrameSignals, SignalDescriptor
from .modeling_sarm import SARMRewardModel
from .processor_sarm import make_sarm_pre_post_processors
from .sarm_utils import interpolate_progress

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset

SPARSE_PROGRESS_SIGNAL = "reward.sarm.sparse_progress"
DENSE_PROGRESS_SIGNAL = "reward.sarm.dense_progress"
_HEAD_SIGNALS = {"sparse": SPARSE_PROGRESS_SIGNAL, "dense": DENSE_PROGRESS_SIGNAL}

SARM_SIGNAL_DESCRIPTORS = {
    signal: SignalDescriptor(
        description=f"SARM task progress from its {head} head, predicted from the window of frames around this frame.",
        greater_is_better=True,
        bounds=(0.0, 1.0),
        allow_nan=True,
    )
    for head, signal in _HEAD_SIGNALS.items()
}

logger = logging.getLogger(__name__)


class SARMFrameScorer:
    """Score each frame with SARM's progress at the center of the window around it."""

    name = "sarm"

    def __init__(
        self,
        model: SARMRewardModel,
        *,
        head_mode: Literal["sparse", "dense", "both"] = "sparse",
        stride: int = 1,
        batch_size: int = 32,
    ) -> None:
        if head_mode not in ("sparse", "dense", "both"):
            raise ValueError(f"head_mode must be 'sparse', 'dense' or 'both', got {head_mode!r}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model = model
        self.head_mode = head_mode
        self.stride = stride
        self.batch_size = batch_size
        self._preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self._preprocessor_dataset: LeRobotDataset | None = None

    @property
    def heads(self) -> list[Literal["sparse", "dense"]]:
        """The heads to score: the dense head only exists for dual-head models."""
        dual_heads = self.model.config.uses_dual_heads
        heads: list[Literal["sparse", "dense"]] = []
        if self.head_mode in ("sparse", "both") or not dual_heads:
            heads.append("sparse")
        if self.head_mode in ("dense", "both") and dual_heads:
            heads.append("dense")
        return heads

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
                    "state_key": config.state_key,
                    "heads": self.heads,
                    "stride": self.stride,
                },
            },
        }

    @property
    def descriptors(self) -> dict[str, SignalDescriptor]:
        """Describe the progress signals for the selected heads."""
        return {_HEAD_SIGNALS[head]: SARM_SIGNAL_DESCRIPTORS[_HEAD_SIGNALS[head]] for head in self.heads}

    def _get_preprocessor(
        self, dataset: LeRobotDataset
    ) -> PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]:
        # SARM normalizes state with the scored dataset's statistics. Without dataset metadata the
        # processor needs no subtask annotations, which only training targets use.
        if self._preprocessor is None or self._preprocessor_dataset is not dataset:
            preprocessor, _ = make_sarm_pre_post_processors(
                self.model.config, dataset_stats=dataset.meta.stats
            )
            for step in preprocessor.steps:
                if hasattr(step, "eval"):
                    step.eval()
            self._preprocessor, self._preprocessor_dataset = preprocessor, dataset
        return self._preprocessor

    def score_episode(self, dataset: LeRobotDataset, episode_index: int) -> FrameSignals:
        config = self.model.config
        preprocessor = self._get_preprocessor(dataset)
        episode = dataset.meta.episodes[episode_index]
        episode_start = int(episode["dataset_from_index"])
        episode_end = int(episode["dataset_to_index"])
        num_frames = episode_end - episode_start

        task_index = dataset.get_frames(episode_index, ["task_index"], [0])["task_index"][0]
        task = str(dataset.meta.tasks.iloc[int(task_index)].name)
        keys = [config.image_key]
        if config.state_key in dataset.meta.features:
            keys.append(config.state_key)
        episode_frames = dataset.get_frames(episode_index, keys, list(range(num_frames)))
        frames = episode_frames[config.image_key]
        states = episode_frames.get(config.state_key)

        query_frames = list(range(0, num_frames, self.stride))
        if query_frames[-1] != num_frames - 1:
            query_frames.append(num_frames - 1)
        center = config.n_obs_steps // 2
        values = {head: np.full(len(query_frames), np.nan) for head in self.heads}

        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(query_frames), self.batch_size):
                chunk = query_frames[start : start + self.batch_size]
                # Windows are clamped to the episode, as `LeRobotDataset` does for `delta_timestamps`.
                windows = torch.tensor(
                    [
                        delta_window(frame, config.observation_delta_indices, 0, num_frames)[0]
                        for frame in chunk
                    ]
                )
                batch: dict[str, Any] = {
                    config.image_key: frames[windows],
                    "task": task,
                    "index": torch.tensor([episode_start + frame for frame in chunk]),
                    "episode_index": torch.full((len(chunk),), episode_index),
                }
                if states is not None:
                    batch[config.state_key] = states[windows]
                try:
                    processed = preprocessor(batch)
                    features = {
                        key: processed[key]
                        for key in ("text_features", "video_features", "state_features", "lengths")
                    }
                    for head in self.heads:
                        prediction = self.model.predict_progress(features, head_mode=head)
                        values[head][start : start + len(chunk)] = (
                            prediction.progress[:, center].cpu().numpy()
                        )
                except Exception as exc:
                    logger.warning("Failed to score frames %s of episode %d: %s", chunk, episode_index, exc)

        all_frames = np.arange(num_frames)
        signals: dict[str, NDArray[np.float32]] = {}
        for head, computed in values.items():
            if len(query_frames) == num_frames:
                signals[_HEAD_SIGNALS[head]] = computed.astype(np.float32)
            else:
                signals[_HEAD_SIGNALS[head]] = interpolate_progress(
                    np.asarray(query_frames), computed, all_frames
                )
        return FrameSignals(
            frame_indices=all_frames.astype(np.int64),
            signals=signals,
        )


def make_sarm_frame_scorer(
    model: SARMRewardModel,
    *,
    head_mode: Literal["sparse", "dense", "both"] = "sparse",
    stride: int = 1,
    batch_size: int = 32,
) -> SARMFrameScorer:
    """Construct the standard SARM offline frame scorer."""
    return SARMFrameScorer(model, head_mode=head_mode, stride=stride, batch_size=batch_size)

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

"""One transition representation and small recorder lifecycle for every endpoint."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

from lerobot.utils.constants import ACTION, DEFAULT_FEATURES, DONE, REWARD, SUCCESS, TRUNCATED


@dataclass(frozen=True)
class StepRecord:
    """Canonical experience captured for one applied action."""

    observation: dict[str, Any]
    next_observation: dict[str, Any]
    action: torch.Tensor
    applied_action: torch.Tensor
    reward: float | None
    terminated: bool
    truncated: bool
    success: bool | None
    info: dict[str, Any]
    episode_id: str
    step_index: int
    actor_id: str | None = None
    timestamps: dict[str, float] = field(default_factory=dict)


class EpisodeRecorder(Protocol):
    """Lifecycle implemented by in-memory and LeRobotDataset recorders."""

    def start_episode(self, episode_id: str) -> None:
        """Prepare to receive one episode."""
        ...

    def add(self, record: StepRecord) -> None:
        """Record one transition."""
        ...

    def end_episode(self, episode_id: str) -> None:
        """Commit a successfully completed episode."""
        ...

    def abort_episode(self, episode_id: str) -> None:
        """Discard a partial episode after an exception."""
        ...


class NullRecorder:
    """Recorder used when the caller does not need experience storage."""

    def start_episode(self, episode_id: str) -> None:
        """Ignore episode start."""
        del episode_id

    def add(self, record: StepRecord) -> None:
        """Ignore a transition."""
        del record

    def end_episode(self, episode_id: str) -> None:
        """Ignore successful completion."""
        del episode_id

    def abort_episode(self, episode_id: str) -> None:
        """Ignore aborted completion."""
        del episode_id


class ListRecorder:
    """In-memory recorder useful for tests and small integrations."""

    def __init__(self) -> None:
        """Create an empty recorder."""
        self.records: list[StepRecord] = []
        self.episode_ids: list[str] = []
        self.aborted_episode_ids: list[str] = []
        self._episode_start: int | None = None

    def start_episode(self, episode_id: str) -> None:
        """Open an episode without clearing records from earlier episodes."""
        if self._episode_start is not None:
            raise RuntimeError("An episode is already active")
        self.episode_ids.append(episode_id)
        self._episode_start = len(self.records)

    def add(self, record: StepRecord) -> None:
        """Append one transition to the active episode."""
        if self._episode_start is None:
            raise RuntimeError("Call start_episode() before add()")
        self.records.append(record)

    def end_episode(self, episode_id: str) -> None:
        """Commit the records already held in memory."""
        if self._episode_start is None or self.episode_ids[-1] != episode_id:
            raise RuntimeError(f"Episode {episode_id!r} is not active")
        self._episode_start = None

    def abort_episode(self, episode_id: str) -> None:
        """Remove records appended since the active episode started."""
        if self._episode_start is None or self.episode_ids[-1] != episode_id:
            raise RuntimeError(f"Episode {episode_id!r} is not active")
        del self.records[self._episode_start :]
        self.aborted_episode_ids.append(episode_id)
        self._episode_start = None


class EpisodeDataset(Protocol):
    """Writer subset implemented by ``LeRobotDataset``."""

    @property
    def features(self) -> Mapping[str, Any]:
        """Configured dataset schema."""
        ...

    def add_frame(self, frame: dict[str, Any]) -> None:
        """Buffer one dataset frame."""
        ...

    def save_episode(self) -> None:
        """Persist and clear the current episode buffer."""
        ...

    def clear_episode_buffer(self) -> None:
        """Discard buffered frames."""
        ...


def step_record_to_dataset_frame(record: StepRecord, *, task: str) -> dict[str, Any]:
    """Convert a common record to LeRobotDataset's standard frame names."""
    frame = dict(record.observation)
    frame.update(
        {
            ACTION: record.applied_action.detach().cpu().numpy(),
            REWARD: np.atleast_1d(np.float32(np.nan if record.reward is None else record.reward)),
            DONE: np.atleast_1d(np.bool_(record.terminated or record.truncated)),
            TRUNCATED: np.atleast_1d(np.bool_(record.truncated)),
            "task": task,
            "action.requested": record.action.detach().cpu().numpy(),
        }
    )
    if record.success is not None:
        frame[SUCCESS] = np.atleast_1d(np.bool_(record.success))
    return frame


class DatasetRecorder:
    """Write the common record path into a writable ``LeRobotDataset``."""

    def __init__(
        self,
        dataset: EpisodeDataset,
        *,
        task: str,
        frame_encoder: Callable[[StepRecord], dict[str, Any]] | None = None,
    ) -> None:
        """Bind a dataset, task label and optional custom frame encoder."""
        self.dataset = dataset
        self.task = task
        self.frame_encoder = frame_encoder
        self._episode_id: str | None = None
        self._num_frames = 0

    def start_episode(self, episode_id: str) -> None:
        """Open one episode and reject overlapping writes."""
        if self._episode_id is not None:
            raise RuntimeError(f"Episode {self._episode_id!r} is already active")
        has_pending_frames = getattr(self.dataset, "has_pending_frames", None)
        if has_pending_frames is not None and has_pending_frames():
            raise RuntimeError("Dataset already contains an unfinished episode")
        self._episode_id = episode_id
        self._num_frames = 0

    def add(self, record: StepRecord) -> None:
        """Encode one record and keep fields declared by the dataset schema."""
        if record.episode_id != self._episode_id:
            raise RuntimeError(
                f"Record belongs to episode {record.episode_id!r}, expected {self._episode_id!r}"
            )
        frame = (
            self.frame_encoder(record)
            if self.frame_encoder is not None
            else step_record_to_dataset_frame(record, task=self.task)
        )
        frame["task"] = self.task
        allowed = (set(self.dataset.features) - set(DEFAULT_FEATURES)) | {"task"}
        self.dataset.add_frame({key: value for key, value in frame.items() if key in allowed})
        self._num_frames += 1

    def end_episode(self, episode_id: str) -> None:
        """Persist a non-empty completed episode."""
        self._check_active(episode_id)
        if self._num_frames > 0:
            self.dataset.save_episode()
        self._episode_id = None
        self._num_frames = 0

    def abort_episode(self, episode_id: str) -> None:
        """Clear buffered frames instead of saving a partial failed episode."""
        self._check_active(episode_id)
        # add_frame() can mutate a dataset writer before raising, so cleanup cannot
        # depend on our counter (which advances only after add_frame returns).
        self.dataset.clear_episode_buffer()
        self._episode_id = None
        self._num_frames = 0

    def _check_active(self, episode_id: str) -> None:
        if episode_id != self._episode_id:
            raise RuntimeError(f"Episode {episode_id!r} is not active; active={self._episode_id!r}")


class CallbackRecorder:
    """Forward records to replay, metrics or transport code."""

    def __init__(self, add_record: Callable[[StepRecord], None]) -> None:
        """Bind the callback invoked for each record."""
        self.add_record = add_record

    def start_episode(self, episode_id: str) -> None:
        """Require no setup."""
        del episode_id

    def add(self, record: StepRecord) -> None:
        """Forward one record."""
        self.add_record(record)

    def end_episode(self, episode_id: str) -> None:
        """Require no successful teardown."""
        del episode_id

    def abort_episode(self, episode_id: str) -> None:
        """Require no aborted teardown."""
        del episode_id

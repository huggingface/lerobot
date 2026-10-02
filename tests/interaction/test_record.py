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

from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.interaction import DatasetRecorder, StepRecord, step_record_to_dataset_frame


def make_record(*, reward: float | None = 1.5, success: bool | None = True) -> StepRecord:
    return StepRecord(
        observation={"observation.state": np.array([1.0, 2.0], dtype=np.float32)},
        next_observation={"observation.state": np.array([2.0, 3.0], dtype=np.float32)},
        action=torch.tensor([2.0, 3.0]),
        applied_action=torch.tensor([2.0, 2.5]),
        reward=reward,
        terminated=True,
        truncated=False,
        success=success,
        info={"provider": "test"},
        episode_id="episode-1",
        step_index=0,
        actor_id="actor-1",
    )


def test_dataset_frame_uses_applied_action_and_standard_outcome_fields() -> None:
    frame = step_record_to_dataset_frame(make_record(), task="pick cube")

    np.testing.assert_array_equal(frame["action"], np.array([2.0, 2.5], dtype=np.float32))
    np.testing.assert_array_equal(frame["action.requested"], np.array([2.0, 3.0], dtype=np.float32))
    assert frame["next.reward"].item() == 1.5
    assert frame["next.done"].item() is True
    assert frame["next.truncated"].item() is False
    assert frame["next.success"].item() is True
    assert frame["task"] == "pick cube"


def test_missing_real_world_reward_is_recorded_as_nan() -> None:
    frame = step_record_to_dataset_frame(make_record(reward=None), task="pick cube")
    assert np.isnan(frame["next.reward"]).item()


def test_unknown_real_world_success_is_not_fabricated_as_failure() -> None:
    frame = step_record_to_dataset_frame(make_record(success=None), task="pick cube")
    assert "next.success" not in frame


class FakeDataset:
    def __init__(self) -> None:
        self.features = {
            "observation.state": {},
            "action": {},
            "next.reward": {},
            "next.done": {},
        }
        self.frames: list[dict[str, Any]] = []
        self.saved_episodes = 0
        self.cleared_episodes = 0

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.saved_episodes += 1

    def clear_episode_buffer(self) -> None:
        self.cleared_episodes += 1
        self.frames.clear()

    def has_pending_frames(self) -> bool:
        return bool(self.frames)


def test_dataset_recorder_schema_gates_fields_and_saves_completed_episode() -> None:
    dataset = FakeDataset()
    recorder = DatasetRecorder(dataset, task="pick cube")
    record = make_record()

    recorder.start_episode(record.episode_id)
    recorder.add(record)
    recorder.end_episode(record.episode_id)

    assert set(dataset.frames[0]) == {
        "observation.state",
        "action",
        "next.reward",
        "next.done",
        "task",
    }
    assert dataset.saved_episodes == 1
    assert dataset.cleared_episodes == 0


def test_dataset_recorder_owns_task_and_drops_auto_managed_fields() -> None:
    dataset = FakeDataset()
    dataset.features["timestamp"] = {}
    recorder = DatasetRecorder(
        dataset,
        task="configured task",
        frame_encoder=lambda record: {
            "observation.state": record.observation["observation.state"],
            "action": record.applied_action.numpy(),
            "next.reward": np.atleast_1d(np.float32(record.reward)),
            "next.done": np.atleast_1d(np.bool_(True)),
            "task": "wrong task",
            "timestamp": np.atleast_1d(np.float32(123.0)),
        },
    )
    record = make_record()

    recorder.start_episode(record.episode_id)
    recorder.add(record)
    recorder.end_episode(record.episode_id)

    assert dataset.frames[0]["task"] == "configured task"
    assert "timestamp" not in dataset.frames[0]


def test_dataset_recorder_discards_partial_episode_on_abort() -> None:
    dataset = FakeDataset()
    recorder = DatasetRecorder(dataset, task="pick cube")
    record = make_record()

    recorder.start_episode(record.episode_id)
    recorder.add(record)
    recorder.abort_episode(record.episode_id)

    assert dataset.frames == []
    assert dataset.saved_episodes == 0
    assert dataset.cleared_episodes == 1


def test_dataset_recorder_clears_first_frame_that_fails_after_buffering() -> None:
    class FailingDataset(FakeDataset):
        def add_frame(self, frame: dict[str, Any]) -> None:
            super().add_frame(frame)
            raise OSError("image writer failed")

    dataset = FailingDataset()
    recorder = DatasetRecorder(dataset, task="pick cube")
    record = make_record()
    recorder.start_episode(record.episode_id)

    try:
        recorder.add(record)
    except OSError:
        recorder.abort_episode(record.episode_id)

    assert dataset.frames == []
    assert dataset.cleared_episodes == 1


def test_dataset_recorder_writes_a_real_lerobot_dataset(tmp_path) -> None:
    dataset = LeRobotDataset.create(
        repo_id="tests/shared-runtime",
        fps=30,
        root=tmp_path / "dataset",
        use_videos=False,
        features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
            "action": {"dtype": "float32", "shape": (2,), "names": None},
            "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
            "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        },
    )
    recorder = DatasetRecorder(dataset, task="pick cube")
    record = make_record()

    recorder.start_episode(record.episode_id)
    recorder.add(record)
    recorder.end_episode(record.episode_id)

    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 1
    assert dataset.meta.tasks is not None
    assert "pick cube" in dataset.meta.tasks.index
    dataset.finalize()

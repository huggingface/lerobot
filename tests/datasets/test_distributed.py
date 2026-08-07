#!/usr/bin/env python

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

"""Contracts for distributed LeRobot dataset planning and writing."""

import multiprocessing as mp
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.datasets.distributed import DistributedWritePlan, publish_distributed_metadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
    "action": {"dtype": "float32", "shape": (2,), "names": None},
}


def add_episode_frames(dataset: LeRobotDataset, tasks: tuple[str, ...] | list[str]) -> None:
    """Append deterministic numeric frames matching a task sequence."""
    for frame_index, task in enumerate(tasks):
        dataset.add_frame(
            {
                "observation.state": np.array([frame_index, 1], dtype=np.float32),
                "action": np.array([frame_index, 2], dtype=np.float32),
                "task": task,
            }
        )


def add_video_frames(dataset: LeRobotDataset, tasks: tuple[str, ...] | list[str], video_key: str) -> None:
    """Append deterministic RGB frames matching a task sequence."""
    for frame_index, task in enumerate(tasks):
        dataset.add_frame(
            {
                video_key: np.full((32, 32, 3), frame_index * 40, dtype=np.uint8),
                "action": np.array([frame_index, 2], dtype=np.float32),
                "task": task,
            }
        )


def _write_distributed_spec(root: str, repo_id: str, spec, queue) -> None:
    """Run one worker in an isolated spawned process."""
    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root)
    worker.start_distributed_episode(spec)
    add_episode_frames(worker, spec.tasks)
    result = worker.save_distributed_episode(spec)
    worker.finalize()
    queue.put(result)


def write_specs_in_processes(root: Path, repo_id: str, plan: DistributedWritePlan):
    """Write every plan entry with a separate spawned process."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_write_distributed_spec, args=(str(root), repo_id, spec, queue))
        for spec in plan.episodes
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    return results


def test_write_plan_allocates_contiguous_ranges_and_unique_files():
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[3, 2, 4],
        episode_tasks=[
            ["pick"] * 3,
            ["place"] * 2,
            ["pick"] * 4,
        ],
        chunks_size=2,
        video_keys=["observation.images.top"],
    )

    assert [
        (spec.episode_index, spec.dataset_from_index, spec.dataset_to_index) for spec in plan.episodes
    ] == [
        (0, 0, 3),
        (1, 3, 5),
        (2, 5, 9),
    ]
    assert [(spec.data_chunk_index, spec.data_file_index) for spec in plan.episodes] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert plan.task_to_index == {"pick": 0, "place": 1}


@pytest.mark.parametrize(
    ("episode_lengths", "episode_tasks", "message"),
    [
        ([], [], "at least one episode"),
        ([0], [[]], "must be positive"),
        ([2], [["pick"]], "length"),
    ],
)
def test_write_plan_rejects_invalid_episode_specs(episode_lengths, episode_tasks, message):
    with pytest.raises(ValueError, match=message):
        DistributedWritePlan.from_episode_lengths(
            episode_lengths=episode_lengths,
            episode_tasks=episode_tasks,
            chunks_size=2,
        )


def test_distributed_worker_writes_only_its_planned_artifact(tmp_path):
    root = tmp_path / "dataset"
    driver = LeRobotDataset.create(
        "test/distributed",
        fps=30,
        features=FEATURES,
        root=root,
        use_videos=False,
    )
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[3],
        episode_tasks=[["pick"] * 3],
        chunks_size=1000,
    )
    worker = LeRobotDataset.open_distributed_writer("test/distributed", root=root)
    worker.start_distributed_episode(plan.episodes[0])
    add_episode_frames(worker, plan.episodes[0].tasks)

    result = worker.save_distributed_episode(plan.episodes[0])
    worker.finalize()

    assert result.episode_index == 0
    assert result.data_path == root / Path("data/chunk-000/file-000.parquet")
    assert result.data_path.exists()
    assert not (root / "meta" / "tasks.parquet").exists()
    assert driver.meta.total_episodes == 0


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        (["pick"], "length"),
        (["pick", "place"], "task sequence"),
    ],
)
def test_distributed_worker_rejects_length_and_task_mismatch(tmp_path, tasks, message):
    root = tmp_path / "dataset"
    LeRobotDataset.create(
        "test/distributed",
        fps=30,
        features=FEATURES,
        root=root,
        use_videos=False,
    )
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2],
        episode_tasks=[["pick", "pick"]],
        chunks_size=1000,
    )
    worker = LeRobotDataset.open_distributed_writer("test/distributed", root=root)
    worker.start_distributed_episode(plan.episodes[0])
    add_episode_frames(worker, tasks)

    with pytest.raises(ValueError, match=message):
        worker.save_distributed_episode(plan.episodes[0])


def test_distributed_worker_requires_explicit_episode_start(tmp_path):
    root = tmp_path / "dataset"
    LeRobotDataset.create(
        "test/distributed",
        fps=30,
        features=FEATURES,
        root=root,
        use_videos=False,
    )
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[1],
        episode_tasks=[["pick"]],
        chunks_size=1000,
    )
    worker = LeRobotDataset.open_distributed_writer("test/distributed", root=root)
    add_episode_frames(worker, plan.episodes[0].tasks)

    with pytest.raises(RuntimeError, match="start_distributed_episode"):
        worker.save_distributed_episode(plan.episodes[0])


def test_distributed_worker_cleans_artifacts_when_video_encoding_fails(tmp_path):
    video_key = "observation.images.top"
    features = {
        video_key: {"dtype": "video", "shape": (32, 32, 3), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    root = tmp_path / "video-dataset"
    LeRobotDataset.create("test/distributed-video", fps=10, features=features, root=root, use_videos=True)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[1],
        episode_tasks=[["pick"]],
        chunks_size=1000,
        video_keys=[video_key],
    )
    worker = LeRobotDataset.open_distributed_writer("test/distributed-video", root=root)
    worker.start_distributed_episode(plan.episodes[0])
    add_video_frames(worker, plan.episodes[0].tasks, video_key)

    with (
        patch.object(
            worker.writer, "_encode_temporary_episode_video", side_effect=RuntimeError("encode failed")
        ),
        pytest.raises(RuntimeError, match="encode failed"),
    ):
        worker.save_distributed_episode(plan.episodes[0])

    assert not (root / "data/chunk-000/file-000.parquet").exists()
    assert not (root / f"videos/{video_key}/chunk-000/file-000.mp4").exists()


def test_metadata_publish_restores_previous_metadata_when_swap_fails(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    old_meta = root / "meta"
    old_meta.mkdir(parents=True)
    (old_meta / "marker.txt").write_text("old")
    staging_root = root / "staging"
    staged_meta = staging_root / "meta"
    staged_meta.mkdir(parents=True)
    (staged_meta / "marker.txt").write_text("new")

    real_replace = __import__("os").replace

    def fail_staged_publish(source, destination):
        if Path(source) == staged_meta and Path(destination) == old_meta:
            raise OSError("publish failed")
        return real_replace(source, destination)

    monkeypatch.setattr("lerobot.datasets.distributed.os.replace", fail_staged_publish)

    with pytest.raises(OSError, match="publish failed"):
        publish_distributed_metadata(root, staging_root)

    assert (old_meta / "marker.txt").read_text() == "old"


def test_driver_commits_concurrent_worker_results_into_readable_dataset(tmp_path):
    root = tmp_path / "dataset"
    repo_id = "test/distributed"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[3, 2, 4, 1],
        episode_tasks=[
            ["pick"] * 3,
            ["place"] * 2,
            ["pick", "pick", "place", "place"],
            ["place"],
        ],
        chunks_size=2,
    )
    results = write_specs_in_processes(root, repo_id, plan)

    driver.commit_distributed_results(plan, results)
    driver.finalize()
    loaded = LeRobotDataset(repo_id, root=root)

    assert loaded.num_episodes == 4
    assert loaded.num_frames == 10
    expected_metadata_paths = [
        root / "meta/episodes/chunk-000/file-000.parquet",
        root / "meta/episodes/chunk-000/file-001.parquet",
        root / "meta/episodes/chunk-001/file-000.parquet",
        root / "meta/episodes/chunk-001/file-001.parquet",
    ]
    assert all(path.is_file() for path in expected_metadata_paths)
    assert [loaded[index]["index"].item() for index in range(len(loaded))] == list(range(10))
    assert [loaded[index]["task"] for index in range(len(loaded))] == [
        "pick",
        "pick",
        "pick",
        "place",
        "place",
        "pick",
        "pick",
        "place",
        "place",
        "place",
    ]


@pytest.mark.parametrize("tamper", ["missing", "duplicate", "wrong_range", "missing_artifact"])
def test_driver_rejects_invalid_results_without_publishing_metadata(tmp_path, tamper):
    root = tmp_path / "dataset"
    repo_id = "test/distributed"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2, 2],
        episode_tasks=[["pick"] * 2, ["place"] * 2],
        chunks_size=2,
    )
    results = write_specs_in_processes(root, repo_id, plan)

    if tamper == "missing":
        invalid_results = results[:-1]
    elif tamper == "duplicate":
        invalid_results = [results[0], results[0]]
    elif tamper == "wrong_range":
        invalid_results = [
            replace(result, dataset_from_index=0) if result.episode_index == 1 else result
            for result in results
        ]
    else:
        results[0].data_path.unlink()
        invalid_results = results

    with pytest.raises(ValueError):
        driver.commit_distributed_results(plan, invalid_results)

    assert not (root / "meta" / "tasks.parquet").exists()
    assert not (root / "meta" / "episodes").exists()


def test_distributed_video_results_have_independent_files_and_readable_frames(tmp_path):
    video_key = "observation.images.top"
    features = {
        video_key: {"dtype": "video", "shape": (32, 32, 3), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    root = tmp_path / "video-dataset"
    repo_id = "test/distributed-video"
    driver = LeRobotDataset.create(repo_id, fps=10, features=features, root=root, use_videos=True)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2, 2],
        episode_tasks=[["pick"] * 2, ["place"] * 2],
        chunks_size=1000,
        video_keys=[video_key],
    )
    results = []
    for spec in plan.episodes:
        worker = LeRobotDataset.open_distributed_writer(repo_id, root=root)
        worker.start_distributed_episode(spec)
        add_video_frames(worker, spec.tasks, video_key)
        results.append(worker.save_distributed_episode(spec))
        worker.finalize()

    driver.commit_distributed_results(plan, results)
    driver.finalize()
    loaded = LeRobotDataset(repo_id, root=root, video_backend="pyav")

    video_paths = [loaded.root / loaded.meta.get_video_file_path(index, video_key) for index in range(2)]
    assert video_paths[0] != video_paths[1]
    assert all(path.exists() for path in video_paths)
    assert results[0].video_info[video_key]["video.codec"]
    assert loaded.meta.features[video_key]["info"]["video.codec"]
    assert loaded[0][video_key].shape == (3, 32, 32)


def test_driver_rejects_plan_with_video_keys_that_differ_from_dataset_schema(tmp_path):
    video_key = "observation.images.top"
    root = tmp_path / "video-dataset"
    LeRobotDataset.create(
        "test/distributed-video",
        fps=10,
        features={
            video_key: {"dtype": "video", "shape": (32, 32, 3), "names": None},
            "action": {"dtype": "float32", "shape": (2,), "names": None},
        },
        root=root,
        use_videos=True,
    )
    driver = LeRobotDataset.open_distributed_writer("test/distributed-video", root=root)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[1],
        episode_tasks=[["pick"]],
        chunks_size=1000,
        video_keys=[],
    )

    with pytest.raises(ValueError, match="video_keys"):
        driver.commit_distributed_results(plan, [])

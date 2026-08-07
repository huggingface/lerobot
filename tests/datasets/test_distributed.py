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

import json
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.configs import DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT, DepthEncoderConfig
from lerobot.datasets.distributed import (
    DistributedWritePlan,
    DistributedWriteSession,
    publish_distributed_metadata,
)
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pyav_utils import get_codec

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
    "action": {"dtype": "float32", "shape": (2,), "names": None},
}

RGB_IMAGE_KEY = "observation.images.left"
RGB_VIDEO_KEY = "observation.images.right"
DEPTH_IMAGE_KEY = "observation.depth.front"
DEPTH_VIDEO_KEY = "observation.depth.wrist"


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


def add_image_frames(dataset: LeRobotDataset, tasks: tuple[str, ...] | list[str], image_key: str) -> None:
    """Append deterministic raw image frames matching a task sequence."""
    for frame_index, task in enumerate(tasks):
        dataset.add_frame(
            {
                image_key: np.full((32, 32, 3), frame_index * 40, dtype=np.uint8),
                "action": np.array([frame_index, 2], dtype=np.float32),
                "task": task,
            }
        )


def add_multimodal_frames(dataset: LeRobotDataset, tasks: tuple[str, ...] | list[str]) -> None:
    """Append deterministic frames containing raw-image, video, string, and tensor features."""
    for frame_index, task in enumerate(tasks):
        dataset.add_frame(
            {
                RGB_IMAGE_KEY: np.full((3, 24, 32), 20 + frame_index, dtype=np.uint8),
                RGB_VIDEO_KEY: np.full((24, 32, 3), 80 + frame_index, dtype=np.uint8),
                "observation.history": np.full((2, 3, 4), frame_index, dtype=np.float32),
                "caption": f"frame-{frame_index}",
                "action": np.array([frame_index, 2], dtype=np.float32),
                "task": task,
            }
        )


def add_depth_frames(dataset: LeRobotDataset, tasks: tuple[str, ...] | list[str], depth_key: str) -> None:
    """Append deterministic uint16 depth frames in millimetres."""
    for frame_index, task in enumerate(tasks):
        dataset.add_frame(
            {
                depth_key: np.full((1, 24, 32), 1_000 + frame_index * 100, dtype=np.uint16),
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


def _write_distributed_spec_with_session(root: str, repo_id: str, spec, session, queue) -> None:
    """Run one session-aware worker in an isolated spawned process."""
    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root, session=session)
    worker.start_distributed_episode(spec)
    add_episode_frames(worker, spec.tasks)
    result = worker.save_distributed_episode(spec)
    worker.finalize()
    queue.put(result.episode_index)


def _write_multimodal_spec_from_shared_root(root: str, repo_id: str, spec, queue) -> None:
    """Simulate a remote worker that reconstructs all state from shared storage."""
    session = LeRobotDataset.resume_distributed_write_session(root)
    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root, session=session)
    worker.start_distributed_episode(spec)
    add_multimodal_frames(worker, spec.tasks)
    result = worker.save_distributed_episode(spec)
    worker.finalize()
    queue.put(result.episode_index)


def _write_depth_spec_from_shared_root(
    root: str, repo_id: str, spec, depth_key: str, use_video: bool, queue
) -> None:
    """Simulate a remote depth worker using only shared-storage state."""
    session = LeRobotDataset.resume_distributed_write_session(root)
    writer_kwargs = {"depth_encoder": DepthEncoderConfig(use_log=False)} if use_video else {}
    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root, session=session, **writer_kwargs)
    worker.start_distributed_episode(spec)
    add_depth_frames(worker, spec.tasks, depth_key)
    result = worker.save_distributed_episode(spec)
    worker.finalize()
    queue.put(result.episode_index)


def run_one_depth_worker(root: Path, repo_id: str, spec, depth_key: str, *, use_video: bool) -> None:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_write_depth_spec_from_shared_root,
        args=(str(root), repo_id, spec, depth_key, use_video, queue),
    )
    process.start()
    assert queue.get(timeout=60) == spec.episode_index
    process.join(timeout=60)
    assert process.exitcode == 0


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


def test_distributed_worker_removes_raw_image_staging_after_success(tmp_path):
    image_key = "observation.images.top"
    features = {
        image_key: {"dtype": "image", "shape": (32, 32, 3), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    root = tmp_path / "image-dataset"
    repo_id = "test/distributed-image"
    LeRobotDataset.create(repo_id, fps=30, features=features, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2], episode_tasks=[["pick", "pick"]], chunks_size=1000
    )
    spec = plan.episodes[0]
    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root)
    worker.start_distributed_episode(spec)
    add_image_frames(worker, spec.tasks, image_key)
    staging_dir = worker.writer._get_image_file_dir(spec.episode_index, image_key)

    result = worker.save_distributed_episode(spec)
    worker.finalize()

    assert result.data_path.is_file()
    assert not staging_dir.exists()


def test_distributed_result_records_depth_input_unit(tmp_path):
    depth_key = "observation.depth.front"
    features = {
        depth_key: {
            "dtype": "image",
            "shape": (1, 24, 32),
            "names": None,
            "info": {"is_depth_map": True},
        },
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    root = tmp_path / "depth-dataset"
    repo_id = "test/distributed-depth-result"
    LeRobotDataset.create(repo_id, fps=10, features=features, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[1], episode_tasks=[["pick"]], chunks_size=1000
    )
    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root)
    worker.start_distributed_episode(plan.episodes[0])
    worker.add_frame(
        {
            depth_key: np.full((1, 24, 32), 1_000, dtype=np.uint16),
            "action": np.zeros(2, dtype=np.float32),
            "task": "pick",
        }
    )

    result = worker.save_distributed_episode(plan.episodes[0])
    worker.finalize()

    assert result.depth_units == {depth_key: DEPTH_MILLIMETER_UNIT}


def test_distributed_commit_rejects_inconsistent_depth_units(tmp_path):
    depth_key = "observation.depth.front"
    features = {
        depth_key: {
            "dtype": "image",
            "shape": (1, 24, 32),
            "names": None,
            "info": {"is_depth_map": True},
        },
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    root = tmp_path / "mixed-depth-units"
    repo_id = "test/distributed-mixed-depth-units"
    driver = LeRobotDataset.create(repo_id, fps=10, features=features, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[1, 1], episode_tasks=[["pick"], ["place"]], chunks_size=1000
    )
    results = []
    for spec, value in zip(
        plan.episodes,
        [np.full((1, 24, 32), 1_000, dtype=np.uint16), np.full((1, 24, 32), 1.0, dtype=np.float32)],
        strict=True,
    ):
        worker = LeRobotDataset.open_distributed_writer(repo_id, root=root)
        worker.start_distributed_episode(spec)
        worker.add_frame({depth_key: value, "action": np.zeros(2, dtype=np.float32), "task": spec.tasks[0]})
        results.append(worker.save_distributed_episode(spec))
        worker.finalize()

    assert results[0].depth_units == {depth_key: DEPTH_MILLIMETER_UNIT}
    assert results[1].depth_units == {depth_key: DEPTH_METER_UNIT}
    with pytest.raises(ValueError, match="disagree on depth unit"):
        driver.commit_distributed_results(plan, results)

    assert not (root / "meta" / "tasks.parquet").exists()


@pytest.mark.parametrize(
    ("depth_units", "message"),
    [
        ({}, "missing depth units"),
        ({DEPTH_IMAGE_KEY: "cm"}, "unsupported depth unit"),
        ({DEPTH_IMAGE_KEY: DEPTH_MILLIMETER_UNIT, "observation.depth.unknown": "mm"}, "unknown depth keys"),
    ],
)
def test_distributed_commit_rejects_invalid_depth_unit_metadata(tmp_path, depth_units, message):
    features = {
        DEPTH_IMAGE_KEY: {
            "dtype": "image",
            "shape": (1, 24, 32),
            "names": None,
            "info": {"is_depth_map": True},
        },
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    root = tmp_path / "invalid-depth-units"
    repo_id = "test/distributed-invalid-depth-units"
    driver = LeRobotDataset.create(repo_id, fps=10, features=features, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[1], episode_tasks=[["pick"]], chunks_size=1000
    )
    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root)
    worker.start_distributed_episode(plan.episodes[0])
    worker.add_frame(
        {
            DEPTH_IMAGE_KEY: np.full((1, 24, 32), 1_000, dtype=np.uint16),
            "action": np.zeros(2, dtype=np.float32),
            "task": "pick",
        }
    )
    result = worker.save_distributed_episode(plan.episodes[0])
    worker.finalize()

    with pytest.raises(ValueError, match=message):
        driver.commit_distributed_results(plan, [replace(result, depth_units=depth_units)])

    assert not (root / "meta" / "tasks.parquet").exists()


def test_distributed_commit_uses_declared_depth_unit_for_legacy_session_result(tmp_path):
    features = {
        DEPTH_IMAGE_KEY: {
            "dtype": "image",
            "shape": (1, 24, 32),
            "names": None,
            "info": {"is_depth_map": True, "depth_unit": DEPTH_MILLIMETER_UNIT},
        },
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    root = tmp_path / "legacy-depth-session"
    repo_id = "test/distributed-legacy-depth-session"
    driver = LeRobotDataset.create(repo_id, fps=10, features=features, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[1], episode_tasks=[["pick"]], chunks_size=1000
    )
    session = driver.create_distributed_write_session(plan)
    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root, session=session)
    worker.start_distributed_episode(plan.episodes[0])
    worker.add_frame(
        {
            DEPTH_IMAGE_KEY: np.full((1, 24, 32), 1_000, dtype=np.uint16),
            "action": np.zeros(2, dtype=np.float32),
            "task": "pick",
        }
    )
    result = worker.save_distributed_episode(plan.episodes[0])
    worker.finalize()

    result_path = session.result_path(result.episode_index)
    legacy_payload = json.loads(result_path.read_text())
    legacy_payload.pop("depth_units")
    result_path.write_text(json.dumps(legacy_payload))

    resumed = LeRobotDataset.resume_distributed_write_session(root)
    restored_results = resumed.load_results()
    assert restored_results[0].depth_units == {}

    driver.commit_distributed_results(resumed.plan, restored_results)
    driver.finalize()
    loaded = LeRobotDataset(repo_id, root=root, depth_output_unit=DEPTH_MILLIMETER_UNIT)

    assert loaded.meta.features[DEPTH_IMAGE_KEY]["info"]["depth_unit"] == DEPTH_MILLIMETER_UNIT
    np.testing.assert_allclose(loaded[0][DEPTH_IMAGE_KEY].numpy(), 1_000)


def test_distributed_spawn_workers_write_multicamera_and_complex_features(tmp_path):
    root = tmp_path / "multimodal-dataset"
    repo_id = "test/distributed-multimodal"
    features = {
        RGB_IMAGE_KEY: {"dtype": "image", "shape": (3, 24, 32), "names": None},
        RGB_VIDEO_KEY: {"dtype": "video", "shape": (3, 24, 32), "names": None},
        "observation.history": {"dtype": "float32", "shape": (2, 3, 4), "names": None},
        "caption": {"dtype": "string", "shape": (1,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    driver = LeRobotDataset.create(repo_id, fps=10, features=features, root=root, use_videos=True)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2, 2],
        episode_tasks=[["pick", "pick"], ["place", "place"]],
        chunks_size=1000,
        video_keys=[RGB_VIDEO_KEY],
    )
    driver.create_distributed_write_session(plan)

    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_write_multimodal_spec_from_shared_root,
            args=(str(root), repo_id, spec, queue),
        )
        for spec in plan.episodes
    ]
    for process in processes:
        process.start()
    completed = sorted(queue.get(timeout=60) for _ in processes)
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    resumed = LeRobotDataset.resume_distributed_write_session(root)
    driver.commit_distributed_results(resumed.plan, resumed.load_results())
    driver.finalize()
    loaded = LeRobotDataset(repo_id, root=root, video_backend="pyav")

    assert completed == [0, 1]
    assert loaded.num_episodes == 2
    assert loaded[0][RGB_IMAGE_KEY].shape == (3, 24, 32)
    assert loaded[0][RGB_VIDEO_KEY].shape == (3, 24, 32)
    assert loaded[0]["observation.history"].shape == (2, 3, 4)
    assert loaded[0]["caption"] == "frame-0"
    assert loaded[2]["task"] == "place"
    assert not list((root / "images" / RGB_IMAGE_KEY).rglob("*.png"))
    assert len(list((root / "videos" / RGB_VIDEO_KEY).rglob("*.mp4"))) == 2


def test_distributed_depth_image_preserves_units_stats_and_values(tmp_path):
    root = tmp_path / "depth-image-dataset"
    repo_id = "test/distributed-depth-image"
    features = {
        DEPTH_IMAGE_KEY: {
            "dtype": "image",
            "shape": (1, 24, 32),
            "names": None,
            "info": {"is_depth_map": True},
        },
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    driver = LeRobotDataset.create(repo_id, fps=10, features=features, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2], episode_tasks=[["pick", "pick"]], chunks_size=1000
    )
    driver.create_distributed_write_session(plan)
    run_one_depth_worker(root, repo_id, plan.episodes[0], DEPTH_IMAGE_KEY, use_video=False)
    resumed = LeRobotDataset.resume_distributed_write_session(root)
    driver.commit_distributed_results(resumed.plan, resumed.load_results())
    driver.finalize()
    loaded = LeRobotDataset(repo_id, root=root, depth_output_unit=DEPTH_MILLIMETER_UNIT)

    assert loaded.meta.features[DEPTH_IMAGE_KEY]["info"]["depth_unit"] == DEPTH_MILLIMETER_UNIT
    np.testing.assert_allclose(loaded[0][DEPTH_IMAGE_KEY].numpy(), 1_000)
    np.testing.assert_allclose(loaded[1][DEPTH_IMAGE_KEY].numpy(), 1_100)
    np.testing.assert_allclose(loaded.meta.stats[DEPTH_IMAGE_KEY]["mean"], 1_050)
    assert not list((root / "images" / DEPTH_IMAGE_KEY).rglob("*.tiff"))


@pytest.mark.skipif(get_codec("hevc") is None, reason="'hevc' not in local FFmpeg build")
def test_distributed_depth_video_preserves_depth_metadata_and_values(tmp_path):
    root = tmp_path / "depth-video-dataset"
    repo_id = "test/distributed-depth-video"
    features = {
        DEPTH_VIDEO_KEY: {
            "dtype": "video",
            "shape": (1, 24, 32),
            "names": None,
            "info": {"is_depth_map": True},
        },
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    driver = LeRobotDataset.create(repo_id, fps=10, features=features, root=root, use_videos=True)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[4],
        episode_tasks=[["pick"] * 4],
        chunks_size=1000,
        video_keys=[DEPTH_VIDEO_KEY],
    )
    driver.create_distributed_write_session(plan)
    run_one_depth_worker(root, repo_id, plan.episodes[0], DEPTH_VIDEO_KEY, use_video=True)
    resumed = LeRobotDataset.resume_distributed_write_session(root)
    driver.commit_distributed_results(resumed.plan, resumed.load_results())
    driver.finalize()
    loaded = LeRobotDataset(repo_id, root=root, video_backend="pyav", depth_output_unit=DEPTH_MILLIMETER_UNIT)

    info = loaded.meta.features[DEPTH_VIDEO_KEY]["info"]
    assert info["is_depth_map"] is True
    assert info["depth_unit"] == DEPTH_MILLIMETER_UNIT
    assert info["video.codec"] == "hevc"
    assert info["video.pix_fmt"] == "gray12le"
    assert loaded[0][DEPTH_VIDEO_KEY].shape == (1, 24, 32)
    np.testing.assert_allclose(loaded[0][DEPTH_VIDEO_KEY].numpy(), 1_000, atol=10)


def test_distributed_driver_rejects_result_from_different_session_mount_path(tmp_path):
    root = tmp_path / "dataset"
    alternate_mount = tmp_path / "alternate-mount"
    repo_id = "test/distributed-mount-path"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[1], episode_tasks=[["pick"]], chunks_size=1000
    )
    driver.create_distributed_write_session(plan)
    alternate_mount.symlink_to(root, target_is_directory=True)
    worker_session = LeRobotDataset.resume_distributed_write_session(alternate_mount)
    worker = LeRobotDataset.open_distributed_writer(repo_id, root=alternate_mount, session=worker_session)
    worker.start_distributed_episode(plan.episodes[0])
    add_episode_frames(worker, plan.episodes[0].tasks)
    worker.save_distributed_episode(plan.episodes[0])
    worker.finalize()
    driver_session = LeRobotDataset.resume_distributed_write_session(root)

    with pytest.raises(ValueError, match="wrote unexpected path"):
        driver_session.load_results()


def test_distributed_write_session_persists_worker_result_and_resumes_pending_specs(tmp_path):
    root = tmp_path / "dataset"
    repo_id = "test/distributed-session"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2, 2],
        episode_tasks=[["pick"] * 2, ["place"] * 2],
        chunks_size=1000,
    )
    session = driver.create_distributed_write_session(plan)

    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root, session=session)
    worker.start_distributed_episode(plan.episodes[0])
    add_episode_frames(worker, plan.episodes[0].tasks)
    result = worker.save_distributed_episode(plan.episodes[0])
    worker.finalize()

    assert session.result_path(result.episode_index).is_file()

    resumed = LeRobotDataset.resume_distributed_write_session(root)

    assert resumed.plan == plan
    restored_results = resumed.load_results()
    assert len(restored_results) == 1
    assert restored_results[0].episode_index == result.episode_index
    assert restored_results[0].data_path == result.data_path
    assert restored_results[0].tasks == result.tasks
    assert restored_results[0].stats.keys() == result.stats.keys()
    assert resumed.pending_specs() == [plan.episodes[1]]


def test_distributed_write_session_publishes_plan_and_results_directory_atomically(tmp_path):
    root = tmp_path / "dataset"
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2],
        episode_tasks=[["pick"] * 2],
        chunks_size=1000,
    )

    session = DistributedWriteSession.create(root, plan)

    assert session.session_root.is_dir()
    assert (session.session_root / "plan.json").is_file()
    assert session.results_root.is_dir()
    assert not list(root.glob(".distributed-write-session.*.tmp"))


def test_distributed_write_apis_reject_non_default_storage_format(tmp_path):
    root = tmp_path / "dataset"
    repo_id = "test/distributed-storage-format"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[1],
        episode_tasks=[["pick"]],
        chunks_size=1000,
    )
    driver.meta.info.storage_format = "lance"
    write_info(driver.meta.info, root)

    with pytest.raises(ValueError, match=r"storage_format='lerobot'"):
        driver.create_distributed_write_session(plan)
    with pytest.raises(ValueError, match=r"storage_format='lerobot'"):
        driver.commit_distributed_results(plan, [])
    with pytest.raises(ValueError, match=r"storage_format='lerobot'"):
        LeRobotDataset.open_distributed_writer(repo_id, root=root)

    assert not (root / "data").exists()
    assert not (root / ".distributed-write-session").exists()


def test_distributed_writer_rejects_remote_root_before_io():
    with pytest.raises(ValueError, match="local dataset root"):
        LeRobotDataset.open_distributed_writer(
            "test/distributed-remote",
            root="hf://datasets/test/distributed-remote",
        )


def test_distributed_write_session_persists_result_from_spawned_worker(tmp_path):
    root = tmp_path / "dataset"
    repo_id = "test/distributed-session"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2, 2],
        episode_tasks=[["pick"] * 2, ["place"] * 2],
        chunks_size=1000,
    )
    session = driver.create_distributed_write_session(plan)
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_write_distributed_spec_with_session,
        args=(str(root), repo_id, plan.episodes[0], session, queue),
    )

    process.start()
    assert queue.get(timeout=30) == plan.episodes[0].episode_index
    process.join(timeout=30)
    assert process.exitcode == 0

    resumed = LeRobotDataset.resume_distributed_write_session(root)

    assert [result.episode_index for result in resumed.load_results()] == [0]
    assert resumed.pending_specs() == [plan.episodes[1]]


def test_distributed_write_session_does_not_reuse_artifact_without_result_record(tmp_path):
    root = tmp_path / "dataset"
    repo_id = "test/distributed-session"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2, 2],
        episode_tasks=[["pick"] * 2, ["place"] * 2],
        chunks_size=1000,
    )
    driver.create_distributed_write_session(plan)

    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root)
    worker.start_distributed_episode(plan.episodes[0])
    add_episode_frames(worker, plan.episodes[0].tasks)
    worker.save_distributed_episode(plan.episodes[0])
    worker.finalize()

    resumed = DistributedWriteSession.resume(root)

    assert resumed.pending_specs() == list(plan.episodes)
    assert resumed.load_results() == []
    assert (root / "data/chunk-000/file-000.parquet").is_file()


def test_distributed_write_session_cleans_orphan_artifacts_for_pending_specs(tmp_path):
    root = tmp_path / "dataset"
    repo_id = "test/distributed-session"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2],
        episode_tasks=[["pick"] * 2],
        chunks_size=1000,
    )
    session = driver.create_distributed_write_session(plan)

    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root)
    worker.start_distributed_episode(plan.episodes[0])
    add_episode_frames(worker, plan.episodes[0].tasks)
    worker.save_distributed_episode(plan.episodes[0])
    worker.finalize()

    session.cleanup_orphaned_artifacts()

    assert session.pending_specs() == list(plan.episodes)
    assert not (root / "data/chunk-000/file-000.parquet").exists()


def test_distributed_write_session_cleanup_discards_invalid_result_record(tmp_path):
    root = tmp_path / "dataset"
    repo_id = "test/distributed-session"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2],
        episode_tasks=[["pick"] * 2],
        chunks_size=1000,
    )
    session = driver.create_distributed_write_session(plan)

    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root, session=session)
    worker.start_distributed_episode(plan.episodes[0])
    add_episode_frames(worker, plan.episodes[0].tasks)
    result = worker.save_distributed_episode(plan.episodes[0])
    worker.finalize()
    result.data_path.unlink()

    session.cleanup_orphaned_artifacts()

    assert not session.result_path(result.episode_index).exists()
    assert session.pending_specs() == list(plan.episodes)


def test_distributed_write_session_cleanup_discards_corrupt_artifact_record(tmp_path):
    root = tmp_path / "dataset"
    repo_id = "test/distributed-session"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2],
        episode_tasks=[["pick"] * 2],
        chunks_size=1000,
    )
    session = driver.create_distributed_write_session(plan)

    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root, session=session)
    worker.start_distributed_episode(plan.episodes[0])
    add_episode_frames(worker, plan.episodes[0].tasks)
    result = worker.save_distributed_episode(plan.episodes[0])
    worker.finalize()
    result.data_path.write_text("corrupt")

    session.cleanup_orphaned_artifacts()

    assert not session.result_path(result.episode_index).exists()
    assert not result.data_path.exists()
    assert session.pending_specs() == list(plan.episodes)


def test_distributed_write_session_resumes_and_commits_reused_worker_results(tmp_path):
    root = tmp_path / "dataset"
    repo_id = "test/distributed-session"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2, 2],
        episode_tasks=[["pick"] * 2, ["place"] * 2],
        chunks_size=1000,
    )
    session = driver.create_distributed_write_session(plan)

    first_worker = LeRobotDataset.open_distributed_writer(repo_id, root=root, session=session)
    first_worker.start_distributed_episode(plan.episodes[0])
    add_episode_frames(first_worker, plan.episodes[0].tasks)
    first_worker.save_distributed_episode(plan.episodes[0])
    first_worker.finalize()

    resumed = LeRobotDataset.resume_distributed_write_session(root)
    for spec in resumed.pending_specs():
        worker = LeRobotDataset.open_distributed_writer(repo_id, root=root, session=resumed)
        worker.start_distributed_episode(spec)
        add_episode_frames(worker, spec.tasks)
        worker.save_distributed_episode(spec)
        worker.finalize()

    driver.commit_distributed_results(resumed.plan, resumed.load_results())
    driver.finalize()
    loaded = LeRobotDataset(repo_id, root=root)

    assert loaded.num_episodes == 2
    assert loaded.num_frames == 4
    assert [loaded[index]["task"] for index in range(len(loaded))] == [
        "pick",
        "pick",
        "place",
        "place",
    ]


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


def test_metadata_publish_rejects_missing_destination_metadata(tmp_path):
    root = tmp_path / "dataset"
    staging_root = root / "staging"
    staged_meta = staging_root / "meta"
    staged_meta.mkdir(parents=True)
    (staged_meta / "marker.txt").write_text("new")

    with pytest.raises(FileNotFoundError, match=r"LeRobotDataset\.create\(\)"):
        publish_distributed_metadata(root, staging_root)

    assert staged_meta.is_dir()
    assert not list(root.glob(".meta-distributed-backup-*"))


def test_commit_preserves_staged_metadata_when_destination_metadata_is_missing(tmp_path):
    root = tmp_path / "dataset"
    repo_id = "test/distributed-missing-meta"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[1],
        episode_tasks=[["pick"]],
        chunks_size=1000,
    )
    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root)
    worker.start_distributed_episode(plan.episodes[0])
    add_episode_frames(worker, plan.episodes[0].tasks)
    result = worker.save_distributed_episode(plan.episodes[0])
    worker.finalize()
    (root / "meta").rename(root / "removed-meta")

    with pytest.raises(RuntimeError, match=r"Staged metadata remains at .*\.distributed-metadata-.*meta"):
        driver.commit_distributed_results(plan, [result])

    staging_roots = list(root.glob(".distributed-metadata-*"))
    assert len(staging_roots) == 1
    assert (staging_roots[0] / "meta" / "info.json").is_file()
    assert not list(root.glob(".meta-distributed-backup-*"))


def test_commit_reports_staged_metadata_when_atomic_publish_fails(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    repo_id = "test/distributed-publish-failure"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[1],
        episode_tasks=[["pick"]],
        chunks_size=1000,
    )
    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root)
    worker.start_distributed_episode(plan.episodes[0])
    add_episode_frames(worker, plan.episodes[0].tasks)
    result = worker.save_distributed_episode(plan.episodes[0])
    worker.finalize()

    real_replace = __import__("os").replace

    def fail_staged_publish(source, destination):
        source = Path(source)
        if source.name == "meta" and source.parent.name.startswith(".distributed-metadata-"):
            raise OSError("publish failed")
        return real_replace(source, destination)

    monkeypatch.setattr("lerobot.datasets.distributed.os.replace", fail_staged_publish)

    with pytest.raises(RuntimeError, match=r"Staged metadata remains at .*\.distributed-metadata-.*meta"):
        driver.commit_distributed_results(plan, [result])

    assert (root / "meta" / "info.json").is_file()
    staging_roots = list(root.glob(".distributed-metadata-*"))
    assert len(staging_roots) == 1
    assert (staging_roots[0] / "meta" / "info.json").is_file()
    assert not list(root.glob(".meta-distributed-backup-*"))


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

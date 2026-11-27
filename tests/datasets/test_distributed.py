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

"""Tests for unified distributed write planning, session recovery, depth/multimodal features, and metadata commit."""

import multiprocessing as mp
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from lerobot.configs import DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT
from lerobot.datasets.distributed import (
    DistributedShardSpec,
    DistributedWritePlan,
    publish_distributed_metadata,
    recover_distributed_metadata,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
    "action": {"dtype": "float32", "shape": (2,), "names": None},
}
VIDEO_KEY = "observation.images.top"
IMAGE_KEY = "observation.images.wrist"
DEPTH_IMAGE_KEY = "observation.depth.wrist"
DEPTH_VIDEO_KEY = "observation.depth.top"
VIDEO_FEATURES = {
    **FEATURES,
    VIDEO_KEY: {"dtype": "video", "shape": (24, 32, 3), "names": None},
}
MULTIMODAL_FEATURES = {
    **FEATURES,
    VIDEO_KEY: {"dtype": "video", "shape": (24, 32, 3), "names": None},
    IMAGE_KEY: {"dtype": "image", "shape": (24, 32, 3), "names": None},
    "caption": {"dtype": "string", "shape": (1,), "names": None},
    "history": {"dtype": "float32", "shape": (2, 3), "names": None},
}
DEPTH_IMAGE_FEATURES = {
    **FEATURES,
    DEPTH_IMAGE_KEY: {"dtype": "image", "shape": (1, 24, 32), "names": None, "info": {"is_depth_map": True}},
}
DEPTH_VIDEO_FEATURES = {
    **FEATURES,
    DEPTH_VIDEO_KEY: {"dtype": "video", "shape": (1, 24, 32), "names": None, "info": {"is_depth_map": True}},
}


def make_episode_frames(tasks: tuple[str, ...] | list[str]) -> list[dict]:
    return [
        {
            "observation.state": np.array([index, index + 1], dtype=np.float32),
            "action": np.array([index * 2, index * 2 + 1], dtype=np.float32),
            "task": task,
        }
        for index, task in enumerate(tasks)
    ]


def make_video_frames(tasks: tuple[str, ...] | list[str], video_key: str) -> list[dict]:
    return [
        {
            "observation.state": np.array([index, index + 1], dtype=np.float32),
            "action": np.array([index * 2, index * 2 + 1], dtype=np.float32),
            video_key: np.full((24, 32, 3), (index + 1) * 40, dtype=np.uint8),
            "task": task,
        }
        for index, task in enumerate(tasks)
    ]


def make_image_frames(tasks: tuple[str, ...] | list[str], image_key: str) -> list[dict]:
    return [
        {
            "observation.state": np.array([index, index + 1], dtype=np.float32),
            "action": np.array([index * 2, index * 2 + 1], dtype=np.float32),
            image_key: np.full((24, 32, 3), (index + 1) * 30, dtype=np.uint8),
            "task": task,
        }
        for index, task in enumerate(tasks)
    ]


def make_multimodal_frames(tasks: tuple[str, ...] | list[str]) -> list[dict]:
    return [
        {
            "observation.state": np.array([index, index + 1], dtype=np.float32),
            "action": np.array([index * 2, index * 2 + 1], dtype=np.float32),
            VIDEO_KEY: np.full((24, 32, 3), (index + 1) * 40, dtype=np.uint8),
            IMAGE_KEY: np.full((24, 32, 3), (index + 1) * 30, dtype=np.uint8),
            "caption": f"{task}-{index}",
            "history": np.full((2, 3), index + 0.5, dtype=np.float32),
            "task": task,
        }
        for index, task in enumerate(tasks)
    ]


def make_depth_frames(
    tasks: tuple[str, ...] | list[str], depth_key: str, *, use_meters: bool = False
) -> list[dict]:
    frames = []
    for index, task in enumerate(tasks):
        if use_meters:
            depth = np.full((1, 24, 32), 1.0 + index * 0.5, dtype=np.float32)
        else:
            depth = np.full((1, 24, 32), 1_000 + index * 500, dtype=np.uint16)
        frames.append(
            {
                "observation.state": np.array([index, index + 1], dtype=np.float32),
                "action": np.array([index * 2, index * 2 + 1], dtype=np.float32),
                depth_key: depth,
                "task": task,
            }
        )
    return frames


def write_shard_in_process(
    root: str, repo_id: str, shard: DistributedShardSpec, frame_kind: str = "numeric"
) -> None:
    session = LeRobotDataset.resume_distributed_write_session(root)
    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root, session=session)
    try:
        if frame_kind == "multimodal":
            inputs = [make_multimodal_frames(spec.tasks) for spec in shard.episodes]
        elif frame_kind == "depth_image":
            inputs = [make_depth_frames(spec.tasks, DEPTH_IMAGE_KEY) for spec in shard.episodes]
        elif frame_kind == "depth_video":
            inputs = [make_depth_frames(spec.tasks, DEPTH_VIDEO_KEY) for spec in shard.episodes]
        elif frame_kind == "video":
            inputs = [make_video_frames(spec.tasks, VIDEO_KEY) for spec in shard.episodes]
        else:
            inputs = [make_episode_frames(spec.tasks) for spec in shard.episodes]
        worker.write_distributed_shard(shard, inputs)
    finally:
        worker.finalize()


def write_all_shards_in_processes(
    root: Path, repo_id: str, plan: DistributedWritePlan, frame_kind: str = "numeric"
) -> list:
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(target=write_shard_in_process, args=(str(root), repo_id, shard, frame_kind))
        for shard in plan.shards
    ]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join(timeout=60)
        assert proc.exitcode == 0
    return LeRobotDataset.resume_distributed_write_session(root).load_results()


def _publish_metadata_then_hard_exit_after_backup(root: str, staging_root: str, crash_point: str) -> None:
    root_path = Path(root)
    staging_path = Path(staging_root)
    real_replace = os.replace
    calls = 0

    def crashing_replace(source, destination):
        nonlocal calls
        real_replace(source, destination)
        if Path(source) == root_path / "meta" or Path(destination) == root_path / "meta":
            calls += 1
            if (crash_point == "between_swaps" and calls == 1) or (
                crash_point == "after_publish" and calls == 2
            ):
                os._exit(91)

    os.replace = crashing_replace
    publish_distributed_metadata(root_path, staging_path)


def test_write_plan_allocates_contiguous_ranges_and_unique_files():
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[3, 2, 4],
        episode_tasks=[
            ["pick", "pick", "place"],
            ["place", "place"],
            ["stack", "stack", "pick", "stack"],
        ],
        chunks_size=2,
        pack_episodes=False,
    )

    assert plan.task_to_index == {"pick": 0, "place": 1, "stack": 2}
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
    # pack_episodes=False unifies into 1-episode shards.
    assert len(plan.shards) == 3
    assert [shard.episodes for shard in plan.shards] == [(spec,) for spec in plan.episodes]


def test_write_plan_can_start_after_existing_dataset_indices():
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2, 3],
        episode_tasks=[["place", "pick"], ["stack", "stack", "place"]],
        chunks_size=2,
        start_episode_index=4,
        start_dataset_index=10,
        start_chunk_index=1,
        start_file_index=1,
        task_to_index={"pick": 0, "place": 1},
        pack_episodes=False,
    )

    assert plan.task_to_index == {"pick": 0, "place": 1, "stack": 2}
    assert [
        (spec.episode_index, spec.dataset_from_index, spec.dataset_to_index) for spec in plan.episodes
    ] == [
        (4, 10, 12),
        (5, 12, 15),
    ]
    assert [(spec.data_chunk_index, spec.data_file_index) for spec in plan.episodes] == [
        (1, 1),
        (2, 0),
    ]


@pytest.mark.parametrize(
    ("episode_lengths", "episode_tasks", "message"),
    [
        ([], [], "at least one episode"),
        ([0], [[]], "must be positive"),
        ([2], [["pick"]], "length"),
        ([1], [[""]], "non-empty strings"),
    ],
)
def test_write_plan_rejects_invalid_episode_specs(episode_lengths, episode_tasks, message):
    with pytest.raises(ValueError, match=message):
        DistributedWritePlan.from_episode_lengths(
            episode_lengths=episode_lengths,
            episode_tasks=episode_tasks,
            chunks_size=10,
        )


def test_write_plan_rejects_rollover_without_packing():
    with pytest.raises(ValueError, match="rollover_in_shard=True requires pack_episodes=True"):
        DistributedWritePlan.from_episode_lengths(
            episode_lengths=[2],
            episode_tasks=[["pick", "pick"]],
            chunks_size=10,
            pack_episodes=False,
            rollover_in_shard=True,
        )


def test_distributed_write_apis_reject_non_default_storage_format(tmp_path):
    root = tmp_path / "dataset"
    driver = LeRobotDataset.create("test/non-default", fps=30, features=FEATURES, root=root, use_videos=False)
    driver.meta.info.storage_format = "lance"

    with pytest.raises(ValueError, match="storage_format"):
        driver.create_distributed_write_plan(episode_lengths=[2], episode_tasks=[["pick", "pick"]])

    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2], episode_tasks=[["pick", "pick"]], chunks_size=driver.meta.chunks_size
    )
    with pytest.raises(ValueError, match="storage_format"):
        driver.create_distributed_write_session(plan)
    with pytest.raises(ValueError, match="storage_format"):
        driver.commit_distributed_results(plan, [])
    driver.finalize()


def test_distributed_writer_rejects_remote_root_before_io():
    with pytest.raises(ValueError, match="object-store URIs are not supported"):
        LeRobotDataset.open_distributed_writer("test/remote", root="s3://bucket/dataset")


def test_distributed_driver_rejects_result_from_different_session_mount_path(tmp_path):
    root = tmp_path / "dataset"
    alternate_mount = tmp_path / "alternate-mount"
    repo_id = "test/mount-path"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = driver.create_distributed_write_plan(
        episode_lengths=[2], episode_tasks=[["pick", "pick"]], pack_episodes=False
    )
    session = driver.create_distributed_write_session(plan)
    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root, session=session)
    results = worker.write_distributed_shard(plan.shards[0], [make_episode_frames(plan.episodes[0].tasks)])
    worker.finalize()

    wrong_mount_result = replace(results[0], data_path=alternate_mount / "data/chunk-000/file-000.parquet")
    with pytest.raises(ValueError, match="unexpected path"):
        driver.commit_distributed_results(plan, [wrong_mount_result])
    driver.finalize()


def test_distributed_write_session_publishes_plan_and_cleans_on_create_failure(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    driver = LeRobotDataset.create(
        "test/session-atomic", fps=30, features=FEATURES, root=root, use_videos=False
    )
    plan = driver.create_distributed_write_plan(
        episode_lengths=[2], episode_tasks=[["pick", "pick"]], pack_episodes=False
    )

    real_replace = os.replace

    def fail_session_publish(src, dst):
        if Path(dst) == root / ".session":
            raise OSError("simulated session publish failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_session_publish)
    with pytest.raises(OSError, match="simulated session publish failure"):
        driver.create_distributed_write_session(plan)
    assert not (root / ".session").exists()
    assert list(root.glob(".session.*.tmp")) == []

    monkeypatch.setattr(os, "replace", real_replace)
    session = driver.create_distributed_write_session(plan)
    assert (session.session_root / "plan.json").is_file()
    assert (session.session_root / "plan_digest").is_file()
    assert session.shards_root.is_dir()
    driver.finalize()


def test_distributed_worker_removes_raw_image_staging_after_success(tmp_path):
    root = tmp_path / "dataset"
    repo_id = "test/image-staging"
    features = {**FEATURES, IMAGE_KEY: {"dtype": "image", "shape": (24, 32, 3), "names": None}}
    driver = LeRobotDataset.create(repo_id, fps=30, features=features, root=root, use_videos=False)
    plan = driver.create_distributed_write_plan(
        episode_lengths=[2], episode_tasks=[["pick", "pick"]], pack_episodes=False
    )
    session = driver.create_distributed_write_session(plan)
    worker = LeRobotDataset.open_distributed_writer(repo_id, root=root, session=session)
    results = worker.write_distributed_shard(
        plan.shards[0], [make_image_frames(plan.episodes[0].tasks, IMAGE_KEY)]
    )
    worker.finalize()

    assert results[0].data_path.is_file()
    assert not (root / "images").exists()
    driver.commit_distributed_results(plan, results)
    driver.finalize()


def test_distributed_depth_unit_validation_and_roundtrip(tmp_path):
    # 1. Depth image roundtrip + inconsistent depth unit rejection.
    img_root = tmp_path / "depth-image"
    driver = LeRobotDataset.create(
        "test/depth-img", fps=30, features=DEPTH_IMAGE_FEATURES, root=img_root, use_videos=False
    )
    plan = driver.create_distributed_write_plan(
        episode_lengths=[2, 2],
        episode_tasks=[["pick", "pick"], ["place", "place"]],
        pack_episodes=False,
    )
    driver.create_distributed_write_session(plan)
    results = write_all_shards_in_processes(img_root, "test/depth-img", plan, frame_kind="depth_image")
    assert all(r.depth_units == {DEPTH_IMAGE_KEY: DEPTH_MILLIMETER_UNIT} for r in results)

    # Reject inconsistent depth units across workers.
    conflicting = [results[0], replace(results[1], depth_units={DEPTH_IMAGE_KEY: DEPTH_METER_UNIT})]
    with pytest.raises(ValueError, match="disagree on depth unit"):
        driver.commit_distributed_results(plan, conflicting)
    # Reject unknown / unsupported / missing depth units.
    with pytest.raises(ValueError, match="unknown depth keys"):
        driver.commit_distributed_results(
            plan,
            [
                replace(results[0], depth_units={**results[0].depth_units, "extra": DEPTH_METER_UNIT}),
                results[1],
            ],
        )
    with pytest.raises(ValueError, match="unsupported depth unit"):
        driver.commit_distributed_results(
            plan, [replace(results[0], depth_units={DEPTH_IMAGE_KEY: "cm"}), results[1]]
        )
    with pytest.raises(ValueError, match="missing depth units"):
        driver.commit_distributed_results(plan, [replace(results[0], depth_units={}), results[1]])

    driver.commit_distributed_results(plan, results)
    driver.finalize()

    loaded = LeRobotDataset("test/depth-img", root=img_root, depth_output_unit=DEPTH_MILLIMETER_UNIT)
    assert loaded.meta.features[DEPTH_IMAGE_KEY]["info"]["depth_unit"] == DEPTH_MILLIMETER_UNIT
    np.testing.assert_array_equal(loaded[0][DEPTH_IMAGE_KEY].numpy(), 1_000)
    np.testing.assert_array_equal(loaded[1][DEPTH_IMAGE_KEY].numpy(), 1_500)

    # Append with conflicting depth unit must fail.
    resumed = LeRobotDataset.resume("test/depth-img", root=img_root)
    append_plan = resumed.create_distributed_write_plan(
        episode_lengths=[2], episode_tasks=[["pick", "pick"]], pack_episodes=False
    )
    append_session = resumed.create_distributed_write_session(append_plan)
    worker = LeRobotDataset.open_distributed_writer("test/depth-img", root=img_root, session=append_session)
    meter_results = worker.write_distributed_shard(
        append_plan.shards[0],
        [make_depth_frames(append_plan.episodes[0].tasks, DEPTH_IMAGE_KEY, use_meters=True)],
    )
    worker.finalize()
    with pytest.raises(ValueError, match="disagrees with the existing depth unit"):
        resumed.commit_distributed_results(append_plan, meter_results)
    resumed.finalize()

    # 2. Depth video roundtrip.
    vid_root = tmp_path / "depth-video"
    vdriver = LeRobotDataset.create(
        "test/depth-vid", fps=10, features=DEPTH_VIDEO_FEATURES, root=vid_root, use_videos=True
    )
    vplan = vdriver.create_distributed_write_plan(
        episode_lengths=[2], episode_tasks=[["pick", "pick"]], pack_episodes=False
    )
    vdriver.create_distributed_write_session(vplan)
    vresults = write_all_shards_in_processes(vid_root, "test/depth-vid", vplan, frame_kind="depth_video")
    vdriver.commit_distributed_results(vplan, vresults)
    vdriver.finalize()

    vloaded = LeRobotDataset(
        "test/depth-vid", root=vid_root, video_backend="pyav", depth_output_unit=DEPTH_MILLIMETER_UNIT
    )
    assert vloaded.meta.features[DEPTH_VIDEO_KEY]["info"]["video.pix_fmt"] == "gray12le"
    np.testing.assert_allclose(vloaded[0][DEPTH_VIDEO_KEY].numpy(), 1_000, atol=10)


def test_distributed_spawn_workers_write_multicamera_and_complex_features(tmp_path):
    root = tmp_path / "multimodal"
    repo_id = "test/multimodal"
    driver = LeRobotDataset.create(repo_id, fps=10, features=MULTIMODAL_FEATURES, root=root, use_videos=True)
    plan = driver.create_distributed_write_plan(
        episode_lengths=[2, 2],
        episode_tasks=[["pick", "place"], ["place", "stack"]],
        pack_episodes=False,
    )
    driver.create_distributed_write_session(plan)
    results = write_all_shards_in_processes(root, repo_id, plan, frame_kind="multimodal")
    driver.commit_distributed_results(plan, results)
    driver.finalize()

    loaded = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    assert loaded.meta.total_episodes == 2
    assert len(loaded) == 4
    assert loaded[0]["caption"] == "pick-0"
    assert loaded[3]["caption"] == "stack-1"
    np.testing.assert_array_equal(loaded[0]["history"].numpy(), np.full((2, 3), 0.5, dtype=np.float32))


def test_commit_rejects_changed_results_after_success(tmp_path):
    root = tmp_path / "idempotent"
    repo_id = "test/idempotent"
    driver = LeRobotDataset.create(repo_id, fps=30, features=FEATURES, root=root, use_videos=False)
    plan = driver.create_distributed_write_plan(
        episode_lengths=[2], episode_tasks=[["pick", "pick"]], pack_episodes=False
    )
    driver.create_distributed_write_session(plan)
    results = write_all_shards_in_processes(root, repo_id, plan)
    driver.commit_distributed_results(plan, results)

    # Repeating identical commit is idempotent.
    driver.commit_distributed_results(plan, results)

    # Conflicting result for an already committed plan is rejected.
    altered = [replace(results[0], stats={**results[0].stats, "action": {"min": np.array([99.0, 99.0])}})]
    with pytest.raises(ValueError, match="Conflicting results"):
        driver.commit_distributed_results(plan, altered)
    driver.finalize()


def test_metadata_publish_rollback_and_crash_recovery(tmp_path, monkeypatch):
    # 1. Rollback when second os.replace fails.
    root = tmp_path / "swap-fail"
    staging = root / ".staged"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "marker.txt").write_text("old")
    (staging / "meta").mkdir(parents=True)
    (staging / "meta" / "marker.txt").write_text("new")

    real_replace = os.replace
    calls = 0

    def fail_second_replace(src, dst):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("second swap failed")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_second_replace)
    with pytest.raises(OSError, match="second swap failed"):
        publish_distributed_metadata(root, staging)
    assert (root / "meta" / "marker.txt").read_text() == "old"
    assert not (root / ".session" / "meta" / "journal.json").exists()

    # 2. Missing destination metadata raises FileNotFoundError.
    monkeypatch.setattr(os, "replace", real_replace)
    shutil.rmtree(root / "meta")
    with pytest.raises(FileNotFoundError, match="is missing"):
        publish_distributed_metadata(root, staging)


@pytest.mark.parametrize("crash_point", ["between_swaps", "after_publish"])
def test_metadata_publish_resumes_after_hard_exit_between_swaps(tmp_path, crash_point):
    root = tmp_path / f"crash-{crash_point}"
    staging = root / ".session" / "meta" / "staging-test"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "marker.txt").write_text("old")
    (staging / "meta").mkdir(parents=True)
    (staging / "meta" / "marker.txt").write_text("new")

    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_publish_metadata_then_hard_exit_after_backup, args=(str(root), str(staging), crash_point)
    )
    proc.start()
    proc.join(timeout=30)
    assert proc.exitcode == 91
    assert (root / ".session" / "meta" / "journal.json").is_file()

    assert recover_distributed_metadata(root) is True
    assert (root / "meta" / "marker.txt").read_text() == "new"
    assert not (root / ".session" / "meta" / "journal.json").exists()
    assert list((root / ".session" / "meta").glob("backup-*")) == []


def test_driver_concurrent_writes_append_batches_and_tamper_rejection(tmp_path):
    root = tmp_path / "concurrent-append"
    repo_id = "test/concurrent-append"
    driver = LeRobotDataset.create(repo_id, fps=10, features=VIDEO_FEATURES, root=root, use_videos=True)

    # Batch 1: 2 independent episodes (pack_episodes=False -> 1-episode shards).
    plan1 = driver.create_distributed_write_plan(
        episode_lengths=[2, 3],
        episode_tasks=[["pick", "pick"], ["place", "place", "pick"]],
        pack_episodes=False,
    )
    driver.create_distributed_write_session(plan1)
    results1 = write_all_shards_in_processes(root, repo_id, plan1, frame_kind="video")

    # Tampered artifact is rejected without publishing metadata.
    original_bytes = results1[0].data_path.read_bytes()
    results1[0].data_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        driver.commit_distributed_results(plan1, results1)
    assert driver.meta.total_episodes == 0
    results1[0].data_path.write_bytes(original_bytes)

    driver.commit_distributed_results(plan1, results1)
    assert not (root / ".session").exists()
    assert not (root / "meta" / "distributed-commits").exists()
    assert [p.name for p in root.iterdir() if p.name.startswith(".")] == []
    driver.finalize()

    # Batch 2: append 2 more episodes with video.
    resumed = LeRobotDataset.resume(repo_id, root=root)
    plan2 = resumed.create_distributed_write_plan(
        episode_lengths=[2, 2],
        episode_tasks=[["stack", "stack"], ["pick", "place"]],
        pack_episodes=False,
    )
    resumed.create_distributed_write_session(plan2)
    results2 = write_all_shards_in_processes(root, repo_id, plan2, frame_kind="video")
    resumed.commit_distributed_results(plan2, results2)

    # Stale plan after dataset advances is rejected.
    with pytest.raises(ValueError, match="starts at episode"):
        resumed.create_distributed_write_session(plan1)
    resumed.finalize()

    loaded = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    assert loaded.meta.total_episodes == 4
    assert loaded.meta.total_frames == 9
    assert len(loaded) == 9
    assert pq.ParquetFile(results1[0].data_path).metadata.num_rows == 2
    assert pq.ParquetFile(results2[1].data_path).metadata.num_rows == 2


def test_driver_rejects_plan_with_mismatched_video_keys(tmp_path):
    root = tmp_path / "mismatch-video-keys"
    driver = LeRobotDataset.create(
        "test/mismatch", fps=10, features=VIDEO_FEATURES, root=root, use_videos=True
    )
    bad_plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2],
        episode_tasks=[["pick", "pick"]],
        chunks_size=driver.meta.chunks_size,
        video_keys=[],
    )
    with pytest.raises(ValueError, match="video_keys must match"):
        driver.create_distributed_write_session(bad_plan)
    driver.finalize()

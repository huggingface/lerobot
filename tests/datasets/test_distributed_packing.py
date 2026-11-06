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

"""Serial-v3 parity, shared-shard ownership, and process recovery round trips."""

import multiprocessing as mp
import os
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest
from filelock import Timeout

import lerobot.datasets.distributed_writer as shard_writer_module
from lerobot.configs import DEPTH_MILLIMETER_UNIT, DepthEncoderConfig
from lerobot.datasets.distributed import DistributedShardSpec, DistributedWritePlan, artifact_digest
from lerobot.datasets.io_utils import get_file_size_in_mb
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pyav_utils import get_codec

VIDEO = "observation.images.front"
FEATURES = {
    "action": {"dtype": "float32", "shape": (2,), "names": None},
    VIDEO: {"dtype": "video", "shape": (32, 32, 3), "names": None},
}
EXIT_CODE = 86
_SLOW_TASK_RELEASE = None


def initialize_scheduler_worker(release):
    global _SLOW_TASK_RELEASE
    _SLOW_TASK_RELEASE = release


def scheduled_numeric_task(root, shard):
    if shard.shard_index == 0 and not _SLOW_TASK_RELEASE.wait(timeout=60):
        raise RuntimeError("A completed continuation should release the slow initial task.")
    consumed = []

    def inputs():
        for spec in shard.episodes:
            consumed.append(spec.episode_index)
            yield [{"action": np.array([spec.episode_index, 0], dtype=np.float32), "task": "pick"}]

    results = write_shard(root, shard, inputs())
    return os.getpid(), consumed, [r.episode_index for r in results]


def frames(episode, count=2):
    return [
        {
            "action": np.array([episode, i], dtype=np.float32),
            VIDEO: np.full((32, 32, 3), episode * 40 + i * 8, dtype=np.uint8),
            "task": "z-pick" if episode % 2 == 0 else "a-place",
        }
        for i in range(count)
    ]


def create_driver(root, *, data_limit=100, video_limit=100):
    return LeRobotDataset.create(
        "test/packed",
        fps=10,
        features=FEATURES,
        root=root,
        data_files_size_in_mb=data_limit,
        video_files_size_in_mb=video_limit,
    )


def make_plan(driver, episodes, **kwargs):
    return driver.create_distributed_write_plan(
        episode_lengths=[len(e) for e in episodes],
        episode_tasks=[[f["task"] for f in e] for e in episodes],
        **kwargs,
    )


def write_shard(root, shard, episodes):
    session = LeRobotDataset.resume_distributed_write_session(root)
    worker = LeRobotDataset.open_distributed_writer("test/packed", root=root, session=session)
    try:
        return worker.write_distributed_shard(shard, episodes)
    finally:
        worker.finalize()


def write_shard_process(root, shard, episodes):
    write_shard(root, shard, episodes)


def write_pending_shards(root, episodes):
    """Run the evolving task queue to completion; each call consumes only its prefix."""
    session = LeRobotDataset.resume_distributed_write_session(root)
    base = session.plan.episodes[0].episode_index
    while pending := session.pending_shards():
        for shard in pending:
            results = write_shard(root, shard, [episodes[s.episode_index - base] for s in shard.episodes])
            assert len({r.data_path for r in results}) == 1
            for key in session.plan.video_keys:
                assert (
                    len(
                        {
                            (
                                r.video_metadata[f"videos/{key}/chunk_index"],
                                r.video_metadata[f"videos/{key}/file_index"],
                            )
                            for r in results
                        }
                    )
                    == 1
                )
    return session.load_results()


def crash_worker(root, shard, episodes, stage):
    session = LeRobotDataset.resume_distributed_write_session(root)
    worker = LeRobotDataset.open_distributed_writer("test/packed", root=root, session=session)
    if stage == "writing":

        def interrupted_input():
            yield episodes[0]
            os._exit(EXIT_CODE)

        worker.write_distributed_shard(shard, interrupted_input())
    elif stage == "boundary":
        original_boundary = session.record_shard_boundary

        def interrupt_after_boundary(spec, count):
            original_boundary(spec, count)
            os._exit(EXIT_CODE)

        with patch.object(session, "record_shard_boundary", side_effect=interrupt_after_boundary):
            worker.write_distributed_shard(shard, episodes)
    elif stage in {"publish", "video_publish"}:
        original = shard_writer_module.os.replace

        def interrupt_publish(source, destination):
            original(source, destination)
            directory = "videos" if stage == "video_publish" else "data"
            if Path(destination).is_relative_to(Path(root) / directory):
                os._exit(EXIT_CODE)

        with patch.object(shard_writer_module.os, "replace", side_effect=interrupt_publish):
            worker.write_distributed_shard(shard, episodes)
    else:
        original_record = session.record_shard_results

        def interrupt_after_record(spec, results):
            original_record(spec, results)
            os._exit(EXIT_CODE)

        with patch.object(session, "record_shard_results", side_effect=interrupt_after_record):
            worker.write_distributed_shard(shard, episodes)


def crash_driver(root, stage):
    session = LeRobotDataset.resume_distributed_write_session(root)
    driver = LeRobotDataset.resume("test/packed", root=root)
    original = os.replace

    def interrupted_publish(source, destination):
        original(source, destination)
        if (
            stage == "backup"
            and Path(source) == Path(root) / "meta"
            or stage == "published"
            and Path(destination) == Path(root) / "meta"
        ):
            os._exit(EXIT_CODE)

    with patch.object(os, "replace", side_effect=interrupted_publish):
        driver.commit_distributed_results(session.plan, session.load_results())


def numeric_frames(episode, count):
    return [
        {
            "action": np.array([episode, index], dtype=np.float32),
            "task": "z-existing" if episode < 2 else ("a-new", "z-existing", "m-new")[index % 3],
        }
        for index in range(count)
    ]


def data_and_video_hashes(root):
    return {
        path: artifact_digest(path)
        for pattern in ("data/**/*.parquet", "videos/**/*.mp4")
        for path in root.glob(pattern)
    }


def block_inside_task(root, shard, episodes, entered, release):
    def inputs():
        yield episodes[0]
        entered.set()
        if not release.wait(timeout=60):
            raise RuntimeError("Timed out waiting for the test to release the active task.")
        yield from episodes[1:]

    write_shard(root, shard, inputs())


@pytest.mark.parametrize("append", [False, True], ids=["new", "append"])
@pytest.mark.parametrize(
    "lengths,budget,limit,group_lengths",
    [
        ([1], 1, 100, [1]),
        ([1, 9, 2, 1], 4, 100, [1, 1, 2]),
        ([1] * 7, 3, 100, [3, 3, 1]),
        ([2, 3, 1, 4, 2], 5, 100, [2, 2, 1]),
        ([11], 3, 100, [1]),
        ([1, 4, 2, 3], 20, 0.000001, [1, 1, 1, 1]),
    ],
    ids=["single", "uneven", "many-short", "exact-budget", "oversized", "soft-split"],
)
def test_packing_scenario_matrix_roundtrip(tmp_path, append, lengths, budget, limit, group_lengths):
    driver = LeRobotDataset.create(
        "test/packed",
        root=tmp_path / "dataset",
        fps=10,
        features={"action": FEATURES["action"]},
        use_videos=False,
        data_files_size_in_mb=limit,
    )
    driver.meta.update_chunk_settings(chunks_size=2)
    previous = [numeric_frames(0, 2), numeric_frames(1, 3)] if append else []
    for episode in previous:
        for frame in episode:
            driver.add_frame(dict(frame))
        driver.save_episode()
    if append:
        driver.finalize()
        driver = LeRobotDataset.resume("test/packed", root=driver.root)
    original_hashes = data_and_video_hashes(driver.root)
    base = len(previous)
    episodes = [numeric_frames(base + index, length) for index, length in enumerate(lengths)]
    plan = make_plan(driver, episodes, max_frames_per_worker=budget)
    session = driver.create_distributed_write_session(plan)
    completion_order = []
    while pending := session.pending_shards():
        # Finish higher episode indices first; metadata must still follow global order.
        for task in reversed(pending):
            inputs = [episodes[spec.episode_index - base] for spec in task.episodes]
            written = write_shard(driver.root, task, inputs)
            completion_order.extend(result.episode_index for result in written)
            assert len({result.data_path for result in written}) == 1
            assert pq.ParquetFile(written[0].data_path).num_row_groups == len(written)
            # Duplicate delivery after success must not need the source again.
            assert [r.episode_index for r in write_shard(driver.root, task, [])] == [
                r.episode_index for r in written
            ]
    assert sorted(completion_order) == list(range(base, base + len(episodes)))
    assert [len(task.episodes) for task in session.task_specs()] == group_lengths
    if len(plan.shards) > 1:
        assert completion_order[0] > base
    results = session.load_results()
    driver.commit_distributed_results(plan, list(reversed(results)))
    driver.finalize()
    assert all(artifact_digest(path) == digest for path, digest in original_hashes.items())
    assert len(list(driver.root.glob("data/**/*.parquet"))) == len(original_hashes) + len(group_lengths)
    loaded = LeRobotDataset("test/packed", root=driver.root)
    all_episodes = previous + episodes
    expected_values = []
    offset = 0
    for episode_index, episode in enumerate(all_episodes):
        metadata = loaded.meta.episodes[episode_index]
        assert metadata["dataset_from_index"] == offset
        assert metadata["dataset_to_index"] == offset + len(episode)
        for frame_index, frame in enumerate(episode):
            row = loaded[offset + frame_index]
            np.testing.assert_array_equal(row["action"], frame["action"])
            assert row["episode_index"].item() == episode_index
            assert row["frame_index"].item() == frame_index
            assert row["index"].item() == offset + frame_index
            assert row["task"] == frame["task"]
            assert row["task_index"].item() == loaded.meta.tasks.loc[frame["task"], "task_index"]
            expected_values.append(frame["action"])
        offset += len(episode)
    assert len(loaded) == offset
    assert loaded.num_episodes == len(all_episodes)
    for statistic, values in (
        ("min", np.min(expected_values, axis=0)),
        ("max", np.max(expected_values, axis=0)),
        ("mean", np.mean(expected_values, axis=0)),
    ):
        np.testing.assert_allclose(loaded.meta.stats["action"][statistic], values, atol=1e-6)


@pytest.mark.parametrize("relation,expected_prefix", [("below", 1), ("equal", 1), ("above", 2)])
def test_actual_parquet_size_threshold_boundary(tmp_path, relation, expected_prefix):
    features = {"action": FEATURES["action"]}
    probe = LeRobotDataset.create("test/packed", root=tmp_path / "probe", fps=10, features=features)
    for frame in numeric_frames(0, 2):
        probe.add_frame(dict(frame))
    probe.save_episode()
    size = get_file_size_in_mb(next(probe.root.glob("data/**/*.parquet")))
    probe.finalize()
    assert size > 0
    limit = {"below": np.nextafter(size, 0), "equal": size, "above": np.nextafter(size, np.inf)}[relation]
    driver = LeRobotDataset.create(
        "test/packed",
        root=tmp_path / "dataset",
        fps=10,
        features=features,
        data_files_size_in_mb=float(limit),
    )
    episodes = [numeric_frames(index, 2) for index in range(3)]
    plan = make_plan(driver, episodes)
    session = driver.create_distributed_write_session(plan)

    def limited_source():
        yield from episodes[:expected_prefix]
        pytest.fail("The task consumed input past its size boundary.")

    results = write_shard(driver.root, plan.shards[0], limited_source())
    assert len(results) == expected_prefix
    assert session.pending_shards()[0].shard_index == expected_prefix
    results = write_pending_shards(driver.root, episodes)
    driver.commit_distributed_results(plan, results)
    driver.finalize()
    loaded = LeRobotDataset("test/packed", root=driver.root)
    assert len(loaded) == 6
    for index in range(6):
        np.testing.assert_array_equal(loaded[index]["action"], [index // 2, index % 2])


@pytest.mark.parametrize("failure", ["short", "long", "wrong-task", "source-error"])
def test_failed_second_episode_keeps_task_unpublished_and_retryable(tmp_path, failure):
    driver = LeRobotDataset.create(
        "test/packed",
        root=tmp_path / "dataset",
        fps=10,
        features={"action": FEATURES["action"]},
        use_videos=False,
    )
    episodes = [numeric_frames(index, 2) for index in range(4)]
    plan = make_plan(driver, episodes, max_frames_per_worker=4)
    session = driver.create_distributed_write_session(plan)
    other = write_shard(driver.root, plan.shards[1], episodes[2:])
    original_hashes = data_and_video_hashes(driver.root)
    metadata_before = (driver.root / "meta/info.json").read_bytes()

    def failing_frames():
        yield dict(episodes[1][0])
        raise OSError("Injected source read failure")

    invalid = {
        "short": episodes[1][:1],
        "long": episodes[1] + episodes[1][:1],
        "wrong-task": [{**frame, "task": "wrong"} for frame in episodes[1]],
        "source-error": failing_frames(),
    }[failure]
    error = OSError if failure == "source-error" else ValueError
    with pytest.raises(error):
        write_shard(driver.root, plan.shards[0], [episodes[0], invalid])
    assert [result.episode_index for result in session.load_results()] == [2, 3]
    assert not session.shard_boundary_path(plan.shards[0]).exists()
    assert data_and_video_hashes(driver.root) == original_hashes
    with pytest.raises(ValueError):
        driver.commit_distributed_results(plan, other)
    assert (driver.root / "meta/info.json").read_bytes() == metadata_before
    write_shard(driver.root, plan.shards[0], episodes[:2])
    assert all(artifact_digest(path) == digest for path, digest in original_hashes.items())
    driver.commit_distributed_results(plan, session.load_results())
    driver.finalize()
    loaded = LeRobotDataset("test/packed", root=driver.root)
    assert len(loaded) == 8
    for index in range(8):
        np.testing.assert_array_equal(loaded[index]["action"], [index // 2, index % 2])


def test_live_worker_lock_blocks_duplicate_and_releases_on_process_death(tmp_path):
    driver = LeRobotDataset.create(
        "test/packed",
        root=tmp_path / "dataset",
        fps=10,
        features={"action": FEATURES["action"]},
        use_videos=False,
    )
    episodes = [numeric_frames(index, 2) for index in range(2)]
    plan = make_plan(driver, episodes)
    session = driver.create_distributed_write_session(plan)
    context = mp.get_context("spawn")
    entered, release = context.Event(), context.Event()
    process = context.Process(
        target=block_inside_task,
        args=(driver.root, plan.shards[0], episodes, entered, release),
    )
    process.start()
    try:
        assert entered.wait(timeout=60), "Worker never reached its second episode."
        with pytest.raises(Timeout):
            write_shard(driver.root, plan.shards[0], [])
        with pytest.raises(Timeout):
            session.cleanup_orphaned_artifacts()
        assert session.load_results() == []
    finally:
        process.terminate()
        process.join(timeout=30)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
    assert process.exitcode is not None and process.exitcode != 0
    session.cleanup_orphaned_artifacts()
    write_shard(driver.root, plan.shards[0], episodes)
    driver.commit_distributed_results(plan, session.load_results())
    driver.finalize()
    assert LeRobotDataset("test/packed", root=driver.root).num_frames == 4


@pytest.mark.parametrize("append", [False, True], ids=["new", "append"])
def test_crash_between_camera_publications_preserves_completed_tail(tmp_path, append):
    rear = "observation.images.rear"
    driver = LeRobotDataset.create(
        "test/packed",
        root=tmp_path / "dataset",
        fps=10,
        features={**FEATURES, rear: FEATURES[VIDEO]},
        data_files_size_in_mb=0.000001,
    )
    episodes = [frames(index, count) for index, count in enumerate([2, 3, 1, 4])]
    for episode in episodes:
        for frame in episode:
            frame[rear] = np.full((32, 32, 3), 200, dtype=np.uint8) - frame[VIDEO]
    previous = episodes[:1] if append else []
    for episode in previous:
        for frame in episode:
            driver.add_frame(dict(frame))
        driver.save_episode()
    if append:
        driver.finalize()
        driver = LeRobotDataset.resume("test/packed", root=driver.root)
    original_hashes = data_and_video_hashes(driver.root)
    batch = episodes[len(previous) :]
    plan = make_plan(driver, batch)
    session = driver.create_distributed_write_session(plan)
    process = mp.get_context("spawn").Process(
        target=crash_worker,
        args=(driver.root, plan.shards[0], batch, "video_publish"),
    )
    process.start()
    process.join(timeout=60)
    if process.is_alive():
        process.kill()
        process.join(timeout=10)
    assert process.exitcode == EXIT_CODE
    published = set(data_and_video_hashes(driver.root)) - set(original_hashes)
    assert sum(path.suffix == ".parquet" for path in published) == 1
    assert sum(path.suffix == ".mp4" for path in published) == 1
    first = plan.episodes[0].episode_index
    while tails := [task for task in session.pending_shards() if task.shard_index != first]:
        for task in tails:
            write_shard(driver.root, task, [batch[s.episode_index - first] for s in task.episodes])
    protected = {
        driver.root / path: digest
        for result in session.load_results()
        for path, digest in result.artifact_digests.items()
    }
    protected.update(original_hashes)
    session.cleanup_orphaned_artifacts()
    assert all(not path.exists() for path in published)
    assert all(artifact_digest(path) == digest for path, digest in protected.items())
    assert len(write_shard(driver.root, plan.shards[0], batch[:1])) == 1
    driver.commit_distributed_results(plan, session.load_results())
    driver.finalize()
    assert all(artifact_digest(path) == digest for path, digest in protected.items())
    loaded = LeRobotDataset("test/packed", root=driver.root, video_backend="pyav")
    assert loaded.num_episodes == len(episodes)
    expected_frames = [frame for episode in episodes for frame in episode]
    assert len(loaded) == len(expected_frames)
    for index, expected in enumerate(expected_frames):
        np.testing.assert_array_equal(loaded[index]["action"], expected["action"])
        for key in (VIDEO, rear):
            np.testing.assert_allclose(loaded[index][key].mean(), expected[key].mean() / 255, atol=3 / 255)


def test_plan_groups_contiguous_frames_and_reserves_rollover_slots():
    plan = DistributedWritePlan.from_episode_lengths(
        episode_lengths=[2, 2, 8, 1],
        episode_tasks=[["pick"] * n for n in [2, 2, 8, 1]],
        chunks_size=2,
        start_chunk_index=3,
        start_file_index=1,
        start_episode_index=5,
        start_dataset_index=20,
        max_frames_per_worker=4,
    )
    assert [[s.episode_index for s in shard.episodes] for shard in plan.shards] == [[5, 6], [7], [8]]
    assert [shard.num_frames for shard in plan.shards] == [4, 8, 1]
    assert [(s.data_chunk_index, s.data_file_index) for s in plan.episodes] == [
        (3, 1),
        (4, 0),
        (4, 1),
        (5, 0),
    ]


@pytest.mark.parametrize(
    "data_limit,video_limit", [(100, 100), (0.0005, 100), (100, 0.0005), (0.0005, 0.0005)]
)
@pytest.mark.parametrize("append", [False, True])
def test_task_soft_limits_preserve_serial_values_and_global_indices(
    tmp_path, data_limit, video_limit, append
):
    serial = create_driver(tmp_path / "serial", data_limit=data_limit, video_limit=video_limit)
    driver = create_driver(tmp_path / "distributed", data_limit=data_limit, video_limit=video_limit)
    serial.meta.update_chunk_settings(chunks_size=2)
    driver.meta.update_chunk_settings(chunks_size=2)
    if append:
        for dataset in (serial, driver):
            for frame in frames(0):
                dataset.add_frame(dict(frame))
            dataset.save_episode()
            dataset.finalize()
        serial = LeRobotDataset.resume("test/packed", root=serial.root)
        driver = LeRobotDataset.resume("test/packed", root=driver.root)
    episodes = [frames(i) for i in range(int(append), int(append) + 3)]
    for episode in episodes:
        for frame in episode:
            serial.add_frame(dict(frame))
        serial.save_episode()
    serial.finalize()
    plan = make_plan(driver, episodes)
    assert len(plan.shards) == 1
    session = driver.create_distributed_write_session(plan)
    results = write_pending_shards(driver.root, episodes)
    assert len(session.load_results()) == 3
    assert session.pending_shards() == []
    driver.commit_distributed_results(plan, results)
    driver.finalize()
    actual = LeRobotDataset("test/packed", root=driver.root, video_backend="pyav")
    expected = LeRobotDataset("test/packed", root=serial.root, video_backend="pyav")
    for key in (
        "episode_index",
        "dataset_from_index",
        "dataset_to_index",
    ):
        np.testing.assert_allclose(actual.meta.episodes[key], expected.meta.episodes[key])
    assert len(actual) == len(expected)
    for feature in ("action", VIDEO, "index", "episode_index", "task_index"):
        for statistic in ("min", "max", "mean", "std", "count"):
            np.testing.assert_allclose(
                actual.meta.stats[feature][statistic], expected.meta.stats[feature][statistic], atol=1e-5
            )
    for index in range(len(actual)):
        assert actual[index]["task"] == expected[index]["task"]
        np.testing.assert_array_equal(actual[index]["action"], expected[index]["action"])
        np.testing.assert_allclose(actual[index][VIDEO], expected[index][VIDEO], atol=2 / 255)
    expected_files = 3 if min(data_limit, video_limit) < 1 else 1
    assert len({result.data_path for result in results}) == expected_files
    assert (
        len(
            {
                (
                    r.video_metadata[f"videos/{VIDEO}/chunk_index"],
                    r.video_metadata[f"videos/{VIDEO}/file_index"],
                )
                for r in results
            }
        )
        == expected_files
    )
    if expected_files == 1:
        assert pq.ParquetFile(results[0].data_path).num_row_groups == 3


def test_spawn_workers_pack_multiple_episodes_without_overlapping_files(tmp_path):
    driver = create_driver(tmp_path / "dataset")
    episodes = [frames(i) for i in range(4)]
    plan = make_plan(driver, episodes, max_frames_per_worker=4)
    session = driver.create_distributed_write_session(plan)
    processes = [
        mp.get_context("spawn").Process(
            target=write_shard_process,
            args=(driver.root, shard, episodes[i * 2 : i * 2 + 2]),
        )
        for i, shard in enumerate(plan.shards)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0
    results = session.load_results()
    assert len({r.data_path for r in results}) == 2
    assert len(list(driver.root.glob("videos/**/*.mp4"))) == 2
    driver.commit_distributed_results(plan, results)
    driver.finalize()
    loaded = LeRobotDataset("test/packed", root=driver.root, video_backend="pyav")
    for i in range(8):
        np.testing.assert_array_equal(loaded[i]["action"], [i // 2, i % 2])
        np.testing.assert_allclose(loaded[i][VIDEO].mean(), ((i // 2) * 40 + (i % 2) * 8) / 255, atol=3 / 255)


def test_soft_size_target_seals_two_episodes_and_schedules_the_untouched_tail(tmp_path):
    # Measure real artifacts to choose a boundary independent of codec/library
    # versions: episode one is below target; episode two crosses both targets.
    serial = create_driver(tmp_path / "serial")
    data_sizes, video_sizes = [], []
    for _ in range(2):
        for frame in frames(0):
            serial.add_frame(frame)
        serial.save_episode()
        data_sizes.append(get_file_size_in_mb(next(serial.root.glob("data/**/*.parquet"))))
        video_sizes.append(get_file_size_in_mb(next(serial.root.glob("videos/**/*.mp4"))))
    data_limit = sum(data_sizes) / 2
    video_limit = sum(video_sizes) / 2
    assert data_sizes[0] < data_limit < data_sizes[1]
    assert video_sizes[0] < video_limit < video_sizes[1]
    serial.meta.update_chunk_settings(data_files_size_in_mb=data_limit, video_files_size_in_mb=video_limit)
    for frame in frames(0):
        serial.add_frame(frame)
    serial.save_episode()
    serial.finalize()

    driver = create_driver(tmp_path / "distributed", data_limit=data_limit, video_limit=video_limit)
    episodes = [frames(0) for _ in range(3)]
    plan = make_plan(driver, episodes)
    session = driver.create_distributed_write_session(plan)

    def inputs_without_lookahead():
        yield episodes[0]
        yield episodes[1]
        pytest.fail("The first task must not consume an episode from its continuation.")

    first = write_shard(driver.root, plan.shards[0], inputs_without_lookahead())
    assert len(first) == 2
    pending = session.pending_shards()
    assert [[s.episode_index for s in task.episodes] for task in pending] == [[2]]
    # A different writer can immediately take the unstarted continuation.
    write_shard(driver.root, pending[0], episodes[2:])
    results = session.load_results()
    assert [r.data_file_index for r in results] == [0, 0, 2]
    assert [r.video_metadata[f"videos/{VIDEO}/file_index"] for r in results] == [0, 0, 2]
    assert [r.video_metadata[f"videos/{VIDEO}/from_timestamp"] for r in results] == [0, 0.2, 0]
    assert pq.ParquetFile(results[0].data_path).num_row_groups == 2
    driver.commit_distributed_results(plan, results)
    driver.finalize()
    actual = LeRobotDataset("test/packed", root=driver.root, video_backend="pyav")
    expected = LeRobotDataset("test/packed", root=serial.root, video_backend="pyav")
    for i in range(6):
        np.testing.assert_array_equal(actual[i]["action"], expected[i]["action"])
        np.testing.assert_allclose(actual[i][VIDEO], expected[i][VIDEO], atol=2 / 255)


@pytest.mark.parametrize("stage", ["writing", "publish", "record"])
def test_hard_killed_shard_is_retried_as_a_whole(tmp_path, stage):
    driver = create_driver(tmp_path / "dataset")
    episodes = [frames(i) for i in range(2)]
    plan = make_plan(driver, episodes)
    session = driver.create_distributed_write_session(plan)
    process = mp.get_context("spawn").Process(
        target=crash_worker, args=(driver.root, plan.shards[0], episodes, stage)
    )
    process.start()
    process.join(timeout=60)
    assert process.exitcode == EXIT_CODE
    before = session.load_results()
    assert len(before) == (2 if stage == "record" else 0)
    original_hash = artifact_digest(before[0].data_path) if before else None
    results = write_shard(driver.root, plan.shards[0], episodes)
    if original_hash is not None:
        assert artifact_digest(results[0].data_path) == original_hash
    assert len({r.data_path for r in results}) == 1
    assert not session.shard_staging_root(plan.shards[0]).exists()
    driver.commit_distributed_results(plan, session.load_results())
    driver.finalize()
    assert LeRobotDataset("test/packed", root=driver.root).num_frames == 4


def test_corrupt_shared_shard_retry_preserves_other_worker_and_committed_data(tmp_path):
    driver = create_driver(tmp_path / "dataset")
    episodes = [frames(i) for i in range(4)]
    plan = make_plan(driver, episodes, max_frames_per_worker=4)
    session = driver.create_distributed_write_session(plan)
    first = write_shard(driver.root, plan.shards[0], episodes[:2])
    second = write_shard(driver.root, plan.shards[1], episodes[2:])
    other_hash = artifact_digest(second[0].data_path)
    first[0].data_path.write_bytes(b"partial parquet")
    retry = write_shard(driver.root, plan.shards[0], episodes[:2])
    assert artifact_digest(second[0].data_path) == other_hash
    assert len(retry) == 2
    driver.commit_distributed_results(plan, session.load_results())
    # Retained session objects can replay an already committed packed batch.
    worker = LeRobotDataset.open_distributed_writer("test/packed", root=driver.root, session=session)
    assert len(worker.write_distributed_shard(plan.shards[0], [])) == 2
    retry[0].data_path.write_bytes(b"corrupt committed data")
    with pytest.raises(ValueError):
        worker.write_distributed_shard(plan.shards[0], episodes[:2])
    assert retry[0].data_path.read_bytes() == b"corrupt committed data"
    worker.finalize()
    driver.finalize()


def test_packed_plan_excludes_episode_writes_and_duplicate_shard_attempts(tmp_path):
    driver = create_driver(tmp_path / "dataset")
    episodes = [frames(i) for i in range(2)]
    plan = make_plan(driver, episodes)
    session = driver.create_distributed_write_session(plan)
    worker = LeRobotDataset.open_distributed_writer("test/packed", root=driver.root, session=session)
    with pytest.raises(ValueError, match="write_distributed_shard"):
        worker.start_distributed_episode(plan.episodes[0])
    with session.shard_lock(plan.shards[0]):
        with pytest.raises(Timeout):
            worker.write_distributed_shard(plan.shards[0], episodes)
        with pytest.raises(Timeout):
            session.cleanup_orphaned_artifacts()
    worker.finalize()
    driver.finalize()


def test_packed_commit_rejects_file_outside_reservation(tmp_path):
    driver = create_driver(tmp_path / "dataset")
    episodes = [frames(0), frames(1)]
    plan = make_plan(driver, episodes)
    driver.create_distributed_write_session(plan)
    results = write_shard(driver.root, plan.shards[0], episodes)
    with pytest.raises(ValueError, match="reservation"):
        driver.commit_distributed_results(plan, [replace(results[0], data_file_index=99), results[1]])
    assert driver.meta.total_episodes == 0
    driver.finalize()


def test_three_workers_dynamically_schedule_thirty_whole_episodes(tmp_path):
    driver = LeRobotDataset.create(
        "test/packed",
        fps=10,
        features={"action": FEATURES["action"]},
        root=tmp_path / "dataset",
        use_videos=False,
        data_files_size_in_mb=0.0008,
    )
    plan = driver.create_distributed_write_plan(
        episode_lengths=[1] * 30,
        episode_tasks=[["pick"] for _ in range(30)],
        max_frames_per_worker=10,
    )
    session = driver.create_distributed_write_session(plan)
    context = mp.get_context("spawn")
    release = context.Event()
    completed = []
    with ProcessPoolExecutor(
        max_workers=3,
        mp_context=context,
        initializer=initialize_scheduler_worker,
        initargs=(release,),
    ) as pool:
        inflight = {}
        try:
            while True:
                running_indices = {task.shard_index for task in inflight.values()}
                for task in session.pending_shards():
                    if len(inflight) == 3:
                        break
                    if task.shard_index not in running_indices:
                        inflight[pool.submit(scheduled_numeric_task, driver.root, task)] = task
                        running_indices.add(task.shard_index)
                if not inflight:
                    break
                done, _ = wait(inflight, timeout=60, return_when=FIRST_COMPLETED)
                assert done, "Task queue stalled while a worker should be available."
                for future in done:
                    task = inflight.pop(future)
                    pid, consumed, written = future.result()
                    assert consumed == written  # No lookahead or partial episode transfer.
                    completed.append((task.shard_index, pid, written))
                    if task.shard_index not in {0, 10, 20}:
                        release.set()
        finally:
            release.set()
    assert len({pid for _, pid, _ in completed}) <= 3
    assert sorted(index for _, _, indices in completed for index in indices) == list(range(30))
    assert len(completed) > 3
    first_task_position = next(i for i, (index, _, _) in enumerate(completed) if index == 0)
    assert any(index not in {0, 10, 20} for index, _, _ in completed[:first_task_position])
    results = session.load_results()
    assert len({r.data_path for r in results}) == len(completed)
    driver.commit_distributed_results(plan, results)
    driver.finalize()
    loaded = LeRobotDataset("test/packed", root=driver.root)
    for index in range(30):
        np.testing.assert_array_equal(loaded[index]["action"], [index, 0])


@pytest.mark.parametrize("stage", ["boundary", "publish", "record"])
def test_task_boundary_survives_crash_and_retry_preserves_completed_continuation(tmp_path, stage):
    driver = create_driver(tmp_path / "dataset", data_limit=0.0001)
    episodes = [frames(i) for i in range(3)]
    plan = make_plan(driver, episodes)
    session = driver.create_distributed_write_session(plan)
    process = mp.get_context("spawn").Process(
        target=crash_worker,
        args=(driver.root, plan.shards[0], episodes, stage),
    )
    process.start()
    process.join(timeout=60)
    assert process.exitcode == EXIT_CODE
    assert [[s.episode_index for s in task.episodes] for task in session.task_specs()] == [[0], [1, 2]]
    # Complete the suffix on another writer before repairing the failed prefix.
    for start in (1, 2):
        task = next(task for task in session.pending_shards() if task.shard_index == start)
        write_shard(driver.root, task, episodes[start:])
    tail = [r for r in session.load_results() if r.episode_index > 0]
    before = {path: artifact_digest(path) for r in tail for path in [r.data_path]}
    boundary_before = session.shard_boundary_path(plan.shards[0]).read_bytes()
    session.cleanup_orphaned_artifacts()

    def only_first_episode():
        yield episodes[0]
        pytest.fail("A retry must respect the durable prefix boundary.")

    retried = write_shard(driver.root, plan.shards[0], only_first_episode())
    assert len(retried) == 1
    assert session.shard_boundary_path(plan.shards[0]).read_bytes() == boundary_before
    assert all(artifact_digest(path) == digest for path, digest in before.items())
    driver.commit_distributed_results(plan, session.load_results())
    assert not list((driver.root / ".distributed-shards").glob("shard-*"))
    # After session cleanup, the commit receipt still preserves task boundaries.
    worker = LeRobotDataset.open_distributed_writer("test/packed", root=driver.root, session=session)
    assert len(worker.write_distributed_shard(plan.shards[0], [])) == 1
    worker.finalize()
    driver.finalize()
    assert LeRobotDataset("test/packed", root=driver.root).num_frames == 6


def test_corrupt_prefix_retries_without_reabsorbing_its_completed_tail(tmp_path):
    driver = create_driver(tmp_path / "dataset", data_limit=0.0001)
    episodes = [frames(0), frames(1), frames(2)]
    plan = make_plan(driver, episodes)
    session = driver.create_distributed_write_session(plan)
    results = write_pending_shards(driver.root, episodes)
    hashes = {r.data_path: artifact_digest(r.data_path) for r in results[1:]}
    results[0].data_path.write_bytes(b"corrupt")
    retried = write_shard(driver.root, plan.shards[0], episodes)
    assert [r.episode_index for r in retried] == [0]
    assert all(artifact_digest(path) == digest for path, digest in hashes.items())
    driver.commit_distributed_results(plan, session.load_results())
    driver.finalize()


def test_unreleased_episode_cannot_be_stolen_by_another_task(tmp_path):
    driver = create_driver(tmp_path / "dataset")
    episodes = [frames(0), frames(1)]
    plan = make_plan(driver, episodes)
    driver.create_distributed_write_session(plan)
    with pytest.raises(ValueError, match="allocated task"):
        write_shard(driver.root, DistributedShardSpec(plan.episodes[1:]), episodes[1:])
    assert not list(driver.root.glob("data/**/*.parquet"))
    driver.finalize()


@pytest.mark.parametrize("corruption", ["indices", "missing", "overlap"])
def test_invalid_task_boundary_fails_closed_without_deleting_other_tasks(tmp_path, corruption):
    driver = create_driver(tmp_path / "dataset", data_limit=0.0001)
    episodes = [frames(0), frames(1), frames(2)]
    plan = make_plan(driver, episodes)
    session = driver.create_distributed_write_session(plan)
    results = write_pending_shards(driver.root, episodes)
    hashes = {r.data_path: artifact_digest(r.data_path) for r in results}
    if corruption == "missing":
        session.shard_boundary_path(plan.shards[0]).unlink()
    elif corruption == "overlap":
        session.shard_boundary_path(plan.shards[0]).write_text('{"episode_indices": [0, 1]}')
    else:
        session.shard_boundary_path(plan.shards[0]).write_text('{"episode_indices": [0, 2]}')
    with pytest.raises(ValueError, match="persisted task boundary"):
        session.cleanup_orphaned_artifacts()
    assert all(artifact_digest(path) == digest for path, digest in hashes.items())
    driver.finalize()


@pytest.mark.parametrize("stream", ["data", "video"])
def test_commit_rejects_multiple_file_slots_inside_one_task(tmp_path, stream):
    driver = create_driver(tmp_path / "dataset")
    episodes = [frames(0), frames(1)]
    plan = make_plan(driver, episodes)
    driver.create_distributed_write_session(plan)
    results = write_shard(driver.root, plan.shards[0], episodes)
    result = results[0]
    source = result.data_path if stream == "data" else next(driver.root.glob("videos/**/*.mp4"))
    target = source.with_name("file-001" + source.suffix)
    shutil.copyfile(source, target)
    digests = dict(result.artifact_digests)
    digest = digests.pop(str(source.relative_to(driver.root)))
    digests[str(target.relative_to(driver.root))] = digest
    if stream == "data":
        altered = replace(result, data_file_index=1, data_path=target, artifact_digests=digests)
    else:
        altered = replace(
            result,
            video_metadata={**result.video_metadata, f"videos/{VIDEO}/file_index": 1},
            artifact_digests=digests,
        )
    with pytest.raises(ValueError, match="exactly one"):
        driver.commit_distributed_results(plan, [altered, results[1]])
    assert driver.meta.total_episodes == 0
    target.unlink()
    driver.commit_distributed_results(plan, results)
    driver.finalize()


def test_one_camera_reaching_target_seals_all_streams_together(tmp_path):
    noisy = "observation.images.noisy"
    features = {**FEATURES, noisy: FEATURES[VIDEO]}
    episodes = [frames(0), frames(1)]
    rng = np.random.default_rng(42)
    for episode in episodes:
        for frame in episode:
            frame[noisy] = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    probe = LeRobotDataset.create("test/packed", root=tmp_path / "probe", fps=10, features=features)
    for frame in episodes[0]:
        probe.add_frame(dict(frame))
    probe.save_episode(parallel_encoding=False)
    probe.finalize()
    small = get_file_size_in_mb(next((probe.root / "videos" / VIDEO).rglob("*.mp4")))
    large = get_file_size_in_mb(next((probe.root / "videos" / noisy).rglob("*.mp4")))
    assert small < large
    driver = LeRobotDataset.create(
        "test/packed",
        root=tmp_path / "dataset",
        fps=10,
        features=features,
        video_files_size_in_mb=(small + large) / 2,
    )
    plan = make_plan(driver, episodes)
    session = driver.create_distributed_write_session(plan)
    first = write_shard(driver.root, plan.shards[0], episodes)
    assert len(first) == 1
    assert [t.shard_index for t in session.pending_shards()] == [1]
    write_shard(driver.root, session.pending_shards()[0], episodes[1:])
    results = session.load_results()
    for result in results:
        for key in (VIDEO, noisy):
            assert result.video_metadata[f"videos/{key}/file_index"] == result.data_file_index
            assert result.video_metadata[f"videos/{key}/from_timestamp"] == 0
    assert len(list(driver.root.glob("data/**/*.parquet"))) == 2
    assert len(list(driver.root.glob("videos/**/*.mp4"))) == 4
    driver.commit_distributed_results(plan, results)
    driver.finalize()
    loaded = LeRobotDataset("test/packed", root=driver.root, video_backend="pyav")
    assert len(loaded) == 4
    assert loaded[2][noisy].shape == (3, 32, 32)


@pytest.mark.parametrize("stage", ["backup", "published"])
@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize("split", [False, True], ids=["packed", "continuations"])
def test_driver_recovers_packed_metadata_publish(tmp_path, stage, append, split):
    driver = create_driver(tmp_path / "dataset", data_limit=0.0001 if split else 100)
    if append:
        for frame in frames(0):
            driver.add_frame(frame)
        driver.save_episode()
        driver.finalize()
        driver = LeRobotDataset.resume("test/packed", root=driver.root)
    episodes = [frames(i) for i in range(int(append), int(append) + 2)]
    plan = make_plan(driver, episodes)
    initial_session = driver.create_distributed_write_session(plan)
    results = write_pending_shards(driver.root, episodes)
    assert len(initial_session.task_specs()) == (2 if split else 1)
    hashes = {
        driver.root / path: digest for result in results for path, digest in result.artifact_digests.items()
    }
    driver.finalize()
    process = mp.get_context("spawn").Process(target=crash_driver, args=(driver.root, stage))
    process.start()
    process.join(timeout=60)
    assert process.exitcode == EXIT_CODE
    session = LeRobotDataset.resume_distributed_write_session(driver.root)
    resumed = LeRobotDataset.resume("test/packed", root=driver.root)
    resumed.commit_distributed_results(session.plan, session.load_results())
    resumed.commit_distributed_results(session.plan, session.load_results())
    resumed.finalize()
    assert all(artifact_digest(path) == digest for path, digest in hashes.items())
    loaded = LeRobotDataset("test/packed", root=driver.root, video_backend="pyav")
    assert loaded.num_episodes == 2 + int(append)
    np.testing.assert_array_equal(loaded[-1]["action"], [1 + int(append), 1])
    assert loaded[-1][VIDEO].shape == (3, 32, 32)


def test_numeric_shard_retries_bad_input_and_supports_multiple_append_batches(tmp_path):
    driver = LeRobotDataset.create(
        "test/packed",
        fps=10,
        features={"action": FEATURES["action"]},
        root=tmp_path / "dataset",
        use_videos=False,
    )
    for batch in range(2):
        episodes = [[{"action": np.array([batch, ep], dtype=np.float32), "task": "pick"}] for ep in range(2)]
        plan = make_plan(driver, episodes)
        session = driver.create_distributed_write_session(plan)
        with pytest.raises(ValueError, match="Missing input"):
            write_shard(driver.root, plan.shards[0], episodes[:1])
        assert session.load_results() == []
        results = write_shard(driver.root, plan.shards[0], episodes)
        assert results[0].data_path == results[1].data_path
        driver.commit_distributed_results(plan, results)
    driver.finalize()
    loaded = LeRobotDataset("test/packed", root=driver.root)
    assert loaded.num_episodes == 4
    for index in range(4):
        np.testing.assert_array_equal(loaded[index]["action"], [index // 2, index % 2])


@pytest.mark.skipif(get_codec("hevc") is None, reason="'hevc' not in local FFmpeg build")
def test_packed_multimodal_depth_and_rgb_roundtrip(tmp_path):
    depth_image = "observation.depth.raw"
    depth_video = "observation.depth.video"
    rgb_image = "observation.images.raw"
    features = {
        **FEATURES,
        rgb_image: {"dtype": "image", "shape": (3, 32, 32), "names": None},
        depth_image: {"dtype": "image", "shape": (1, 32, 32), "names": None, "info": {"is_depth_map": True}},
        depth_video: {"dtype": "video", "shape": (1, 32, 32), "names": None, "info": {"is_depth_map": True}},
        "caption": {"dtype": "string", "shape": (1,), "names": None},
        "history": {"dtype": "float32", "shape": (2, 3), "names": None},
    }
    driver = LeRobotDataset.create("test/packed", fps=10, features=features, root=tmp_path / "dataset")
    episodes = [frames(i) for i in range(2)]
    for ep, episode in enumerate(episodes):
        for index, frame in enumerate(episode):
            depth = np.full((1, 32, 32), 1_000 + 100 * (ep * 2 + index), dtype=np.uint16)
            frame.update(
                {
                    rgb_image: np.full((3, 32, 32), ep * 30 + index, dtype=np.uint8),
                    depth_image: depth,
                    depth_video: depth.copy(),
                    "caption": f"episode-{ep}-frame-{index}",
                    "history": np.full((2, 3), ep + index, dtype=np.float32),
                }
            )
    plan = make_plan(driver, episodes)
    session = driver.create_distributed_write_session(plan)
    worker = LeRobotDataset.open_distributed_writer(
        "test/packed",
        root=driver.root,
        session=session,
        # The local HEVC build drops a boundary frame when concatenating default
        # GOP=2 clips, including with the ordinary serial writer. All-intra clips
        # let this test isolate packing, units, and offsets from that upstream issue.
        depth_encoder=DepthEncoderConfig(use_log=False, g=1),
    )
    results = worker.write_distributed_shard(plan.shards[0], episodes)
    worker.finalize()
    driver.commit_distributed_results(plan, results)
    driver.finalize()
    assert results[0].data_path == results[1].data_path
    assert len(list(driver.root.glob("videos/**/*.mp4"))) == 2
    loaded = LeRobotDataset(
        "test/packed", root=driver.root, video_backend="pyav", depth_output_unit=DEPTH_MILLIMETER_UNIT
    )
    for i in range(4):
        np.testing.assert_array_equal(loaded[i][depth_image], 1_000 + 100 * i)
        np.testing.assert_allclose(loaded[i][depth_video], 1_000 + 100 * i, atol=5)
        assert loaded[i]["caption"] == f"episode-{i // 2}-frame-{i % 2}"
        np.testing.assert_allclose(loaded[i][rgb_image], ((i // 2) * 30 + i % 2) / 255)
        np.testing.assert_array_equal(loaded[i]["history"], np.full((2, 3), i // 2 + i % 2))
    for key in (depth_image, depth_video):
        assert loaded.meta.features[key]["info"]["depth_unit"] == DEPTH_MILLIMETER_UNIT

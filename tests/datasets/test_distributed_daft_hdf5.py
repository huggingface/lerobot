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

"""Real HDF5 -> Daft on Ray -> distributed v3 writer -> standard reader tests.

Run explicitly with LEROBOT_RUN_DAFT_E2E=1 and daft, ray, h5py installed.
The fixture owns a local head and two logical worker nodes, not physical hosts.
Only task/source descriptors travel through Daft; workers open the HDF5 files.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from lerobot.datasets.distributed import artifact_digest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

daft = pytest.importorskip("daft")
ray = pytest.importorskip("ray")
h5py = pytest.importorskip("h5py")
cluster_utils = pytest.importorskip("ray.cluster_utils")

pytestmark = [
    pytest.mark.dataset,
    pytest.mark.skipif(
        os.environ.get("LEROBOT_RUN_DAFT_E2E") != "1",
        reason="Set LEROBOT_RUN_DAFT_E2E=1 to start the isolated Daft/Ray test cluster.",
    ),
]

REPO_ID = "test/daft-hdf5"
CAMERAS = ("front", "wrist")
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
    "action": {"dtype": "float32", "shape": (2,), "names": None},
    **{
        f"observation.images.{camera}": {"dtype": "video", "shape": (32, 32, 3), "names": None}
        for camera in CAMERAS
    },
}


def create_hdf5(path, episode, length):
    """A small ALOHA-style source with changing actions, states, tasks and cameras."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as source:
        source.attrs["fps"] = 10
        source.create_dataset(
            "action", data=np.array([[episode, i] for i in range(length)], dtype=np.float32)
        )
        source.create_dataset(
            "observations/qpos",
            data=np.array([[episode, i, episode + i] for i in range(length)], dtype=np.float32),
        )
        source.create_dataset(
            "tasks",
            data=["pick" if (episode + i) % 2 == 0 else "place" for i in range(length)],
            dtype=h5py.string_dtype(),
        )
        for camera_index, camera in enumerate(CAMERAS):
            pixels = np.stack(
                [
                    np.full((32, 32, 3), (episode * 7 + i * 3 + camera_index * 31) % 200, dtype=np.uint8)
                    for i in range(length)
                ]
            )
            source.create_dataset(f"observations/images/{camera}", data=pixels, compression="gzip")
    return path


def read_hdf5_frames(path):
    with h5py.File(path, "r") as source:
        for index in range(len(source["action"])):
            yield {
                "action": source["action"][index],
                "observation.state": source["observations/qpos"][index],
                "task": source["tasks"].asstr()[index],
                **{
                    f"observation.images.{camera}": source[f"observations/images/{camera}"][index]
                    for camera in CAMERAS
                },
            }


def prepare_batch(driver, source_paths, *, budget, manifest_path):
    lengths, tasks = [], []
    for path in source_paths:
        with h5py.File(path, "r") as source:
            lengths.append(len(source["action"]))
            tasks.append(source["tasks"].asstr()[:].tolist())
    plan = driver.create_distributed_write_plan(
        episode_lengths=lengths,
        episode_tasks=tasks,
        max_frames_per_worker=budget,
    )
    session = driver.create_distributed_write_session(plan)
    manifest_path.write_text(
        json.dumps(
            {
                str(spec.episode_index): str(path)
                for spec, path in zip(plan.episodes, source_paths, strict=True)
            }
        )
    )
    return session


@daft.func(cpus=1, use_process=False, on_error="raise")
def convert_hdf5_task(root: str, manifest_path: str, task_id: int) -> str:
    """One Daft row invokes one logical write task, possibly completing a prefix."""
    started_ns = time.monotonic_ns()
    session = LeRobotDataset.resume_distributed_write_session(root)
    task = next(task for task in session.task_specs() if task.shard_index == task_id)
    source_paths = json.loads(Path(manifest_path).read_text())
    consumed = []

    def episodes():
        for spec in task.episodes:
            consumed.append(spec.episode_index)
            yield read_hdf5_frames(source_paths[str(spec.episode_index)])

    worker = LeRobotDataset.open_distributed_writer(REPO_ID, root=root, session=session)
    try:
        results = worker.write_distributed_shard(task, episodes())
    finally:
        worker.finalize()
    return json.dumps(
        {
            "task_id": task_id,
            "completed": [result.episode_index for result in results],
            "consumed": consumed,
            "pid": os.getpid(),
            "node_id": ray.get_runtime_context().get_node_id(),
            "started_ns": started_ns,
            "finished_ns": time.monotonic_ns(),
        }
    )


def run_round(session, manifest, tasks):
    table = daft.from_pydict(
        {
            "root": [str(session.root)] * len(tasks),
            "manifest": [str(manifest)] * len(tasks),
            "task_id": [task.shard_index for task in tasks],
        }
    ).repartition(len(tasks))
    results = table.select(
        convert_hdf5_task(table["root"], table["manifest"], table["task_id"]).alias("result")
    ).to_pydict()["result"]
    return [json.loads(result) for result in results]


def run_pending(session, manifest):
    rounds = []
    while tasks := session.pending_shards():
        assert len(rounds) < len(session.plan.episodes), "Continuation scheduling did not make progress."
        records = run_round(session, manifest, tasks)
        assert all(record["completed"] == record["consumed"] for record in records)
        rounds.append(records)
    completed = [episode for records in rounds for record in records for episode in record["completed"]]
    assert sorted(completed) == [spec.episode_index for spec in session.plan.episodes]
    return rounds


def artifact_hashes(root):
    return {
        path: artifact_digest(path)
        for pattern in ("data/**/*.parquet", "videos/**/*.mp4")
        for path in root.glob(pattern)
    }


def assert_roundtrip(root, paths):
    reader = LeRobotDataset(REPO_ID, root=root, video_backend="pyav")
    assert reader.num_episodes == len(paths)
    offset = 0
    actions = []
    for episode_index, path in enumerate(paths):
        expected = list(read_hdf5_frames(path))
        metadata = reader.meta.episodes[episode_index]
        assert metadata["dataset_from_index"] == offset
        assert metadata["dataset_to_index"] == offset + len(expected)
        for frame_index, frame in enumerate(expected):
            row = reader[offset + frame_index]
            np.testing.assert_array_equal(row["action"], frame["action"])
            np.testing.assert_array_equal(row["observation.state"], frame["observation.state"])
            assert row["episode_index"].item() == episode_index
            assert row["frame_index"].item() == frame_index
            assert row["index"].item() == offset + frame_index
            assert row["task"] == frame["task"]
            assert row["task_index"].item() == reader.meta.tasks.loc[frame["task"], "task_index"]
            for camera in CAMERAS:
                key = f"observation.images.{camera}"
                np.testing.assert_allclose(
                    row[key],
                    frame[key].transpose(2, 0, 1) / 255,
                    atol=3 / 255,
                )
            actions.append(frame["action"])
        offset += len(expected)
    assert len(reader) == offset
    np.testing.assert_allclose(reader.meta.stats["action"]["mean"], np.mean(actions, axis=0), atol=1e-6)


@pytest.fixture(scope="module")
def daft_cluster():
    assert not ray.is_initialized(), "Run these integration tests in a separate Python process."
    cluster = cluster_utils.Cluster()
    evidence = {"daft": daft.__version__, "ray": ray.__version__, "cases": []}
    with (
        tempfile.TemporaryDirectory(prefix="lr-ray-", dir=os.environ.get("LEROBOT_RAY_TMPDIR")) as ray_root,
        pytest.MonkeyPatch.context() as environment,
    ):
        # Isolate Ray's cluster-address file as well as node logs and sockets.
        environment.setenv("RAY_TMPDIR", ray_root)
        # Pytest imports this file as a top-level module; remote UDF unpickling
        # needs the same module search path in the worker interpreters.
        environment.setenv(
            "PYTHONPATH",
            os.pathsep.join(filter(None, [str(Path(__file__).parent), os.environ.get("PYTHONPATH")])),
        )
        ray_temp = Path(ray_root) / "ray"
        ray_temp.mkdir()
        try:
            cluster.add_node(
                num_cpus=0,
                include_dashboard=False,
                temp_dir=str(ray_temp),
                object_store_memory=128 * 1024**2,
            )
            for _ in range(2):
                cluster.add_node(num_cpus=2, object_store_memory=256 * 1024**2)
            cluster.wait_for_nodes()
            ray.init(address=cluster.address, log_to_driver=False)
            daft.set_runner_ray(address=cluster.address)
            evidence["worker_nodes"] = [
                node["NodeID"]
                for node in ray.nodes()
                if node["Alive"] and node["Resources"].get("CPU", 0) > 0
            ]
            assert len(evidence["worker_nodes"]) == 2
            yield evidence
        finally:
            report = os.environ.get("LEROBOT_DAFT_E2E_REPORT")
            if report:
                Path(report).write_text(json.dumps(evidence, indent=2))
            ray.shutdown()
            cluster.shutdown()


@pytest.mark.parametrize("append", [False, True], ids=["new", "append"])
@pytest.mark.parametrize(
    "lengths,budget,data_limit,video_limit,task_lengths,round_count",
    [
        ([3], 5, 100, 100, [1], 1),
        ([2] * 6, 2, 100, 100, [1] * 6, 1),
        ([2] * 6, 6, 100, 100, [3, 3], 1),
        ([1, 7, 2, 1, 4], 4, 100, 100, [1, 1, 2, 1], 1),
        ([2] * 30, 10, 100, 100, [5] * 6, 1),
        ([2] * 12, 8, 0.000001, 100, [1] * 12, 4),
        ([2] * 12, 8, 100, 0.000001, [1] * 12, 4),
    ],
    ids=[
        "single",
        "separate",
        "packed",
        "uneven",
        "thirty-episodes",
        "data-continuations",
        "video-continuations",
    ],
)
def test_daft_hdf5_conversion(
    tmp_path,
    daft_cluster,
    append,
    lengths,
    budget,
    data_limit,
    video_limit,
    task_lengths,
    round_count,
):
    driver = LeRobotDataset.create(
        REPO_ID,
        root=tmp_path / "dataset",
        fps=10,
        features=FEATURES,
        data_files_size_in_mb=data_limit,
        video_files_size_in_mb=video_limit,
    )
    driver.meta.update_chunk_settings(chunks_size=2)
    previous = []
    if append:
        previous = [create_hdf5(tmp_path / "source/seed.hdf5", 0, 3)]
        for frame in read_hdf5_frames(previous[0]):
            driver.add_frame(frame)
        driver.save_episode()
        driver.finalize()
        driver = LeRobotDataset.resume(REPO_ID, root=driver.root)
    old_hashes = artifact_hashes(driver.root)
    paths = [
        create_hdf5(tmp_path / f"source/episode-{index}.hdf5", index + len(previous), length)
        for index, length in enumerate(lengths)
    ]
    manifest = tmp_path / "sources.json"
    session = prepare_batch(driver, paths, budget=budget, manifest_path=manifest)
    rounds = run_pending(session, manifest)
    assert len(rounds) == round_count
    assert [len(task.episodes) for task in session.task_specs()] == task_lengths
    records = [record for batch in rounds for record in batch]
    assert all(record["pid"] != os.getpid() for record in records)
    assert all(record["node_id"] in daft_cluster["worker_nodes"] for record in records)
    if len(lengths) == 30:
        assert len({record["node_id"] for record in records}) == 2
        # Both nodes run on this host, so monotonic clocks are comparable.
        assert any(
            left["node_id"] != right["node_id"]
            and max(left["started_ns"], right["started_ns"]) < min(left["finished_ns"], right["finished_ns"])
            for left in records
            for right in records
        ), "The two worker nodes did not execute conversion tasks concurrently."
    results = session.load_results()
    assert len({result.data_path for result in results}) == len(task_lengths)
    for task in session.task_specs():
        owned = [
            result
            for result in results
            if result.episode_index in {spec.episode_index for spec in task.episodes}
        ]
        assert len({result.data_path for result in owned}) == 1
        assert pq.ParquetFile(owned[0].data_path).num_row_groups == len(task.episodes)
        for camera in CAMERAS:
            key = f"videos/observation.images.{camera}"
            assert (
                len(
                    {
                        (
                            result.video_metadata[f"{key}/chunk_index"],
                            result.video_metadata[f"{key}/file_index"],
                        )
                        for result in owned
                    }
                )
                == 1
            )
    driver.commit_distributed_results(session.plan, list(reversed(results)))
    driver.finalize()
    assert all(artifact_digest(path) == digest for path, digest in old_hashes.items())
    assert len(list(driver.root.glob("data/**/*.parquet"))) == len(task_lengths) + len(previous)
    assert len(list(driver.root.glob("videos/**/*.mp4"))) == len(CAMERAS) * (
        len(task_lengths) + len(previous)
    )
    assert_roundtrip(driver.root, previous + paths)
    daft_cluster["cases"].append(
        {
            "case": tmp_path.name,
            "episodes": len(paths),
            "append": append,
            "tasks": len(task_lengths),
            "rounds": rounds,
        }
    )


def test_daft_hdf5_missing_source_retries_without_rewriting_other_task(tmp_path, daft_cluster):
    driver = LeRobotDataset.create(REPO_ID, root=tmp_path / "dataset", fps=10, features=FEATURES)
    paths = [create_hdf5(tmp_path / f"episode-{index}.hdf5", index, 2) for index in range(4)]
    manifest = tmp_path / "sources.json"
    session = prepare_batch(driver, paths, budget=4, manifest_path=manifest)
    completed = run_round(session, manifest, [session.plan.shards[1]])
    protected = artifact_hashes(driver.root)
    metadata = (driver.root / "meta/info.json").read_bytes()
    missing = paths[1]
    backup = missing.with_suffix(".backup")
    missing.rename(backup)
    try:
        with pytest.raises(Exception, match="episode-1.hdf5|No such file"):
            run_round(session, manifest, [session.plan.shards[0]])
    finally:
        backup.rename(missing)
    assert [result.episode_index for result in session.load_results()] == [2, 3]
    assert artifact_hashes(driver.root) == protected
    assert (driver.root / "meta/info.json").read_bytes() == metadata
    assert not (driver.root / ".distributed-shards").exists()
    retried = run_round(session, manifest, session.pending_shards())
    assert retried[0]["completed"] == retried[0]["consumed"] == [0, 1]
    assert all(artifact_digest(path) == digest for path, digest in protected.items())
    driver.commit_distributed_results(session.plan, session.load_results())
    driver.finalize()
    assert_roundtrip(driver.root, paths)
    daft_cluster["cases"].append({"case": "missing-source-retry", "rounds": [completed, retried]})


def test_daft_hdf5_duplicate_delivery_reuses_files_without_opening_sources(tmp_path, daft_cluster):
    driver = LeRobotDataset.create(REPO_ID, root=tmp_path / "dataset", fps=10, features=FEATURES)
    paths = [create_hdf5(tmp_path / f"episode-{index}.hdf5", index, 2) for index in range(6)]
    manifest = tmp_path / "sources.json"
    session = prepare_batch(driver, paths, budget=6, manifest_path=manifest)
    first = run_pending(session, manifest)
    before = artifact_hashes(driver.root)
    saved = []
    try:
        for path in paths:
            backup = path.with_suffix(".backup")
            path.rename(backup)
            saved.append((path, backup))
        replay = run_round(session, manifest, session.task_specs())
    finally:
        for path, backup in saved:
            backup.rename(path)
    assert all(record["consumed"] == [] for record in replay)
    assert sorted(episode for record in replay for episode in record["completed"]) == list(range(6))
    assert artifact_hashes(driver.root) == before
    loaded_results = session.load_results()
    driver.commit_distributed_results(session.plan, loaded_results)
    driver.commit_distributed_results(session.plan, loaded_results)
    driver.finalize()
    assert_roundtrip(driver.root, paths)
    daft_cluster["cases"].append({"case": "duplicate-delivery", "rounds": first + [replay]})

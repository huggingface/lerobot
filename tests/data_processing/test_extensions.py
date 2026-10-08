# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
import copy
import json
import runpy
import time
from pathlib import Path

import pytest

pytest.importorskip("datasets")
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from lerobot.data_processing.artifacts import ArtifactStore, file_checksum  # noqa: E402
from lerobot.data_processing.metrics import report_stage  # noqa: E402
from lerobot.data_processing.modules.pair_links import INVENTORY  # noqa: E402
from lerobot.data_processing.planner import seal_plan  # noqa: E402
from lerobot.data_processing.runtime import gpu_assignments, run_local  # noqa: E402
from lerobot.data_processing.types import DatasetRef, InputItem, artifact_identities  # noqa: E402
from lerobot.data_processing.worker import accepted_in_shard  # noqa: E402
from tests.data_processing.test_conversion import raw_source  # noqa: E402

REF = DatasetRef("fixture/input", "a" * 40)


def test_exported_model_requires_explicit_trust():
    from lerobot.data_processing.modules.video_embeddings import VideoEmbeddings

    with pytest.raises(ValueError, match="trusted_checkpoint"):
        VideoEmbeddings("untrusted.pt2", "a" * 64, 3)


def test_model_batch_real_decode_embedding_and_metrics(tmp_path):
    import numpy as np
    import torch

    manifest = raw_source(tmp_path)

    class Mean(torch.nn.Module):
        def forward(self, image):
            return image.mean(dim=(2, 3))

    checkpoint = tmp_path / "trusted-model.pt2"
    model = torch.export.export(
        Mean().eval(),
        (torch.zeros(3, 3, 32, 32),),
        dynamic_shapes={"image": {0: torch.export.Dim("frames", min=1, max=128)}},
    )
    torch.export.save(model, checkpoint)
    config = {
        "checkpoint_uri": str(checkpoint),
        "checkpoint_sha256": file_checksum(checkpoint)[0],
        "embedding_dim": 3,
        "trusted_checkpoint": True,
        "image_size": 32,
    }
    results = []
    for batch_size in (1, 2):
        store = ArtifactStore(tmp_path / f"run-{batch_size}")
        items = []
        for episode in range(2):
            video = manifest.parent / f"{episode}.mp4"
            items.append(
                InputItem(
                    str(episode),
                    {
                        "video_path": str(video),
                        "video_sha256": file_checksum(video)[0],
                        "episode_index": episode,
                        "camera": "front",
                        "frame_indices": [0, 3, 5],
                    },
                    physical_seconds=1,
                    camera_seconds=1,
                )
            )
        start = time.perf_counter()
        plan = seal_plan(
            store,
            REF,
            "lerobot.data_processing.modules.video_embeddings:VideoEmbeddings",
            config,
            items,
            shard_size=2,
        )
        summary = run_local(store, plan, batch_size=batch_size)
        accepted = list(accepted_in_shard(store, plan, 0).values())
        rows = []
        for result in accepted:
            with store.open(result.artifacts[0].path) as stream:
                rows.extend(pq.read_table(stream).to_pylist())
        assert [row["frame_index"] for row in rows] == [0, 3, 5, 0, 3, 5]
        assert len(rows) == 6 and np.isfinite([row["embedding"] for row in rows]).all()
        results.append(rows)
        report = report_stage(store, plan, summary, wall_seconds=time.perf_counter() - start)
        assert report["timer_seconds"]["inference"] > 0 and report["timer_seconds"]["decode"] > 0
        assert report["requested_gpu_hours"] == 0 and report["requested_cpu_hours"] > 0
        assert report["physical_input_hours"] == 2 / 3600
        assert report["realtime_multiplier"] is not None
        previous = set(store.list(f"metrics/{plan.plan_id}/workers/*.parquet"))
        run_local(store, plan)
        resumed = report_stage(
            store,
            plan,
            summary,
            wall_seconds=1,
            worker_paths=set(store.list(f"metrics/{plan.plan_id}/workers/*.parquet")) - previous,
        )
        assert resumed["realtime_multiplier"] is None  # caching is not processing throughput
    assert results[0] == results[1]


def test_gpu_assignment_preserves_scheduler_ids_and_rejects_oversubscription(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    assert gpu_assignments(2, 2) == [("GPU-a", "GPU-b"), ("GPU-c", "GPU-d")]
    with pytest.raises(ValueError, match="exceed"):
        gpu_assignments(3, 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError):
        gpu_assignments(1, 1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,0")
    with pytest.raises(ValueError, match="unique"):
        gpu_assignments(2, 1)


def test_artifact_binding_identity_and_input_invalidation(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    ids = []
    for path, digest in (
        ("attempt-a/file", "a" * 64),
        ("attempt-b/file", "a" * 64),
        ("attempt-c/file", "b" * 64),
    ):
        payload = {"upstream": {"path": path, "sha256": digest, "size": 123, "name": "atoms", "rows": 1}}
        plan = seal_plan(
            store,
            REF,
            "tests.data_processing._modules:Echo",
            {},
            [InputItem("one", payload, identity_payload=artifact_identities(payload))],
        )
        ids.append(plan.read_shard(store, 0)[0].item_id)
    assert ids[0] == ids[1] and ids[2] != ids[1]


def test_pair_reducer_validated_links_weak_pairs_and_bad_ranges(tmp_path):
    inventory = tmp_path / "inventory.parquet"
    source = {
        "dataset_uri": "hf://datasets/owner/ego",
        "revision": "a" * 40,
        "episode_id": "source",
        "frames": 100,
        "cameras": ["ego"],
        "kind": "ego",
        "has_actions": False,
        "split": "train",
        "origin_group": None,
        "simulation_validation": None,
    }
    target = {
        **source,
        "dataset_uri": "hf://datasets/owner/robot",
        "episode_id": "target",
        "cameras": ["front"],
        "kind": "robot",
        "has_actions": True,
    }
    pq.write_table(pa.Table.from_pylist([source, target], schema=INVENTORY), inventory)
    source_ref = {key: source[key] for key in ("dataset_uri", "revision", "episode_id")}
    target_ref = {key: target[key] for key in ("dataset_uri", "revision", "episode_id")}
    payload = {
        "source": {**source_ref, "camera": "ego", "start": 0, "end": 50},
        "target": {**target_ref, "camera": "front", "start": 10, "end": 80},
        "pair_type": "ego_to_robot",
        "alignment_validated": True,
        "alignment_evidence": "reviewed procedure/order/outcome match",
    }
    weak = {**payload, "pair_type": "weak_semantic"}
    invalid = copy.deepcopy(payload)
    invalid["target"]["end"] = 101
    missing = copy.deepcopy(payload)
    missing["target"]["episode_id"] = "unknown"
    store = ArtifactStore(tmp_path / "run")
    plan = seal_plan(
        store,
        REF,
        "lerobot.data_processing.modules.pair_links:PairLinks",
        {"inventory_uri": str(inventory), "inventory_sha256": file_checksum(inventory)[0]},
        [InputItem(str(index), value) for index, value in enumerate((payload, weak, invalid, missing))],
        shard_size=4,
    )
    summary = run_local(store, plan, batch_size=4)
    assert summary.completed == 2 and summary.rejected == 2
    accepted = accepted_in_shard(store, plan, 0)
    outputs = []
    for result in accepted.values():
        for artifact in result.artifacts:
            with store.open(artifact.path) as stream:
                outputs.extend(pq.read_table(stream).to_pylist())
    assert [row["action_loss"] for row in outputs] == [True, False]
    assert all(artifact.name == "pairs" for result in accepted.values() for artifact in result.artifacts)


def test_video_only_ego_conversion(tmp_path):
    from lerobot.data_processing.conversion import ConvertConfig, convert_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = raw_source(tmp_path)
    value = json.loads(manifest.read_text())
    value["features"] = {}
    for episode in value["episodes"]:
        episode["arrays"], episode["frames"] = {}, 6
    manifest.write_text(json.dumps(value))
    root = convert_dataset(
        ConvertConfig(source={"manifest": str(manifest)}, output=tmp_path / "converted-ego", size=64)
    )
    dataset = LeRobotDataset("processing/ego", root=root, video_backend="pyav")
    assert len(dataset) == 12 and "action" not in dataset.features


def test_remote_store_without_rename_or_immediate_listing(monkeypatch):
    import uuid

    from lerobot.data_processing.runtime import finalize_stage
    from lerobot.data_processing.worker import run_worker_group
    from tests.data_processing.test_runtime import make_plan

    store = ArtifactStore("memory://eventual-" + uuid.uuid4().hex)
    plan = make_plan(store, size=2)
    monkeypatch.setattr(store.fs, "mv", lambda *a, **k: pytest.fail("remote rename was required"))
    run_worker_group(store.uri, plan.plan_id, [0], 2, 0)
    real_list = store.list
    monkeypatch.setattr(store, "list", lambda pattern: [] if "checkpoints" in pattern else real_list(pattern))
    with pytest.raises(RuntimeError, match="incomplete"):
        finalize_stage(store, plan)
    assert not store.list(f"{plan.prefix}/accepted/*.parquet")
    monkeypatch.setattr(store, "list", real_list)
    assert finalize_stage(store, plan).completed == 2


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"split": "test"}, "cross_split_pair"),
        ({"has_actions": False}, "target_has_no_robot_actions"),
        ({"kind": "ego"}, "target_has_no_robot_actions"),
        ({"kind": "sim_robot"}, "unvalidated_generated_robot"),
    ],
)
def test_pair_target_validation(tmp_path, changes, reason):
    source = {
        "dataset_uri": "source",
        "revision": "a" * 40,
        "episode_id": "0",
        "frames": 100,
        "cameras": ["front"],
        "kind": "ego",
        "has_actions": False,
        "split": "train",
        "origin_group": None,
        "simulation_validation": None,
    }
    target = {**source, "dataset_uri": "target", "kind": "robot", "has_actions": True, **changes}
    path = tmp_path / "inventory.parquet"
    pq.write_table(pa.Table.from_pylist([source, target], schema=INVENTORY), path)
    refs = [
        {
            **{key: row[key] for key in ("dataset_uri", "revision", "episode_id")},
            "camera": "front",
            "start": 0,
            "end": 100,
        }
        for row in (source, target)
    ]
    store = ArtifactStore(tmp_path / "run")
    plan = seal_plan(
        store,
        REF,
        "lerobot.data_processing.modules.pair_links:PairLinks",
        {"inventory_uri": str(path), "inventory_sha256": file_checksum(path)[0]},
        [InputItem("pair", {"source": refs[0], "target": refs[1], "pair_type": "ego_to_robot"})],
    )
    summary = run_local(store, plan)
    result = next(iter(accepted_in_shard(store, plan, 0).values()))
    assert summary.rejected == 1 and result.reason == reason and not result.artifacts


def test_local_cpu_admission(tmp_path, monkeypatch):
    from lerobot.data_processing import runtime
    from tests.data_processing.test_runtime import make_plan

    monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda _: {0}, raising=False)
    store = ArtifactStore(tmp_path / "run")
    plan = make_plan(store, size=4)
    with pytest.raises(ValueError, match="CPU cores"):
        run_local(store, plan, workers=2)
    assert not store.list(f"{plan.prefix}/attempts/*/*/checkpoints/*.json")


def test_full_conversion_benchmark_and_unmeasured_matrix(tmp_path):
    from lerobot.configs import RGBEncoderConfig
    from lerobot.data_processing.conversion import ConvertConfig

    benchmark = runpy.run_path(str(Path(__file__).parents[2] / "benchmarks/benchmark_processing.py"))
    matrix = benchmark["scale_matrix"](8)
    assert matrix["measured"] is False and matrix["cpu"][-1]["requested_cores"] == 8192
    cfg = ConvertConfig(
        source={"manifest": str(raw_source(tmp_path))},
        size=64,
        encoder=RGBEncoderConfig(vcodec="h264"),
    )
    cfg.runtime.shard_size = 1
    result = benchmark["benchmark_conversion"](cfg, [1, 2])
    assert all(row["physical_input_hours"] == 2 / 3600 for row in result["results"])
    assert all(row["output_bytes"] > 0 and row["realtime_multiplier"] > 0 for row in result["results"])
    print(json.dumps(result, indent=2))

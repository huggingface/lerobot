# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

import subprocess
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

from lerobot.data_processing.artifacts import ArtifactStore
from lerobot.data_processing.planner import StagePlan, seal_plan
from lerobot.data_processing.runtime import finalize_stage, run_local
from lerobot.data_processing.types import DatasetRef, InputItem, ItemResult, Outcome, Resources
from lerobot.data_processing.worker import accepted_in_shard

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

FACTORY = "tests.data_processing._modules:Echo"


def make_plan(store, config=None, *, size=6, shard_size=2):
    return seal_plan(
        store,
        DatasetRef("local/input", "a" * 64),
        FACTORY,
        config or {},
        (InputItem(str(i), {"value": i}) for i in range(size)),
        shard_size=shard_size,
    )


def output_values(store, plan):
    rows = []
    for shard in range(plan.shards):
        results = accepted_in_shard(store, plan, shard)
        for item in plan.read_shard(store, shard):
            for artifact in results[item.item_id].artifacts:
                with store.open(artifact.path) as stream:
                    rows.extend(pq.read_table(stream).to_pylist())
    return rows


def test_sealed_plan_roundtrip_and_shard_independent_ids(tmp_path):
    store = ArtifactStore(tmp_path)
    one = make_plan(store, shard_size=1)
    three = make_plan(store, shard_size=3)
    a = [item for shard in range(one.shards) for item in one.read_shard(store, shard)]
    b = [item for shard in range(three.shards) for item in three.read_shard(store, shard)]
    assert a == b
    assert StagePlan.load(store, one.plan_id) == one
    assert make_plan(store, shard_size=1) == one
    selected = make_plan(store, size=5, shard_size=1)
    assert selected.plan_id != one.plan_id
    assert selected.read_shard(store, 0) == one.read_shard(store, 0)


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_worker_counts_preserve_outputs_and_identity(tmp_path, workers):
    store = ArtifactStore(tmp_path)
    plan = make_plan(store)
    summary = run_local(store, plan, workers=workers)
    assert summary.items == summary.completed == 6
    expected = [{"key": str(i), "seed": plan.read_shard(store, i // 2)[i % 2].seed} for i in range(6)]
    assert output_values(store, plan) == expected


def test_model_lifecycle_once_across_shards_and_noop_resume(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    log = tmp_path / "calls"
    plan = make_plan(store, {"log_path": str(log)})
    first = run_local(store, plan)
    assert log.read_text().splitlines() == ["setup", "0", "1", "2", "3", "4", "5", "teardown"]
    before = log.read_bytes()
    second = run_local(store, plan)
    assert second == first
    assert log.read_bytes() == before


def test_interrupted_shard_resumes_only_uncommitted_batches(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    log = tmp_path / "calls"
    plan = make_plan(
        store,
        {"log_path": str(log), "fail_key": "1", "failure_marker": str(tmp_path / "failed")},
        shard_size=6,
    )
    with pytest.raises(RuntimeError, match="injected interruption"):
        run_local(store, plan, max_retries=0)
    assert set(accepted_in_shard(store, plan, 0)) == {plan.read_shard(store, 0)[0].item_id}
    assert log.read_text().splitlines() == ["setup", "0", "1", "teardown"]
    run_local(store, plan)
    assert log.read_text().splitlines().count("0") == 1
    assert output_values(store, plan)[-1]["key"] == "5"


@pytest.mark.parametrize("batch_size", [1, 2])
def test_corrupt_output_recomputed_without_repeating_valid_items(tmp_path, batch_size):
    store = ArtifactStore(tmp_path / "run")
    log = tmp_path / "calls"
    plan = make_plan(store, {"log_path": str(log)})
    first = run_local(store, plan, batch_size=batch_size)
    result = next(iter(accepted_in_shard(store, plan, 0).values()))
    Path(store.path(result.artifacts[0].path)).write_bytes(b"corrupt")
    second = run_local(store, plan, batch_size=batch_size)
    assert second.completed == 6
    assert first.accepted_path != second.accepted_path
    calls = log.read_text().splitlines()
    assert calls.count("0") == 2 and calls.count("1") == 1


def test_missing_and_empty_outcomes_are_not_failures(tmp_path):
    store = ArtifactStore(tmp_path)
    plan = seal_plan(
        store, DatasetRef("input", "b" * 64), FACTORY, {}, [InputItem("missing", {"missing": True})]
    )
    summary = run_local(store, plan)
    assert summary.masked == 1 and summary.completed == 0
    result = next(iter(accepted_in_shard(store, plan, 0).values()))
    assert result.reason == "missing_camera"
    with pytest.raises(ValueError, match="require a reason"):
        ItemResult("id", Outcome.MASKED)


def test_schema_mismatch_never_commits(tmp_path):
    store = ArtifactStore(tmp_path)
    plan = make_plan(store, {"bad_schema": True}, size=1)
    with pytest.raises(ValueError, match="schema mismatch"):
        run_local(store, plan, max_retries=0)
    with pytest.raises(RuntimeError, match="incomplete"):
        finalize_stage(store, plan)


def test_duplicate_keys_and_incomplete_plans_rejected(tmp_path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        seal_plan(
            store, DatasetRef("input", "a" * 64), FACTORY, {}, [InputItem("same", {}), InputItem("same", {})]
        )
    plan = make_plan(ArtifactStore(tmp_path / "other"))
    with pytest.raises(RuntimeError, match="incomplete"):
        finalize_stage(ArtifactStore(tmp_path / "other"), plan)


def test_artifact_paths_and_immutable_keys(tmp_path):
    store = ArtifactStore(tmp_path)
    artifact = store.put_json("record.json", {"a": 1})
    assert store.verify(artifact)
    with pytest.raises(FileExistsError):
        store.put_json("record.json", {"a": 2})
    for path in ("../escape", "/escape", "x/../../escape", "x\\escape"):
        with pytest.raises(ValueError, match="Unsafe"):
            store.path(path)


def test_object_store_without_rename(tmp_path, monkeypatch):
    store = ArtifactStore(f"memory://processing-{uuid.uuid4().hex}")

    def no_rename(*args, **kwargs):
        raise AssertionError("Object storage must not depend on rename")

    monkeypatch.setattr(store.fs, "mv", no_rename)
    plan = make_plan(store, size=2)
    assert run_local(store, plan).completed == 2
    assert run_local(store, plan).completed == 2


def test_plan_corruption_detected(tmp_path):
    store = ArtifactStore(tmp_path)
    plan = make_plan(store)
    path = Path(store.path(plan.work_path))
    table = pq.read_table(path)
    table = table.set_column(2, "payload", pa.array(["{}"] * table.num_rows))
    pq.write_table(table, path, row_group_size=2)
    with pytest.raises(ValueError, match="checksum mismatch"):
        StagePlan.load(store, plan.plan_id)
    with pytest.raises(ValueError, match="shard checksum mismatch"):
        plan.read_shard(store, 0)


def test_cpu_runtime_imports_no_vlm_or_torch():
    script = "import sys; import lerobot.data_processing.runtime; assert not any(k == 'torch' or 'vlm_client' in k for k in sys.modules); from lerobot.utils import get_safe_torch_device; assert callable(get_safe_torch_device)"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_empty_plan_retains_typed_accepted_schema(tmp_path):
    store = ArtifactStore(tmp_path)
    plan = make_plan(store, size=0)
    summary = run_local(store, plan)
    assert summary.items == summary.completed == 0
    with store.open(summary.accepted_path) as stream:
        table = pq.read_table(stream)
    assert table.schema.field("item_id").type == pa.string()


def test_local_plan_rejects_a_second_controller(tmp_path):
    from filelock import FileLock

    store = ArtifactStore(tmp_path)
    plan = make_plan(store)
    lock_path = Path(store.path(f"locks/{plan.plan_id}.lock"))
    lock_path.parent.mkdir(parents=True)
    with FileLock(lock_path), pytest.raises(RuntimeError, match="Another controller"):
        run_local(store, plan)


@pytest.mark.parametrize("kwargs", [{"cpus": 0}, {"gpus": -1}, {"memory_gb": float("nan")}])
def test_invalid_resources(kwargs):
    with pytest.raises(ValueError):
        Resources(**kwargs)


def test_real_video_quality_black_frame_counts(tmp_path):
    video = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    for value in [0, 0, 240, 240]:
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()
    store = ArtifactStore(tmp_path / "run")
    plan = seal_plan(
        store,
        DatasetRef(str(video), "c" * 64),
        "lerobot.data_processing.modules.video_quality:VideoQuality",
        {"sample_every": 1},
        [InputItem("camera", {"video_path": str(video), "expected_frames": 4})],
    )
    run_local(store, plan)
    result = next(iter(accepted_in_shard(store, plan, 0).values()))
    with store.open(result.artifacts[0].path) as stream:
        row = pq.read_table(stream).to_pylist()[0]
    assert row["decoded_frames"] == row["sampled_frames"] == 4
    assert row["black_frames"] == 2

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

import subprocess
import sys
import uuid
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.annotations.runtime import worker
from lerobot.annotations.runtime.artifacts import ArtifactStore
from lerobot.annotations.runtime.planner import StagePlan, seal_plan
from lerobot.annotations.runtime.runtime import finalize_stage, gpu_assignments, run_local
from lerobot.annotations.runtime.types import (
    DatasetRef,
    InputItem,
    ItemResult,
    Outcome,
    Resources,
    artifact_identities,
    canonical_json,
)
from lerobot.annotations.runtime.worker import accepted_in_shard, accepted_items

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

FACTORY = "tests.annotations.runtime._modules:Echo"


def make_plan(store, config=None, *, size=6, shard_size=2):
    return seal_plan(
        store,
        DatasetRef("local/input", "a" * 64),
        FACTORY,
        config or {},
        (InputItem(str(i), {"value": i}) for i in range(size)),
        shard_size=shard_size,
    )


@pytest.mark.parametrize("missing_directory_error", [False, True])
def test_remote_listing_sees_checkpoints_written_by_other_workers(
    tmp_path, monkeypatch, missing_directory_error
):
    store = ArtifactStore(tmp_path)
    store.is_local = False
    first = store.path("attempts/worker-0/checkpoint.json")
    second = store.path("attempts/worker-1/checkpoint.json")
    visible = [first]
    cached = None

    def glob(path):
        nonlocal cached
        assert path == store.path("attempts/*/checkpoint.json")
        if cached is None:
            cached = list(visible)
        elif missing_directory_error and cached != visible:
            raise FileNotFoundError("New worker directory absent from cached parent listing")
        return cached

    def invalidate_cache():
        nonlocal cached
        cached = None

    monkeypatch.setattr(store.fs, "glob", glob)
    monkeypatch.setattr(store.fs, "invalidate_cache", invalidate_cache)
    assert store.list("attempts/*/checkpoint.json") == ["attempts/worker-0/checkpoint.json"]
    visible.append(second)  # another process publishes a checkpoint, bypassing this client's cache
    assert store.list("attempts/*/checkpoint.json") == [
        "attempts/worker-0/checkpoint.json",
        "attempts/worker-1/checkpoint.json",
    ]


def test_local_listing_does_not_invalidate_filesystem_cache(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path)
    store.put_json("checkpoint.json", {})
    monkeypatch.setattr(store.fs, "invalidate_cache", lambda: pytest.fail("Local listings need no refresh"))
    assert store.list("*.json") == ["checkpoint.json"]


def test_local_publication_needs_no_hardlinks_and_cannot_clobber(tmp_path, monkeypatch):
    import os
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(os, "link", lambda *args: pytest.fail("Hard links are not portable"))
    source = tmp_path / "source"
    source.write_bytes(b"result")
    store = ArtifactStore(tmp_path / "run")

    def publish(index):
        try:
            return store.put_file("./same-key" if index % 2 else "same-key", source)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(publish, range(4)))
    assert sum(result is not None for result in results) == 1
    artifact = next(result for result in results if result)
    assert store.verify(artifact)
    Path(store.path(artifact.path)).write_bytes(b"broken")
    # Downloads hash bytes even when a coarse filesystem clock hides same-size edits.
    with pytest.raises(ValueError, match="download checksum mismatch"):
        store.download(artifact, tmp_path / "download")
    assert not (tmp_path / "download").exists()
    # Give the stat-keyed memo an observable change, independent of clock granularity.
    Path(store.path(artifact.path)).write_bytes(b"broken-output")
    assert not store.verify(artifact)


def test_publication_lock_count_is_bounded(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    source = tmp_path / "source"
    source.write_bytes(b"result")
    for index in range(300):
        store.put_file(f"items/{index}/output", source)
    assert len(list((tmp_path / "run").rglob("*.lock"))) <= 64


def test_verification_cache_survives_large_second_pass(tmp_path, monkeypatch):
    from lerobot.annotations.runtime.artifacts import file_checksum
    from lerobot.annotations.runtime.types import Artifact

    store = ArtifactStore(tmp_path / "run")
    (tmp_path / "run").mkdir()
    artifacts = []
    for index in range(10000):
        path = tmp_path / "run" / str(index)
        path.write_bytes(b"checked")
        digest, size = file_checksum(path)
        artifacts.append(Artifact(str(index), digest, size))
    assert all(store.verify(artifact) for artifact in artifacts)
    monkeypatch.setattr(store, "open", lambda *args: pytest.fail("Unchanged artifacts should be memoized"))
    assert all(store.verify(artifact) for artifact in artifacts)


def test_complete_item_pins_exclude_global_provenance_but_unpinned_items_do_not(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    for pinned in (False, True):
        items = [InputItem("0", {"value": 0}, content_digest="c" * 64 if pinned else None)]
        first = seal_plan(store, DatasetRef("input", "a" * 64), FACTORY, {}, items, upstream={"parent": "x"})
        changed = seal_plan(
            store, DatasetRef("input", "b" * 64), FACTORY, {}, items, upstream={"parent": "y"}
        )
        assert first.plan_id != changed.plan_id
        assert (first.read_shard(store, 0)[0].item_id == changed.read_shard(store, 0)[0].item_id) is pinned


def test_verification_cache_evicts_lru_not_the_entire_index(tmp_path):
    store = ArtifactStore(tmp_path / "run", verification_cache_size=2)
    source = tmp_path / "source"
    source.write_bytes(b"data")
    a, b = store.put_file("a", source), store.put_file("b", source)
    assert store.verify(a)  # a, not b, must survive the next insertion.
    c = store.put_file("c", source)
    assert {path.rsplit("/", 1)[-1] for path, _ in store._verified} == {"a", "c"}
    assert store.verify(a) and store.verify(c) and store.verify(b)


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


def test_repacking_reuses_logical_items_without_model_startup(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    log = tmp_path / "calls"
    first = make_plan(store, {"log_path": str(log)}, shard_size=2)
    summary = run_local(store, first)
    calls = log.read_bytes()
    repacked = make_plan(store, {"log_path": str(log)}, shard_size=3)
    assert repacked.plan_id != first.plan_id
    assert run_local(store, repacked).content_digest == summary.content_digest
    assert log.read_bytes() == calls
    artifact = next(iter(accepted_in_shard(store, first, 0).values())).artifacts[0]
    Path(store.path(artifact.path)).write_bytes(b"corrupt")
    assert run_local(store, repacked).content_digest == summary.content_digest
    assert log.read_text().splitlines().count("0") == 2
    assert all(log.read_text().splitlines().count(str(key)) == 1 for key in range(1, 6))


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


@pytest.mark.parametrize("fail_key", ["0", "1"])
def test_spawn_failure_stops_sibling_batches_and_preserves_resume(tmp_path, fail_key):
    store = ArtifactStore(tmp_path / "run")
    log = tmp_path / "calls"
    plan = seal_plan(
        store,
        DatasetRef("local/input", "a" * 64),
        "tests.annotations.runtime._modules:FailFastEcho",
        {"log_path": str(log), "fail_key": fail_key, "failure_marker": str(tmp_path / "failed")},
        (InputItem(str(i), {}) for i in range(20)),
        shard_size=1,
    )
    with pytest.raises(ValueError, match="permanent sibling failure"):
        run_local(store, plan, workers=2, batch_size=1)
    lines = log.read_text().splitlines()
    completed = [line for line in lines if line.isdigit()]
    assert 1 <= len(completed) < 10  # Do not drain the healthy group's ten shards.
    assert lines.count("setup") == lines.count("teardown") == 2
    assert not store.list(f"{plan.prefix}/accepted/*.parquet")
    assert run_local(store, plan, workers=2, batch_size=1).completed == 20
    final = log.read_text().splitlines()
    assert all(final.count(key) == 1 for key in completed)  # Cancelled siblings checkpoint first.


@pytest.mark.parametrize("permanent", [False, True])
def test_diagnostic_store_outage_does_not_override_retry_decision(tmp_path, monkeypatch, caplog, permanent):
    store = ArtifactStore(tmp_path / "run")
    plan = make_plan(store, size=1)
    original = ArtifactStore.put_file
    failures = []

    def put(self, relative, source, **kwargs):
        if relative.endswith("/error.json"):
            raise ConnectionError("diagnostic store is still unavailable")
        if "/outputs/" in relative and not failures:
            failures.append(relative)
            if permanent:
                raise ValueError("original invalid schema")
            raise ConnectionError("original output store outage")
        return original(self, relative, source, **kwargs)

    monkeypatch.setattr(ArtifactStore, "put_file", put)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)
    if permanent:
        with pytest.raises(ValueError, match="original invalid schema"):
            run_local(store, plan, max_retries=1)
    else:
        assert run_local(store, plan, max_retries=1).completed == 1
    assert len(failures) == 1
    assert "Could not persist worker error diagnostic" in caplog.text


@pytest.mark.parametrize("malformed_field", ["results", "artifacts"])
@pytest.mark.parametrize("replace_existing", [False, True], ids=["cached", "repair"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_resume_skips_malformed_checkpoints(tmp_path, malformed_field, replace_existing, batch_size, caplog):
    store = ArtifactStore(tmp_path / "run")
    log = tmp_path / "calls"
    plan = make_plan(store, {"log_path": str(log)}, size=2)
    first = run_local(store, plan, batch_size=batch_size)
    before = log.read_bytes()
    checkpoint_path = store.list(f"{plan.prefix}/attempts/00000000/*/checkpoints/*.json")[0]
    checkpoint = store.read_json(checkpoint_path)
    if malformed_field == "results":
        checkpoint["results"] = None
    else:
        checkpoint["results"][0]["artifacts"] = None
    if replace_existing:
        Path(store.path(checkpoint_path)).write_bytes(canonical_json(checkpoint))
    else:
        store.put_json(
            f"{plan.prefix}/attempts/00000000/00000000000000000000-invalid/checkpoints/00000000.json",
            checkpoint,
        )

    second = run_local(store, plan, batch_size=batch_size)
    assert "Ignoring invalid" in caplog.text and plan.plan_id in caplog.text
    assert second.completed == 2 and second.content_digest == first.content_digest
    calls = log.read_text().splitlines()
    assert calls.count("0") == (2 if replace_existing else 1)
    # Null results invalidate the whole batch; a bad artifact invalidates only its item.
    assert calls.count("1") == (
        2 if replace_existing and malformed_field == "results" and batch_size == 2 else 1
    )
    if not replace_existing:
        assert second == first and log.read_bytes() == before  # No model setup on cached resume.


@pytest.mark.parametrize("batch_size", [1, 2])
def test_corrupt_output_recomputed_without_repeating_valid_items(tmp_path, monkeypatch, batch_size):
    store = ArtifactStore(tmp_path / "run")
    log = tmp_path / "calls"
    plan = make_plan(store, {"log_path": str(log)})
    first = run_local(store, plan, batch_size=batch_size)
    result = next(iter(accepted_in_shard(store, plan, 0).values()))
    Path(store.path(result.artifacts[0].path)).write_bytes(b"corrupt")
    second = run_local(store, plan, batch_size=batch_size)
    assert second.completed == 6
    assert first.accepted_path != second.accepted_path
    assert first.content_digest == second.content_digest
    decode_schema = worker.pa.ipc.read_schema
    decoded = []

    def read_schema(*args, **kwargs):
        decoded.append(True)
        return decode_schema(*args, **kwargs)

    monkeypatch.setattr(worker.pa.ipc, "read_schema", read_schema)
    replay = accepted_in_shard(store, plan, 0)
    assert replay[plan.read_shard(store, 0)[0].item_id].artifacts[0].path != result.artifacts[0].path
    assert len(replay) == 2 and len(decoded) == 1  # Both attempts validated with one decoded schema.
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


@pytest.mark.parametrize("remote", [False, True])
def test_verified_download_is_single_pass_and_does_not_publish_corruption(tmp_path, monkeypatch, remote):
    store = ArtifactStore(f"memory://download-{uuid.uuid4().hex}" if remote else tmp_path / "store")
    source = tmp_path / "source"
    source.write_bytes(b"x" * (2 * 1024 * 1024 + 7))
    artifact = store.put_file("source", source)
    destination = tmp_path / "release" / "result"
    opened = []
    original = store.open

    def counted_open(path):
        opened.append(path)
        return original(path)

    monkeypatch.setattr(store, "open", counted_open)
    assert store.download(artifact, destination) == destination
    assert destination.read_bytes() == source.read_bytes()
    assert opened == [artifact.path]
    with store.fs.open(store.path(artifact.path), "wb") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.download(artifact, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert sorted(destination.parent.iterdir()) == [destination]
    missing = tmp_path / "missing-result"
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.download(artifact, missing)
    assert not missing.exists()


def test_accepted_items_are_ordered_complete_and_shard_bounded(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path)
    plan = make_plan(store, size=5, shard_size=2)
    run_local(store, plan)
    visited = []
    original = StagePlan.read_shard

    def read_shard(self, store, shard):
        visited.append(shard)
        return original(self, store, shard)

    monkeypatch.setattr(StagePlan, "read_shard", read_shard)
    results = accepted_items(store, plan)
    first, result = next(results)
    assert first.key == "0" and result.item_id == first.item_id
    assert set(visited) == {0}  # No eager scan of later shards.
    assert [item.key for item, _ in results] == ["1", "2", "3", "4"]
    assert set(visited) == {0, 1, 2}
    Path(store.path(result.artifacts[0].path)).unlink()
    with pytest.raises(RuntimeError, match="Stage incomplete"):
        list(accepted_items(store, plan))


@pytest.mark.parametrize("gpus", [0, 1])
def test_spawn_dispatch_shares_cpu_pool_but_isolates_gpu_groups(tmp_path, monkeypatch, gpus):
    from lerobot.annotations.runtime import runtime

    pools, calls = [], []

    class Pool:
        def __init__(self, *, max_workers, mp_context, initializer, initargs):
            assert mp_context.get_start_method() == "spawn"
            assert initializer is worker._initialize_worker and len(initargs) == 1
            pools.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, *args):
            calls.append((self, function, args))
            future = Future()
            future.set_result(None)
            return future

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setattr(runtime, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(
        runtime,
        "load_module",
        lambda *a: SimpleNamespace(spec=SimpleNamespace(resources=Resources(gpus=gpus))),
    )
    monkeypatch.setattr(runtime, "finalize_stage", lambda *a: "finalized")
    store = ArtifactStore(tmp_path)
    plan = make_plan(store)
    assert run_local(store, plan, workers=2) == "finalized"
    assert pools == ([1, 1] if gpus else [2])
    assert (calls[0][0] is calls[1][0]) == (not gpus)
    assert [args[2] for _, _, args in calls] == [[0, 2], [1]]
    expected_devices = [("GPU-a",), ("GPU-b",)] if gpus else [None, None]
    assert [args[-1] for _, _, args in calls] == expected_devices


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
    script = """
import sys
import lerobot.annotations.runtime.runtime
assert not any(k == 'torch' or 'vlm_client' in k for k in sys.modules)
from lerobot import utils
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging
assert utils.ActionInterpolator is ActionInterpolator
assert utils.get_safe_torch_device is get_safe_torch_device
assert utils.register_third_party_plugins is register_third_party_plugins
assert utils.init_logging is init_logging
assert all(hasattr(utils, name) for name in utils.__all__)
"""
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
            DatasetRef("fixture/input", "a" * 40),
            FACTORY,
            {},
            [InputItem("one", payload, identity_payload=artifact_identities(payload))],
        )
        ids.append(plan.read_shard(store, 0)[0].item_id)
    assert ids[0] == ids[1] and ids[2] != ids[1]


def test_remote_store_without_rename_or_immediate_listing(monkeypatch):
    from lerobot.annotations.runtime.worker import run_worker_group

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
    result = finalize_stage(store, plan)
    assert result.completed == 2
    assert run_local(store, plan) == result  # Cached resume also needs no remote rename.


def test_local_cpu_admission(tmp_path, monkeypatch):
    from lerobot.annotations.runtime import runtime

    monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda _: {0}, raising=False)
    store = ArtifactStore(tmp_path / "run")
    plan = make_plan(store, size=4)
    with pytest.raises(ValueError, match="CPU cores"):
        run_local(store, plan, workers=2)
    assert not store.list(f"{plan.prefix}/attempts/*/*/checkpoints/*.json")

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Real worker admission, quality masks, classified retries and window ownership."""

import pytest

from lerobot.data_processing.artifacts import ArtifactStore
from lerobot.data_processing.configs import RuntimeConfig, StageConfig
from lerobot.data_processing.errors import ErrorCategory, UndecodableInputError, classify_error, retry_delay
from lerobot.data_processing.pipeline import run_pipeline
from lerobot.data_processing.planner import seal_plan
from lerobot.data_processing.runtime import run_local
from lerobot.data_processing.types import DatasetRef, InputItem, Resources
from lerobot.data_processing.windows import reconcile_frame_rows, reconcile_spans, time_windows
from lerobot.data_processing.worker import accepted_in_shard

pytest.importorskip("pyarrow")


class Source:
    dataset_ref = DatasetRef("fixture", "a" * 40)

    def discover(self, stage, store, upstream):
        yield InputItem(stage.id, {"value": 1, "quality": {"usable": False}})


def test_independent_stages_overlap_and_join_waits(tmp_path):
    stages = [
        StageConfig(
            name,
            "tests.data_processing._modules:RendezvousEcho",
            {
                "signal": str(tmp_path / name),
                "peer": str(tmp_path / peer),
            },
        )
        for name, peer in (("hands", "subtasks"), ("subtasks", "hands"))
    ]
    stages.append(
        StageConfig("join", "tests.data_processing._modules:Echo", depends_on=("hands", "subtasks"))
    )
    store, completed = run_pipeline(
        Source(),
        stages,
        RuntimeConfig(
            run_uri=str(tmp_path / "run"),
            max_parallel_stages=2,
            resource_budget=Resources(cpus=2, memory_gb=8),
        ),
    )
    assert set(completed) == {"hands", "subtasks", "join"}
    assert all(summary.completed == 1 for _, summary in completed.values())
    assert len(store.list("stages/*/accepted/*.parquet")) == 3


def test_budget_serializes_ready_stages_and_rejects_oversized_request(tmp_path, monkeypatch):
    import threading
    import time

    from lerobot.data_processing import pipeline

    original = pipeline.run_local
    active, maximum = 0, 0
    lock = threading.Lock()

    def execute(*args, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        result = original(*args, **kwargs)
        with lock:
            active -= 1
        return result

    monkeypatch.setattr(pipeline, "run_local", execute)
    stages = [StageConfig(name, "tests.data_processing._modules:Echo") for name in ("a", "b")]
    runtime = RuntimeConfig(
        run_uri=str(tmp_path / "run"), max_parallel_stages=2, resource_budget=Resources(cpus=1, memory_gb=4)
    )
    assert len(run_pipeline(Source(), stages, runtime)[1]) == 2
    assert maximum == 1
    runtime.resource_budget = Resources(cpus=1, memory_gb=1)
    with pytest.raises(ValueError, match="exceeds resource_budget"):
        run_pipeline(Source(), stages, runtime)


def test_condition_masks_without_model_setup_and_missing_condition_errors(tmp_path):
    log = tmp_path / "calls"
    stages = [
        StageConfig("quality", "tests.data_processing._modules:Echo"),
        StageConfig(
            "hands",
            "tests.data_processing._modules:Echo",
            {"log_path": str(log)},
            depends_on=("quality",),
            when="quality.usable",
            skip_reason="black_frames",
        ),
    ]
    store, completed = run_pipeline(Source(), stages, RuntimeConfig(run_uri=str(tmp_path / "run")))
    plan, summary = completed["hands"]
    assert summary.masked == summary.items == 1 and summary.completed == 0
    assert not log.exists()  # Not even model setup or teardown ran.
    result = next(iter(accepted_in_shard(store, plan, 0).values()))
    assert result.reason == "black_frames" and not result.artifacts
    stages[1].when = "quality.missing"
    with pytest.raises(ValueError, match="Missing condition"):
        run_pipeline(Source(), stages, RuntimeConfig(run_uri=str(tmp_path / "missing")))


def test_transient_retry_and_permanent_fail_fast(tmp_path, monkeypatch):
    from lerobot.data_processing import worker

    delays = []
    monkeypatch.setattr(worker.time, "sleep", delays.append)
    for permanent in (False, True):
        store = ArtifactStore(tmp_path / str(permanent))
        plan = seal_plan(
            store,
            Source.dataset_ref,
            "tests.data_processing._modules:TemporaryEcho",
            {"marker": str(tmp_path / "marker"), "permanent": permanent},
            [InputItem("0", {})],
        )
        if permanent:
            with pytest.raises(ValueError, match="schema"):
                run_local(store, plan)
        else:
            assert run_local(store, plan).completed == 1
        errors = [store.read_json(path) for path in store.list("stages/*/attempts/*/*/error.json")]
        assert len(errors) == 1
        assert errors[0]["category"] == ("invalid_schema" if permanent else "temporary_network")
        assert errors[0]["retryable"] is not permanent
    assert delays == [1.0]
    assert retry_delay(100) == 30
    assert classify_error(UndecodableInputError("bad bytes")) == ErrorCategory.INPUT
    assert classify_error(RuntimeError("bug")) == ErrorCategory.UNKNOWN


def test_overlap_owns_every_irregular_source_frame_exactly_once():
    timestamps = (1.0, 1.03, 1.09, 1.2, 1.28, 1.43, 1.61, 1.8)
    indices = (3, 4, 7, 8, 9, 11, 13, 15)
    windows = time_windows(4, indices, timestamps, context_seconds=0.6, overlap_seconds=0.1)
    owned = [frame for window in windows for frame in window.output_frames]
    assert [(f.frame_index, f.timestamp) for f in owned] == list(zip(indices, timestamps, strict=True))
    assert windows[0].context_end > windows[0].output_end
    output = []
    for window in windows:
        rows = [
            {"episode_index": f.episode_index, "frame_index": f.frame_index, "timestamp": f.timestamp}
            for f in window.frames
        ]
        output.extend(reconcile_frame_rows(window, rows))
    assert [row["frame_index"] for row in output] == list(indices)
    rows[0]["timestamp"] += 1
    with pytest.raises(ValueError, match="altered"):
        reconcile_frame_rows(windows[-1], rows)
    clipped = reconcile_spans(windows[0], [{"start": 0, "end": 2, "text": "hold"}], timestamps)
    assert clipped == [{"start": 1.0, "end": timestamps[windows[0].output_end], "text": "hold"}]

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Typed operational measurements, never part of scientific result identities."""

import logging
import tempfile
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path

logger = logging.getLogger(__name__)
TIMERS = (
    "setup",
    "process",
    "validation",
    "checkpoint",
    "resume_scan",
    "teardown",
    "read",
    "decode",
    "inference",
    "write",
)


@contextmanager
def measure(metrics, name, lock=None):
    if name not in TIMERS:
        raise ValueError(f"Unknown metric timer: {name}")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        with lock or nullcontext():
            metrics[name + "_seconds"] = metrics.get(name + "_seconds", 0.0) + elapsed


def worker_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("plan_id", pa.string()),
            ("worker_id", pa.string()),
            ("status", pa.string()),
            ("wall_seconds", pa.float64()),
            *((timer + "_seconds", pa.float64()) for timer in TIMERS),
            ("batches", pa.int64()),
            ("items_computed", pa.int64()),
            ("items_reused", pa.int64()),
            ("retries", pa.int64()),
            ("bytes_written", pa.int64()),
            ("requested_cpu_hours", pa.float64()),
            ("requested_gpu_hours", pa.float64()),
        ]
    )


def record_worker(store, plan_id, metrics, resources):
    """Best-effort telemetry must never turn accepted work into a failed result."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    identifier = uuid.uuid4().hex
    row = {
        "plan_id": plan_id,
        "worker_id": identifier,
        **{timer + "_seconds": 0.0 for timer in TIMERS},
        "batches": 0,
        "items_computed": 0,
        "items_reused": 0,
        "retries": 0,
        "bytes_written": 0,
        **metrics,
        "requested_cpu_hours": metrics["wall_seconds"] * resources.cpus / 3600,
        "requested_gpu_hours": metrics["wall_seconds"] * resources.gpus / 3600,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="lerobot-metrics-") as directory:
            path = Path(directory) / "worker.parquet"
            pq.write_table(pa.Table.from_pylist([row], schema=worker_schema()), path)
            store.put_file(f"metrics/{plan_id}/workers/{identifier}.parquet", path)
    except Exception:
        logger.warning("Could not persist worker telemetry for %s", plan_id, exc_info=True)


def report_stage(store, plan, summary, *, wall_seconds, worker_paths=None):
    """Controller wall time includes workers, I/O, retries and finalization.

    Requested worker hours are estimates from declared resources, not billing,
    utilization, scheduler queue time, or controller/service GPU ownership.
    """
    import pyarrow.parquet as pq

    physical = camera = 0.0
    unknown = 0
    for shard in range(plan.shards):
        for item in plan.read_shard(store, shard):
            physical += item.physical_seconds or 0
            camera += item.camera_seconds or 0
            unknown += int(item.physical_seconds is None)
    records = []
    for path in (
        worker_paths if worker_paths is not None else store.list(f"metrics/{plan.plan_id}/workers/*.parquet")
    ):
        with store.open(path) as stream:
            records.extend(pq.read_table(stream).to_pylist())
    return {
        "plan_id": plan.plan_id,
        "items": summary.items,
        "completed": summary.completed,
        "masked": summary.masked,
        "rejected": summary.rejected,
        "wall_seconds": wall_seconds,
        "physical_input_hours": physical / 3600,
        "camera_input_hours": camera / 3600,
        "unknown_duration_items": unknown,
        "realtime_multiplier": physical / wall_seconds
        if wall_seconds > 0
        and not unknown
        and records
        and not any(row["items_reused"] for row in records)
        and sum(row["items_computed"] for row in records) >= summary.items
        else None,
        "requested_cpu_hours": sum(row["requested_cpu_hours"] for row in records),
        "requested_gpu_hours": sum(row["requested_gpu_hours"] for row in records),
        "worker_attempts": len(records),
        "bytes_written": sum(row["bytes_written"] for row in records),
        "retries": sum(row["retries"] for row in records),
        "timer_seconds": {timer: sum(row[timer + "_seconds"] for row in records) for timer in TIMERS},
        "resource_accounting": "declared worker resources only; controller/services/queue not measured",
    }

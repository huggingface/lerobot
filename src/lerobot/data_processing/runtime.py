# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Bounded local execution and single-owner stage finalization."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock, Timeout

from lerobot.utils.import_utils import _pyarrow_available, require_package

from .artifacts import ArtifactStore, file_checksum
from .planner import StagePlan, load_module
from .types import Outcome, canonical_json
from .worker import accepted_in_shard, run_worker_group

if TYPE_CHECKING or _pyarrow_available:
    import pyarrow as pa
    import pyarrow.parquet as pq


@dataclass(frozen=True)
class StageSummary:
    plan_id: str
    items: int
    completed: int
    masked: int
    rejected: int
    accepted_path: str
    content_digest: str


def finalize_stage(store: ArtifactStore, plan: StagePlan) -> StageSummary:
    """Validate complete coverage before publishing an immutable accepted index."""
    require_package("pyarrow", "dataset")
    schema = pa.schema(
        [
            ("item_id", pa.string()),
            ("outcome", pa.string()),
            ("reason", pa.string()),
            ("artifacts", pa.string()),
        ]
    )
    counts = dict.fromkeys(Outcome, 0)
    semantic_digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="lerobot-finalize-") as directory:
        path = Path(directory) / "accepted.parquet"
        with pq.ParquetWriter(path, schema) as writer:
            for shard in range(plan.shards):
                accepted = accepted_in_shard(store, plan, shard)
                items = plan.read_shard(store, shard)
                if len(accepted) != len(items):
                    raise RuntimeError(
                        f"Stage incomplete: shard {shard}, {len(accepted)}/{len(items)} accepted"
                    )
                rows = []
                for item in items:
                    result = accepted[item.item_id]
                    counts[result.outcome] += 1
                    semantic_digest.update(
                        canonical_json(
                            {
                                "item_id": item.item_id,
                                "outcome": result.outcome.value,
                                "reason": result.reason,
                                "artifacts": [
                                    {key: value for key, value in artifact.items() if key != "path"}
                                    for artifact in sorted(
                                        result.to_dict()["artifacts"], key=lambda value: value["name"]
                                    )
                                ],
                            }
                        )
                        + b"\n"
                    )
                    rows.append(
                        {
                            "item_id": item.item_id,
                            "outcome": result.outcome.value,
                            "reason": result.reason,
                            "artifacts": canonical_json(result.to_dict()["artifacts"]).decode(),
                        }
                    )
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        digest, size = file_checksum(path)
        accepted_path = f"{plan.prefix}/accepted/{digest}.parquet"
        if store.exists(accepted_path):
            if store.checksum(accepted_path) != (digest, size):
                raise RuntimeError("Accepted index checksum mismatch")
        else:
            store.put_file(accepted_path, path)
    return StageSummary(
        plan.plan_id,
        plan.items,
        counts[Outcome.COMPLETED],
        counts[Outcome.MASKED],
        counts[Outcome.REJECTED],
        accepted_path,
        semantic_digest.hexdigest(),
    )


def run_local(
    store: ArtifactStore, plan: StagePlan, *, workers: int = 1, batch_size: int = 1, max_retries: int = 2
) -> StageSummary:
    """Run static bounded groups with spawn; no CUDA/VLM construction in the coordinator.

    One controller owns this plan. Each worker initializes its module once, even
    when processing many shards. Resource admission belongs to the recipe/launcher.
    """
    if workers < 1 or batch_size < 1 or max_retries < 0:
        raise ValueError("Invalid workers, batch size or retry limit")
    if not store.is_local:
        lock = nullcontext()
    else:
        lock_path = Path(store.path(f"locks/{plan.plan_id}.lock"))
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(lock_path, timeout=0)
    try:
        with lock:
            return _run_local_owned(store, plan, workers, batch_size, max_retries)
    except Timeout as exc:
        raise RuntimeError("Another controller owns this local processing plan") from exc


def _run_local_owned(store, plan, workers, batch_size, max_retries):
    workers = min(workers, max(1, plan.shards))
    if workers > 1 and store.uri.startswith("memory://"):
        raise ValueError("Spawn workers require persistent shared storage, not memory://")
    groups = [list(range(index, plan.shards, workers)) for index in range(workers)]
    resources = load_module(plan.factory, plan.config).spec.resources
    capacity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    if workers * resources.cpus > capacity:
        raise ValueError(f"Requested {workers * resources.cpus} CPU cores, but this process has {capacity}")
    if resources.gpus:
        devices = gpu_assignments(workers, resources.gpus)
        # One pool per GPU group guarantees a fresh process will never switch
        # CUDA visibility after it has initialized the CUDA runtime.
        with ExitStack() as stack:
            pools = [
                stack.enter_context(
                    ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
                )
                for _ in groups
            ]
            futures = [
                pool.submit(
                    run_worker_group,
                    store.uri,
                    plan.plan_id,
                    group,
                    batch_size,
                    max_retries,
                    store.storage_options,
                    devices[index],
                )
                for index, (pool, group) in enumerate(zip(pools, groups, strict=True))
            ]
            for future in futures:
                future.result()
    elif workers == 1:
        run_worker_group(store.uri, plan.plan_id, groups[0], batch_size, max_retries, store.storage_options)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            futures = [
                executor.submit(
                    run_worker_group,
                    store.uri,
                    plan.plan_id,
                    group,
                    batch_size,
                    max_retries,
                    store.storage_options,
                )
                for group in groups
            ]
            for future in futures:
                future.result()
    return finalize_stage(store, plan)


def gpu_assignments(workers, gpus_per_worker):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        import torch

        devices = [str(index) for index in range(torch.cuda.device_count())]
    else:
        devices = [entry.strip() for entry in visible.split(",") if entry.strip() and entry.strip() != "-1"]
    if len(set(devices)) != len(devices):
        raise ValueError("Visible GPU IDs must be unique; duplicate IDs would oversubscribe one GPU")
    if workers * gpus_per_worker > len(devices):
        raise ValueError("Local GPU workers exceed visible GPUs; reduce workers or use Slurm/HF Jobs")
    return [
        tuple(devices[index * gpus_per_worker : (index + 1) * gpus_per_worker]) for index in range(workers)
    ]

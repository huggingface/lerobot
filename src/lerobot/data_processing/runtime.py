# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Bounded local execution and single-owner stage finalization."""

from __future__ import annotations

import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock, Timeout

from lerobot.utils.import_utils import _pyarrow_available, require_package

from .artifacts import ArtifactStore, file_checksum
from .planner import StagePlan
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
    if workers == 1:
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

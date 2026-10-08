# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Persistent worker lifecycle and schema-checked, manifest-last batch checkpoints."""

from __future__ import annotations

import base64
import json
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _pyarrow_available, require_package

from .artifacts import ArtifactStore
from .planner import StagePlan, load_module
from .types import Artifact, ItemResult, Outcome, WorkItem, checked_name

if TYPE_CHECKING or _pyarrow_available:
    import pyarrow as pa
    import pyarrow.parquet as pq


class WorkerContext:
    """Worker-scoped output access; modules must not mutate source or global metadata."""

    def __init__(self, store: ArtifactStore, plan: StagePlan, attempt: str, scratch: Path):
        require_package("pyarrow", "dataset")
        self.store, self.plan, self.attempt, self.scratch = store, plan, attempt, scratch
        self.schemas = {
            name: pa.ipc.read_schema(pa.BufferReader(base64.b64decode(schema))) if schema != "asset" else None
            for name, schema in plan.outputs.items()
        }

    def write_parquet(self, item: WorkItem, name: str, table: pa.Table) -> Artifact:
        if (
            name not in self.schemas
            or self.schemas[name] is None
            or not table.schema.equals(self.schemas[name], check_metadata=True)
        ):
            raise ValueError(f"Undeclared output or schema mismatch: {name}")
        path = self.scratch / f"{uuid.uuid4().hex}.parquet"
        try:
            pq.write_table(table, path, use_compliant_nested_type=False)
            return self.store.put_file(
                f"{self.attempt}/outputs/{item.item_id}/{checked_name(name)}.parquet",
                path,
                name=name,
                rows=table.num_rows,
            )
        finally:
            path.unlink(missing_ok=True)

    def write_asset(self, item: WorkItem, name: str, path: Path) -> Artifact:
        """Upload a closed noncanonical media/file asset; its module validates semantics."""
        if name not in self.schemas or self.schemas[name] is not None:
            raise ValueError(f"Undeclared asset output: {name}")
        return self.store.put_file(
            f"{self.attempt}/outputs/{item.item_id}/{checked_name(name)}{path.suffix}", path, name=name
        )


def validate_result(result: ItemResult, item: WorkItem, context: WorkerContext) -> None:
    if result.item_id != item.item_id:
        raise ValueError("Module returned the wrong item identity")
    names = [artifact.name for artifact in result.artifacts]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate output names")
    if result.outcome == Outcome.COMPLETED and set(names) != set(context.schemas):
        raise ValueError("Completed item is missing declared outputs")
    for artifact in result.artifacts:
        if not artifact.path.startswith(context.attempt + "/outputs/" + item.item_id + "/"):
            raise ValueError("Output belongs to another item or attempt")
        if artifact.name not in context.schemas or not context.store.verify(artifact):
            raise ValueError("Invalid artifact checksum or name")
        if context.schemas[artifact.name] is None:
            continue
        with context.store.open(artifact.path) as stream:
            parquet = pq.ParquetFile(stream)
            if (
                not parquet.schema_arrow.equals(context.schemas[artifact.name], check_metadata=True)
                or parquet.metadata.num_rows != artifact.rows
            ):
                raise ValueError("Invalid artifact schema or row count")


def accepted_in_shard(store: ArtifactStore, plan: StagePlan, shard: int) -> dict[str, ItemResult]:
    """Replay bounded manifests, accepting the first valid result per logical item."""
    items = {item.item_id: item for item in plan.read_shard(store, shard)}
    accepted: dict[str, ItemResult] = {}
    for path in store.list(f"{plan.prefix}/attempts/{shard:08d}/*/checkpoints/*.json"):
        try:
            checkpoint = store.read_json(path)
            if checkpoint["plan_id"] != plan.plan_id or checkpoint["shard"] != shard:
                continue
            attempt = path.split("/checkpoints/")[0]
            with tempfile.TemporaryDirectory(prefix="lerobot-verify-") as directory:
                context = WorkerContext(store, plan, attempt, Path(directory))
                for value in checkpoint["results"]:
                    result = ItemResult.from_dict(value)
                    if (
                        result.item_id in accepted
                        or result.outcome == Outcome.FAILED
                        or result.item_id not in items
                    ):
                        continue
                    try:
                        validate_result(result, items[result.item_id], context)
                    except (OSError, ValueError, KeyError):
                        continue
                    accepted[result.item_id] = result
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            # An incomplete, stale or corrupt checkpoint is not completion evidence.
            continue
    return accepted


def run_worker_group(
    store_uri: str,
    plan_id: str,
    shards: list[int],
    batch_size: int,
    max_retries: int,
    storage_options: dict | None = None,
) -> None:
    """One model setup across all assigned shards; retries preserve validated batches."""
    if batch_size < 1 or max_retries < 0:
        raise ValueError("Invalid batch size or retry limit")
    store = ArtifactStore(store_uri, storage_options=storage_options)
    # The controller verifies the complete work file once. Workers validate their
    # row-group digests instead of re-downloading/hash-reading the whole plan.
    plan = StagePlan.load(store, plan_id, verify_work=False)
    module = load_module(plan.factory, plan.config)
    with tempfile.TemporaryDirectory(prefix="lerobot-worker-") as directory:
        context = WorkerContext(store, plan, "", Path(directory))
        initialized = False
        try:
            for shard in shards:
                for retry in range(max_retries + 1):
                    accepted = accepted_in_shard(store, plan, shard)
                    pending = [item for item in plan.read_shard(store, shard) if item.item_id not in accepted]
                    if not pending:
                        break
                    if not initialized:
                        initialized = True
                        module.setup(context)
                    # UUID gives each concurrently requeued worker its own output
                    # namespace. Timestamp orders persisted attempts, not item IDs.
                    attempt = f"{plan.prefix}/attempts/{shard:08d}/{time.time_ns():020d}-{uuid.uuid4().hex}"
                    context.attempt = attempt
                    try:
                        for offset in range(0, len(pending), batch_size):
                            batch = pending[offset : offset + batch_size]
                            results = module.process_batch(batch, context)
                            if len(results) != len(batch) or len({r.item_id for r in results}) != len(batch):
                                raise ValueError("Module must return exactly one result per input item")
                            by_id = {result.item_id: result for result in results}
                            for item in batch:
                                validate_result(by_id[item.item_id], item, context)
                            if any(r.outcome == Outcome.FAILED for r in results):
                                raise RuntimeError("Module reported failed work")
                            store.put_json(
                                f"{attempt}/checkpoints/{offset:08d}.json",
                                {"plan_id": plan_id, "shard": shard, "results": [asdict(r) for r in results]},
                            )
                        break
                    except Exception:
                        if retry == max_retries:
                            raise
        finally:
            if initialized:
                module.teardown()

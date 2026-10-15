# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Persistent worker lifecycle and schema-checked, manifest-last batch checkpoints."""

from __future__ import annotations

import base64
import logging
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _pyarrow_available, require_package

from .artifacts import ArtifactStore
from .errors import ErrorCategory, classify_error, retry_delay
from .metrics import measure, record_worker
from .planner import StagePlan, load_module
from .types import Artifact, ItemResult, Outcome, WorkItem, checked_name, fingerprint

logger = logging.getLogger(__name__)
_stop_event: Event | None = None


def _initialize_worker(stop_event: Event) -> None:
    """Share cancellation through spawn initialization, never through a task queue."""
    global _stop_event
    _stop_event = stop_event


if TYPE_CHECKING or _pyarrow_available:
    import pyarrow as pa
    import pyarrow.parquet as pq


class WorkerContext:
    """Worker-scoped output access; modules must not mutate source or global metadata."""

    def __init__(self, store: ArtifactStore, plan: StagePlan, attempt: str, scratch: Path):
        require_package("pyarrow", "dataset")
        self.store, self.plan, self.attempt, self.scratch = store, plan, attempt, scratch
        self.metrics: dict[str, float | int | str] = {}
        self._metric_lock = threading.Lock()
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
            with self.measure("write"):
                artifact = self.store.put_file(
                    f"{self.attempt}/outputs/{item.item_id}/{checked_name(name)}.parquet",
                    path,
                    name=name,
                    rows=table.num_rows,
                )
            self.add_metric("bytes_written", artifact.size)
            return artifact
        finally:
            path.unlink(missing_ok=True)

    def write_asset(self, item: WorkItem, name: str, path: Path) -> Artifact:
        """Upload a closed noncanonical media/file asset; its module validates semantics."""
        if name not in self.schemas or self.schemas[name] is not None:
            raise ValueError(f"Undeclared asset output: {name}")
        with self.measure("write"):
            artifact = self.store.put_file(
                f"{self.attempt}/outputs/{item.item_id}/{checked_name(name)}{path.suffix}", path, name=name
            )
        self.add_metric("bytes_written", artifact.size)
        return artifact

    def measure(self, name):
        return measure(self.metrics, name, self._metric_lock)

    def add_metric(self, name, value):
        with self._metric_lock:
            self.metrics[name] = self.metrics.get(name, 0) + value


class _InvalidResultError(ValueError):
    """Safe diagnostic messages, never model output or arbitrary persisted content."""


def validate_result(result: ItemResult, item: WorkItem, context: WorkerContext) -> None:
    if result.item_id != item.item_id:
        raise _InvalidResultError("Module returned the wrong item identity")
    names = [artifact.name for artifact in result.artifacts]
    if len(names) != len(set(names)):
        raise _InvalidResultError("Duplicate output names")
    if result.outcome == Outcome.COMPLETED and set(names) != set(context.schemas):
        raise _InvalidResultError("Completed item is missing declared outputs")
    for artifact in result.artifacts:
        if not artifact.path.startswith(context.attempt + "/outputs/" + item.item_id + "/"):
            raise _InvalidResultError("Output belongs to another item or attempt")
        if artifact.name not in context.schemas or not context.store.verify(artifact):
            raise _InvalidResultError("Invalid artifact checksum or name")
        if context.schemas[artifact.name] is None:
            continue
        with context.store.open(artifact.path) as stream:
            parquet = pq.ParquetFile(stream)
            if (
                not parquet.schema_arrow.equals(context.schemas[artifact.name], check_metadata=True)
                or parquet.metadata.num_rows != artifact.rows
            ):
                raise _InvalidResultError("Invalid artifact schema or row count")


def accepted_in_shard(store: ArtifactStore, plan: StagePlan, shard: int) -> dict[str, ItemResult]:
    """Replay bounded manifests, accepting the first valid result per logical item."""
    items = {item.item_id: item for item in plan.read_shard(store, shard)}
    accepted: dict[str, ItemResult] = {}
    context = None

    def checkpoint_paths():
        yield from store.list(f"{plan.prefix}/attempts/{shard:08d}/*/checkpoints/*.json")
        # Bind current execution settings, but reuse checked logical results from
        # other plans. A healthy same-plan resume needs no shared index lookups.
        for item_id in sorted(items.keys() - accepted.keys()):
            for reference in store.list(f"cache/items/{item_id}/*.json"):
                try:
                    path = store.read_json(reference)["checkpoint"]
                    if not path.startswith(plan.prefix + "/"):
                        yield path
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    logger.warning("Ignoring invalid cache reference %s (%s)", reference, type(exc).__name__)

    seen = set()
    for path in checkpoint_paths():
        if path in seen:
            continue
        seen.add(path)
        if store.exists(path.split("/checkpoints/")[0] + "/pruned.json"):
            logger.debug("Skipping intentionally pruned materialization checkpoint %s", path)
            continue
        try:
            checkpoint = store.read_json(path)
            if not path.startswith(f"stages/{checkpoint['plan_id']}/attempts/{checkpoint['shard']:08d}/"):
                continue
            # Replay is read-only: decode schemas once, without creating scratch files.
            if context is None:
                context = WorkerContext(store, plan, "", Path())
            context.attempt = path.split("/checkpoints/")[0]
            for value in checkpoint["results"]:
                try:
                    result = ItemResult.from_dict(value)
                    if (
                        result.item_id in accepted
                        or result.outcome == Outcome.FAILED
                        or result.item_id not in items
                    ):
                        continue
                    validate_result(result, items[result.item_id], context)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    logger.warning(
                        "Ignoring invalid result: plan=%s shard=%s checkpoint=%s (%s)",
                        plan.plan_id,
                        shard,
                        path,
                        str(exc) if isinstance(exc, _InvalidResultError) else type(exc).__name__,
                    )
                    continue
                accepted[result.item_id] = result
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # An incomplete, stale or corrupt checkpoint is not completion evidence.
            logger.warning(
                "Ignoring invalid checkpoint: plan=%s shard=%s checkpoint=%s (%s)",
                plan.plan_id,
                shard,
                path,
                type(exc).__name__,
            )
            continue
    return accepted


def accepted_items(store: ArtifactStore, plan: StagePlan) -> Iterator[tuple[WorkItem, ItemResult]]:
    """Replay complete results in source order, retaining at most one shard at a time."""
    for shard in range(plan.shards):
        items = plan.read_shard(store, shard)
        accepted = accepted_in_shard(store, plan, shard)
        if len(accepted) != len(items):
            raise RuntimeError(f"Stage incomplete: shard {shard}, {len(accepted)}/{len(items)} accepted")
        for item in items:
            yield item, accepted[item.item_id]


def run_worker_group(
    store_uri: str,
    plan_id: str,
    shards: list[int],
    batch_size: int,
    max_retries: int,
    storage_options: dict | None = None,
    device_ids: tuple[str, ...] | None = None,
) -> None:
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_ids)
    try:
        return _run_worker_group(store_uri, plan_id, shards, batch_size, max_retries, storage_options)
    finally:
        if device_ids is not None:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _run_worker_group(store_uri, plan_id, shards, batch_size, max_retries, storage_options):
    """One model setup across all assigned shards; retries preserve validated batches."""
    start = time.perf_counter()
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
        setup_complete = False
        status = "failed"
        reused = set()
        try:
            for shard in shards:
                for retry in range(max_retries + 1):
                    if _stop_event is not None and _stop_event.is_set():
                        status = "cancelled"
                        return
                    with context.measure("resume_scan"):
                        accepted = accepted_in_shard(store, plan, shard)
                    reused.update(accepted)
                    pending = [item for item in plan.read_shard(store, shard) if item.item_id not in accepted]
                    if not pending:
                        break
                    # UUID gives each concurrently requeued worker its own output
                    # namespace. Timestamp orders persisted attempts, not item IDs.
                    attempt = f"{plan.prefix}/attempts/{shard:08d}/{time.time_ns():020d}-{uuid.uuid4().hex}"
                    context.attempt = attempt
                    try:
                        for offset in range(0, len(pending), batch_size):
                            if _stop_event is not None and _stop_event.is_set():
                                status = "cancelled"
                                return
                            batch = pending[offset : offset + batch_size]
                            active = [item for item in batch if item.mask_reason is None]
                            if active and not initialized:
                                initialized = True
                                with context.measure("setup"):
                                    module.setup(context)
                                setup_complete = True
                            with context.measure("process"):
                                results = module.process_batch(active, context) if active else []
                            results = list(results)
                            results.extend(
                                ItemResult(item.item_id, Outcome.MASKED, reason=item.mask_reason)
                                for item in batch
                                if item.mask_reason is not None
                            )
                            context.add_metric("batches", 1)
                            context.add_metric("items_computed", len(active))
                            by_id = {result.item_id: result for result in results}
                            if len(results) != len(batch) or len(by_id) != len(batch):
                                raise ValueError("Module must return exactly one result per input item")
                            with context.measure("validation"):
                                for item in batch:
                                    validate_result(by_id[item.item_id], item, context)
                            if any(r.outcome == Outcome.FAILED for r in results):
                                raise RuntimeError("Module reported failed work")
                            with context.measure("checkpoint"):
                                checkpoint_path = f"{attempt}/checkpoints/{offset:08d}.json"
                                store.put_json(
                                    checkpoint_path,
                                    {
                                        "plan_id": plan_id,
                                        "shard": shard,
                                        "results": [asdict(r) for r in results],
                                    },
                                )
                                for result in results:
                                    store.put_json(
                                        f"cache/items/{result.item_id}/{fingerprint(checkpoint_path)}.json",
                                        {"checkpoint": checkpoint_path},
                                    )
                        break
                    except Exception as exc:
                        category = classify_error(exc)
                        retryable = category in {ErrorCategory.NETWORK, ErrorCategory.SERVICE}
                        delay = retry_delay(retry) if retryable and retry < max_retries else 0.0
                        try:
                            store.put_json(
                                f"{attempt}/error.json",
                                {
                                    "category": category.value,
                                    "exception_type": type(exc).__name__,
                                    "retry": retry,
                                    "retryable": retryable,
                                    "backoff_seconds": delay,
                                },
                            )
                        except Exception as diagnostic_error:
                            logger.warning(
                                "Could not persist worker error diagnostic: plan=%s shard=%s (%s)",
                                plan_id,
                                shard,
                                type(diagnostic_error).__name__,
                            )
                        context.metrics["last_error_category"] = category.value
                        if not retryable or retry == max_retries:
                            raise
                        if initialized and not setup_complete:
                            # Never retry using a partly initialized model/client.
                            # Its teardown contract also covers failed setup.
                            module.teardown()
                            initialized = False
                            module = load_module(plan.factory, plan.config)
                        context.add_metric("retries", 1)
                        if _stop_event is None:
                            time.sleep(delay)
                        else:
                            _stop_event.wait(delay)
            status = "completed"
        finally:
            try:
                if initialized:
                    with context.measure("teardown"):
                        module.teardown()
            except BaseException:
                status = "failed"
                raise
            finally:
                context.metrics.update(
                    status=status, wall_seconds=time.perf_counter() - start, items_reused=len(reused)
                )
                record_worker(store, plan_id, context.metrics, module.spec.resources)

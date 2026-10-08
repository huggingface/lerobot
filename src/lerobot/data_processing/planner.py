# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Stream source metadata into immutable, row-group-addressable work plans."""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _pyarrow_available, require_package

from .artifacts import ArtifactStore, file_checksum
from .types import (
    Artifact,
    DatasetRef,
    InputItem,
    ProcessingModule,
    WorkItem,
    canonical_json,
    checked_name,
    fingerprint,
)

if TYPE_CHECKING or _pyarrow_available:
    import pyarrow as pa
    import pyarrow.parquet as pq


def load_module(factory: str, config: dict[str, Any]) -> ProcessingModule:
    """Load explicitly trusted Python code, without model setup or device imports."""
    module, separator, name = factory.partition(":")
    if not separator or not module or not name:
        raise ValueError("A module factory must be 'importable.module:ClassName'")
    return getattr(importlib.import_module(module), name)(**config)


@dataclass(frozen=True)
class StagePlan:
    plan_id: str
    factory: str
    config: dict[str, Any]
    work_path: str
    shards: int
    items: int
    outputs: dict[str, str]
    shard_hashes: list[str]

    @property
    def prefix(self) -> str:
        return f"stages/{self.plan_id}"

    @classmethod
    def load(cls, store: ArtifactStore, plan_id: str, *, verify_work: bool = True) -> StagePlan:
        checked_name(plan_id)
        manifest = store.read_json(f"stages/{plan_id}/plan.json")
        plan = cls(**manifest["plan"])
        if plan.plan_id != plan_id:
            raise ValueError("Plan identity mismatch")
        if verify_work and not store.verify(Artifact(**manifest["work"])):
            raise ValueError("Work plan checksum mismatch")
        return plan

    def read_shard(self, store: ArtifactStore, shard: int) -> list[WorkItem]:
        require_package("pyarrow", "dataset")
        if not 0 <= shard < self.shards:
            raise ValueError("Shard index outside the sealed plan")
        with store.open(self.work_path) as stream:
            rows = pq.ParquetFile(stream).read_row_group(shard).to_pylist()
        if fingerprint(rows) != self.shard_hashes[shard]:
            raise ValueError("Work shard checksum mismatch")
        return [
            WorkItem(row["item_id"], row["key"], json.loads(row["payload"]), row["cost"], row["seed"])
            for row in rows
        ]


def seal_plan(
    store: ArtifactStore,
    dataset: DatasetRef,
    factory: str,
    config: dict[str, Any],
    items: Iterable[InputItem],
    *,
    shard_size: int = 64,
    upstream: dict[str, str] | None = None,
) -> StagePlan:
    """Seal metadata once; item identity excludes worker count and scheduler IDs.

    Source order must be stable. Each row group is one bounded shard. The item-key
    set detects duplicates but no episode tensors or timestamps are collected.
    """
    require_package("pyarrow", "dataset")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    module = load_module(factory, config)
    schemas = {
        name: base64.b64encode(schema.serialize().to_pybytes()).decode()
        for name, schema in module.spec.outputs.items()
    }
    identity = {
        "dataset": asdict(dataset),
        "factory": factory,
        "config": config,
        "module": {"name": module.spec.name, "version": module.spec.version, "scope": module.spec.scope},
        "schemas": schemas,
        "upstream": upstream or {},
    }
    semantic_id = fingerprint(identity)
    # Packing belongs to the plan, but never to a logical item's identity/seed.
    schema = pa.schema(
        [
            ("item_id", pa.string()),
            ("key", pa.string()),
            ("payload", pa.string()),
            ("cost", pa.float64()),
            ("seed", pa.int64()),
        ]
    )
    selection = hashlib.sha256()
    seen: set[str] = set()
    count = shards = 0
    shard_hashes = []
    with tempfile.TemporaryDirectory(prefix="lerobot-plan-") as directory:
        path = Path(directory) / "work.parquet"
        with pq.ParquetWriter(path, schema) as writer:
            batch = []
            for item in items:
                if item.key in seen:
                    raise ValueError(f"Duplicate source item key: {item.key}")
                seen.add(item.key)
                selection.update(canonical_json(asdict(item)) + b"\n")
                item_id = fingerprint({"stage": semantic_id, "key": item.key, "payload": item.payload})
                batch.append(
                    {
                        "item_id": item_id,
                        "key": item.key,
                        "payload": canonical_json(item.payload).decode(),
                        "cost": float(item.cost),
                        "seed": int(item_id[:15], 16),
                    }
                )
                count += 1
                if len(batch) == shard_size:
                    shard_hashes.append(fingerprint(batch))
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema), row_group_size=shard_size)
                    batch = []
                    shards += 1
            if batch:
                shard_hashes.append(fingerprint(batch))
                writer.write_table(pa.Table.from_pylist(batch, schema=schema), row_group_size=shard_size)
                shards += 1
        plan_id = fingerprint(
            {"identity": identity, "shard_size": shard_size, "selection": selection.hexdigest()}
        )
        manifest_path = f"stages/{plan_id}/plan.json"
        if store.exists(manifest_path):
            return StagePlan.load(store, plan_id)
        work_path = f"stages/{plan_id}/work_items.parquet"
        # A killed discovery can leave an unreferenced immutable file. It is safe
        # to reuse only if the complete newly discovered bytes match.
        if store.exists(work_path):
            digest, size = file_checksum(path)
            if store.checksum(work_path) != (digest, size):
                raise ValueError("Unsealed work artifact differs from rediscovered inputs")
            artifact = Artifact(work_path, digest, size)
        else:
            artifact = store.put_file(work_path, path)
    plan = StagePlan(plan_id, factory, config, work_path, shards, count, schemas, shard_hashes)
    store.put_json(
        manifest_path,
        {
            "plan": asdict(plan),
            "identity": identity,
            "selection": selection.hexdigest(),
            "work": asdict(artifact),
        },
    )
    return plan

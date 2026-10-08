# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Validate HumanGen-style links to robot targets, without generating/copying video.

The supplied pinned inventory comes from validated episode metadata. This reducer
does not infer physical correctness from semantic similarity or invent robot actions.
"""

import pyarrow as pa
import pyarrow.parquet as pq

from ..sources.acquisition import acquire_file
from ..types import DatasetRef, ItemResult, ModuleSpec, Outcome, fingerprint

INVENTORY = pa.schema(
    [
        ("dataset_uri", pa.string()),
        ("revision", pa.string()),
        ("episode_id", pa.string()),
        ("frames", pa.int64()),
        ("cameras", pa.list_(pa.string())),
        ("kind", pa.string()),
        ("has_actions", pa.bool_()),
        ("split", pa.string()),
        ("origin_group", pa.string()),
        ("simulation_validation", pa.string()),
    ]
)
PAIRS = pa.schema(
    [
        ("pair_id", pa.string()),
        ("pair_type", pa.string()),
        ("split", pa.string()),
        ("source_dataset", pa.string()),
        ("source_revision", pa.string()),
        ("source_episode", pa.string()),
        ("source_camera", pa.string()),
        ("source_start", pa.int64()),
        ("source_end", pa.int64()),
        ("target_dataset", pa.string()),
        ("target_revision", pa.string()),
        ("target_episode", pa.string()),
        ("target_camera", pa.string()),
        ("target_start", pa.int64()),
        ("target_end", pa.int64()),
        ("action_loss", pa.bool_()),
        ("validation_evidence", pa.string()),
    ]
)


class PairLinks:
    def __init__(self, inventory_uri, inventory_sha256):
        self.inventory_uri, self.inventory_sha256 = inventory_uri, inventory_sha256
        self.spec = ModuleSpec("pair_links", "1", "reduction", {"pairs": PAIRS})

    def setup(self, context):
        path = acquire_file(self.inventory_uri, self.inventory_sha256, context.scratch / "inventory.parquet")
        table = pq.read_table(path)
        if not table.schema.equals(INVENTORY):
            raise ValueError("Pair inventory schema mismatch")
        self.inventory, origins = {}, {}
        for row in table.to_pylist():
            DatasetRef(row["dataset_uri"], row["revision"])
            key = (row["dataset_uri"], row["revision"], row["episode_id"])
            if key in self.inventory or not row["frames"] or row["frames"] < 0:
                raise ValueError("Duplicated/invalid pair inventory episode")
            if row["split"] not in {"train", "validation", "test"}:
                raise ValueError("Pair inventory must have fixed splits before pairing")
            origin = row["origin_group"]
            if origin and origin in origins and origins[origin] != row["split"]:
                raise ValueError("Generated relatives cross evaluation splits")
            origins[origin] = row["split"]
            self.inventory[key] = row

    def teardown(self):
        self.inventory = {}

    def _resolve(self, ref):
        row = self.inventory.get((ref["dataset_uri"], ref["revision"], ref["episode_id"]))
        if row is None:
            raise ValueError("unresolved_episode")
        if (
            not all(type(ref[key]) is int for key in ("start", "end"))
            or not 0 <= ref["start"] < ref["end"] <= row["frames"]
        ):
            raise ValueError("invalid_frame_range")
        if ref["camera"] not in row["cameras"]:
            raise ValueError("unresolved_camera")
        return row

    def process_batch(self, items, context):
        results = []
        for item in items:
            payload = item.payload
            try:
                source, target = self._resolve(payload["source"]), self._resolve(payload["target"])
                if source["split"] != target["split"]:
                    raise ValueError("cross_split_pair")
                if target["kind"] not in {"robot", "sim_robot"} or not target["has_actions"]:
                    raise ValueError("target_has_no_robot_actions")
                if target["kind"] == "sim_robot" and not target["simulation_validation"]:
                    raise ValueError("unvalidated_generated_robot")
                if not payload["pair_type"]:
                    raise ValueError("missing_pair_type")
                row = {
                    "pair_id": fingerprint(payload),
                    "pair_type": payload["pair_type"],
                    "split": source["split"],
                    "action_loss": payload["pair_type"] != "weak_semantic"
                    and bool(payload.get("alignment_validated"))
                    and bool(payload.get("alignment_evidence")),
                    "validation_evidence": payload.get("alignment_evidence")
                    or target["simulation_validation"],
                }
                for side in ("source", "target"):
                    ref = payload[side]
                    row.update(
                        {
                            side + "_dataset": ref["dataset_uri"],
                            side + "_revision": ref["revision"],
                            side + "_episode": ref["episode_id"],
                            side + "_camera": ref["camera"],
                            side + "_start": ref["start"],
                            side + "_end": ref["end"],
                        }
                    )
            except (KeyError, ValueError) as exc:
                results.append(ItemResult(item.item_id, Outcome.REJECTED, reason=str(exc)))
                continue
            artifact = context.write_parquet(item, "pairs", pa.Table.from_pylist([row], schema=PAIRS))
            results.append(ItemResult(item.item_id, Outcome.COMPLETED, (artifact,)))
        return results

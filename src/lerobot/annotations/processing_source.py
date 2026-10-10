# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Native episode discovery and lineage for annotation recipes."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lerobot.data_processing.artifacts import file_checksum
from lerobot.data_processing.types import (
    Artifact,
    DatasetRef,
    InputItem,
    artifact_identities,
    fingerprint,
)
from lerobot.data_processing.worker import accepted_items
from lerobot.utils.constants import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT

from .steerable_pipeline.reader import EpisodeRecord, _load_tasks_lookup


def _load_v3_info(root: Path) -> dict[str, Any]:
    info = json.loads((root / "meta/info.json").read_text())
    version = info.get("codebase_version") if isinstance(info, dict) else None
    if not isinstance(version, str) or not version.startswith("v3."):
        raise ValueError(
            "Scalable annotation requires a LeRobot v3 dataset; "
            "use the standalone conversion/format-upgrade scripts first"
        )
    return info


def _load_record(root, payload):
    tables = []
    for relative in payload["paths"]:
        table = pq.read_table(root / relative, columns=["episode_index", "frame_index", "timestamp"])
        tables.append(table.filter(pc.equal(table["episode_index"], payload["episode_index"])))
    table = pa.concat_tables(tables).sort_by([("frame_index", "ascending")])
    indices = table["frame_index"].to_pylist()
    if len(indices) != payload["rows"] or indices != list(range(len(indices))):
        raise ValueError("Incomplete, duplicated or noncontiguous episode frames")
    timestamps = table["timestamp"].to_pylist()
    if not timestamps or any(b <= a for a, b in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("Invalid episode clock")
    return EpisodeRecord(
        payload["episode_index"],
        payload["task"],
        tuple(timestamps),
        tuple(indices),
        root / payload["paths"][0],
        0,
        len(indices),
        data_paths=tuple(root / path for path in payload["paths"]),
    )


class LeRobotEpisodeSource:
    """Discover small episode/file references; only workers collect episode clocks."""

    def __init__(self, root: Path, only_episodes=None):
        self.root = root
        info = _load_v3_info(root)
        self.episodes: dict[int, dict[str, Any]] = {}
        tasks = _load_tasks_lookup(root)
        manifest: list[dict[str, Any]] = []
        selected = set(only_episodes) if only_episodes is not None else None
        # Explicit frame-file pattern excludes sparse annotation Parquet tables.
        for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
            file = pq.ParquetFile(path)
            native_columns = [
                name for name in file.schema_arrow.names if name not in {LANGUAGE_PERSISTENT, LANGUAGE_EVENTS}
            ]
            digest = hashlib.sha256()
            for batch in file.iter_batches(batch_size=1024, columns=native_columns):
                table = pa.Table.from_batches([batch]).replace_schema_metadata(None)
                buffer = pa.BufferOutputStream()
                with pa.ipc.new_stream(buffer, table.schema) as writer:
                    writer.write_table(table)
                digest.update(buffer.getvalue())
            manifest.append({"path": str(path.relative_to(root)), "native": digest.hexdigest()})
            columns = ["episode_index"] + (["task_index"] if "task_index" in file.schema_arrow.names else [])
            for batch in file.iter_batches(columns=columns):
                for row in batch.to_pylist():
                    ep = row["episode_index"]
                    if selected is not None and ep not in selected:
                        continue
                    value = self.episodes.setdefault(
                        ep,
                        {
                            "episode_index": ep,
                            "paths": [],
                            "rows": 0,
                            "task": tasks.get(row.get("task_index"), ""),
                        },
                    )
                    relative = str(path.relative_to(root))
                    if relative not in value["paths"]:
                        value["paths"].append(relative)
                    value["rows"] += 1
        self.fps = info["fps"]
        self.cameras = sum(
            feature["dtype"] in {"image", "video"} for feature in info.get("features", {}).values()
        )
        self.info_checksum = file_checksum(root / "meta/info.json")
        info["features"] = {
            key: value
            for key, value in info.get("features", {}).items()
            if key not in {LANGUAGE_PERSISTENT, LANGUAGE_EVENTS}
        }
        info.pop("tools", None)
        manifest.append({"info": info})
        for directory in ("videos", "meta/episodes"):
            for path in sorted((root / directory).rglob("*")):
                if path.is_file():
                    manifest.append({"path": str(path.relative_to(root)), "checksum": file_checksum(path)})
        manifest.append({"tasks": tasks})
        self.dataset_ref = DatasetRef(str(root.resolve()), fingerprint(manifest))

    def discover(self, stage, store, upstream):
        bindings = {}
        for name, (plan, _) in upstream.items():
            bindings[name] = {
                item.key: {artifact.name: asdict(artifact) for artifact in result.artifacts}
                for item, result in accepted_items(store, plan)
            }
        if stage.id == "materialize":
            by_file = defaultdict(dict)
            for ep, payload in self.episodes.items():
                owner_outputs = {}
                for owner in ("plan", "interjections", "vqa"):
                    stage_name = "plan_update" if owner == "plan" and "plan_update" in bindings else owner
                    if stage_name in bindings and "atoms" in bindings[stage_name][str(ep)]:
                        owner_outputs[owner] = bindings[stage_name][str(ep)]["atoms"]
                for path in payload["paths"]:
                    by_file[path][str(ep)] = owner_outputs
            # info.json declares one dataset-wide schema. Unselected files must
            # also contain typed empty language columns before that schema is
            # published: datasets cannot synthesize nested Arrow JSON nulls for
            # a missing column. This adds no labels or inference to those files.
            for path in sorted((self.root / "data").rglob("*.parquet")):
                if {LANGUAGE_PERSISTENT, LANGUAGE_EVENTS} - set(pq.read_schema(path).names):
                    by_file.setdefault(str(path.relative_to(self.root)), {})
            for path, episodes in sorted(by_file.items()):
                ownership_path = self.root / _ownership_path(path)
                payload = {
                    "path": path,
                    "episodes": episodes,
                    "source_checksum": file_checksum(self.root / path),
                    "ownership_checksum": file_checksum(ownership_path) if ownership_path.exists() else None,
                }
                yield InputItem(
                    path,
                    payload,
                    sum(self.episodes[int(ep)]["rows"] for ep in episodes),
                    identity_payload=artifact_identities(payload),
                )
        else:
            for ep, payload in sorted(self.episodes.items()):
                parent = {
                    name: outputs[str(ep)]["atoms"]
                    for name, outputs in bindings.items()
                    if "atoms" in outputs[str(ep)]
                }
                quality = None
                if "quality" in bindings:
                    artifact = Artifact(**bindings["quality"][str(ep)]["quality"])
                    if not store.verify(artifact):
                        raise ValueError("Quality artifact checksum mismatch")
                    with store.open(artifact.path) as stream:
                        quality = pq.read_table(stream).to_pylist()[0]
                mask = (
                    quality["reason"]
                    if quality and not quality["usable"] and stage.id not in {"validate", "materialize"}
                    else None
                )
                metadata = {**payload, "upstream": parent, **({"quality": quality} if quality else {})}
                yield InputItem(
                    str(ep),
                    metadata,
                    payload["rows"],
                    identity_payload=artifact_identities(metadata),
                    physical_seconds=payload["rows"] / self.fps,
                    camera_seconds=payload["rows"] / self.fps * self.cameras,
                    mask_reason=mask,
                )


def _ownership_path(relative):
    return "meta/annotations/ownership/" + relative.removeprefix("data/")

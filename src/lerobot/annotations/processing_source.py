# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Native episode discovery and lineage for annotation recipes."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lerobot.annotations.runtime.artifacts import file_checksum, local_file_token
from lerobot.annotations.runtime.types import (
    DatasetRef,
    InputItem,
    artifact_identities,
    fingerprint,
)
from lerobot.annotations.runtime.worker import accepted_items
from lerobot.datasets.utils import DEFAULT_VIDEO_PATH
from lerobot.utils.constants import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT

from .steerable_pipeline.reader import EpisodeRecord, RejectedEpisodeError, _load_tasks_lookup
from .storage import atomic_json


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
        raise RejectedEpisodeError(
            "invalid_episode_frames", "Incomplete, duplicated or noncontiguous episode frames"
        )
    timestamps = table["timestamp"].to_pylist()
    if (
        not timestamps
        or any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in timestamps)
        or any(b <= a for a, b in zip(timestamps, timestamps[1:], strict=False))
    ):
        raise RejectedEpisodeError("invalid_episode_clock", "Invalid episode clock")
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

    def __init__(
        self, root: Path, only_episodes=None, *, verify_sources=False, cache_dir: Path | None = None
    ):
        self.root = root
        info = _load_v3_info(root)
        self.episodes: dict[int, dict[str, Any]] = {}
        tasks = _load_tasks_lookup(root)
        manifest: list[dict[str, Any]] = []
        selected = set(only_episodes) if only_episodes is not None else None
        cache_path = (
            cache_dir if cache_dir is not None else root / ".annotate_staging"
        ) / "input-fingerprints.json"
        try:
            cache = json.loads(cache_path.read_text()) if not verify_sources else {}
            if not isinstance(cache, dict):
                cache = {}
        except (OSError, ValueError):
            cache = {}
        refreshed = {}

        def cached(path, kind, compute):
            key = kind + ":" + str(path.relative_to(root))
            token = list(local_file_token(path))
            old = cache.get(key, {})
            value = (
                old["value"]
                if isinstance(old, dict) and "value" in old and old.get("token") == token
                else compute()
            )
            if list(local_file_token(path)) != token:
                raise RuntimeError(f"Source changed during fingerprinting: {path.relative_to(root)}")
            refreshed[key] = {"token": token, "value": value}
            return value

        def inspect_data(path):
            file = pq.ParquetFile(path)
            native_columns = [
                name for name in file.schema_arrow.names if name not in {LANGUAGE_PERSISTENT, LANGUAGE_EVENTS}
            ]
            episodes = {}
            episode_digests = {}
            # LeRobot episodes are contiguous. Carry only their final partial
            # 1024-row block across reads, not a whole long episode.
            current = None
            pending = None

            def hash_rows(ep, rows):
                # One fixed-size block, not an IPC stream per reader batch.
                # Combine chunks so physical Parquet/Arrow splits are invisible.
                rows = rows.combine_chunks()
                buffer = pa.BufferOutputStream()
                with pa.ipc.new_stream(buffer, rows.schema) as writer:
                    writer.write_table(rows)
                episode_digests[ep].update(buffer.getvalue())

            for batch in file.iter_batches(batch_size=1024, columns=native_columns):
                table = pa.Table.from_batches([batch]).replace_schema_metadata(None)
                for ep in pc.unique(table["episode_index"]).to_pylist():
                    native = table.filter(pc.equal(table["episode_index"], ep))
                    ep = str(ep)
                    if ep != current:
                        if pending is not None:
                            hash_rows(current, pending)
                        current, pending = ep, None
                    value = episodes.setdefault(
                        ep,
                        {
                            "rows": 0,
                            "task_index": native["task_index"][0].as_py()
                            if "task_index" in native.column_names
                            else None,
                        },
                    )
                    value["rows"] += native.num_rows
                    episode_digests.setdefault(ep, hashlib.sha256())
                    pending = pa.concat_tables([pending, native]) if pending is not None else native
                    full_rows = pending.num_rows // 1024 * 1024
                    for start in range(0, full_rows, 1024):
                        hash_rows(ep, pending.slice(start, 1024))
                    pending = pending.slice(full_rows) if full_rows < pending.num_rows else None
            if pending is not None:
                hash_rows(current, pending)
            for ep, value in episodes.items():
                value["digest"] = episode_digests[ep].hexdigest()
            # File provenance uses the same canonical episode content, not
            # reader batch framing that changes when materialization repacks it.
            return {"native": fingerprint(episodes), "episodes": episodes}

        # Explicit frame-file pattern excludes sparse annotation Parquet tables.
        for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
            data = cached(path, "native-v3", lambda path=path: inspect_data(path))
            manifest.append({"path": str(path.relative_to(root)), "native": data["native"]})
            for ep_text, row in data["episodes"].items():
                ep = int(ep_text)
                if selected is not None and ep not in selected:
                    continue
                value = self.episodes.setdefault(
                    ep,
                    {
                        "episode_index": ep,
                        "paths": [],
                        "rows": 0,
                        "native_parts": [],
                        "task": tasks.get(row.get("task_index"), ""),
                    },
                )
                relative = str(path.relative_to(root))
                if relative not in value["paths"]:
                    value["paths"].append(relative)
                value["rows"] += row["rows"]
                value["native_parts"].append(row["digest"])
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
        videos = {}
        for directory in ("videos", "meta/episodes"):
            for path in sorted((root / directory).rglob("*")):
                if path.is_file():
                    digest = cached(path, "sha256", lambda path=path: file_checksum(path))
                    manifest.append({"path": str(path.relative_to(root)), "checksum": digest})
                    if directory == "videos":
                        videos[str(path.relative_to(root))] = digest
        manifest.append({"tasks": tasks})
        self.settings = {
            "fps": self.fps,
            "features": [
                (key, {name: value.get(name) for name in ("dtype", "shape", "names")})
                for key, value in sorted(info["features"].items())
            ],
        }
        self.video_bindings = defaultdict(list)
        video_keys = [key for key, value in info["features"].items() if value["dtype"] == "video"]
        if video_keys:
            for path in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")):
                for batch in pq.ParquetFile(path).iter_batches():
                    for row in batch.to_pylist():
                        ep = row["episode_index"]
                        if ep not in self.episodes:
                            continue
                        for key in video_keys:
                            prefix = f"videos/{key}/"
                            relative = info.get("video_path", DEFAULT_VIDEO_PATH).format(
                                video_key=key,
                                chunk_index=row[prefix + "chunk_index"],
                                file_index=row[prefix + "file_index"],
                            )
                            self.video_bindings[ep].append(
                                {
                                    "camera": key,
                                    "checksum": videos[relative],
                                    "from_timestamp": row[prefix + "from_timestamp"],
                                    "to_timestamp": row.get(prefix + "to_timestamp"),
                                }
                            )
        self.dataset_ref = DatasetRef("lerobot-native", fingerprint(manifest))
        atomic_json(cache_path, refreshed)

    def discover(self, stage, store, upstream):
        bindings = {}
        outcomes = defaultdict(dict)
        for name, (plan, _) in upstream.items():
            bindings[name] = {}
            for item, result in accepted_items(store, plan):
                bindings[name][item.key] = {artifact.name: asdict(artifact) for artifact in result.artifacts}
                if result.reason:
                    outcomes[item.key][name] = {"outcome": result.outcome.value, "reason": result.reason}
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
            for path in self.materialization_paths():
                by_file.setdefault(path, {})
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
                    content_digest=fingerprint(artifact_identities(payload)),
                )
        else:
            for ep, payload in sorted(self.episodes.items()):
                parent = {
                    name: outputs[str(ep)]["atoms"]
                    for name, outputs in bindings.items()
                    if "atoms" in outputs[str(ep)]
                }
                unavailable = outcomes[str(ep)]
                metadata = {**payload, "upstream": parent, "unavailable": unavailable}
                identity = {
                    "native_parts": payload["native_parts"],
                    "task": payload["task"],
                    "settings": self.settings,
                    "videos": self.video_bindings[ep],
                    "upstream": artifact_identities(parent),
                    "unavailable": unavailable,
                }
                if stage.id == "human_video" and "plan" not in bindings:
                    # Native identities deliberately omit language columns. This
                    # stage also consumes publisher subtasks when plan is off.
                    record = _load_record(self.root, metadata)
                    frames = record.frames_df()
                    stored = (
                        frames["language_persistent"].iloc[0]
                        if "language_persistent" in frames.columns and len(frames)
                        else []
                    )
                    identity["existing_subtasks"] = [
                        {key: atom.get(key) for key in ("content", "timestamp")}
                        for atom in (list(stored) if stored is not None else [])
                        if atom.get("style") == "subtask"
                    ]
                yield InputItem(
                    str(ep),
                    metadata,
                    payload["rows"],
                    identity_payload=identity,
                    content_digest=fingerprint(identity),
                    physical_seconds=payload["rows"] / self.fps,
                    camera_seconds=payload["rows"] / self.fps * self.cameras,
                    mask_reason=("upstream_unavailable:" + ",".join(sorted(unavailable)))
                    if unavailable and stage.id not in {"validate", "materialize"}
                    else None,
                )

    def materialization_paths(self):
        """All shards that selection/schema normalization can rewrite, known before inference."""
        paths = {path for episode in self.episodes.values() for path in episode["paths"]}
        for path in sorted((self.root / "data").glob("chunk-*/file-*.parquet")):
            if {LANGUAGE_PERSISTENT, LANGUAGE_EVENTS} - set(pq.read_schema(path).names):
                paths.add(str(path.relative_to(self.root)))
        return sorted(paths)

    def publication_paths(self, *, plan_enabled):
        """Conservative output names, including ownership and consolidated window metadata."""
        paths = {"meta/info.json"}
        for path in self.materialization_paths():
            paths.update((path, _ownership_path(path)))
        if plan_enabled:
            paths.update(_windows_path(episode["paths"][0]) for episode in self.episodes.values())
        return paths


def _ownership_path(relative):
    return "meta/annotations/ownership/" + relative.removeprefix("data/")


def _windows_path(relative):
    return "meta/annotations/windows/" + relative.removeprefix("data/")

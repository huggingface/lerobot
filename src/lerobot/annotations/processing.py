# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Language-domain recipes and modules over LeRobot's shared offline runtime."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lerobot.data_processing.artifacts import file_checksum
from lerobot.data_processing.configs import StageConfig
from lerobot.data_processing.pipeline import run_pipeline
from lerobot.data_processing.sinks.local_commit import commit_local_files, recover_local_commit
from lerobot.data_processing.types import (
    Artifact,
    DatasetRef,
    InputItem,
    ItemResult,
    ModuleSpec,
    Outcome,
    artifact_identities,
    canonical_json,
    fingerprint,
)
from lerobot.data_processing.worker import accepted_in_shard
from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
from lerobot.datasets.language import (
    language_array,
    language_events_arrow_type,
    language_feature_info,
    language_persistent_arrow_type,
)
from lerobot.utils.constants import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT

from .steerable_pipeline.config import AnnotationPipelineConfig
from .steerable_pipeline.executor import PhaseResult, PipelineRunSummary
from .steerable_pipeline.frames import make_frame_provider
from .steerable_pipeline.modules import (
    GeneralVqaModule,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from .steerable_pipeline.modules.plan_subtasks_memory import subtask_windows
from .steerable_pipeline.reader import EpisodeRecord, _load_tasks_lookup
from .steerable_pipeline.staging import EpisodeStaging
from .steerable_pipeline.validator import StagingValidator, ValidationReport
from .steerable_pipeline.vlm_client import make_vlm_client
from .steerable_pipeline.writer import (
    _normalize_event_row,
    _normalize_persistent_row,
    _validate_atom_invariants,
    _validate_speech_atom,
)

ATOM_SCHEMA = pa.schema(
    [
        ("role", pa.string()),
        ("content", pa.string()),
        ("style", pa.string()),
        ("timestamp", pa.float64()),
        ("camera", pa.string()),
        ("tool_calls", pa.list_(pa.string())),
    ]
)
OWNERSHIP_SCHEMA = pa.schema(
    [
        ("episode_index", pa.int64()),
        ("frame_index", pa.int64()),
        ("column", pa.string()),
        ("atom_hash", pa.string()),
        ("owner", pa.string()),
        ("count", pa.int64()),
    ]
)
WINDOW_SCHEMA = pa.schema(
    [
        ("window_index", pa.int64()),
        ("context_start", pa.int64()),
        ("context_end", pa.int64()),
        ("output_start", pa.int64()),
        ("output_end", pa.int64()),
        ("episode_index", pa.int64()),
        ("frame_index", pa.int64()),
        ("timestamp", pa.float64()),
        ("owned", pa.bool_()),
    ]
)


def _read_atoms(store, artifact):
    artifact = Artifact(**artifact)
    if not store.verify(artifact):
        raise ValueError("Upstream annotation checksum mismatch")
    with store.open(artifact.path) as stream:
        rows = pq.read_table(stream).to_pylist()
    for row in rows:
        row["tool_calls"] = (
            [json.loads(call) for call in row["tool_calls"]] if row["tool_calls"] is not None else None
        )
    return rows


def _staged_table(rows):
    normalized = []
    for row in rows:
        _validate_atom_invariants(row)
        _validate_speech_atom(row)
        normalized.append(
            {
                **{name: row.get(name) for name in ATOM_SCHEMA.names},
                "tool_calls": [canonical_json(call).decode() for call in row["tool_calls"]]
                if row.get("tool_calls") is not None
                else None,
            }
        )
    return pa.Table.from_pylist(normalized, schema=ATOM_SCHEMA)


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
        info = json.loads((root / "meta/info.json").read_text())
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
            values = {}
            for shard in range(plan.shards):
                results = accepted_in_shard(store, plan, shard)
                for item in plan.read_shard(store, shard):
                    values[item.key] = {
                        artifact.name: asdict(artifact) for artifact in results[item.item_id].artifacts
                    }
            bindings[name] = values
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
                yield InputItem(
                    path,
                    {
                        "path": path,
                        "episodes": episodes,
                        "source_checksum": file_checksum(self.root / path),
                        "ownership_checksum": file_checksum(ownership_path)
                        if ownership_path.exists()
                        else None,
                    },
                    sum(self.episodes[int(ep)]["rows"] for ep in episodes),
                    identity_payload={
                        "path": path,
                        "episodes": artifact_identities(episodes),
                        "source_checksum": file_checksum(self.root / path),
                        "ownership_checksum": file_checksum(ownership_path)
                        if ownership_path.exists()
                        else None,
                    },
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


class EpisodeQuality:
    """Cheap sampled gate before language inference, with an auditable decision."""

    def __init__(self, root, quality_config, camera_key=None, video_backend=None):
        from .steerable_pipeline.config import QualityConfig

        self.root = Path(root)
        self.config = QualityConfig(**quality_config)
        self.camera_key, self.video_backend = camera_key, video_backend
        self.schema = pa.schema(
            [
                ("episode_index", pa.int64()),
                ("usable", pa.bool_()),
                ("reason", pa.string()),
                ("sampled_frames", pa.int64()),
                ("black_frames", pa.int64()),
                ("frame_indices", pa.list_(pa.int64())),
                ("timestamps", pa.list_(pa.float64())),
            ]
        )
        self.spec = ModuleSpec("episode_quality", "1", "episode", {"quality": self.schema})

    def setup(self, context):
        self.provider = make_frame_provider(
            self.root, camera_key=self.camera_key, video_backend=self.video_backend
        )

    def teardown(self):
        pass

    def process_batch(self, items, context):
        results = []
        for item in items:
            record = _load_record(self.root, item.payload)
            indices = sorted(
                {
                    round(i * (record.row_count - 1) / max(1, self.config.sample_frames - 1))
                    for i in range(self.config.sample_frames)
                }
            )
            timestamps = [record.frame_timestamps[i] for i in indices]
            frames = self.provider.frames_at(record, timestamps)
            black = sum(float(frame.float().mean()) <= self.config.black_threshold for frame in frames)
            usable = (
                len(frames) == len(indices) and black / max(1, len(frames)) <= self.config.max_black_fraction
            )
            reason = (
                None
                if usable
                else ("missing_camera_frames" if len(frames) != len(indices) else "black_frame_fraction")
            )
            row = {
                "episode_index": record.episode_index,
                "usable": usable,
                "reason": reason,
                "sampled_frames": len(frames),
                "black_frames": black,
                "frame_indices": [record.frame_indices[i] for i in indices],
                "timestamps": timestamps,
            }
            artifact = context.write_parquet(item, "quality", pa.Table.from_pylist([row], schema=self.schema))
            results.append(ItemResult(item.item_id, Outcome.COMPLETED, (artifact,)))
        return results


class LanguageModule:
    """An episode language stage; shared execution knows nothing about its prompts."""

    def __init__(self, root, phase, annotation_config, client_factory=None):
        self.root = Path(root)
        self.phase = phase
        import draccus

        self.config = draccus.decode(AnnotationPipelineConfig, annotation_config)
        self.client_factory = client_factory
        outputs = {"atoms": ATOM_SCHEMA, **({"windows": WINDOW_SCHEMA} if phase == "plan" else {})}
        self.spec = ModuleSpec(f"language_{phase}", "3", "episode", outputs)

    def setup(self, context):
        cfg = self.config
        if cfg.vlm.api_key_env:
            cfg.vlm.api_key = os.environ[cfg.vlm.api_key_env]
        if os.environ.get("LEROBOT_ANNOTATION_ENDPOINTS"):
            cfg.vlm.api_bases = tuple(json.loads(os.environ["LEROBOT_ANNOTATION_ENDPOINTS"]))
        if self.client_factory:
            module, name = self.client_factory.split(":", 1)
            vlm = getattr(importlib.import_module(module), name)()
        else:
            vlm = make_vlm_client(replace(cfg.vlm, auto_serve=False))
        self.client = vlm
        provider = make_frame_provider(
            self.root, camera_key=cfg.vlm.camera_key, video_backend=cfg.video_backend
        )
        if self.phase in {"plan", "plan_update"}:
            self.module = PlanSubtasksMemoryModule(vlm=vlm, config=cfg.plan, frame_provider=provider)
        elif self.phase == "interjections":
            self.module = InterjectionsAndSpeechModule(
                vlm=vlm, config=cfg.interjections, seed=cfg.seed, frame_provider=provider
            )
        elif self.phase == "vqa":
            self.module = GeneralVqaModule(vlm=vlm, config=cfg.vqa, seed=cfg.seed, frame_provider=provider)
        else:
            raise ValueError("Unknown language phase")

    def scientific_config(self):
        import draccus

        values = draccus.encode(self.config)
        shared = {key: values[key] for key in ("vlm", "seed", "video_backend")}
        shared["vlm"].pop("client_concurrency", None)
        for key in (
            "endpoint_limit_url",
            "endpoint_limit_key",
            "endpoint_limit_token_env",
            "endpoint_limit_timeout_s",
        ):
            shared["vlm"].pop(key, None)
        names = ("plan",) if self.phase in {"plan", "plan_update"} else (self.phase,)
        return {
            "root": str(self.root),
            "phase": self.phase,
            "client_factory": self.client_factory,
            "annotation_config": {**shared, **{name: values[name] for name in names}},
        }

    def teardown(self):
        close = getattr(self.client, "close", None) if hasattr(self, "client") else None
        if close:
            close()

    def process_batch(self, items, context):
        def process(item):
            record = _load_record(self.root, item.payload)
            staging = EpisodeStaging(context.scratch / item.item_id, record.episode_index)
            for name, artifact in item.payload["upstream"].items():
                staging.write(name, _read_atoms(context.store, artifact))
            if self.phase == "plan_update":
                rows = staging.read("interjections")
                rows = [row for row in rows if row.get("style") == "interjection"]
                if rows:
                    self.module.run_plan_updates(
                        record,
                        staging,
                        [row["timestamp"] for row in rows],
                        [row.get("content") or "" for row in rows],
                    )
                output_name = "plan"
            else:
                self.module.run_episode(record, staging)
                output_name = self.phase
            # Full cross-family validation happens in materialization, once all
            # enabled stages are present. These artifacts retain exact source clocks.
            output = staging.read(output_name)
            if self.phase == "plan" and not any(row.get("style") == "subtask" for row in output):
                raise ValueError(
                    "Subtask annotation returned no subtasks; refusing a successful empty result"
                )
            artifact = context.write_parquet(item, "atoms", _staged_table(output))
            artifacts = [artifact]
            if self.phase == "plan":
                windows = subtask_windows(record, self.config.plan)
                rows = [
                    {
                        "window_index": window.index,
                        "context_start": window.context_start,
                        "context_end": window.context_end,
                        "output_start": window.output_start,
                        "output_end": window.output_end,
                        "episode_index": frame.episode_index,
                        "frame_index": frame.frame_index,
                        "timestamp": frame.timestamp,
                        "owned": window.output_start <= window.context_start + offset < window.output_end,
                    }
                    for window in windows
                    for offset, frame in enumerate(window.frames)
                ]
                artifacts.append(
                    context.write_parquet(item, "windows", pa.Table.from_pylist(rows, schema=WINDOW_SCHEMA))
                )
            if staging.root.exists():
                shutil.rmtree(staging.root)
            return ItemResult(item.item_id, Outcome.COMPLETED, tuple(artifacts))

        with ThreadPoolExecutor(
            max_workers=min(self.config.executor.episode_parallelism, len(items))
        ) as pool:
            return list(pool.map(process, items))


def _canonical_atom(row, persistent):
    fields = (
        ("role", "content", "style", "timestamp", "camera", "tool_calls")
        if persistent
        else ("role", "content", "style", "camera", "tool_calls")
    )
    if set(row) - set(fields):
        raise ValueError("Source language atom contains unsupported extra fields")
    atom = {name: row.get(name) for name in fields}
    if persistent:
        atom["timestamp"] = pa.scalar(atom["timestamp"], type=pa.float32()).as_py()
    calls = atom["tool_calls"]
    atom["tool_calls"] = (
        [
            canonical_json(call).decode()
            if not isinstance(call, str)
            else canonical_json(json.loads(call)).decode()
            for call in calls
        ]
        if calls is not None
        else None
    )
    return atom


def _language_array(values, dtype):
    # Arrow cannot construct nested extension arrays directly. Construct JSON
    # storage strings, then restore the canonical extension type with a cast.
    return language_array(values, dtype)


def _ownership_path(relative):
    return "meta/annotations/ownership/" + relative.removeprefix("data/")


class LanguageValidator:
    """Validate complete episodes after all enabled families have produced artifacts."""

    def __init__(self, root, annotation_config):
        self.root = Path(root)
        self.skip_validation = annotation_config.get("skip_validation", False)
        self.schema = pa.schema([("episode_index", pa.int64()), ("warnings", pa.list_(pa.string()))])
        self.spec = ModuleSpec("language_validate", "1", "episode", {"validation": self.schema})

    def setup(self, context):
        pass

    def teardown(self):
        pass

    def process_batch(self, items, context):
        results = []
        for item in items:
            record = _load_record(self.root, item.payload)
            staging = EpisodeStaging(context.scratch / item.item_id, record.episode_index)
            for name, artifact in item.payload["upstream"].items():
                staging.write("plan" if name == "plan_update" else name, _read_atoms(context.store, artifact))
            report = StagingValidator().validate([record], staging.root)
            if not report.ok and not self.skip_validation:
                raise ValueError(report.summary() + ": " + "; ".join(report.errors))
            table = pa.Table.from_pylist(
                [{"episode_index": record.episode_index, "warnings": report.warnings + report.errors}],
                schema=self.schema,
            )
            artifact = context.write_parquet(item, "validation", table)
            if staging.root.exists():
                shutil.rmtree(staging.root)
            results.append(ItemResult(item.item_id, Outcome.COMPLETED, (artifact,)))
        return results


class LanguageMaterializer:
    """One physical output file per item; preserve unknown/source-owned atoms."""

    def __init__(self, root, annotation_config):
        self.root = Path(root)
        self.annotation_config = annotation_config
        self.spec = ModuleSpec(
            "language_materialize", "2", "asset", {"data": None, "ownership": OWNERSHIP_SCHEMA}
        )

    def setup(self, context):
        pass

    def teardown(self):
        pass

    def process_batch(self, items, context):
        results = []
        for item in items:
            path = self.root / item.payload["path"]
            if file_checksum(path) != tuple(item.payload["source_checksum"]):
                raise ValueError("Source changed after materialization planning")
            table = pq.read_table(path)
            rows = table.select(["episode_index", "frame_index", "timestamp"]).to_pylist()
            owner_path = self.root / _ownership_path(item.payload["path"])
            checksum = file_checksum(owner_path) if owner_path.exists() else None
            expected = item.payload["ownership_checksum"]
            if checksum != (tuple(expected) if expected is not None else None):
                raise ValueError("Annotation ownership changed after planning")
            old = pq.read_table(owner_path).to_pylist() if owner_path.exists() else []
            additions = defaultdict(list)
            active = {}
            for ep_text, owners in item.payload["episodes"].items():
                ep = int(ep_text)
                active[ep] = set(owners)
                # Validate complete episodes at generation time; file materialization
                # can see just one part of an episode split across physical files.
                staging = EpisodeStaging(context.scratch / item.item_id, ep)
                raw_atoms = []
                for owner, artifact in owners.items():
                    atoms = _read_atoms(context.store, artifact)
                    staging.write(owner, atoms)
                    raw_atoms.extend(atoms)
                    for atom in atoms:
                        persistent = atom.get("style") in {"subtask", "plan", "memory", "motion", "task_aug"}
                        normalized = (
                            _normalize_persistent_row(atom) if persistent else _normalize_event_row(atom)
                        )
                        normalized = _canonical_atom(normalized, persistent)
                        timestamp = None if persistent else float(atom["timestamp"])
                        additions[
                            (ep, LANGUAGE_PERSISTENT if persistent else LANGUAGE_EVENTS, timestamp)
                        ].append((owner, normalized))
                frame_times = {float(row["timestamp"]) for row in rows if row["episode_index"] == ep}
                # Atom routing/invariants have been checked. Reject orphan event
                # timestamps only when this file contains the complete episode;
                # partial episode files receive their own matching event subset.
                if raw_atoms and not frame_times:
                    raise ValueError("Annotations reference a missing episode")
                if staging.root.exists():
                    shutil.rmtree(staging.root)
            removal = defaultdict(Counter)
            kept = []
            for ownership in old:
                if ownership["owner"] in active.get(ownership["episode_index"], set()):
                    removal[(ownership["episode_index"], ownership["frame_index"], ownership["column"])][
                        ownership["atom_hash"]
                    ] += ownership["count"]
                else:
                    kept.append(ownership)
            generated = {}
            for column, persistent in ((LANGUAGE_PERSISTENT, True), (LANGUAGE_EVENTS, False)):
                old_values = table[column].to_pylist() if column in table.column_names else [[] for _ in rows]
                values = []
                for row, atoms in zip(rows, old_values, strict=True):
                    ep, frame = row["episode_index"], -1 if persistent else row["frame_index"]
                    counts = removal[(ep, frame, column)].copy()
                    surviving = []
                    for atom in reversed(atoms or []):
                        normalized = _canonical_atom(atom, persistent)
                        digest = fingerprint(normalized)
                        if counts[digest]:
                            counts[digest] -= 1
                        else:
                            surviving.append(normalized)
                    surviving.reverse()
                    new = additions[(ep, column, None if persistent else float(row["timestamp"]))]
                    surviving.extend(atom for _, atom in new)
                    values.append(surviving)
                    grouped = Counter((owner, fingerprint(atom)) for owner, atom in new)
                    for (owner, digest), count in grouped.items():
                        generated[(ep, frame, column, owner, digest)] = count
                dtype = language_persistent_arrow_type() if persistent else language_events_arrow_type()
                array = _language_array(values, dtype)
                if column in table.column_names:
                    table = table.set_column(table.column_names.index(column), column, array)
                else:
                    table = table.append_column(column, array)
            new_ownership = kept + [
                {
                    "episode_index": ep,
                    "frame_index": frame,
                    "column": column,
                    "owner": owner,
                    "atom_hash": digest,
                    "count": count,
                }
                for (ep, frame, column, owner, digest), count in sorted(generated.items())
            ]
            output = context.scratch / f"{item.item_id}.parquet"
            write_table_one_row_group_per_episode(table, output)
            if not pq.read_table(output).equals(table):
                raise ValueError("Language output failed round-trip validation")
            data = context.write_asset(item, "data", output)
            output.unlink()
            ownership = context.write_parquet(
                item, "ownership", pa.Table.from_pylist(new_ownership, schema=OWNERSHIP_SCHEMA)
            )
            results.append(ItemResult(item.item_id, Outcome.COMPLETED, (data, ownership)))
        return results


def run_annotation_pipeline(cfg: AnnotationPipelineConfig, root: Path, *, client_factory=None):
    """Build a language recipe; current flags remain compatible, execution is shared."""
    root = root.resolve()
    if (
        cfg.runtime.backend == "slurm"
        and cfg.vlm.auto_serve
        and cfg.runtime.mode != "plan"
        and not client_factory
    ):
        raise ValueError(
            "Slurm language clients require reachable external endpoints and vlm.auto_serve=false"
        )
    recover_local_commit(root)
    enabled = [name for name in ("plan", "interjections", "vqa") if getattr(cfg, name).enabled]
    if not enabled:
        return PipelineRunSummary([], [], ValidationReport())
    source = LeRobotEpisodeSource(root, cfg.only_episodes)
    if not source.episodes:
        raise ValueError("No episodes selected")
    # Credentials are environment references, never written to a work plan.
    if cfg.vlm.api_key not in {"", "EMPTY"} and not cfg.vlm.api_key_env:
        raise ValueError("Processing requires vlm.api_key_env rather than persisting an API key")
    import draccus

    config = {
        key: draccus.encode(getattr(cfg, key))
        for key in (
            "vlm",
            "plan",
            "interjections",
            "vqa",
            "seed",
            "video_backend",
            "executor",
            "skip_validation",
        )
    }
    config["vlm"]["api_key"] = "EMPTY"

    def phase_config(name):
        shared = {key: value for key, value in config.items() if key not in {"plan", "interjections", "vqa"}}
        family = "plan" if name == "plan_update" else name
        return {**shared, family: config[family]}

    stages = []
    if cfg.quality.enabled:
        stages.append(
            StageConfig(
                "quality",
                "lerobot.annotations.processing:EpisodeQuality",
                {
                    "root": str(root),
                    "quality_config": draccus.encode(cfg.quality),
                    "camera_key": cfg.vlm.camera_key,
                    "video_backend": cfg.video_backend,
                },
            )
        )
    for name in enabled:
        dependencies = (("quality",) if cfg.quality.enabled else ()) + (
            ("plan",) if name == "interjections" and "plan" in enabled else ()
        )
        stages.append(
            StageConfig(
                name,
                "lerobot.annotations.processing:LanguageModule",
                {
                    "root": str(root),
                    "phase": name,
                    "annotation_config": phase_config(name),
                    "client_factory": client_factory,
                },
                dependencies,
                when="quality.usable" if cfg.quality.enabled else None,
            )
        )
    if "plan" in enabled and "interjections" in enabled:
        stages.append(
            StageConfig(
                "plan_update",
                "lerobot.annotations.processing:LanguageModule",
                {
                    "root": str(root),
                    "phase": "plan_update",
                    "annotation_config": phase_config("plan_update"),
                    "client_factory": client_factory,
                },
                ("plan", "interjections") + (("quality",) if cfg.quality.enabled else ()),
                when="quality.usable" if cfg.quality.enabled else None,
            )
        )
    stages.append(
        StageConfig(
            "validate",
            "lerobot.annotations.processing:LanguageValidator",
            {"root": str(root), "annotation_config": config},
            tuple(stage.id for stage in stages),
        )
    )
    stages.append(
        StageConfig(
            "materialize",
            "lerobot.annotations.processing:LanguageMaterializer",
            {"root": str(root), "annotation_config": config},
            tuple(stage.id for stage in stages),
        )
    )
    runtime = replace(
        cfg.runtime, run_uri=cfg.runtime.run_uri or str(cfg.resolved_staging_dir(root) / "processing")
    )
    service: list[Any] = []
    previous_endpoints = os.environ.get("LEROBOT_ANNOTATION_ENDPOINTS")

    def before_execute(store, plan):
        if not cfg.vlm.auto_serve or client_factory or not plan.factory.endswith("LanguageModule") or service:
            return

        def needs_inference(shard):
            accepted = accepted_in_shard(store, plan, shard)
            return any(
                item.mask_reason is None and item.item_id not in accepted
                for item in plan.read_shard(store, shard)
            )

        if any(needs_inference(shard) for shard in range(plan.shards)):
            client = make_vlm_client(cfg.vlm)
            service.append(client)
            os.environ["LEROBOT_ANNOTATION_ENDPOINTS"] = json.dumps(client.api_bases)

    try:
        store, completed = run_pipeline(source, stages, runtime, before_execute=before_execute)
    finally:
        for client in service:
            close = getattr(client, "close", None)
            if close:
                close()
        if previous_endpoints is None:
            os.environ.pop("LEROBOT_ANNOTATION_ENDPOINTS", None)
        else:
            os.environ["LEROBOT_ANNOTATION_ENDPOINTS"] = previous_endpoints
    if runtime.mode == "plan":
        return PipelineRunSummary([], [], ValidationReport())
    plan, _ = completed["materialize"]
    files = {}
    expected: dict[str, tuple[str, int] | None] = {"meta/info.json": source.info_checksum}
    with tempfile.TemporaryDirectory(prefix="lerobot-language-release-") as temporary_dir:
        directory = Path(temporary_dir)
        for shard in range(plan.shards):
            results = accepted_in_shard(store, plan, shard)
            for item in plan.read_shard(store, shard):
                expected[item.payload["path"]] = tuple(item.payload["source_checksum"])
                checksum = item.payload["ownership_checksum"]
                expected[_ownership_path(item.payload["path"])] = tuple(checksum) if checksum else None
                for artifact in results[item.item_id].artifacts:
                    relative = (
                        item.payload["path"]
                        if artifact.name == "data"
                        else _ownership_path(item.payload["path"])
                    )
                    target = directory / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with store.open(artifact.path) as stream, target.open("wb") as output:
                        shutil.copyfileobj(stream, output)
                    if file_checksum(target) != (artifact.sha256, artifact.size):
                        raise ValueError("Accepted release file was corrupted during download")
                    files[relative] = target
        info = json.loads((root / "meta/info.json").read_text())
        # Small provenance tables live beside the enriched data, not only in the
        # runtime cache. Keep native videos untouched and publication explicit.
        for stage_name, output_name in (("quality", "quality"), ("plan", "windows")):
            if stage_name not in completed:
                continue
            provenance_plan, _ = completed[stage_name]
            for shard in range(provenance_plan.shards):
                results = accepted_in_shard(store, provenance_plan, shard)
                for item in provenance_plan.read_shard(store, shard):
                    for artifact in results[item.item_id].artifacts:
                        if artifact.name != output_name:
                            continue
                        relative = f"meta/annotations/{output_name}/episode-{item.key}.parquet"
                        target = directory / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with store.open(artifact.path) as stream, target.open("wb") as output:
                            shutil.copyfileobj(stream, output)
                        if file_checksum(target) != (artifact.sha256, artifact.size):
                            raise ValueError("Annotation provenance checksum mismatch")
                        expected[relative] = (
                            file_checksum(root / relative) if (root / relative).exists() else None
                        )
                        files[relative] = target
        info["features"] = {**info.get("features", {}), **language_feature_info()}
        from lerobot.datasets.language import SAY_TOOL_SCHEMA

        tools = info.get("tools") or []
        if not any(tool.get("function", {}).get("name") == "say" for tool in tools):
            info["tools"] = [*tools, SAY_TOOL_SCHEMA]
        info_path = directory / "info.json"
        info_path.write_bytes(canonical_json(info))
        files["meta/info.json"] = info_path
        # Preparation lives outside the transient release directory so a crash
        # after publishing the journal can finish the commit on the next invocation.
        prepared = root / ".annotate_staging" / "prepared" / plan.plan_id
        commit_local_files(root, files, prepared, expected=expected)
    phases = [
        PhaseResult(name, summary.completed, summary.masked + summary.rejected)
        for name, (_, summary) in completed.items()
        if name != "materialize"
    ]
    validation = ValidationReport(episodes_checked=len(source.episodes))
    validation_plan, _ = completed["validate"]
    for shard in range(validation_plan.shards):
        for result in accepted_in_shard(store, validation_plan, shard).values():
            with store.open(result.artifacts[0].path) as stream:
                for row in pq.read_table(stream).to_pylist():
                    validation.warnings.extend(row["warnings"])
    return PipelineRunSummary(
        phases,
        [root / path for path in files if path.startswith("data/")],
        validation,
        metadata_paths=[root / path for path in files if path.startswith("meta/")],
    )

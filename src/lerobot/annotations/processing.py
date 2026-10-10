# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Language-domain processing modules; factories remain checkpoint-compatible."""

from __future__ import annotations

import importlib
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.data_processing.artifacts import file_checksum
from lerobot.data_processing.types import (
    Artifact,
    ItemResult,
    ModuleSpec,
    Outcome,
    canonical_json,
    fingerprint,
)
from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
from lerobot.datasets.language import (
    language_array,
    language_events_arrow_type,
    language_persistent_arrow_type,
)
from lerobot.utils.constants import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT

from .processing_recipe import run_annotation_pipeline as run_annotation_pipeline
from .processing_source import LeRobotEpisodeSource as LeRobotEpisodeSource, _load_record, _ownership_path
from .steerable_pipeline.config import AnnotationPipelineConfig
from .steerable_pipeline.frames import make_frame_provider
from .steerable_pipeline.modules import (
    GeneralVqaModule,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from .steerable_pipeline.modules.plan_subtasks_memory import subtask_windows
from .steerable_pipeline.staging import EpisodeStaging
from .steerable_pipeline.validator import StagingValidator
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
        self.spec = ModuleSpec(f"language_{phase}", "4" if phase == "plan" else "3", "episode", outputs)

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


_language_array = language_array  # Compatibility for callers of the original adapter.


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
            present_episodes = {row["episode_index"] for row in rows}
            for ep_text, owners in item.payload["episodes"].items():
                ep = int(ep_text)
                active[ep] = set(owners)
                for owner, artifact in owners.items():
                    atoms = _read_atoms(context.store, artifact)
                    if atoms and ep not in present_episodes:
                        raise ValueError("Annotations reference a missing episode")
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
                array = language_array(values, dtype)
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

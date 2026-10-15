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

from lerobot.annotations.runtime.artifacts import file_checksum
from lerobot.annotations.runtime.types import (
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
from .steerable_pipeline.prompts import _DIR as PROMPT_DIR, load as load_prompt
from .steerable_pipeline.reader import RejectedEpisodeError
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
        shared = {key: values[key] for key in ("seed", "video_backend")}
        shared["vlm"] = {
            key: values["vlm"][key]
            for key in (
                "backend",
                "model_id",
                "model_revision",
                "max_new_tokens",
                "temperature",
                "max_model_len",
                "chat_template_kwargs",
                "reasoning_effort",
                "camera_key",
            )
        }
        names = ("plan",) if self.phase in {"plan", "plan_update"} else (self.phase,)
        return {
            "phase": self.phase,
            "client_factory": self.client_factory,
            "annotation_config": {**shared, **{name: values[name] for name in names}},
            "prompts": fingerprint(
                {
                    path.stem: load_prompt(path.stem)
                    for path in sorted(PROMPT_DIR.glob("*.txt"))
                    if path.stem.startswith(names)
                }
            ),
        }

    def teardown(self):
        close = getattr(self.client, "close", None) if hasattr(self, "client") else None
        if close:
            close()

    def process_batch(self, items, context):
        def process(item):
            try:
                return annotate(item)
            except RejectedEpisodeError as exc:
                return ItemResult(item.item_id, Outcome.REJECTED, reason=exc.reason)
            finally:
                shutil.rmtree(context.scratch / item.item_id, ignore_errors=True)

        def annotate(item):
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
                return ItemResult(item.item_id, Outcome.REJECTED, reason="no_subtasks")
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
        [canonical_json(json.loads(call) if isinstance(call, str) else call).decode() for call in calls]
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

    def scientific_config(self):
        return {"skip_validation": self.skip_validation}

    def setup(self, context):
        pass

    def teardown(self):
        pass

    def process_batch(self, items, context):
        results = []
        for item in items:
            try:
                record = _load_record(self.root, item.payload)
            except RejectedEpisodeError as exc:
                table = pa.Table.from_pylist(
                    [
                        {
                            "episode_index": item.payload["episode_index"],
                            "warnings": [f"Episode {item.key}: rejected ({exc.reason})"],
                        }
                    ],
                    schema=self.schema,
                )
                artifact = context.write_parquet(item, "validation", table)
                results.append(ItemResult(item.item_id, Outcome.COMPLETED, (artifact,)))
                continue
            staging = EpisodeStaging(context.scratch / item.item_id, record.episode_index)
            for name, artifact in item.payload["upstream"].items():
                staging.write("plan" if name == "plan_update" else name, _read_atoms(context.store, artifact))
            report = StagingValidator().validate([record], staging.root)
            report.warnings.extend(
                f"Episode {record.episode_index}: {name} {value['outcome']} ({value['reason']})"
                for name, value in item.payload.get("unavailable", {}).items()
            )
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

    def scientific_config(self):
        return {}  # All materialization inputs are content-bound in the work items.

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

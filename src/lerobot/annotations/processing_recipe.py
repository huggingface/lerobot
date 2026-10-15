# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Language recipe orchestration and transactional dataset publication."""

from __future__ import annotations

import base64
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lerobot.annotations.runtime.artifacts import file_checksum
from lerobot.annotations.runtime.configs import StageConfig
from lerobot.annotations.runtime.pipeline import run_pipeline
from lerobot.annotations.runtime.types import Artifact, canonical_json
from lerobot.annotations.runtime.worker import accepted_in_shard, accepted_items
from lerobot.datasets.language import language_feature_info

from .processing_source import LeRobotEpisodeSource, _load_v3_info, _ownership_path, _windows_path
from .steerable_pipeline.config import AnnotationPipelineConfig
from .steerable_pipeline.executor import PhaseResult, PipelineRunSummary
from .steerable_pipeline.validator import ValidationReport
from .steerable_pipeline.vlm_client import make_vlm_client
from .storage import commit_local_files, recover_local_commit


def run_annotation_pipeline(cfg: AnnotationPipelineConfig, root: Path, *, client_factory=None):
    """Build a language recipe; current flags remain compatible, execution is shared."""
    root = root.resolve()
    _load_v3_info(root)
    if (
        cfg.runtime.backend == "slurm"
        and cfg.vlm.auto_serve
        and cfg.runtime.mode != "plan"
        and not client_factory
    ):
        raise ValueError(
            "Slurm language clients require reachable external endpoints and vlm.auto_serve=false"
        )
    if cfg.runtime.mode == "plan":
        if (root / ".processing_commit.json").exists():
            raise RuntimeError("Recover the pending dataset commit before planning")
    else:
        recover_local_commit(root)
    enabled = [name for name in ("plan", "interjections", "vqa") if getattr(cfg, name).enabled]
    if not enabled:
        return PipelineRunSummary([], [], ValidationReport())
    source = LeRobotEpisodeSource(
        root,
        cfg.only_episodes,
        verify_sources=cfg.runtime.verify_sources,
        cache_dir=cfg.resolved_staging_dir(root),
    )
    if not source.episodes:
        raise ValueError("No episodes selected")
    if cfg.push_to_hub and cfg.runtime.mode != "plan":
        from .hub import preflight_publication

        preflight_publication(
            root, cfg, planned_paths=source.publication_paths(plan_enabled=cfg.plan.enabled)
        )
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

    stages = []
    phase_dependencies: dict[str, tuple[str, ...]] = {
        name: ("plan",) if name == "interjections" and "plan" in enabled else () for name in enabled
    }
    if "plan" in enabled and "interjections" in enabled:
        phase_dependencies["plan_update"] = ("plan", "interjections")
    shared = {key: value for key, value in config.items() if key not in {"plan", "interjections", "vqa"}}
    for name, parents in phase_dependencies.items():
        family = "plan" if name == "plan_update" else name
        stages.append(
            StageConfig(
                name,
                "lerobot.annotations.processing:LanguageModule",
                {
                    "root": str(root),
                    "phase": name,
                    "annotation_config": {**shared, family: config[family]},
                    "client_factory": client_factory,
                },
                parents,
            )
        )
    for name, module in (("validate", "LanguageValidator"), ("materialize", "LanguageMaterializer")):
        stages.append(
            StageConfig(
                name,
                f"lerobot.annotations.processing:{module}",
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
        for item, result in accepted_items(store, plan):
            expected[item.payload["path"]] = tuple(item.payload["source_checksum"])
            checksum = item.payload["ownership_checksum"]
            expected[_ownership_path(item.payload["path"])] = tuple(checksum) if checksum else None
            for artifact in result.artifacts:
                relative = (
                    item.payload["path"] if artifact.name == "data" else _ownership_path(item.payload["path"])
                )
                files[relative] = store.download(artifact, directory / relative)
        info = json.loads((root / "meta/info.json").read_text())
        # Small provenance tables live beside the enriched data, not only in the
        # runtime cache. Keep native videos untouched and publication explicit.
        if "plan" in completed:
            provenance_plan, _ = completed["plan"]
            by_file: dict[str, dict[int, Artifact]] = defaultdict(dict)
            for item, result in accepted_items(store, provenance_plan):
                for artifact in result.artifacts:
                    if artifact.name == "windows":
                        relative = _windows_path(source.episodes[int(item.key)]["paths"][0])
                        by_file[relative][int(item.key)] = artifact
            for relative, updates in sorted(by_file.items()):
                previous = root / relative
                expected[relative] = file_checksum(previous) if previous.exists() else None
                destination = directory / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                schema = pa.ipc.read_schema(
                    pa.BufferReader(base64.b64decode(provenance_plan.outputs["windows"]))
                )
                with pq.ParquetWriter(destination, schema) as writer:
                    if previous.exists():
                        for batch in pq.ParquetFile(previous).iter_batches():
                            table = pa.Table.from_batches([batch])
                            writer.write_table(
                                table.filter(
                                    pc.invert(pc.is_in(table["episode_index"], pa.array(sorted(updates))))
                                )
                            )
                    for artifact in updates.values():
                        temporary = store.download(artifact, directory / "window-input.parquet")
                        for batch in pq.ParquetFile(temporary).iter_batches():
                            writer.write_table(pa.Table.from_batches([batch]))
                        temporary.unlink()
                files[relative] = destination
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
        prepared = cfg.resolved_staging_dir(root) / "prepared" / plan.plan_id
        commit_local_files(root, files, prepared, expected=expected)
    if not runtime.retain_materialized_outputs:
        # Opt-in cache retention policy, only after durable local commit. Keep
        # model outputs/provenance; missing materialization assets rebuild on CPU.
        attempts = {
            artifact.path.split("/outputs/")[0]
            for _, result in accepted_items(store, plan)
            for artifact in result.artifacts
        }
        for attempt in sorted(attempts):
            marker = f"{attempt}/pruned.json"
            if not store.exists(marker):
                store.put_json(marker, {"reason": "materialized_outputs_pruned"})
            for path in store.list(f"{attempt}/outputs/*/*.parquet"):
                store.fs.rm(store.path(path))
    phases = [
        PhaseResult(name, summary.completed, summary.masked + summary.rejected)
        for name, (_, summary) in completed.items()
        if name != "materialize"
    ]
    validation = ValidationReport(episodes_checked=len(source.episodes))
    validation_plan, _ = completed["validate"]
    for _, result in accepted_items(store, validation_plan):
        with store.open(result.artifacts[0].path) as stream:
            for row in pq.read_table(stream).to_pylist():
                validation.warnings.extend(row["warnings"])
    return PipelineRunSummary(
        phases,
        [root / path for path in files if path.startswith("data/")],
        validation,
        metadata_paths=[root / path for path in files if path.startswith("meta/")],
    )

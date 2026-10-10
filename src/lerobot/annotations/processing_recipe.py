# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Language recipe orchestration and transactional dataset publication."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from lerobot.data_processing.artifacts import file_checksum
from lerobot.data_processing.configs import StageConfig
from lerobot.data_processing.pipeline import run_pipeline
from lerobot.data_processing.sinks.local_commit import commit_local_files, recover_local_commit
from lerobot.data_processing.types import canonical_json
from lerobot.data_processing.worker import accepted_in_shard, accepted_items
from lerobot.datasets.language import language_feature_info

from .processing_source import LeRobotEpisodeSource, _ownership_path
from .steerable_pipeline.config import AnnotationPipelineConfig
from .steerable_pipeline.executor import PhaseResult, PipelineRunSummary
from .steerable_pipeline.validator import ValidationReport
from .steerable_pipeline.vlm_client import make_vlm_client


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
    phase_dependencies: dict[str, tuple[str, ...]] = {
        name: ("plan",) if name == "interjections" and "plan" in enabled else () for name in enabled
    }
    if "plan" in enabled and "interjections" in enabled:
        phase_dependencies["plan_update"] = ("plan", "interjections")
    shared = {key: value for key, value in config.items() if key not in {"plan", "interjections", "vqa"}}
    quality = ("quality",) if cfg.quality.enabled else ()
    for name, parents in phase_dependencies.items():
        family = "plan" if name == "plan_update" else name
        # Preserve the sealed dependency ordering used by existing checkpoints.
        dependencies = parents + quality if name == "plan_update" else quality + parents
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
                dependencies,
                when="quality.usable" if cfg.quality.enabled else None,
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
        for stage_name, output_name in (("quality", "quality"), ("plan", "windows")):
            if stage_name not in completed:
                continue
            provenance_plan, _ = completed[stage_name]
            for item, result in accepted_items(store, provenance_plan):
                for artifact in result.artifacts:
                    if artifact.name != output_name:
                        continue
                    relative = f"meta/annotations/{output_name}/episode-{item.key}.parquet"
                    expected[relative] = (
                        file_checksum(root / relative) if (root / relative).exists() else None
                    )
                    files[relative] = store.download(artifact, directory / relative)
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

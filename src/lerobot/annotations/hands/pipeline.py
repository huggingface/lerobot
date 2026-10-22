# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Run raw human-POV groups with the shared annotation execution engine."""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import tempfile
import time
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from filelock import FileLock

from ..runtime.artifacts import file_checksum
from ..runtime.configs import StageConfig
from ..runtime.pipeline import run_pipeline
from ..runtime.types import Outcome
from ..runtime.worker import accepted_in_shard, accepted_items
from .config import COMPUTE, HandPipelineConfig, estimate
from .source import RawHandVideoSource


def extract_dataset(bundle: Path, root: Path) -> None:
    """Materialize only ordinary dataset files, never links or absolute paths."""
    with tarfile.open(bundle) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
                or not path.parts
                or path.parts[0] not in {"meta", "data", "videos"}
            ):
                raise ValueError("Unexpected hand dataset bundle member")
            destination = root / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Missing hand dataset bundle contents")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)


def remap_source_ids(root: Path) -> list[dict]:
    """Convert each native group's local video IDs to the frozen input IDs."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    metadata = json.loads((root / "meta/egocentric_workflow.json").read_text())
    mapping = {int(key): int(Path(value).stem) for key, value in metadata["source_video_index"].items()}
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        temporary = path.with_suffix(".remap.parquet")
        reader = pq.ParquetFile(path)
        with pq.ParquetWriter(temporary, reader.schema_arrow) as writer:
            for batch in reader.iter_batches(batch_size=1024):
                table = pa.Table.from_batches([batch])
                column = table["annotation.source_video_id"]
                ids = [
                    [mapping[row[0]]] if isinstance(row, list) else mapping[row] for row in column.to_pylist()
                ]
                table = table.set_column(
                    table.schema.get_field_index("annotation.source_video_id"),
                    "annotation.source_video_id",
                    pa.array(ids, type=column.type),
                )
                writer.write_table(table)
        temporary.replace(path)
    return [
        {**entry, "source_video_id": mapping[entry["source_video_id"]]}
        for entry in metadata["episode_sources"]
    ]


def materialize(
    store, plan, config: HandPipelineConfig, started: float, planning: dict, *, fresh: bool = False
) -> dict:
    from lerobot.datasets.aggregate import aggregate_datasets

    output = config.output_root.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".hands.lock"):
        if output.exists():
            report_path = output / "meta/hand_annotations.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text())
                if report.get("plan_id") == plan.plan_id:
                    return {
                        **report,
                        "reused": True,
                        "this_invocation_seconds": time.perf_counter() - started,
                    }
            raise FileExistsError("Choose a fresh output_root; existing datasets are never overwritten")
        with tempfile.TemporaryDirectory(prefix=".hands-", dir=output.parent) as temporary:
            directory = Path(temporary)
            roots: list[Path] = []
            reports: list[dict] = []
            lineage: list[dict] = []
            episode_offset = 0
            for item, result in accepted_items(store, plan):
                artifacts = {a.name: a for a in result.artifacts}
                report_path = store.download(artifacts["report"], directory / (item.key + ".json"))
                report = json.loads(report_path.read_text())
                reports.append(report)
                if result.outcome != Outcome.COMPLETED:
                    continue
                bundle = store.download(artifacts["dataset"], directory / (item.key + ".tar"))
                root = directory / item.key
                extract_dataset(bundle, root)
                episodes = remap_source_ids(root)
                lineage.extend(
                    {**episode, "episode_index": episode_offset + episode["episode_index"]}
                    for episode in episodes
                )
                episode_offset += len(episodes)
                roots.append(root)
            dataset = directory / "result"
            if len(roots) == 1:
                shutil.copytree(roots[0], dataset)
            elif roots:
                aggregate_datasets(
                    repo_ids=[f"local/{root.name}" for root in roots],
                    aggr_repo_id=config.repo_id,
                    roots=roots,
                    aggr_root=dataset,
                    concatenate_videos=False,
                    concatenate_data=False,
                )
            (dataset / "meta").mkdir(parents=True, exist_ok=True)
            videos = [video for report in reports for video in report["videos"]]
            source_videos = [video for report in reports for video in report["source_videos"]]
            # Bind native staged names back to user-visible paths, including
            # videos rejected because neither/both hands were not observed.
            by_id = {v["source_video_id"]: v for v in source_videos}
            video_results = [
                {
                    **video,
                    "source_video_id": int(Path(video["video"]).stem),
                    "source_path": by_id[int(Path(video["video"]).stem)]["relative_path"],
                }
                for video in videos
            ]
            rejected = [v["source_video_id"] for v in video_results if v["status"] == "rejected"]
            frames = sum(report["summary"]["rectified_frames"] for report in reports)
            report = {
                "status": "rejected"
                if not lineage
                else "completed_with_rejections"
                if rejected
                else "completed",
                "plan_id": plan.plan_id,
                "estimate": planning,
                "recipe": reports[0]["lerobot_hand_recipe"],
                "input_videos": len(videos),
                "rejected_video_ids": rejected,
                "failed_video_ids": [],
                "episodes": len(lineage),
                "source_frames": frames,
                "accepted_frames": sum(r["summary"]["accepted_two_hand_frames"] for r in reports),
                "videos": video_results,
                "source_videos": source_videos,
                "episode_sources": lineage,
                "wall_seconds": time.perf_counter() - started,
                "fps_scope": "This invocation includes input hashing/staging, models, export and final materialization; "
                "withheld on cached resume. Queue time is included when the controller waits for remote jobs.",
                "reused": False,
            }
            # Stage cache accounting is explicit; never claim inference FPS
            # from a fast re-materialization of already-computed groups.
            report["measured_fps"] = frames / report["wall_seconds"] if fresh and frames else None
            (dataset / "meta/hand_annotations.json").write_text(json.dumps(report, indent=2, allow_nan=False))
            dataset.rename(output)
            return report


def run_hand_pipeline(config: HandPipelineConfig) -> dict:
    started = time.perf_counter()
    config.validate()
    if config.output_root.resolve().is_relative_to(config.input_root.resolve()):
        raise ValueError("output_root must be outside input_root")
    if not config.backend.research_terms_accepted and config.runtime.mode != "plan":
        raise ValueError(
            "The research pipeline requires upstream model rights and research_terms_accepted=true"
        )
    source = RawHandVideoSource(config.input_root, config.videos_per_group)
    planning = estimate(config, source.groups)
    if not config.backend.assets_sha256 and Path(config.backend.assets_manifest).is_file():
        config.backend.assets_sha256 = file_checksum(Path(config.backend.assets_manifest))[0]
    if config.runtime.mode != "plan":
        if not re.fullmatch(r"[0-9a-f]{64}", config.backend.assets_sha256):
            raise ValueError("Pin the provisioned asset manifest with backend.assets_sha256 before execution")
        for name, value in asdict(config.backend).items():
            if isinstance(value, str) and not value:
                raise ValueError(f"Provision backend.{name} before execution")
    if not config.runtime.run_uri:
        config.runtime.run_uri = str(config.output_root.resolve().with_name(config.output_root.name + ".run"))
    if config.runtime.backend == "hf_jobs":
        flavor = COMPUTE[config.compute]["hf_flavor"]
        if flavor is None:
            raise ValueError("This is a local/Slurm compute profile; select available HF Jobs hardware")
        config.runtime.hf_jobs.gpu_flavor = flavor
    stage = StageConfig(
        "hands",
        "lerobot.annotations.hands.processing:HandReconstructionStage",
        {"backend": asdict(config.backend), "minimum_segment_frames": config.minimum_segment_frames},
    )
    fresh = False

    def before_execute(store, plan):
        nonlocal fresh
        fresh = not any(accepted_in_shard(store, plan, shard) for shard in range(plan.shards))
        source.stage_inputs(store, plan)

    store, completed = run_pipeline(source, [stage], config.runtime, before_execute=before_execute)
    if config.runtime.mode == "plan":
        return {"status": "planned", "estimate": planning, "input_revision": source.dataset_ref.revision}
    plan, summary = completed["hands"]
    return materialize(store, plan, config, started, planning, fresh=fresh)

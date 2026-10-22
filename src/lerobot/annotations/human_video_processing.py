# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Checkpointed human-demo generation; workers write assets, never source files."""

import importlib
import json
import os
import tarfile
from dataclasses import replace
from pathlib import Path, PurePosixPath

import draccus

from .processing import LanguageModule, _read_atoms
from .processing_source import _load_record
from .runtime.artifacts import file_checksum
from .runtime.errors import EndpointLimitError
from .runtime.types import ItemResult, ModuleSpec, Outcome
from .runtime.worker import accepted_items
from .steerable_pipeline.config import AnnotationPipelineConfig
from .steerable_pipeline.frames import make_frame_provider
from .steerable_pipeline.modules.human_video import (
    MANIFEST_FILENAME,
    HumanVideoModule,
    InferenceProvidersGenerator,
)
from .steerable_pipeline.staging import EpisodeStaging
from .steerable_pipeline.vlm_client import make_vlm_client


class HumanVideoStage(LanguageModule):
    """One accepted bundle per episode, including explicit failed segment records.

    Provider calls may be billed. A completed episode is reused on resume, even
    when its manifest contains failed segments. Interrupted, uncommitted work can
    repeat; do not increase runtime retries for uncertain paid provider failures.
    """

    def __init__(self, root, annotation_config, client_factory=None, generator_factory=None):
        self.root = Path(root)
        self.phase = "human_video"
        self.config = draccus.decode(AnnotationPipelineConfig, annotation_config)
        output = PurePosixPath(self.config.human_video.output_dir)
        if output.is_absolute() or ".." in output.parts or not output.parts:
            raise ValueError("human_video.output_dir must be a relative directory")
        self.client_factory = client_factory
        self.generator_factory = generator_factory
        self.spec = ModuleSpec("human_video", "1", "episode", {"bundle": None})

    def scientific_config(self):
        return {**super().scientific_config(), "generator_factory": self.generator_factory}

    def setup(self, context):
        cfg = self.config
        if cfg.vlm.api_key_env:
            cfg.vlm.api_key = os.environ[cfg.vlm.api_key_env]
        if os.environ.get("LEROBOT_ANNOTATION_ENDPOINTS"):
            cfg.vlm.api_bases = tuple(json.loads(os.environ["LEROBOT_ANNOTATION_ENDPOINTS"]))
        if self.client_factory:
            module, name = self.client_factory.split(":", 1)
            self.client = getattr(importlib.import_module(module), name)()
        else:
            self.client = make_vlm_client(replace(cfg.vlm, auto_serve=False))
        self.frames = make_frame_provider(
            self.root,
            camera_key=cfg.human_video.camera_key or cfg.vlm.camera_key,
            video_backend=cfg.video_backend,
        )
        if self.generator_factory:
            module, name = self.generator_factory.split(":", 1)
            self.generator = getattr(importlib.import_module(module), name)()
        else:
            self.generator = InferenceProvidersGenerator(config=cfg.human_video)

    def process_batch(self, items, context):
        results = []
        for item in items:
            record = _load_record(self.root, item.payload)
            directory = context.scratch / item.item_id
            staging = EpisodeStaging(directory / "staging", record.episode_index)
            for name, artifact in item.payload["upstream"].items():
                staging.write(name, _read_atoms(context.store, artifact))
            module = HumanVideoModule(
                vlm=self.client,
                config=self.config.human_video,
                root=directory,
                generator=self.generator,
                frame_provider=self.frames,
            )
            try:
                module.run_episode(record, staging)
            except Exception as exc:
                # No second provider request after an uncertain billed call.
                raise EndpointLimitError("Human-video generation failed; review before retrying") from exc
            manifest = staging.episode_dir / MANIFEST_FILENAME
            bundle = directory / "bundle.tar.gz"
            with tarfile.open(bundle, "w:gz") as archive:
                archive.add(manifest, arcname="manifest.json")
                for path in sorted((directory / self.config.human_video.output_dir).rglob("*")):
                    if path.is_file():
                        archive.add(path, arcname=path.relative_to(directory).as_posix())
            artifact = context.write_asset(item, "bundle", bundle)
            results.append(ItemResult(item.item_id, Outcome.COMPLETED, (artifact,)))
        return results


def prepare_human_videos(root, directory, store, plan, output_dir):
    """Return source-checked files for the recipe's recoverable local commit."""
    manifest_path = root / "meta/human_videos.jsonl"
    previous = (
        [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        if manifest_path.exists()
        else []
    )
    entries, updated, files, expected = [], set(), {}, {}
    for item, result in accepted_items(store, plan):
        if result.outcome != Outcome.COMPLETED:
            continue
        updated.add(int(item.key))
        bundle = store.download(result.artifacts[0], directory / "human-video-bundle.tar.gz")
        with tarfile.open(bundle) as archive:
            entries.extend(json.load(archive.extractfile("manifest.json")))
            prefix = f"{output_dir}/episode_{int(item.key):06d}/"
            for member in archive.getmembers():
                if member.name == "manifest.json":
                    continue
                path = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or not member.name.startswith(prefix)
                    or ".." in path.parts
                    or path.is_absolute()
                    or path.suffix not in {".png", ".mp4"}
                ):
                    raise ValueError("Invalid human-video bundle member")
                destination = directory / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.extractfile(member).read())
                files[member.name] = destination
                target = root / member.name
                expected[member.name] = file_checksum(target) if target.exists() else None
    entries = [entry for entry in previous if entry["episode_index"] not in updated] + entries
    relative = "meta/human_videos.jsonl"
    destination = directory / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(entry, allow_nan=False) + "\n" for entry in entries))
    files[relative] = destination
    expected[relative] = file_checksum(manifest_path) if manifest_path.exists() else None
    return files, expected

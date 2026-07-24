# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LeRobot metadata adapter for automatic MP4 sidecar resolution."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import fsspec

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.streaming.manifest import EpisodeVideoManifest
from lerobot.streaming.sidecar import SidecarSpec, ensure_mp4_sidecar, sidecar_cache_path
from lerobot.utils.constants import HF_LEROBOT_HOME

DEFAULT_SIDECAR_CACHE = HF_LEROBOT_HOME / "streaming-sidecars"


def range_backend_for_root(data_root: str) -> str:
    """Use direct HTTP only for HF roots; all local and other fsspec protocols stay generic."""
    return "native-http" if data_root.startswith("hf://") else "fsspec"


def streaming_data_root(
    meta: LeRobotDatasetMetadata,
    *,
    requested_root: str | Path | None,
    configured_data_root: str | None,
) -> str:
    if configured_data_root is not None:
        return configured_data_root.rstrip("/")
    if requested_root is not None:
        return str(Path(requested_root).expanduser())
    return f"hf://datasets/{meta.repo_id}@{meta.revision}"


def make_sidecar_spec(meta: LeRobotDatasetMetadata, data_root: str) -> SidecarSpec:
    relative_paths = sorted(
        {
            str(meta.get_video_file_path(episode_index, video_key))
            for episode_index in range(int(meta.total_episodes))
            for video_key in meta.video_keys
        }
    )
    root = Path(data_root).expanduser()
    source_files: tuple[tuple[str, int | None], ...]
    if root.is_dir():
        source_files = tuple((path, (root / path).stat().st_size) for path in relative_paths)
    else:
        source_files = tuple((path, None) for path in relative_paths)
    return SidecarSpec(
        repo_id=meta.repo_id,
        revision=str(meta.revision),
        data_root=data_root.rstrip("/"),
        source_files=source_files,
    )


def build_mp4_sidecar(
    destination: str | Path,
    spec: SidecarSpec,
    *,
    workers: int = 8,
    range_backend: str = "native-http",
    max_probe_bytes: int = 64 * 1024 * 1024,
) -> None:
    EpisodeVideoManifest.write_file_sidecar(
        destination,
        [path for path, _size in spec.source_files],
        spec.data_root,
        spec=spec,
        range_backend=range_backend,
        workers=workers,
        max_probe_bytes=max_probe_bytes,
    )


def published_sidecar_url(spec: SidecarSpec, cache_root: str | Path = DEFAULT_SIDECAR_CACHE) -> str:
    name = sidecar_cache_path(cache_root, spec).name
    return f"{spec.data_root}/meta/mp4-sidecars/{name}"


def download_published_sidecar(
    destination: Path,
    spec: SidecarSpec,
    *,
    cache_root: str | Path = DEFAULT_SIDECAR_CACHE,
) -> bool:
    if Path(spec.data_root).expanduser().is_dir():
        return False
    filesystem, source = fsspec.core.url_to_fs(published_sidecar_url(spec, cache_root))
    if not filesystem.exists(source):
        return False
    with filesystem.open(source, "rb") as remote, destination.open("wb") as local:
        shutil.copyfileobj(remote, local)
    return True


def ensure_dataset_mp4_sidecar(
    meta: LeRobotDatasetMetadata,
    data_root: str,
    *,
    cache_root: str | Path = DEFAULT_SIDECAR_CACHE,
    workers: int = 8,
    range_backend: str = "native-http",
    lock_timeout_s: float = 30 * 60,
) -> Path | None:
    if not meta.video_keys:
        return None

    spec = make_sidecar_spec(meta, data_root)
    logging.info(
        "Resolving training-time MP4 sidecar for %s@%s (%d files)",
        spec.repo_id,
        spec.revision,
        len(spec.source_files),
    )
    return ensure_mp4_sidecar(
        spec,
        cache_root,
        build=lambda path, target_spec: build_mp4_sidecar(
            path,
            target_spec,
            workers=workers,
            range_backend=range_backend,
        ),
        download=lambda path, target_spec: download_published_sidecar(
            path,
            target_spec,
            cache_root=cache_root,
        ),
        lock_timeout_s=lock_timeout_s,
    )

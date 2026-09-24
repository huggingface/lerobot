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
import re
import shutil
from pathlib import Path
from uuid import uuid4

import fsspec
from huggingface_hub import HfApi, HfFileSystem

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
    token: str | bool | None = None,
) -> str:
    """Resolve the payload root, pinning Hub dataset revisions to immutable commits.

    Args:
        meta (`LeRobotDatasetMetadata`):
            Dataset identity and resolved metadata revision.
        requested_root (`str | Path | None`):
            Explicit local payload directory, if provided.
        configured_data_root (`str | None`):
            Payload override, taking precedence over the local root and metadata URL.
        token (`str | bool | None`, *optional*):
            Hub authentication used to resolve mutable revision names.

    Returns:
        `str`: Local path, pinned Hub dataset URL or bucket URL used for payload reads.
    """
    if configured_data_root is not None:
        return _pin_hub_root(configured_data_root.rstrip("/"), token=token)
    if requested_root is not None:
        return str(Path(requested_root).expanduser())
    if getattr(meta, "repo_type", "dataset") == "bucket":
        return meta.url_root
    revision = meta.revision
    metadata_root = Path(meta.root) if hasattr(meta, "root") else None
    if metadata_root is not None and metadata_root.parent.name == "snapshots":
        revision = metadata_root.name
    return _pin_hub_root(
        f"{getattr(meta, 'url_root', f'hf://datasets/{meta.repo_id}')}@{revision}", token=token
    )


def _pin_hub_root(data_root: str, *, token: str | bool | None = None) -> str:
    """Use immutable dataset commits for both sidecar identity and payload reads."""
    if not data_root.startswith("hf://datasets/"):
        return data_root
    if re.match(r"^hf://datasets/[^/]+/[^/@]+@[0-9a-f]{40}(?:/|$)", data_root):
        return data_root
    resolved = HfFileSystem(token=token).resolve_path(data_root)
    sha = HfApi(token=token).dataset_info(resolved.repo_id, revision=resolved.revision).sha
    suffix = f"/{resolved.path_in_repo}" if resolved.path_in_repo else ""
    return f"hf://datasets/{resolved.repo_id}@{sha}{suffix}"


def make_sidecar_spec(
    meta: LeRobotDatasetMetadata, data_root: str, *, token: str | bool | None = None
) -> SidecarSpec:
    """Describe the full dataset's video sources and their revision or object identities.

    Args:
        meta (`LeRobotDatasetMetadata`):
            Metadata for every episode, including each camera's source video path.
        data_root (`str`):
            Root containing the source videos.
        token (`str | bool | None`, *optional*):
            Hub authentication for revision resolution and bucket metadata reads.

    Returns:
        `SidecarSpec`: Source identity used to validate and key the local sidecar cache.
    """
    data_root = _pin_hub_root(data_root.rstrip("/"), token=token)
    relative_paths = sorted(
        {
            str(meta.get_video_file_path(episode_index, video_key))
            for episode_index in range(int(meta.total_episodes))
            for video_key in meta.video_keys
        }
    )
    root = Path(data_root).expanduser()
    source_files: tuple[tuple[str, int | None], ...]
    fingerprints: tuple[tuple[str, str], ...] = ()
    if data_root.startswith("hf://buckets/"):
        parts = data_root.removeprefix("hf://buckets/").split("/", 2)
        bucket_id = "/".join(parts[:2])
        prefix = parts[2].rstrip("/") + "/" if len(parts) == 3 else ""
        wanted = {prefix + path: path for path in relative_paths}
        # One paginated listing, not one HEAD per video. Xet hashes detect even
        # same-size replacements; no payloads are downloaded for validation.
        files = {
            wanted[item.path]: item
            for item in HfApi(token=token).list_bucket_tree(bucket_id, prefix=prefix, recursive=True)
            if item.type == "file" and item.path in wanted
        }
        missing = set(relative_paths) - files.keys()
        if missing:
            raise FileNotFoundError(f"Bucket is missing source video: {min(missing)}")
        source_files = tuple((path, files[path].size) for path in relative_paths)
        fingerprints = tuple((path, files[path].xet_hash) for path in relative_paths)
    elif root.is_dir():
        stats = {path: (root / path).stat() for path in relative_paths}
        source_files = tuple((path, stats[path].st_size) for path in relative_paths)
        fingerprints = tuple(
            (path, f"{stats[path].st_mtime_ns}:{stats[path].st_ctime_ns}") for path in relative_paths
        )
    elif not data_root.startswith("hf://datasets/"):
        filesystem, fs_root = fsspec.core.url_to_fs(data_root, skip_instance_cache=True)
        infos = {path: filesystem.info(f"{fs_root.rstrip('/')}/{path}") for path in relative_paths}
        source_files = tuple((path, int(infos[path]["size"])) for path in relative_paths)
        fingerprints = tuple(
            (
                path,
                str(
                    infos[path].get("ETag")
                    or infos[path].get("etag")
                    or infos[path].get("version_id")
                    or infos[path].get("mtime")
                    or infos[path].get("LastModified")
                    or uuid4().hex
                ),
            )
            for path in relative_paths
        )
        # Without a stable object identity, rebuild rather than trust a stale index.
    else:
        source_files = tuple((path, None) for path in relative_paths)
    return SidecarSpec(
        repo_id=meta.repo_id,
        revision=str(meta.revision),
        data_root=data_root.rstrip("/"),
        source_files=source_files,
        source_fingerprints=fingerprints,
    )


def build_mp4_sidecar(
    destination: str | Path,
    spec: SidecarSpec,
    *,
    workers: int = 8,
    range_backend: str = "native-http",
    max_probe_bytes: int = 64 * 1024 * 1024,
    token: str | bool | None = None,
) -> None:
    """Index the specified videos into a local sidecar without publishing it.

    Args:
        destination (`str | Path`):
            Local NPZ output path. The caller owns locking and atomic installation.
        spec (`SidecarSpec`):
            Source files and dataset identity to record in the sidecar.
        workers (`int`, *optional*, defaults to `8`):
            Concurrent source-indexing workers.
        range_backend (`str`, *optional*, defaults to `"native-http"`):
            Range reader: native-http for Hub sources, or fsspec for other roots.
        max_probe_bytes (`int`, *optional*):
            Maximum header or tail probe size; defaults to 64 MiB.
        token (`str | bool | None`, *optional*):
            Authentication for source reads, never stored in the sidecar.
    """
    EpisodeVideoManifest.write_file_sidecar(
        destination,
        [path for path, _size in spec.source_files],
        spec.data_root,
        spec=spec,
        range_backend=range_backend,
        workers=workers,
        max_probe_bytes=max_probe_bytes,
        token=token,
    )


def published_sidecar_url(spec: SidecarSpec, cache_root: str | Path = DEFAULT_SIDECAR_CACHE) -> str:
    """Return the content-keyed sidecar location under the remote dataset's metadata."""
    name = sidecar_cache_path(cache_root, spec).name
    return f"{spec.data_root}/meta/mp4-sidecars/{name}"


def download_published_sidecar(
    destination: Path,
    spec: SidecarSpec,
    *,
    cache_root: str | Path = DEFAULT_SIDECAR_CACHE,
    token: str | bool | None = None,
) -> bool:
    """Copy a published sidecar for later validation, returning False when absent."""
    if Path(spec.data_root).expanduser().is_dir():
        return False
    source_url = published_sidecar_url(spec, cache_root)
    storage_options = {"token": token} if token is not None and source_url.startswith("hf://") else {}
    filesystem, source = fsspec.core.url_to_fs(source_url, **storage_options)
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
    token: str | bool | None = None,
) -> Path | None:
    """Reuse or safely build a validated local video index.

    Check the local cache, try a published sidecar, then build locally if necessary. A process
    lock serializes installation, and atomic replacement preserves the previous valid file on
    failure. This function never uploads an index.

    Args:
        meta (`LeRobotDatasetMetadata`):
            Full dataset metadata, not a rank-local episode subset.
        data_root (`str`):
            Payload root containing the indexed source videos.
        cache_root (`str | Path`, *optional*):
            Local sidecar cache; defaults to streaming-sidecars under HF_LEROBOT_HOME.
        workers (`int`, *optional*, defaults to `8`):
            Concurrent workers used only when a new index must be built.
        range_backend (`str`, *optional*, defaults to `"native-http"`):
            Range reader used during index construction.
        lock_timeout_s (`float`, *optional*):
            Maximum time to wait for another builder; defaults to 30 minutes.
        token (`str | bool | None`, *optional*):
            Hub authentication for metadata, sidecar and source reads.

    Returns:
        `Path | None`: Validated local sidecar path, or None for datasets without video.
    """
    if not meta.video_keys:
        return None

    spec = make_sidecar_spec(meta, data_root, token=token)
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
            token=token,
        ),
        download=lambda path, target_spec: download_published_sidecar(
            path,
            target_spec,
            cache_root=cache_root,
            token=token,
        ),
        lock_timeout_s=lock_timeout_s,
    )

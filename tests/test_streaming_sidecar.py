#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from filelock import FileLock

from lerobot.streaming.manifest import EpisodeVideoManifest, VideoFileRecord
from lerobot.streaming.mp4 import Mp4Index
from lerobot.streaming.sidecar import (
    SidecarLockTimeoutError,
    SidecarSpec,
    ensure_mp4_sidecar,
    sidecar_cache_path,
)


def _record(path: str = "videos/camera/chunk-000/file-000.mp4", size: int = 128) -> VideoFileRecord:
    arrays = np.array([0], dtype=np.int64)
    index = Mp4Index(
        file_path=path,
        file_size=size,
        ftyp=b"",
        moov_offset=0,
        mdat_offset=0,
        mdat_payload_offset=0,
        mdat_payload_size=size,
        faststart=True,
        codec="avc1",
        timescale=1,
        duration=1,
        track_id=1,
        width=1,
        height=1,
        stsd_body=b"",
        sample_pts=np.array([0.0]),
        sample_durations=arrays,
        sample_sizes=arrays,
        sample_offsets=arrays,
        sync_samples=arrays,
    )
    return VideoFileRecord(path, size, index)


def _spec(revision: str = "rev-a", size: int = 128) -> SidecarSpec:
    return SidecarSpec(
        repo_id="owner/dataset",
        revision=revision,
        data_root="hf://datasets/owner/dataset",
        source_files=(("videos/camera/chunk-000/file-000.mp4", size),),
    )


def _write_valid(path: Path, spec: SidecarSpec) -> None:
    EpisodeVideoManifest.save_file_sidecar(path, [_record(size=spec.source_files[0][1])], spec=spec)


def test_sidecar_cache_path_is_revision_keyed(tmp_path: Path) -> None:
    first = sidecar_cache_path(tmp_path, _spec("rev-a"))
    second = sidecar_cache_path(tmp_path, _spec("rev-b"))

    assert first != second
    assert first.parent == second.parent


def test_ensure_reuses_valid_local_sidecar(tmp_path: Path) -> None:
    spec = _spec()
    path = sidecar_cache_path(tmp_path, spec)
    _write_valid(path, spec)
    build_calls = 0

    def build(_path: Path, _spec: SidecarSpec) -> None:
        nonlocal build_calls
        build_calls += 1

    resolved = ensure_mp4_sidecar(spec, tmp_path, build=build)

    assert resolved == path
    assert build_calls == 0


def test_ensure_downloads_valid_published_sidecar(tmp_path: Path) -> None:
    spec = _spec()
    published = tmp_path / "published.npz"
    _write_valid(published, spec)
    build_calls = 0

    def download(path: Path, _spec: SidecarSpec) -> bool:
        shutil.copyfile(published, path)
        return True

    def build(_path: Path, _spec: SidecarSpec) -> None:
        nonlocal build_calls
        build_calls += 1

    resolved = ensure_mp4_sidecar(spec, tmp_path / "cache", build=build, download=download)

    assert EpisodeVideoManifest.validate_file_sidecar(resolved, spec)
    assert build_calls == 0


@pytest.mark.parametrize("invalid_kind", ["corrupt", "stale"])
def test_ensure_rebuilds_invalid_local_sidecar(tmp_path: Path, invalid_kind: str) -> None:
    spec = _spec()
    path = sidecar_cache_path(tmp_path, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    if invalid_kind == "corrupt":
        path.write_bytes(b"not-an-npz")
    else:
        _write_valid(path, _spec(revision="other-revision"))
    build_calls = 0

    def build(target: Path, target_spec: SidecarSpec) -> None:
        nonlocal build_calls
        build_calls += 1
        _write_valid(target, target_spec)

    resolved = ensure_mp4_sidecar(spec, tmp_path, build=build)

    assert EpisodeVideoManifest.validate_file_sidecar(resolved, spec)
    assert build_calls == 1


def test_concurrent_ensure_builds_once(tmp_path: Path) -> None:
    spec = _spec()
    start = threading.Barrier(2)
    build_calls = 0
    build_lock = threading.Lock()

    def build(path: Path, target_spec: SidecarSpec) -> None:
        nonlocal build_calls
        with build_lock:
            build_calls += 1
        _write_valid(path, target_spec)

    def ensure() -> Path:
        start.wait()
        return ensure_mp4_sidecar(spec, tmp_path, build=build)

    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _: ensure(), range(2)))

    assert paths[0] == paths[1]
    assert build_calls == 1


def test_failed_build_does_not_replace_existing_file(tmp_path: Path) -> None:
    spec = _spec()
    path = sidecar_cache_path(tmp_path, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"old-corrupt-file")

    def build(target: Path, _spec: SidecarSpec) -> None:
        target.write_bytes(b"partial")
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        ensure_mp4_sidecar(spec, tmp_path, build=build)

    assert path.read_bytes() == b"old-corrupt-file"
    assert not list(path.parent.glob(f".{path.name}.*.tmp.npz"))


def test_lock_timeout_is_actionable(tmp_path: Path) -> None:
    spec = _spec()
    path = sidecar_cache_path(tmp_path, spec)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        FileLock(lock_path),
        pytest.raises(SidecarLockTimeoutError, match="Timed out waiting"),
    ):
        ensure_mp4_sidecar(spec, tmp_path, build=_write_valid, lock_timeout_s=0.01)

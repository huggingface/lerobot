#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import errno
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pytest
from filelock import FileLock

from lerobot.streaming import _mapped_index
from lerobot.streaming.manifest import EpisodeVideoManifest, VideoFileRecord
from lerobot.streaming.mp4 import Mp4Index
from lerobot.streaming.sidecar import (
    SidecarLockTimeoutError,
    SidecarSpec,
    ensure_mp4_sidecar,
    sidecar_cache_path,
)


@pytest.fixture(autouse=True)
def isolated_index_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "cache-home"))


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
        sample_composition_offsets=arrays,
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


def test_source_generation_round_trip_and_replacement(tmp_path: Path) -> None:
    original = _spec()
    video = original.source_files[0][0]
    first = replace(original, source_fingerprints=((video, "hash-one"),))
    second = replace(original, source_fingerprints=((video, "hash-two"),))
    built = []

    def build(path: Path, spec: SidecarSpec) -> None:
        built.append(spec)
        _write_valid(path, spec)

    first_path = ensure_mp4_sidecar(first, tmp_path, build=build)
    assert ensure_mp4_sidecar(first, tmp_path, build=build) == first_path
    second_path = ensure_mp4_sidecar(second, tmp_path, build=build)
    assert built == [first, second]
    assert first_path != second_path
    assert SidecarSpec.from_dict(second.to_dict()) == second
    assert not EpisodeVideoManifest.validate_file_sidecar(first_path, second)
    assert EpisodeVideoManifest.validate_file_sidecar(second_path, second)
    assert EpisodeVideoManifest.validate_file_sidecar(first_path, first)


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


def test_ensure_rebuilds_sidecar_without_composition_timing(tmp_path: Path) -> None:
    spec = _spec()
    path = sidecar_cache_path(tmp_path, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_spec = SidecarSpec(spec.repo_id, spec.revision, spec.data_root, spec.source_files, schema_version=2)
    # Schema 2 recorded decode timestamps as presentation timestamps and cannot
    # safely be reused even when its source identity and file sizes match.
    np.savez_compressed(
        path, manifest_json=json.dumps({"version": 2, "sidecar": old_spec.to_dict()}).encode()
    )
    assert path != sidecar_cache_path(tmp_path, old_spec)
    assert not EpisodeVideoManifest.validate_file_sidecar(path, spec)
    with pytest.raises(ValueError, match="Unsupported MP4 sidecar schema"):
        EpisodeVideoManifest.load_file_sidecar(path)
    build_calls = []

    def build(target: Path, target_spec: SidecarSpec) -> None:
        build_calls.append(target_spec)
        _write_valid(target, target_spec)

    assert ensure_mp4_sidecar(spec, tmp_path, build=build) == path
    assert build_calls == [spec]
    assert EpisodeVideoManifest.validate_file_sidecar(path, spec)


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


def test_sidecar_arrays_are_read_only_file_backed(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    _write_valid(path, _spec())
    first = EpisodeVideoManifest.load_file_sidecar(path)
    second = EpisodeVideoManifest.load_file_sidecar(path)
    for records in (first, second):
        array = next(iter(records.values())).mp4.sample_pts
        assert not array.flags.writeable
        assert isinstance(array.base, np.memmap)
        np.testing.assert_array_equal(array, [0.0])


def test_warm_validation_does_not_decompress_arrays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "index.npz"
    _write_valid(path, _spec())
    assert EpisodeVideoManifest.validate_file_sidecar(path, _spec())

    def unexpected_load(*args: object, **kwargs: object) -> None:
        pytest.fail("A warm index must not reopen the compressed sidecar")

    monkeypatch.setattr(np, "load", unexpected_load)
    assert EpisodeVideoManifest.validate_file_sidecar(path, _spec())
    records = EpisodeVideoManifest.load_file_sidecar(path)
    np.testing.assert_array_equal(next(iter(records.values())).mp4.sample_pts, [0.0])


def test_sidecar_replacement_preserves_live_arrays(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    _write_valid(path, _spec())
    old = EpisodeVideoManifest.load_file_sidecar(path)
    replacement = _record()
    replacement.mp4.sample_pts[0] = 42.0
    EpisodeVideoManifest.save_file_sidecar(path, [replacement], spec=_spec())
    new = EpisodeVideoManifest.load_file_sidecar(path)
    np.testing.assert_array_equal(next(iter(old.values())).mp4.sample_pts, [0.0])
    np.testing.assert_array_equal(next(iter(new.values())).mp4.sample_pts, [42.0])


@pytest.mark.parametrize("warm", [False, True])
def test_sidecar_generation_uses_open_file_not_stale_path_attributes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warm: bool
) -> None:
    path = tmp_path / "index.npz"
    _write_valid(path, _spec())
    stale_stat = path.stat()
    old = EpisodeVideoManifest.load_file_sidecar(path) if warm else None
    replacement = _record()
    replacement.mp4.sample_pts[0] = 42.0
    EpisodeVideoManifest.save_file_sidecar(path, [replacement], spec=_spec())
    original_stat = Path.stat

    def stale_path_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        # Model a shared filesystem's stale pathname attributes, not stale file data.
        return stale_stat if self == path else original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stale_path_stat)
    new = EpisodeVideoManifest.load_file_sidecar(path)
    np.testing.assert_array_equal(next(iter(new.values())).mp4.sample_pts, [42.0])
    if old is not None:
        np.testing.assert_array_equal(next(iter(old.values())).mp4.sample_pts, [0.0])


def test_sidecar_payload_does_not_reuse_stale_path_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "index.npz"
    _write_valid(path, _spec("old"))
    stale_stat = path.stat()
    _mapped_index.mapped_sidecar(path)
    _write_valid(path, _spec("new"))
    original_stat = Path.stat

    def stale_path_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        return stale_stat if self == path else original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stale_path_stat)
    assert _mapped_index.sidecar_payload(path)["sidecar"]["revision"] == "new"


def test_source_change_during_conversion_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "index.npz"
    _write_valid(path, _spec())
    validate = _mapped_index._validate_arrays

    def change_after_array_read(arrays: dict[str, np.ndarray]) -> None:
        validate(arrays)
        # Mutate the real, still-open source after its arrays have been read.
        with path.open("ab") as changed:
            changed.write(b"concurrent mutation")

    monkeypatch.setattr(_mapped_index, "_validate_arrays", change_after_array_read)
    with pytest.raises(OSError, match="changed during index conversion"):
        _mapped_index.mapped_sidecar(path)
    cache = tmp_path / "cache-home" / "streaming-indexes"
    assert not list(cache.glob("*.bin"))
    assert not list(cache.glob("*.index.tmp"))


def test_sidecar_load_projects_source_files(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    EpisodeVideoManifest.save_file_sidecar(path, [_record("a.mp4"), _record("b.mp4")], spec=_spec())
    records = EpisodeVideoManifest.load_file_sidecar(path, file_paths=["b.mp4"])
    assert list(records) == ["b.mp4"]
    np.testing.assert_array_equal(records["b.mp4"].mp4.sample_pts, [0.0])


def test_concurrent_index_conversion_runs_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "index.npz"
    _write_valid(path, _spec())
    original = np.load
    calls = []

    def counted_load(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(EpisodeVideoManifest.load_file_sidecar, [path] * 4))
    assert len(calls) == 1
    for result in records:
        np.testing.assert_array_equal(next(iter(result.values())).mp4.sample_pts, [0.0])


def test_invalid_derived_index_is_recreated(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    _write_valid(path, _spec())
    cache_path, _ = _mapped_index.mapped_sidecar(path)
    cache_path.write_bytes(b"interrupted")
    records = EpisodeVideoManifest.load_file_sidecar(path)
    np.testing.assert_array_equal(next(iter(records.values())).mp4.sample_pts, [0.0])


@pytest.mark.parametrize("corruption", ["non-object", "fractional-count"])
def test_invalid_derived_metadata_is_recreated(tmp_path: Path, corruption: str) -> None:
    path = tmp_path / "index.npz"
    _write_valid(path, _spec())
    cache_path, payload = _mapped_index.mapped_sidecar(path)
    raw = cache_path.read_bytes()
    old_length = int.from_bytes(raw[-16:-8], "little")
    if corruption == "non-object":
        metadata = b"[]"
    else:
        payload["files"][0]["arrays"]["sample_pts"][1] = 0.5
        metadata = json.dumps(payload).encode()
    cache_path.write_bytes(
        raw[: -16 - old_length] + metadata + len(metadata).to_bytes(8, "little") + raw[-8:]
    )
    records = EpisodeVideoManifest.load_file_sidecar(path)
    np.testing.assert_array_equal(next(iter(records.values())).mp4.sample_pts, [0.0])


def test_temporary_validation_does_not_prepare_index(tmp_path: Path) -> None:
    path = tmp_path / "temporary.npz"
    _write_valid(path, _spec())
    assert EpisodeVideoManifest.validate_file_sidecar(path, _spec(), prepare_cache=False)
    assert not list((tmp_path / "cache-home").glob("**/*.bin"))


def test_wrong_revision_does_not_prepare_index(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    _write_valid(path, _spec())
    assert not EpisodeVideoManifest.validate_file_sidecar(path, _spec("wrong"))
    assert not list((tmp_path / "cache-home").glob("**/*.bin"))


def test_missing_array_is_rejected_without_publishing_index(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    _write_valid(path, _spec())
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "0/sample_offsets"}
    np.savez_compressed(path, **arrays)
    assert not EpisodeVideoManifest.validate_file_sidecar(path, _spec())
    assert not list((tmp_path / "cache-home").glob("**/*.bin"))
    assert not list((tmp_path / "cache-home").glob("**/*.tmp"))


def test_read_only_source_uses_writable_local_index_cache(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    path = source_dir / "index.npz"
    _write_valid(path, _spec())
    path.chmod(0o444)
    source_dir.chmod(0o555)
    try:
        records = EpisodeVideoManifest.load_file_sidecar(path)
        assert list(source_dir.iterdir()) == [path]
        np.testing.assert_array_equal(next(iter(records.values())).mp4.sample_pts, [0.0])
    finally:
        source_dir.chmod(0o755)


def test_full_index_cache_does_not_rebuild_valid_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec()
    path = sidecar_cache_path(tmp_path, spec)
    _write_valid(path, spec)

    def full_disk(*args: object, **kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    def unexpected_build(*args: object) -> None:
        pytest.fail("A full derived cache is not an invalid source sidecar")

    monkeypatch.setattr(_mapped_index.tempfile, "NamedTemporaryFile", full_disk)
    with pytest.raises(OSError, match="free space in HF_LEROBOT_HOME"):
        ensure_mp4_sidecar(spec, tmp_path, build=unexpected_build)


@pytest.mark.parametrize("prepare_cache", [False, True])
def test_sidecar_array_reads_do_not_scan_npz_member_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepare_cache: bool
) -> None:
    path = tmp_path / "index.npz"
    _write_valid(path, _spec())
    original = np.lib.npyio.NpzFile.__getitem__
    array_lookups = []

    def counted_lookup(self: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
        if key != "manifest_json":
            array_lookups.append(key)
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", counted_lookup)
    assert EpisodeVideoManifest.validate_file_sidecar(path, _spec(), prepare_cache=prepare_cache)
    assert array_lookups == []


@pytest.mark.parametrize("prepare_cache", [False, True])
def test_sidecar_rejects_object_arrays_without_publication(tmp_path: Path, prepare_cache: bool) -> None:
    path = tmp_path / "index.npz"
    _write_valid(path, _spec())
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["0/sample_pts"] = np.array([object()], dtype=object)
    np.savez_compressed(path, **arrays)
    assert not EpisodeVideoManifest.validate_file_sidecar(path, _spec(), prepare_cache=prepare_cache)
    assert not list((tmp_path / "cache-home").glob("**/*.bin"))


@pytest.mark.parametrize("published", [False, True])
def test_resolved_sidecar_decompresses_each_array_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, published: bool
) -> None:
    original = _mapped_index._read_arrays
    reads = []

    def counted_read(archive: ZipFile, index: int, item: dict[str, Any]) -> dict[str, np.ndarray]:
        reads.append(index)
        return original(archive, index, item)

    def download(path: Path, spec: SidecarSpec) -> bool:
        _write_valid(path, spec)
        return True

    monkeypatch.setattr(_mapped_index, "_read_arrays", counted_read)
    path = ensure_mp4_sidecar(_spec(), tmp_path, build=_write_valid, download=download if published else None)
    assert EpisodeVideoManifest.validate_file_sidecar(path, _spec())
    records = EpisodeVideoManifest.load_file_sidecar(path)
    np.testing.assert_array_equal(next(iter(records.values())).mp4.sample_pts, [0.0])
    assert reads == [0]
    assert len(list((tmp_path / "cache-home").glob("**/*.bin"))) == 1


def test_invalid_build_preserves_previous_source_and_mapped_generation(tmp_path: Path) -> None:
    spec = _spec()
    path = sidecar_cache_path(tmp_path, spec)
    _write_valid(path, _spec("older"))
    old_bytes = path.read_bytes()
    old = EpisodeVideoManifest.load_file_sidecar(path)

    def invalid_build(target: Path, target_spec: SidecarSpec) -> None:
        _write_valid(target, target_spec)
        with np.load(target, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files if key != "0/sample_offsets"}
        np.savez_compressed(target, **arrays)

    with pytest.raises(ValueError, match="failed revision and source validation"):
        ensure_mp4_sidecar(spec, tmp_path, build=invalid_build)
    assert path.read_bytes() == old_bytes
    np.testing.assert_array_equal(next(iter(old.values())).mp4.sample_pts, [0.0])
    assert len(list((tmp_path / "cache-home").glob("**/*.bin"))) == 1
    assert not list((tmp_path / "cache-home").glob("**/*.index.tmp"))

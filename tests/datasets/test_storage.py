# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import fsspec
import pytest

from lerobot.datasets import storage
from lerobot.datasets.storage import localize_lerobot_root, localize_remote_root, remote_meta_cache_dir

pytestmark = pytest.mark.dataset


@pytest.fixture
def remote_meta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[str]:
    """A memory:// dataset root holding only meta/, with the local cache under tmp_path."""
    monkeypatch.setattr(storage, "HF_LEROBOT_HOME", tmp_path)
    namespace = f"storage-{uuid4().hex}"
    fs = fsspec.filesystem("memory")
    fs.pipe(f"/{namespace}/meta/info.json", json.dumps({"version": "new"}).encode())
    fs.pipe(f"/{namespace}/meta/episodes/chunk-000/file-000.parquet", b"episodes")
    yield f"memory://{namespace}"
    fs.rm(f"/{namespace}", recursive=True)


def test_cache_dirs_of_similar_uris_differ() -> None:
    assert remote_meta_cache_dir("s3://b/a_b") != remote_meta_cache_dir("s3://b/a/b")
    assert len(remote_meta_cache_dir("s3://b/" + "x" * 500).name) < 255


def test_localize_copies_meta_and_force_refresh_swaps_in_place(remote_meta: str) -> None:
    cache = localize_lerobot_root(remote_meta)
    meta = cache / "meta"
    assert json.loads((meta / "info.json").read_text()) == {"version": "new"}
    assert (meta / "episodes" / "chunk-000" / "file-000.parquet").read_bytes() == b"episodes"

    (meta / "info.json").write_text(json.dumps({"version": "stale"}))
    (cache / "meta.old-99999").mkdir()  # left behind by a crashed refresh
    assert json.loads((localize_lerobot_root(remote_meta) / "meta/info.json").read_text()) == {
        "version": "stale"
    }  # an existing cache is reused

    localize_lerobot_root(remote_meta, force_cache_sync=True)
    assert json.loads((meta / "info.json").read_text()) == {"version": "new"}
    assert sorted(path.name for path in cache.iterdir()) == ["meta"]  # no meta.tmp / meta.old-*


def test_localize_rejects_roots_without_meta(remote_meta: str) -> None:
    with pytest.raises(FileNotFoundError, match="info.json"):
        localize_lerobot_root(f"{remote_meta}/missing")


def test_localize_leaves_other_storage_formats_to_their_backend(remote_meta: str) -> None:
    fsspec.filesystem("memory").pipe(
        f"/{remote_meta.removeprefix('memory://')}/meta/info.json",
        json.dumps({"storage_format": "lance"}).encode(),
    )
    with pytest.raises(FileNotFoundError, match="'lance'"):
        localize_lerobot_root(remote_meta)


def test_missing_protocol_package_is_not_reported_as_missing_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def url_to_fs(*args, **kwargs):
        raise ImportError("Install s3fs to access S3")

    monkeypatch.setattr(fsspec.core, "url_to_fs", url_to_fs)
    monkeypatch.setattr(storage, "_DATASET_READER_MODULES", {})  # no other format matches
    with pytest.raises(ImportError, match="s3fs"):
        localize_remote_root("owner/data", "s3://bucket/dataset")

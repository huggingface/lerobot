#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.streaming_sidecar import make_sidecar_spec, range_backend_for_root, streaming_data_root


def test_hub_data_root_is_revision_qualified() -> None:
    meta = SimpleNamespace(repo_id="owner/dataset", revision="a" * 40)

    root = streaming_data_root(meta, requested_root=None, configured_data_root=None)

    assert root == f"hf://datasets/owner/dataset@{'a' * 40}"
    assert range_backend_for_root(root) == "native-http"


def test_explicit_bucket_root_is_preserved() -> None:
    meta = SimpleNamespace(repo_id="owner/dataset", revision="commit-sha")
    bucket = "hf://buckets/owner/dataset-bucket/prefix/"

    root = streaming_data_root(meta, requested_root=None, configured_data_root=bucket)

    assert root == bucket.rstrip("/")
    assert range_backend_for_root(root) == "native-http"


def test_bucket_root_is_derived_without_an_override() -> None:
    meta = SimpleNamespace(
        repo_id="owner/bucket", repo_type="bucket", url_root="hf://buckets/owner/bucket", revision="v3.0"
    )

    root = streaming_data_root(meta, requested_root=None, configured_data_root=None)

    assert root == "hf://buckets/owner/bucket"
    assert range_backend_for_root(root) == "native-http"


def test_local_and_generic_remote_roots_use_fsspec(tmp_path: Path) -> None:
    meta = SimpleNamespace(repo_id="owner/dataset", revision="commit-sha")

    local = streaming_data_root(meta, requested_root=tmp_path, configured_data_root=None)

    assert local == str(tmp_path)
    assert range_backend_for_root(local) == "fsspec"
    assert range_backend_for_root("memory://dataset") == "fsspec"


def test_bucket_replacement_invalidates_sidecar_even_at_same_size(monkeypatch: pytest.MonkeyPatch) -> None:
    video = "videos/camera/chunk-000/file-000.mp4"
    current_hash = "generation-one"
    meta = SimpleNamespace(
        repo_id="owner/bucket",
        revision="v3.0",
        total_episodes=1,
        video_keys=["camera"],
        get_video_file_path=lambda *_args: video,
        ensure_readable=lambda: None,
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        episodes=Dataset.from_dict({"videos/camera/chunk_index": [0], "videos/camera/file_index": [0]}),
    )
    monkeypatch.setattr(
        "lerobot.datasets.streaming_sidecar.HfApi",
        lambda **_kwargs: SimpleNamespace(
            list_bucket_tree=lambda *_args, **_kw: [
                SimpleNamespace(type="file", path="prefix/" + video, size=128, xet_hash=current_hash)
            ]
        ),
        raising=False,
    )
    first = make_sidecar_spec(meta, "hf://buckets/owner/bucket/prefix")
    current_hash = "generation-two"
    second = make_sidecar_spec(meta, "hf://buckets/owner/bucket/prefix")
    assert not second.matches(first)


def test_default_hub_root_uses_metadata_snapshot_commit(tmp_path: Path) -> None:
    sha = "a" * 40
    meta = SimpleNamespace(
        repo_id="owner/dataset",
        revision="main",
        root=tmp_path / "snapshots" / sha,
    )
    assert streaming_data_root(meta, requested_root=None, configured_data_root=None) == (
        f"hf://datasets/owner/dataset@{sha}"
    )


def test_explicit_hub_branch_is_pinned_and_token_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    sha = "b" * 40
    calls = []

    def filesystem(*, token):
        calls.append(token)
        return SimpleNamespace(
            resolve_path=lambda root: SimpleNamespace(
                repo_id="owner/dataset", revision="main", path_in_repo="payload"
            )
        )

    def api(*, token):
        calls.append(token)
        return SimpleNamespace(dataset_info=lambda repo_id, revision: SimpleNamespace(sha=sha))

    monkeypatch.setattr("lerobot.datasets.streaming_sidecar.HfFileSystem", filesystem)
    monkeypatch.setattr("lerobot.datasets.streaming_sidecar.HfApi", api)
    root = streaming_data_root(
        SimpleNamespace(),
        requested_root=None,
        configured_data_root="hf://datasets/owner/dataset@main/payload",
        token=False,
    )
    assert root == f"hf://datasets/owner/dataset@{sha}/payload"
    assert calls == [False, False]

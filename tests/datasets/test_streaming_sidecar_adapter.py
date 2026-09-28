#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import io
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from datasets import Dataset

from lerobot.datasets.streaming_sidecar import (
    download_published_sidecar,
    make_sidecar_spec,
    published_sidecar_url,
    range_backend_for_root,
    streaming_data_root,
)
from lerobot.scripts.lerobot_build_mp4_sidecar import push_sidecar
from lerobot.streaming.manifest import EpisodeVideoManifest
from lerobot.streaming.sidecar import SidecarSpec, ensure_mp4_sidecar
from tests.test_streaming_sidecar import _write_valid


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


@pytest.mark.parametrize("metadata_revision", ["main", "v3.0", "a" * 40])
def test_published_sidecar_identity_uses_source_commit_not_metadata_alias(
    monkeypatch: pytest.MonkeyPatch, metadata_revision: str
) -> None:
    """The CLI's explicit SHA and training's version alias find the same index."""
    monkeypatch.setattr("lerobot.datasets.streaming_sidecar.video_file_groups", lambda _: {"v.mp4": []})
    meta = SimpleNamespace(repo_id="owner/dataset", revision=metadata_revision)
    spec = make_sidecar_spec(meta, f"hf://datasets/owner/dataset@{'a' * 40}/payload")
    assert spec.revision == metadata_revision  # Existing local cache keys stay valid.
    assert spec.data_root == f"hf://datasets/owner/dataset@{'a' * 40}/payload"
    canonical = replace(spec, revision="a" * 40)
    assert published_sidecar_url(spec) == published_sidecar_url(canonical)
    assert spec.matches(canonical)


def test_repository_publication_round_trip_keeps_source_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publishing on B leaves source A readable and its sidecar discoverable."""
    source_sha, published_sha = "a" * 40, "b" * 40
    spec = SidecarSpec(
        "owner/dataset",
        source_sha,
        f"hf://datasets/owner/dataset@{source_sha}/payload",
        (("videos/camera/chunk-000/file-000.mp4", 128),),
    )
    source = tmp_path / "source.npz"
    _write_valid(source, spec)
    branches = {"main": source_sha}
    objects: dict[str, bytes] = {}
    calls: list[dict[str, object]] = []

    def create_branch(**kwargs: object) -> None:
        calls.append(kwargs)
        assert kwargs == {
            "repo_id": "owner/dataset",
            "repo_type": "dataset",
            "branch": "lerobot-sidecars",
            "revision": source_sha,
            "exist_ok": True,
        }
        branches.setdefault("lerobot-sidecars", source_sha)

    class RemoteFiles:
        def resolve_path(self, path: str) -> SimpleNamespace:
            assert path == spec.data_root
            return SimpleNamespace(repo_id="owner/dataset", revision=source_sha)

        def put(self, local: str, remote: str) -> None:
            assert remote.startswith("hf://datasets/owner/dataset@lerobot-sidecars/payload/meta/")
            assert "lerobot-sidecars" in branches
            objects[remote] = Path(local).read_bytes()
            branches["lerobot-sidecars"] = published_sha

        def exists(self, path: str) -> bool:
            return path in objects

        def open(self, path: str, mode: str) -> io.BytesIO:
            assert mode == "rb"
            return io.BytesIO(objects[path])

    remote = RemoteFiles()
    monkeypatch.setattr(
        "lerobot.scripts.lerobot_build_mp4_sidecar.HfApi",
        lambda: SimpleNamespace(create_branch=create_branch),
        raising=False,
    )
    monkeypatch.setattr("lerobot.scripts.lerobot_build_mp4_sidecar.fsspec.filesystem", lambda _: remote)
    monkeypatch.setattr(
        "lerobot.datasets.streaming_sidecar.fsspec.core.url_to_fs", lambda path, **_: (remote, path)
    )
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "mapped"))
    destinations = push_sidecar(str(source), spec)
    assert calls
    assert branches == {"main": source_sha, "lerobot-sidecars": published_sha}
    assert destinations == [published_sidecar_url(spec)]

    def unexpected_build(path: Path, expected: SidecarSpec) -> None:
        pytest.fail("A published sidecar must be reused without rebuilding")

    alias = replace(spec, revision="v3.0")
    result = ensure_mp4_sidecar(
        alias, tmp_path / "cache", build=unexpected_build, download=download_published_sidecar
    )
    assert EpisodeVideoManifest.validate_file_sidecar(result, alias)
    changed = replace(spec, revision="c" * 40, data_root=spec.data_root.replace(source_sha, "c" * 40))
    assert not download_published_sidecar(tmp_path / "missing.npz", changed)
    assert not EpisodeVideoManifest.validate_file_sidecar(result, changed)


def test_bucket_publication_does_not_create_a_git_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Buckets continue publishing directly under their content-keyed metadata path."""
    spec = SidecarSpec("owner/bucket", "v3.0", "hf://buckets/owner/bucket/prefix", (("v.mp4", 128),))
    writes = []
    monkeypatch.setattr(
        "lerobot.scripts.lerobot_build_mp4_sidecar.HfApi",
        lambda: pytest.fail("Buckets have no Git branches"),
        raising=False,
    )
    monkeypatch.setattr(
        "lerobot.scripts.lerobot_build_mp4_sidecar.fsspec.filesystem",
        lambda _: SimpleNamespace(put=lambda local, remote: writes.append((local, remote))),
    )
    destinations = push_sidecar(str(tmp_path / "sidecar.npz"), spec)
    assert destinations == [published_sidecar_url(spec)]
    assert writes[0][1].startswith("hf://buckets/owner/bucket/prefix/meta/mp4-sidecars/")


@pytest.mark.parametrize("suffix", ["", "@main", "@main/prefix@" + "a" * 40])
def test_repository_publication_rejects_unpinned_root(suffix: str) -> None:
    """Only a SHA in the repository revision position identifies immutable payloads."""
    spec = SidecarSpec("owner/data", "main", f"hf://datasets/owner/data{suffix}", ())
    with pytest.raises(ValueError, match="pinned source commit"):
        published_sidecar_url(spec)

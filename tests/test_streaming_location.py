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

from lerobot.streaming.location import LocationKind, StorageLocation, hf_bucket_uri, hf_dataset_uri

SHA = "a" * 40


@pytest.mark.parametrize(
    ("root", "kind", "repo_id", "revision", "path"),
    [
        ("data/local", LocationKind.LOCAL, None, None, ""),
        (Path("/abs/local"), LocationKind.LOCAL, None, None, ""),
        ("hf://buckets/owner/bucket", LocationKind.HF_BUCKET, "owner/bucket", None, ""),
        ("hf://buckets/owner/bucket/sub/dir/", LocationKind.HF_BUCKET, "owner/bucket", None, "sub/dir"),
        ("hf://datasets/owner/data", LocationKind.HF_DATASET, "owner/data", None, ""),
        ("hf://datasets/canonical@v1", LocationKind.HF_DATASET, "canonical", "v1", ""),
        (f"hf://datasets/owner/data@{SHA}/payload", LocationKind.HF_DATASET, "owner/data", SHA, "payload"),
        ("hf://owner/model", LocationKind.HF_OTHER, None, None, ""),
        ("s3://bucket/prefix", LocationKind.FSSPEC, None, None, ""),
        ("file:///abs/local", LocationKind.FSSPEC, None, None, ""),
    ],
)
def test_parse(root, kind, repo_id, revision, path) -> None:
    location = StorageLocation.parse(root)
    assert (location.kind, location.repo_id, location.revision, location.path_in_repo) == (
        kind,
        repo_id,
        revision,
        path,
    )
    assert location.is_remote == (kind is not LocationKind.LOCAL)
    assert location.range_backend == ("native-http" if str(root).startswith("hf://") else "fsspec")


@pytest.mark.parametrize("root", ["hf://buckets/owner", "hf://buckets//bucket"])
def test_parse_rejects_incomplete_bucket_roots(root: str) -> None:
    with pytest.raises(ValueError, match="Invalid"):
        StorageLocation.parse(root)


def test_storage_options_only_carry_the_token_to_hub_roots() -> None:
    assert StorageLocation.parse("hf://buckets/o/b").storage_options("tok") == {"token": "tok"}
    assert StorageLocation.parse("hf://buckets/o/b").storage_options(None) == {}
    assert StorageLocation.parse("s3://b/p").storage_options("tok") == {}
    assert StorageLocation.parse("local").storage_options("tok") == {}


@pytest.mark.parametrize(
    ("root", "commit"),
    [
        (f"hf://datasets/o/d@{SHA}", SHA),
        (f"hf://datasets/o/d@{SHA}/sub", SHA),
        ("hf://datasets/o/d@main", None),
        (f"hf://datasets/o/d@main/prefix@{SHA}", None),
        (f"hf://buckets/o/b@{SHA}", None),
    ],
)
def test_pinned_commit(root: str, commit: str | None) -> None:
    assert StorageLocation.parse(root).pinned_commit == commit


def test_pinned_resolves_dataset_branches_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lerobot.streaming.location.HfFileSystem",
        lambda token: SimpleNamespace(
            resolve_path=lambda uri: SimpleNamespace(repo_id="o/d", revision="main", path_in_repo="sub")
        ),
    )
    monkeypatch.setattr(
        "lerobot.streaming.location.HfApi",
        lambda token: SimpleNamespace(dataset_info=lambda repo_id, revision: SimpleNamespace(sha=SHA)),
    )
    assert StorageLocation.parse("hf://datasets/o/d@main/sub").pinned().uri == f"hf://datasets/o/d@{SHA}/sub"
    for unchanged in (f"hf://datasets/o/d@{SHA}", "hf://buckets/o/b", "local/dir", "s3://b/p"):
        assert StorageLocation.parse(unchanged).pinned().uri == unchanged


def test_uri_builders() -> None:
    assert hf_dataset_uri("o/d") == "hf://datasets/o/d"
    assert hf_dataset_uri("o/d", SHA) == f"hf://datasets/o/d@{SHA}"
    assert hf_bucket_uri("o/b") == "hf://buckets/o/b"
    assert hf_bucket_uri("o/b", "/meta/") == "hf://buckets/o/b/meta"

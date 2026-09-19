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

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.streaming_sidecar import range_backend_for_root, streaming_data_root


def test_hub_data_root_is_revision_qualified() -> None:
    meta = SimpleNamespace(repo_id="owner/dataset", revision="commit-sha")

    root = streaming_data_root(meta, requested_root=None, configured_data_root=None)

    assert root == "hf://datasets/owner/dataset@commit-sha"
    assert range_backend_for_root(root) == "native-http"


def test_explicit_bucket_root_is_preserved() -> None:
    meta = SimpleNamespace(repo_id="owner/dataset", revision="commit-sha")
    bucket = "hf://buckets/owner/dataset-bucket/prefix/"

    root = streaming_data_root(meta, requested_root=None, configured_data_root=bucket)

    assert root == bucket.rstrip("/")
    assert range_backend_for_root(root) == "native-http"


def test_local_and_generic_remote_roots_use_fsspec(tmp_path: Path) -> None:
    meta = SimpleNamespace(repo_id="owner/dataset", revision="commit-sha")

    local = streaming_data_root(meta, requested_root=tmp_path, configured_data_root=None)

    assert local == str(tmp_path)
    assert range_backend_for_root(local) == "fsspec"
    assert range_backend_for_root("memory://dataset") == "fsspec"

#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

# The script imports ``lerobot.datasets`` (via the annotation frame provider),
# which only ships under the ``dataset`` extra.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from unittest.mock import MagicMock, patch  # noqa: E402

from lerobot.scripts.lerobot_curate_cameras import (  # noqa: E402
    _discover_subpaths,
    _to_uint8_frame,
    _uniform_indices,
)


@pytest.mark.parametrize(
    "n,k,expected",
    [
        (0, 4, []),
        (5, 0, []),
        (3, 5, [0, 1, 2]),  # k >= n -> all frames
        (1, 4, [0]),
        (10, 1, [0]),
        (10, 4, [0, 3, 6, 9]),  # evenly spaced, endpoints included
    ],
)
def test_uniform_indices(n, k, expected):
    assert _uniform_indices(n, k) == expected


def test_to_uint8_frame_scales_floats():
    frame = torch.ones(3, 4, 4, dtype=torch.float32)  # [0,1] float
    out = _to_uint8_frame(frame)
    assert out.dtype == torch.uint8
    assert int(out.max()) == 255


def test_to_uint8_frame_passthrough_uint8():
    frame = torch.zeros(3, 4, 4, dtype=torch.uint8)
    out = _to_uint8_frame(frame)
    assert out is frame  # uint8 passes through untouched


def test_discover_subpaths_nested_and_single():
    api = MagicMock()
    # Nested collection: two sub-datasets, sorted, prefixes stripped.
    api.list_repo_files.return_value = [
        "b/y/meta/info.json",
        "a/x/meta/info.json",
        "a/x/videos/observation.images.cam/chunk-000/file-000.mp4",
        "README.md",
    ]
    with patch("huggingface_hub.HfApi", return_value=api):
        assert _discover_subpaths("repo") == ["a/x", "b/y"]

    # Standard single dataset (root meta/info.json) -> None.
    api.list_repo_files.return_value = ["meta/info.json", "data/chunk-000/file-000.parquet"]
    with patch("huggingface_hub.HfApi", return_value=api):
        assert _discover_subpaths("repo") is None

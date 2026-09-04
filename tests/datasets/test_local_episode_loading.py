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

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets import dataset_reader as dataset_reader_module
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_sidecar import ensure_dataset_mp4_sidecar, make_sidecar_spec
from tests.fixtures.constants import DUMMY_REPO_ID


def test_mutating_local_video_invalidates_sidecar_identity(tmp_path: Path) -> None:
    relative_path = Path("videos/camera/chunk-000/file-000.mp4")
    video_path = tmp_path / relative_path
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"first-payload")
    meta = SimpleNamespace(
        repo_id=DUMMY_REPO_ID,
        revision="v3.0",
        total_episodes=1,
        video_keys=["camera"],
        get_video_file_path=lambda _episode, _key: relative_path,
    )
    original_stat = video_path.stat()
    first = make_sidecar_spec(meta, str(tmp_path))

    video_path.write_bytes(b"other-payload")
    os.utime(video_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second = make_sidecar_spec(meta, str(tmp_path))

    assert first.source_files == second.source_files
    assert first.source_fingerprint != second.source_fingerprint


@pytest.mark.parametrize("video_backend", ["pyav", "torchcodec"])
def test_indexed_local_video_reader_matches_map_style(
    tmp_path: Path,
    lerobot_dataset_factory,
    monkeypatch,
    video_backend: str,
) -> None:
    root = tmp_path / "dataset"
    reference = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=20,
        video_backend=video_backend,
    )
    real_ensure = ensure_dataset_mp4_sidecar
    monkeypatch.setattr(
        dataset_reader_module,
        "ensure_dataset_mp4_sidecar",
        lambda meta, data_root, **kwargs: real_ensure(
            meta, data_root, cache_root=tmp_path / "sidecars", **kwargs
        ),
    )
    indexed = LeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        episodes=[1],
        video_backend=video_backend,
        local_episode_loading=True,
    )

    for relative_index in (0, len(indexed) - 1):
        actual = indexed[relative_index]
        expected = reference[int(actual["index"])]
        for key in indexed.meta.camera_keys:
            assert torch.equal(actual[key], expected[key]), key
        assert actual["task"] == expected["task"]

    assert indexed.reader.local_video_cache is not None
    indexed.close()
    assert indexed.reader.local_video_cache is None

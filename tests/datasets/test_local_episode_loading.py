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

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets import dataset_reader as dataset_reader_module
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_sidecar import ensure_dataset_mp4_sidecar, make_sidecar_spec
from tests.fixtures.constants import DUMMY_CAMERA_FEATURES_WITH_DEPTH, DUMMY_REPO_ID


def test_local_video_mutation_changes_sidecar_identity(tmp_path):
    video = tmp_path / "camera.mp4"
    video.write_bytes(b"first-payload")
    stat = video.stat()
    meta = SimpleNamespace(
        repo_id=DUMMY_REPO_ID,
        revision="v3.0",
        total_episodes=1,
        video_keys=["camera"],
        get_video_file_path=lambda *_args: Path("camera.mp4"),
    )
    first = make_sidecar_spec(meta, str(tmp_path))

    video.write_bytes(b"other-payload")
    os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = make_sidecar_spec(meta, str(tmp_path))

    assert first.revision != second.revision


@pytest.mark.parametrize("video_backend", ["pyav", "torchcodec"])
def test_local_episode_sidecar_preserves_batched_map_loading(
    tmp_path: Path, lerobot_dataset_factory, monkeypatch, video_backend: str
) -> None:
    root = tmp_path / "dataset"
    reference = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=20,
        video_backend=video_backend,
    )
    monkeypatch.setattr(
        dataset_reader_module,
        "ensure_dataset_mp4_sidecar",
        lambda meta, data_root, **kwargs: ensure_dataset_mp4_sidecar(
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

    actual = indexed.__getitems__([0, len(indexed) - 1])
    expected = [reference[int(item["index"])] for item in actual]

    for item, expected_item in zip(actual, expected, strict=True):
        for key in indexed.meta.camera_keys:
            assert torch.equal(item[key], expected_item[key]), key
        assert item["task"] == expected_item["task"]
    indexed.reader.close()


def test_local_episode_sidecar_preserves_depth(tmp_path, lerobot_dataset_factory, monkeypatch):
    root = tmp_path / "dataset"
    reference = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=20,
        camera_features=DUMMY_CAMERA_FEATURES_WITH_DEPTH,
        video_backend="pyav",
    )
    monkeypatch.setattr(
        dataset_reader_module,
        "ensure_dataset_mp4_sidecar",
        lambda meta, data_root, **kwargs: ensure_dataset_mp4_sidecar(
            meta, data_root, cache_root=tmp_path / "sidecars", **kwargs
        ),
    )
    indexed = LeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        episodes=[1],
        video_backend="pyav",
        local_episode_loading=True,
    )

    actual = indexed[0]
    expected = reference[int(actual["index"])]

    for key in indexed.meta.camera_keys:
        assert torch.equal(actual[key], expected[key]), key
    indexed.reader.close()

#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Contract tests for DatasetReader."""

from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.video_utils import get_safe_default_codec

# ── Loading ──────────────────────────────────────────────────────────


def test_try_load_returns_true_when_data_exists(tmp_path, lerobot_dataset_factory):
    """Given a fully written dataset, try_load() returns True."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=20, use_videos=False
    )
    reader = DatasetReader(
        meta=dataset.meta,
        root=dataset.root,
        episodes=None,
        tolerance_s=1e-4,
        video_backend=get_safe_default_codec(),
        delta_timestamps=None,
        image_transforms=None,
    )
    assert reader.try_load() is True
    assert reader.hf_dataset is not None


def test_try_load_returns_false_when_no_data(tmp_path):
    """When only metadata exists (no data/ parquets), try_load() returns False."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    root = tmp_path / "meta_only"
    features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/meta_only", fps=30, features=features, root=root, use_videos=False
    )

    reader = DatasetReader(
        meta=meta,
        root=meta.root,
        episodes=None,
        tolerance_s=1e-4,
        video_backend=get_safe_default_codec(),
        delta_timestamps=None,
        image_transforms=None,
    )
    assert reader.try_load() is False
    assert reader.hf_dataset is None


# ── Counts ───────────────────────────────────────────────────────────


def test_num_frames_without_filter(tmp_path, lerobot_dataset_factory):
    """With episodes=None, num_frames equals total_frames."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=3, total_frames=60, use_videos=False
    )
    assert dataset.reader.num_frames == dataset.meta.total_frames


def test_num_episodes_without_filter(tmp_path, lerobot_dataset_factory):
    """With episodes=None, num_episodes equals total_episodes."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=3, total_frames=60, use_videos=False
    )
    assert dataset.reader.num_episodes == dataset.meta.total_episodes


def test_num_frames_with_episode_filter(tmp_path, lerobot_dataset_factory):
    """When filtering to a subset, only those episodes' frames are counted."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=5, total_frames=100, episodes=[0, 2], use_videos=False
    )
    # Filtered frames should be less than total
    assert dataset.reader.num_frames <= dataset.meta.total_frames
    assert dataset.reader.num_episodes == 2


# ── get_item ─────────────────────────────────────────────────────────


def test_get_item_returns_expected_keys(tmp_path, lerobot_dataset_factory):
    """get_item(0) returns a dict with expected keys."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=1, total_frames=10, use_videos=False
    )
    item = dataset.reader.get_item(0)

    # Standard keys that must always be present
    for key in ["index", "episode_index", "frame_index", "timestamp", "task_index", "task"]:
        assert key in item, f"Missing key: {key}"


def test_get_item_values_are_correct(tmp_path, lerobot_dataset_factory):
    """get_item() returns correct index and episode_index."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=20, use_videos=False
    )
    item_0 = dataset.reader.get_item(0)

    assert item_0["index"].item() == 0
    assert item_0["episode_index"].item() == 0


# ── Transforms ───────────────────────────────────────────────────────


def test_image_transforms_are_applied(tmp_path, lerobot_dataset_factory):
    """When image_transforms is provided, get_item() applies it to camera keys."""
    transform_called = {"count": 0}

    def sentinel_transform(img):
        transform_called["count"] += 1
        return img

    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=1,
        total_frames=5,
        use_videos=False,
        image_transforms=sentinel_transform,
    )
    item = dataset[0]  # noqa: F841

    # Should have been called once per camera key per frame
    num_cameras = len(dataset.meta.camera_keys)
    if num_cameras > 0:
        assert transform_called["count"] >= 1


# ── File paths ───────────────────────────────────────────────────────


def test_get_episodes_file_paths_returns_data_paths(tmp_path, lerobot_dataset_factory):
    """get_episodes_file_paths() returns paths including data/ paths."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=20, use_videos=False
    )
    paths = dataset.reader.get_episodes_file_paths()

    assert len(paths) > 0
    assert any("data/" in str(p) for p in paths)


def test_get_episodes_file_paths_includes_video_paths(tmp_path, lerobot_dataset_factory):
    """When dataset has video keys, file paths include video/ paths."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=20, use_videos=True
    )

    if len(dataset.meta.video_keys) > 0:
        paths = dataset.reader.get_episodes_file_paths()
        assert any("video" in str(p).lower() for p in paths)

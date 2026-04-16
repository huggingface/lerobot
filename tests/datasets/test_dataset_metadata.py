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
"""Contract tests for LeRobotDatasetMetadata."""

import json

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.utils import INFO_PATH
from tests.fixtures.constants import DEFAULT_FPS, DUMMY_ROBOT_TYPE

# ── helpers ──────────────────────────────────────────────────────────

SIMPLE_FEATURES = {
    "state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
}

VIDEO_FEATURES = {
    **SIMPLE_FEATURES,
    "observation.images.laptop": {
        "dtype": "video",
        "shape": (64, 96, 3),
        "names": ["height", "width", "channels"],
        "info": None,
    },
}

IMAGE_FEATURES = {
    **SIMPLE_FEATURES,
    "observation.images.laptop": {
        "dtype": "image",
        "shape": (64, 96, 3),
        "names": ["height", "width", "channels"],
        "info": None,
    },
}


def _make_dummy_stats(features: dict) -> dict:
    """Create minimal episode stats matching the given features."""
    stats = {}
    for key, ft in features.items():
        if ft["dtype"] in ("image", "video"):
            stats[key] = {
                "max": np.ones((3, 1, 1), dtype=np.float32),
                "mean": np.full((3, 1, 1), 0.5, dtype=np.float32),
                "min": np.zeros((3, 1, 1), dtype=np.float32),
                "std": np.full((3, 1, 1), 0.25, dtype=np.float32),
                "count": np.array([5]),
            }
        elif ft["dtype"] in ("float32", "float64", "int64"):
            stats[key] = {
                "max": np.ones(ft["shape"], dtype=np.float32),
                "mean": np.full(ft["shape"], 0.5, dtype=np.float32),
                "min": np.zeros(ft["shape"], dtype=np.float32),
                "std": np.full(ft["shape"], 0.25, dtype=np.float32),
                "count": np.array([5]),
            }
    return stats


# ── Construction contracts ───────────────────────────────────────────


def test_create_produces_valid_info_on_disk(tmp_path):
    """create() writes info.json and the returned object reflects the provided settings."""
    root = tmp_path / "new_ds"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/meta",
        fps=DEFAULT_FPS,
        features=SIMPLE_FEATURES,
        robot_type=DUMMY_ROBOT_TYPE,
        root=root,
        use_videos=False,
    )

    # info.json was written to disk
    assert (root / INFO_PATH).exists()
    with open(root / INFO_PATH) as f:
        info_on_disk = json.load(f)

    assert meta.fps == DEFAULT_FPS
    assert meta.robot_type == DUMMY_ROBOT_TYPE
    assert "state" in meta.features
    assert "action" in meta.features
    assert info_on_disk["fps"] == DEFAULT_FPS


def test_create_starts_with_zero_counts(tmp_path):
    """A freshly created metadata has zero episode/frame/task counts."""
    root = tmp_path / "empty_ds"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/empty", fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root, use_videos=False
    )

    assert meta.total_episodes == 0
    assert meta.total_frames == 0
    assert meta.total_tasks == 0
    assert meta.tasks is None
    assert meta.episodes is None
    assert meta.stats is None


def test_create_with_videos_sets_video_path(tmp_path):
    """When features include video-dtype keys, create() produces a non-None video_path."""
    root = tmp_path / "video_ds"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/video", fps=DEFAULT_FPS, features=VIDEO_FEATURES, root=root, use_videos=True
    )

    assert meta.video_path is not None
    assert len(meta.video_keys) == 1
    assert "observation.images.laptop" in meta.video_keys


def test_create_without_videos_has_no_video_path(tmp_path):
    """When use_videos=False and no video features, video_path is None."""
    root = tmp_path / "no_video"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/novid", fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root, use_videos=False
    )

    assert meta.video_path is None
    assert meta.video_keys == []


def test_create_raises_on_existing_directory(tmp_path):
    """create() raises if root directory already exists."""
    root = tmp_path / "existing"
    root.mkdir()

    with pytest.raises(FileExistsError):
        LeRobotDatasetMetadata.create(
            repo_id="test/exists", fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root, use_videos=False
        )


def test_init_loads_existing_metadata(tmp_path, lerobot_dataset_metadata_factory, info_factory):
    """When metadata files exist on disk, __init__ loads them correctly."""
    root = tmp_path / "load_test"
    info = info_factory(total_episodes=3, total_frames=150, total_tasks=1, use_videos=False)
    meta = lerobot_dataset_metadata_factory(root=root, info=info)

    assert meta.total_episodes == 3
    assert meta.total_frames == 150
    assert meta.fps == info["fps"]


# ── Property accessors ───────────────────────────────────────────────


def test_property_accessors_reflect_info(tmp_path):
    """Properties return values consistent with the info dict."""
    root = tmp_path / "props_ds"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/props",
        fps=DEFAULT_FPS,
        features=IMAGE_FEATURES,
        robot_type=DUMMY_ROBOT_TYPE,
        root=root,
        use_videos=False,
    )

    assert meta.fps == DEFAULT_FPS
    assert meta.robot_type == DUMMY_ROBOT_TYPE
    # shapes should be tuples
    for _key, shape in meta.shapes.items():
        assert isinstance(shape, tuple)
    # image_keys should contain the image feature
    assert "observation.images.laptop" in meta.image_keys
    # camera_keys is a superset of image_keys and video_keys
    assert set(meta.image_keys + meta.video_keys) == set(meta.camera_keys)


def test_data_path_is_formattable(tmp_path):
    """data_path contains format placeholders that can be .format()-ed."""
    root = tmp_path / "fmt_ds"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/fmt", fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root, use_videos=False
    )

    formatted = meta.data_path.format(chunk_index=0, file_index=0)
    assert "chunk" in formatted.lower() or "0" in formatted


# ── Task management ──────────────────────────────────────────────────


def test_save_episode_tasks_creates_tasks_dataframe(tmp_path):
    """On a fresh metadata, save_episode_tasks() creates the tasks DataFrame."""
    root = tmp_path / "task_ds"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/task", fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root, use_videos=False
    )
    assert meta.tasks is None

    meta.save_episode_tasks(["Pick up the cube"])

    assert meta.tasks is not None
    assert len(meta.tasks) == 1
    assert "Pick up the cube" in meta.tasks.index


def test_save_episode_tasks_is_additive(tmp_path):
    """New tasks are added; existing tasks keep their original index."""
    root = tmp_path / "additive_ds"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/add", fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root, use_videos=False
    )

    meta.save_episode_tasks(["Task A"])
    idx_a = meta.get_task_index("Task A")

    meta.save_episode_tasks(["Task A", "Task B"])
    assert meta.get_task_index("Task A") == idx_a  # unchanged
    assert meta.get_task_index("Task B") is not None
    assert len(meta.tasks) == 2


def test_get_task_index_returns_none_for_unknown(tmp_path):
    """get_task_index() returns None for an unknown task."""
    root = tmp_path / "unknown_ds"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/unknown", fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root, use_videos=False
    )
    meta.save_episode_tasks(["Known task"])

    assert meta.get_task_index("Known task") == 0
    assert meta.get_task_index("Unknown task") is None


def test_save_episode_tasks_rejects_duplicates(tmp_path):
    """save_episode_tasks() raises ValueError on duplicate task strings."""
    root = tmp_path / "dup_ds"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/dup", fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root, use_videos=False
    )

    with pytest.raises(ValueError):
        meta.save_episode_tasks(["Same task", "Same task"])


# ── Episode saving ───────────────────────────────────────────────────


def test_save_episode_increments_counters(tmp_path):
    """After save_episode(), total_episodes and total_frames increase."""
    root = tmp_path / "ep_ds"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/ep", fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root, use_videos=False
    )
    meta.save_episode_tasks(["Task 1"])
    stats = _make_dummy_stats(meta.features)

    meta.save_episode(
        episode_index=0,
        episode_length=10,
        episode_tasks=["Task 1"],
        episode_stats=stats,
        episode_metadata={},
    )

    assert meta.total_episodes == 1
    assert meta.total_frames == 10


def test_save_episode_updates_stats(tmp_path):
    """After save_episode(), .stats is non-None and has feature keys."""
    root = tmp_path / "stats_ds"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/stats", fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root, use_videos=False
    )
    meta.save_episode_tasks(["Task 1"])
    stats = _make_dummy_stats(meta.features)

    meta.save_episode(
        episode_index=0,
        episode_length=5,
        episode_tasks=["Task 1"],
        episode_stats=stats,
        episode_metadata={},
    )

    assert meta.stats is not None
    # Stats should contain at least the user-defined feature keys
    for key in SIMPLE_FEATURES:
        assert key in meta.stats


# ── Chunk settings ───────────────────────────────────────────────────


def test_update_chunk_settings_persists(tmp_path):
    """update_chunk_settings() changes values and writes info.json."""
    root = tmp_path / "chunk_ds"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/chunk", fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root, use_videos=False
    )
    original = meta.get_chunk_settings()

    meta.update_chunk_settings(chunks_size=500)
    assert meta.chunks_size == 500
    assert meta.chunks_size != original["chunks_size"] or original["chunks_size"] == 500

    # Verify persisted
    with open(root / INFO_PATH) as f:
        info_on_disk = json.load(f)
    assert info_on_disk["chunks_size"] == 500


def test_update_chunk_settings_rejects_non_positive(tmp_path):
    """update_chunk_settings() raises ValueError for <= 0 values."""
    root = tmp_path / "bad_chunk"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/bad", fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root, use_videos=False
    )

    with pytest.raises(ValueError):
        meta.update_chunk_settings(chunks_size=0)
    with pytest.raises(ValueError):
        meta.update_chunk_settings(data_files_size_in_mb=-1)


# ── Finalization ─────────────────────────────────────────────────────


def test_finalize_is_idempotent(tmp_path):
    """Calling finalize() multiple times does not raise."""
    root = tmp_path / "fin_ds"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/fin", fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root, use_videos=False
    )

    meta.finalize()
    meta.finalize()  # second call should not raise


def test_finalize_flushes_buffered_metadata(tmp_path):
    """Episodes saved before finalize() are written to parquet."""
    root = tmp_path / "flush_ds"
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/flush",
        fps=DEFAULT_FPS,
        features=SIMPLE_FEATURES,
        root=root,
        use_videos=False,
        metadata_buffer_size=100,  # large buffer so nothing auto-flushes
    )
    meta.save_episode_tasks(["Task 1"])
    stats = _make_dummy_stats(meta.features)

    # Save a few episodes (won't auto-flush since buffer_size=100)
    for i in range(3):
        meta.save_episode(
            episode_index=i,
            episode_length=5,
            episode_tasks=["Task 1"],
            episode_stats=stats,
            episode_metadata={},
        )

    # Before finalize, the parquet might not exist yet
    meta.finalize()

    # After finalize, episodes parquet should exist
    episodes_dir = root / "meta" / "episodes"
    assert episodes_dir.exists()
    parquet_files = list(episodes_dir.rglob("*.parquet"))
    assert len(parquet_files) > 0

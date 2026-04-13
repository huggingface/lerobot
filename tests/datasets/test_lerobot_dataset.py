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
"""Contract tests for the LeRobotDataset facade.

Tests focus on mode contracts (read-only, write-only, resume), guards,
property delegation, and the full create-record-finalize-read lifecycle.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import lerobot.datasets.dataset_metadata as dataset_metadata_module
import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.dataset_writer import DatasetWriter
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tests.fixtures.constants import DEFAULT_FPS, DUMMY_REPO_ID

SIMPLE_FEATURES = {
    "state": {"dtype": "float32", "shape": (2,), "names": None},
}
SNAPSHOT_MAIN_FEATURES = {
    **SIMPLE_FEATURES,
    "test": {"dtype": "float32", "shape": (2,), "names": None},
}


def _make_frame(task: str = "Dummy task") -> dict:
    return {"task": task, "state": torch.randn(2)}


def _set_default_cache_root(monkeypatch: pytest.MonkeyPatch, cache_root: Path) -> None:
    monkeypatch.setattr(dataset_metadata_module, "HF_LEROBOT_HOME", cache_root)
    monkeypatch.setattr(dataset_metadata_module, "HF_LEROBOT_HUB_CACHE", cache_root / "hub")
    monkeypatch.setattr(lerobot_dataset_module, "HF_LEROBOT_HUB_CACHE", cache_root / "hub")


def _write_dataset_tree(
    root: Path,
    *,
    motor_features: dict[str, dict],
    info_factory,
    stats_factory,
    tasks_factory,
    episodes_factory,
    hf_dataset_factory,
    create_info,
    create_stats,
    create_tasks,
    create_episodes,
    create_hf_dataset,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    info = info_factory(
        total_episodes=1,
        total_frames=3,
        total_tasks=1,
        use_videos=False,
        motor_features=motor_features,
        camera_features={},
    )
    tasks = tasks_factory(total_tasks=1)
    episodes = episodes_factory(
        features=info["features"],
        fps=info["fps"],
        total_episodes=1,
        total_frames=3,
        tasks=tasks,
    )
    stats = stats_factory(features=info["features"])
    hf_dataset = hf_dataset_factory(
        features=info["features"],
        tasks=tasks,
        episodes=episodes,
        fps=info["fps"],
    )

    create_info(root, info)
    create_stats(root, stats)
    create_tasks(root, tasks)
    create_episodes(root, episodes)
    create_hf_dataset(root, hf_dataset)


# ── Read-only mode (via __init__) ────────────────────────────────────


def test_init_creates_reader_no_writer(tmp_path, lerobot_dataset_factory):
    """__init__() sets reader to a DatasetReader and writer to None."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=1, total_frames=10, use_videos=False
    )
    assert isinstance(dataset.reader, DatasetReader)
    assert dataset.writer is None


def test_init_loads_data(tmp_path, lerobot_dataset_factory):
    """After __init__(), the dataset has data and len > 0."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=1, total_frames=10, use_videos=False
    )
    assert len(dataset) > 0


def test_getitem_works_in_read_mode(tmp_path, lerobot_dataset_factory):
    """dataset[0] returns a dict with expected keys in read-only mode."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=1, total_frames=10, use_videos=False
    )
    item = dataset[0]
    assert isinstance(item, dict)
    assert "index" in item
    assert "task" in item


def test_len_matches_num_frames(tmp_path, lerobot_dataset_factory):
    """len(dataset) equals dataset.num_frames."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=30, use_videos=False
    )
    assert len(dataset) == dataset.num_frames


def test_metadata_without_root_uses_hub_cache_snapshot_download(
    tmp_path,
    info_factory,
    stats_factory,
    tasks_factory,
    episodes_factory,
    hf_dataset_factory,
    create_info,
    create_stats,
    create_tasks,
    create_episodes,
    create_hf_dataset,
    monkeypatch,
):
    """Metadata refresh uses the dedicated Hub cache instead of a shared local_dir mirror."""
    repo_id = DUMMY_REPO_ID
    cache_root = tmp_path / "lerobot_cache"
    snapshot_root = cache_root / "hub" / "datasets--dummy--repo" / "snapshots" / "commit-main"
    _write_dataset_tree(
        snapshot_root,
        motor_features=SNAPSHOT_MAIN_FEATURES,
        info_factory=info_factory,
        stats_factory=stats_factory,
        tasks_factory=tasks_factory,
        episodes_factory=episodes_factory,
        hf_dataset_factory=hf_dataset_factory,
        create_info=create_info,
        create_stats=create_stats,
        create_tasks=create_tasks,
        create_episodes=create_episodes,
        create_hf_dataset=create_hf_dataset,
    )

    _set_default_cache_root(monkeypatch, cache_root)
    snapshot_download = Mock(return_value=str(snapshot_root))
    monkeypatch.setattr(dataset_metadata_module, "snapshot_download", snapshot_download)

    meta = LeRobotDatasetMetadata(repo_id=repo_id, revision="main", force_cache_sync=True)

    assert meta.root == snapshot_root
    assert snapshot_download.call_count == 1
    assert snapshot_download.call_args.args == (repo_id,)
    assert snapshot_download.call_args.kwargs == {
        "repo_type": "dataset",
        "revision": "main",
        "cache_dir": cache_root / "hub",
        "allow_patterns": "meta/",
        "ignore_patterns": None,
    }


def test_without_root_reads_different_revisions_from_distinct_snapshot_roots(
    tmp_path,
    info_factory,
    stats_factory,
    tasks_factory,
    episodes_factory,
    hf_dataset_factory,
    create_info,
    create_stats,
    create_tasks,
    create_episodes,
    create_hf_dataset,
    monkeypatch,
):
    """Different revisions resolve to different on-disk snapshot roots."""
    repo_id = DUMMY_REPO_ID
    old_revision = "b59010db93eb6cc3cf06ef2f7cae1bbe62b726d9"
    cache_root = tmp_path / "lerobot_cache"
    main_root = cache_root / "hub" / "datasets--dummy--repo" / "snapshots" / "commit-main"
    old_root = cache_root / "hub" / "datasets--dummy--repo" / "snapshots" / "commit-old"

    _write_dataset_tree(
        main_root,
        motor_features=SNAPSHOT_MAIN_FEATURES,
        info_factory=info_factory,
        stats_factory=stats_factory,
        tasks_factory=tasks_factory,
        episodes_factory=episodes_factory,
        hf_dataset_factory=hf_dataset_factory,
        create_info=create_info,
        create_stats=create_stats,
        create_tasks=create_tasks,
        create_episodes=create_episodes,
        create_hf_dataset=create_hf_dataset,
    )
    _write_dataset_tree(
        old_root,
        motor_features=SIMPLE_FEATURES,
        info_factory=info_factory,
        stats_factory=stats_factory,
        tasks_factory=tasks_factory,
        episodes_factory=episodes_factory,
        hf_dataset_factory=hf_dataset_factory,
        create_info=create_info,
        create_stats=create_stats,
        create_tasks=create_tasks,
        create_episodes=create_episodes,
        create_hf_dataset=create_hf_dataset,
    )

    _set_default_cache_root(monkeypatch, cache_root)
    snapshot_roots = {
        "main": main_root,
        old_revision: old_root,
    }
    meta_snapshot_download = Mock(
        side_effect=lambda repo_id, **kwargs: str(snapshot_roots[kwargs["revision"]])
    )
    data_snapshot_download = Mock(
        side_effect=lambda repo_id, **kwargs: str(snapshot_roots[kwargs["revision"]])
    )
    monkeypatch.setattr(dataset_metadata_module, "snapshot_download", meta_snapshot_download)
    monkeypatch.setattr(lerobot_dataset_module, "snapshot_download", data_snapshot_download)

    main_dataset = LeRobotDataset(
        repo_id=repo_id, revision="main", download_videos=False, force_cache_sync=True
    )
    old_dataset = LeRobotDataset(
        repo_id=repo_id, revision=old_revision, download_videos=False, force_cache_sync=True
    )

    assert main_dataset.root == main_root
    assert old_dataset.root == old_root
    assert "test" in main_dataset.hf_dataset.column_names
    assert "test" not in old_dataset.hf_dataset.column_names

    # Metadata downloads use cache_dir, not local_dir
    assert meta_snapshot_download.call_count == 2
    for download_call in meta_snapshot_download.call_args_list:
        assert download_call.kwargs["cache_dir"] == cache_root / "hub"
        assert "local_dir" not in download_call.kwargs

    # Data downloads also use cache_dir, not local_dir
    assert data_snapshot_download.call_count == 2
    for download_call in data_snapshot_download.call_args_list:
        assert download_call.kwargs["cache_dir"] == cache_root / "hub"
        assert "local_dir" not in download_call.kwargs


def test_metadata_without_root_ignores_legacy_local_dir_cache(
    tmp_path,
    info_factory,
    stats_factory,
    tasks_factory,
    episodes_factory,
    hf_dataset_factory,
    create_info,
    create_stats,
    create_tasks,
    create_episodes,
    create_hf_dataset,
    monkeypatch,
):
    """Legacy local-dir mirrors are bypassed in favor of revision-safe snapshots."""
    repo_id = DUMMY_REPO_ID
    cache_root = tmp_path / "lerobot_cache"
    legacy_root = cache_root / repo_id
    snapshot_root = cache_root / "hub" / "datasets--dummy--repo" / "snapshots" / "commit-main"

    _write_dataset_tree(
        legacy_root,
        motor_features=SIMPLE_FEATURES,
        info_factory=info_factory,
        stats_factory=stats_factory,
        tasks_factory=tasks_factory,
        episodes_factory=episodes_factory,
        hf_dataset_factory=hf_dataset_factory,
        create_info=create_info,
        create_stats=create_stats,
        create_tasks=create_tasks,
        create_episodes=create_episodes,
        create_hf_dataset=create_hf_dataset,
    )
    (legacy_root / ".cache" / "huggingface" / "download").mkdir(parents=True, exist_ok=True)
    _write_dataset_tree(
        snapshot_root,
        motor_features=SNAPSHOT_MAIN_FEATURES,
        info_factory=info_factory,
        stats_factory=stats_factory,
        tasks_factory=tasks_factory,
        episodes_factory=episodes_factory,
        hf_dataset_factory=hf_dataset_factory,
        create_info=create_info,
        create_stats=create_stats,
        create_tasks=create_tasks,
        create_episodes=create_episodes,
        create_hf_dataset=create_hf_dataset,
    )

    _set_default_cache_root(monkeypatch, cache_root)
    snapshot_download = Mock(return_value=str(snapshot_root))
    monkeypatch.setattr(dataset_metadata_module, "snapshot_download", snapshot_download)

    meta = LeRobotDatasetMetadata(repo_id=repo_id, revision="main")

    assert meta.root == snapshot_root
    assert "test" in meta.features
    assert snapshot_download.call_count == 1


def test_download_without_root_uses_hub_cache(
    tmp_path,
    info_factory,
    stats_factory,
    tasks_factory,
    episodes_factory,
    hf_dataset_factory,
    create_info,
    create_stats,
    create_tasks,
    create_episodes,
    create_hf_dataset,
    monkeypatch,
):
    """LeRobotDataset._download() uses cache_dir (not local_dir) when root is not provided."""
    repo_id = DUMMY_REPO_ID
    cache_root = tmp_path / "lerobot_cache"
    snapshot_root = cache_root / "hub" / "datasets--dummy--repo" / "snapshots" / "commit-main"

    # Pre-populate snapshot directory so metadata loads succeed, but leave
    # data absent so that _download() is triggered.
    _write_dataset_tree(
        snapshot_root,
        motor_features=SIMPLE_FEATURES,
        info_factory=info_factory,
        stats_factory=stats_factory,
        tasks_factory=tasks_factory,
        episodes_factory=episodes_factory,
        hf_dataset_factory=hf_dataset_factory,
        create_info=create_info,
        create_stats=create_stats,
        create_tasks=create_tasks,
        create_episodes=create_episodes,
        create_hf_dataset=create_hf_dataset,
    )

    _set_default_cache_root(monkeypatch, cache_root)
    meta_snapshot_download = Mock(return_value=str(snapshot_root))
    monkeypatch.setattr(dataset_metadata_module, "snapshot_download", meta_snapshot_download)

    # Mock the data snapshot_download to return the same root (data already
    # exists there from _write_dataset_tree).
    data_snapshot_download = Mock(return_value=str(snapshot_root))
    monkeypatch.setattr(lerobot_dataset_module, "snapshot_download", data_snapshot_download)

    LeRobotDataset(repo_id=repo_id, revision="main", force_cache_sync=True)

    # _download() should have called snapshot_download with cache_dir
    assert data_snapshot_download.call_count == 1
    call_kwargs = data_snapshot_download.call_args.kwargs
    assert call_kwargs["cache_dir"] == cache_root / "hub"
    assert "local_dir" not in call_kwargs


# ── Write-only mode (via create()) ──────────────────────────────────


def test_create_sets_writer_no_reader(tmp_path):
    """create() sets writer to a DatasetWriter and reader to None."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    assert isinstance(dataset.writer, DatasetWriter)
    assert dataset.reader is None


def test_create_initial_counts_zero(tmp_path):
    """After create(), num_episodes == 0 and num_frames == 0."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    assert dataset.num_episodes == 0
    assert dataset.num_frames == 0


def test_add_frame_works_in_write_mode(tmp_path):
    """add_frame() succeeds on a dataset created via create()."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    dataset.add_frame(_make_frame())  # should not raise


# ── Resume mode ──────────────────────────────────────────────────────


def test_resume_creates_writer(tmp_path):
    """After resume(), writer is a DatasetWriter."""
    root = tmp_path / "resume_ds"
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root
    )
    for _ in range(3):
        dataset.add_frame(_make_frame())
    dataset.save_episode()
    dataset.finalize()

    resumed = LeRobotDataset.resume(repo_id=DUMMY_REPO_ID, root=root)
    assert isinstance(resumed.writer, DatasetWriter)


def test_resume_preserves_episode_count(tmp_path):
    """After resume(), existing episodes are counted."""
    root = tmp_path / "resume_ds"
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root
    )
    for _ in range(3):
        dataset.add_frame(_make_frame())
    dataset.save_episode()
    dataset.finalize()

    resumed = LeRobotDataset.resume(repo_id=DUMMY_REPO_ID, root=root)
    assert resumed.meta.total_episodes == 1


def test_resume_can_add_more_episodes(tmp_path):
    """After resume(), new episodes can be added."""
    root = tmp_path / "resume_ds"
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root
    )
    for _ in range(3):
        dataset.add_frame(_make_frame())
    dataset.save_episode()
    dataset.finalize()

    resumed = LeRobotDataset.resume(repo_id=DUMMY_REPO_ID, root=root)
    for _ in range(2):
        resumed.add_frame(_make_frame())
    resumed.save_episode()

    assert resumed.meta.total_episodes == 2


# ── Writer guard ─────────────────────────────────────────────────────


def test_add_frame_raises_without_writer(tmp_path, lerobot_dataset_factory):
    """add_frame() raises RuntimeError on a read-only dataset."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=1, total_frames=5, use_videos=False
    )
    with pytest.raises(RuntimeError, match="read-only"):
        dataset.add_frame(_make_frame())


def test_save_episode_raises_without_writer(tmp_path, lerobot_dataset_factory):
    """save_episode() raises RuntimeError on a read-only dataset."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=1, total_frames=5, use_videos=False
    )
    with pytest.raises(RuntimeError, match="read-only"):
        dataset.save_episode()


def test_clear_episode_buffer_raises_without_writer(tmp_path, lerobot_dataset_factory):
    """clear_episode_buffer() raises RuntimeError on a read-only dataset."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=1, total_frames=5, use_videos=False
    )
    with pytest.raises(RuntimeError, match="read-only"):
        dataset.clear_episode_buffer()


# ── Reader guard ─────────────────────────────────────────────────────


def test_getitem_raises_before_finalize(tmp_path):
    """dataset[0] raises RuntimeError while recording (before finalize)."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(3):
        dataset.add_frame(_make_frame())
    dataset.save_episode()

    with pytest.raises(RuntimeError, match="finalize"):
        dataset[0]


def test_getitem_works_after_finalize(tmp_path):
    """After finalize(), dataset[0] returns data."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(3):
        dataset.add_frame(_make_frame())
    dataset.save_episode()
    dataset.finalize()

    item = dataset[0]
    assert "state" in item
    assert "task" in item


def test_getitem_after_finalize_with_delta_timestamps(tmp_path):
    """After finalize(), dataset[0] works when delta_timestamps require episode metadata.

    Regression test for https://github.com/huggingface/lerobot/pull/3305.
    The create -> write -> finalize -> read path left meta.episodes as None
    because the write path flushes episodes to disk without updating them
    in memory.  Features that access meta.episodes (video decoding,
    delta_timestamps) would crash with a TypeError.
    """
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(5):
        dataset.add_frame(_make_frame())
    dataset.save_episode()
    dataset.finalize()

    # Set delta_timestamps so get_item() accesses meta.episodes via _get_query_indices
    dataset.delta_timestamps = {"state": [0.0]}

    item = dataset[0]
    assert "state" in item
    assert "state_is_pad" in item


# ── Property delegation ──────────────────────────────────────────────


def test_fps_delegates_to_meta(tmp_path, lerobot_dataset_factory):
    """dataset.fps == dataset.meta.fps."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=1, total_frames=5, use_videos=False
    )
    assert dataset.fps == dataset.meta.fps


def test_features_delegates_to_meta(tmp_path, lerobot_dataset_factory):
    """dataset.features is dataset.meta.features."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=1, total_frames=5, use_videos=False
    )
    assert dataset.features is dataset.meta.features


def test_num_frames_uses_meta_in_write_mode(tmp_path):
    """In write-only mode (reader=None), num_frames comes from metadata."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    assert dataset.reader is None
    assert dataset.num_frames == dataset.meta.total_frames


# ── Lifecycle ────────────────────────────────────────────────────────


def test_finalize_is_idempotent(tmp_path):
    """Calling finalize() twice does not raise."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    dataset.finalize()
    dataset.finalize()


def test_has_pending_frames_lifecycle(tmp_path):
    """has_pending_frames: False -> True (add_frame) -> False (save_episode)."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    assert dataset.has_pending_frames() is False

    dataset.add_frame(_make_frame())
    assert dataset.has_pending_frames() is True

    dataset.save_episode()
    assert dataset.has_pending_frames() is False


def test_create_record_finalize_read_roundtrip(tmp_path):
    """End-to-end: create, record 2 episodes, finalize, re-open, verify data."""
    root = tmp_path / "roundtrip"
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root
    )

    # Episode 0: 3 frames with known values
    ep0_states = []
    for i in range(3):
        state = torch.tensor([float(i), float(i * 2)])
        ep0_states.append(state)
        dataset.add_frame({"task": "Task A", "state": state})
    dataset.save_episode()

    # Episode 1: 2 frames
    ep1_states = []
    for i in range(2):
        state = torch.tensor([float(i + 100), float(i + 200)])
        ep1_states.append(state)
        dataset.add_frame({"task": "Task B", "state": state})
    dataset.save_episode()

    dataset.finalize()

    # Re-open as read-only
    reopened = LeRobotDataset(repo_id=DUMMY_REPO_ID, root=root)
    assert len(reopened) == 5
    assert reopened.num_episodes == 2

    # Verify episode 0
    for i in range(3):
        item = reopened[i]
        assert torch.allclose(item["state"], ep0_states[i], atol=1e-5)
        assert item["episode_index"].item() == 0

    # Verify episode 1
    for i in range(2):
        item = reopened[3 + i]
        assert torch.allclose(item["state"], ep1_states[i], atol=1e-5)
        assert item["episode_index"].item() == 1

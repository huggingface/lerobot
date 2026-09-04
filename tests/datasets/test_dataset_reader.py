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

import json
import sys
import types

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets import LeRobotDataset, register_dataset_reader
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.io_utils import hf_transform_to_torch
from lerobot.datasets.language import LANGUAGE_EVENTS
from lerobot.datasets.storage import _DATASET_READER_MODULES, DEFAULT_STORAGE_FORMAT, localize_remote_root
from lerobot.utils.import_utils import get_safe_default_video_backend
from tests.fixtures.constants import DEFAULT_FPS, DUMMY_REPO_ID

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
        video_backend=get_safe_default_video_backend(),
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
        video_backend=get_safe_default_video_backend(),
        delta_timestamps=None,
        image_transforms=None,
    )
    assert reader.try_load() is False
    assert reader.hf_dataset is None


def test_load_rejects_language_columns_missing_from_metadata(tmp_path, lerobot_dataset_factory):
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=1, total_frames=10, use_videos=False
    )
    parquet_path = next((dataset.root / "data").glob("*/*.parquet"))
    table = pq.read_table(parquet_path)
    language_events = pa.array([[] for _ in range(len(table))], type=pa.list_(pa.string()))
    pq.write_table(table.append_column(LANGUAGE_EVENTS, language_events), parquet_path)

    with pytest.raises(ValueError, match=r"language feature\(s\) missing from metadata.*language_events"):
        dataset.reader.load_and_activate()


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


# ── Batched get_items ────────────────────────────────────────────────


@pytest.mark.parametrize("use_delta", [False, True])
def test_get_items_batched_matches_single(tmp_path, lerobot_dataset_factory, use_delta):
    """Batched get_items (cross-batch video grouping) must match per-index results."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=20, use_videos=True
    )
    fps = dataset.meta.fps
    delta = {dataset.meta.video_keys[0]: [-1 / fps, 0.0], "action": [0.0, 1 / fps]} if use_delta else None
    reader = DatasetReader(
        meta=dataset.meta,
        root=dataset.root,
        episodes=None,
        tolerance_s=1e-4,
        video_backend=get_safe_default_video_backend(),
        delta_timestamps=delta,
        image_transforms=None,
    )
    reader.load_and_activate()

    # Order mixes episodes and repeats an index to exercise same-file grouping.
    order = [0, 10, 1, 11, 0, 19]
    batched = reader.get_items(order)
    singles = [reader.get_item(i) for i in order]

    assert len(batched) == len(singles)
    for got, want in zip(batched, singles, strict=True):
        assert set(got) == set(want)
        for key in want:
            if isinstance(want[key], torch.Tensor):
                assert torch.equal(got[key], want[key]), key
            else:
                assert got[key] == want[key], key


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


# ── Reader registry ──────────────────────────────────────────────────


class ToyDatasetReader(DatasetReader):
    def __init__(self, revision=None, token=None, **kwargs):
        super().__init__(video_backend="pyav", **kwargs)


def test_register_dataset_reader(tmp_path, lerobot_dataset_factory, monkeypatch):
    root = tmp_path / "src"
    lerobot_dataset_factory(
        root=root, total_episodes=2, total_frames=60, use_videos=False, camera_features={}
    )
    plain_item = LeRobotDataset(DUMMY_REPO_ID, root=root)[0]

    module = types.ModuleType("toyfmt_reader")
    module.DATASET_READER = ToyDatasetReader
    monkeypatch.setitem(sys.modules, "toyfmt_reader", module)
    register_dataset_reader("toyfmt", "toyfmt_reader")
    try:
        info_path = root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["storage_format"] = "toyfmt"
        info_path.write_text(json.dumps(info))

        ds = LeRobotDataset(DUMMY_REPO_ID, root=root)
        assert isinstance(ds.reader, ToyDatasetReader)
        item = ds[0]
        for key, expected in plain_item.items():
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(item[key], expected, rtol=0, atol=0)

        with pytest.raises(ValueError, match="already registered"):
            register_dataset_reader("toyfmt", "some.other.module")
        with pytest.raises(ValueError, match="already registered"):
            register_dataset_reader(DEFAULT_STORAGE_FORMAT, "toyfmt_reader")

        # a format whose optional deps are missing must not block probing the others
        module.localize_root = lambda *args, **kwargs: root
        broken = types.ModuleType("broken_reader")

        def raise_import_error(*args, **kwargs):
            raise ImportError("install some-extra")

        broken.localize_root = raise_import_error
        monkeypatch.setitem(sys.modules, "broken_reader", broken)
        monkeypatch.setattr(
            "lerobot.datasets.storage._DATASET_READER_MODULES",
            {"broken": "broken_reader", "toyfmt": "toyfmt_reader"},
        )
        assert localize_remote_root(DUMMY_REPO_ID, "s3://bucket/ds") == root
    finally:
        _DATASET_READER_MODULES.pop("toyfmt", None)


# ── Delta queries ────────────────────────────────────────────────────


def test_query_hf_dataset_matches_row_query(tmp_path, lerobot_dataset_factory):
    """Delta queries answered through cached column views match a plain row query bit for bit."""
    # A window reaching 2 frames back and 2 forward crosses the episode
    # boundaries at both ends, covering the clamped-index padding cases.
    delta_timestamps = {"action": [i / DEFAULT_FPS for i in range(-2, 3)]}
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=2,
        total_frames=20,
        use_videos=False,
        delta_timestamps=delta_timestamps,
    )
    reader = dataset.reader

    for abs_idx in range(reader.num_frames):
        ep_idx = int(reader.hf_dataset[abs_idx]["episode_index"])
        query_indices, _ = reader._get_query_indices(abs_idx, ep_idx)
        result = reader._query_hf_dataset([query_indices])[0]
        for key, q_idx in query_indices.items():
            expected = torch.stack(reader.hf_dataset[q_idx][key])
            assert torch.equal(result[key], expected)


def test_delta_query_transform_receives_only_requested_column(tmp_path, lerobot_dataset_factory):
    """During a delta query, the transform receives only the requested column.

    This is the contract that makes single-column views safe:
    hf_transform_to_torch is column-wise, so feeding it one column cannot
    change outputs. It also guarantees embedded camera images are not
    fetched and decoded when only a low-dimensional column is queried.
    """
    delta_timestamps = {"action": [i / DEFAULT_FPS for i in range(-1, 2)]}
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=1,
        total_frames=10,
        use_videos=False,
        delta_timestamps=delta_timestamps,
    )
    reader = dataset.reader
    seen_key_sets = []

    def spy_transform(items_dict):
        seen_key_sets.append(set(items_dict))
        return hf_transform_to_torch(items_dict)

    # Set the spy before the first query: column views are built lazily, so
    # they inherit it.
    reader.hf_dataset.set_transform(spy_transform)

    query_indices, _ = reader._get_query_indices(5, 0)
    reader._query_hf_dataset([query_indices])

    assert seen_key_sets, "expected the transform to be invoked"
    assert all(keys == {"action"} for keys in seen_key_sets)


def test_column_views_are_rebuilt_after_set_transform(tmp_path, lerobot_dataset_factory):
    """Cached column views must not outlive a set_transform() on hf_dataset.

    set_transform() mutates the dataset in place, so dataset identity alone
    cannot invalidate the cache: a view built under the previous transform
    would keep answering delta queries with it.
    """
    delta_timestamps = {"action": [i / DEFAULT_FPS for i in range(-1, 2)]}
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=1,
        total_frames=10,
        use_videos=False,
        delta_timestamps=delta_timestamps,
    )
    reader = dataset.reader

    query_indices, _ = reader._get_query_indices(5, 0)
    baseline = reader._query_hf_dataset([query_indices])[0]

    def doubling_transform(items_dict):
        items = hf_transform_to_torch(items_dict)
        return {key: [2 * value for value in values] for key, values in items.items()}

    reader.hf_dataset.set_transform(doubling_transform)

    result = reader._query_hf_dataset([query_indices])[0]
    assert torch.equal(result["action"], 2 * baseline["action"])

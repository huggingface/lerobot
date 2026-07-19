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
"""Parity tests: LanceDBDataset must return the same items as LeRobotDataset."""

import pickle
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.datasets.lancedb_dataset import (
    FRAMES_TABLE,
    VIDEO_BLOB_COLUMN,
    VIDEOS_TABLE,
    LanceDBDataset,
    to_lance_column,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tests.fixtures.constants import DUMMY_REPO_ID

lancedb = pytest.importorskip("lancedb")


def convert_frames_to_lance(src_root: Path, dst_root: Path) -> None:
    """Test-only converter: copy ``meta/`` and build the frames table from parquet data."""
    shutil.copytree(src_root / "meta", dst_root / "meta")
    files = sorted((src_root / "data").rglob("*.parquet"))
    table = pa.concat_tables([pq.read_table(f) for f in files]).sort_by("index")
    fields, arrays = [], []
    for field in table.schema:
        column = table.column(field.name).combine_chunks()
        if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
            column = pa.FixedSizeListArray.from_arrays(column.flatten(), len(column[0]))
        arrays.append(column)
        fields.append(pa.field(to_lance_column(field.name), column.type))
    db = lancedb.connect(str(dst_root))
    db.create_table(FRAMES_TABLE, pa.Table.from_arrays(arrays, schema=pa.schema(fields)))

    video_files = sorted((src_root / "videos").rglob("*.mp4"))
    if video_files:
        schema = pa.schema(
            [
                pa.field("video_key", pa.string()),
                pa.field("chunk_index", pa.int64()),
                pa.field("file_index", pa.int64()),
                lancedb.blob(VIDEO_BLOB_COLUMN),
            ]
        )
        videos_table = db.create_table(VIDEOS_TABLE, schema=schema)
        videos_table.add(
            [
                {
                    "video_key": mp4.parts[-3],
                    "chunk_index": int(mp4.parent.name.split("-")[1]),
                    "file_index": int(mp4.stem.split("-")[1]),
                    VIDEO_BLOB_COLUMN: mp4.read_bytes(),
                }
                for mp4 in video_files
            ]
        )


@pytest.fixture
def dataset_roots(tmp_path, lerobot_dataset_factory) -> tuple[Path, Path]:
    src_root = tmp_path / "src"
    lerobot_dataset_factory(
        root=src_root, total_episodes=3, total_frames=90, use_videos=False, camera_features={}
    )
    lance_root = tmp_path / "lance"
    convert_frames_to_lance(src_root, lance_root)
    return src_root, lance_root


def assert_items_equal(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected)
    for key, expected_val in expected.items():
        if isinstance(expected_val, str):
            assert actual[key] == expected_val, key
        else:
            assert actual[key].dtype == expected_val.dtype, key
            assert actual[key].shape == expected_val.shape, key
            torch.testing.assert_close(actual[key], expected_val, rtol=0, atol=0, msg=key)


def test_item_parity(dataset_roots):
    src_root, lance_root = dataset_roots
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root)
    lance_ds = LanceDBDataset(root=lance_root)

    assert len(lance_ds) == len(upstream)
    for idx in [0, len(upstream) // 2, len(upstream) - 1]:
        assert_items_equal(lance_ds[idx], upstream[idx])


def test_delta_timestamps_parity(dataset_roots):
    src_root, lance_root = dataset_roots
    fps = LeRobotDataset(DUMMY_REPO_ID, root=src_root).meta.fps
    delta_timestamps = {
        "state": [-2 / fps, -1 / fps, 0.0],
        "action": [0.0, 1 / fps, 2 / fps, 3 / fps],
    }
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root, delta_timestamps=delta_timestamps)
    lance_ds = LanceDBDataset(root=lance_root, delta_timestamps=delta_timestamps)

    # First/last frames of an episode exercise clamping and padding masks.
    ep_start = int(upstream.meta.episodes[1]["dataset_from_index"])
    ep_end = int(upstream.meta.episodes[1]["dataset_to_index"])
    for idx in [ep_start, ep_start + 5, ep_end - 1]:
        expected = upstream[idx]
        actual = lance_ds[idx]
        assert actual["state_is_pad"].any() == expected["state_is_pad"].any()
        assert_items_equal(actual, expected)


def test_batched_matches_single(dataset_roots):
    _, lance_root = dataset_roots
    lance_ds = LanceDBDataset(root=lance_root)
    indices = [3, 3, 17, 55]
    batched = lance_ds.__getitems__(indices)
    for idx, item in zip(indices, batched, strict=True):
        assert_items_equal(item, lance_ds[idx])


def test_episode_subset(dataset_roots):
    src_root, lance_root = dataset_roots
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root, episodes=[1])
    lance_ds = LanceDBDataset(root=lance_root, episodes=[1])

    assert len(lance_ds) == len(upstream)
    assert lance_ds.absolute_to_relative_idx == upstream.reader._absolute_to_relative_idx
    for idx in [0, len(upstream) - 1]:
        assert_items_equal(lance_ds[idx], upstream[idx])


@pytest.fixture
def video_dataset_roots(tmp_path, lerobot_dataset_factory) -> tuple[Path, Path]:
    src_root = tmp_path / "src_video"
    lerobot_dataset_factory(root=src_root, total_episodes=2, total_frames=40, use_videos=True)
    lance_root = tmp_path / "lance_video"
    convert_frames_to_lance(src_root, lance_root)
    return src_root, lance_root


def test_video_item_parity(video_dataset_roots):
    src_root, lance_root = video_dataset_roots
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root, video_backend="torchcodec")
    lance_ds = LanceDBDataset(root=lance_root)

    assert len(lance_ds) == len(upstream)
    for idx in [0, len(upstream) // 2, len(upstream) - 1]:
        assert_items_equal(lance_ds[idx], upstream[idx])


def test_video_delta_timestamps_parity(video_dataset_roots):
    src_root, lance_root = video_dataset_roots
    fps = LeRobotDataset(DUMMY_REPO_ID, root=src_root).meta.fps
    video_key = LeRobotDataset(DUMMY_REPO_ID, root=src_root).meta.video_keys[0]
    delta_timestamps = {
        video_key: [-1 / fps, 0.0, 1 / fps],
        "action": [0.0, 1 / fps],
    }
    upstream = LeRobotDataset(
        DUMMY_REPO_ID, root=src_root, delta_timestamps=delta_timestamps, video_backend="torchcodec"
    )
    lance_ds = LanceDBDataset(root=lance_root, delta_timestamps=delta_timestamps)

    ep_start = int(upstream.meta.episodes[1]["dataset_from_index"])
    for idx in [0, ep_start - 1, ep_start]:
        expected = upstream[idx]
        actual = lance_ds[idx]
        assert actual[f"{video_key}_is_pad"].tolist() == expected[f"{video_key}_is_pad"].tolist()
        assert_items_equal(actual, expected)


def test_video_return_uint8(video_dataset_roots):
    _, lance_root = video_dataset_roots
    lance_ds = LanceDBDataset(root=lance_root, return_uint8=True)
    item = lance_ds[0]
    video_key = lance_ds.meta.video_keys[0]
    assert item[video_key].dtype == torch.uint8


def test_pickle_and_dataloader(dataset_roots):
    _, lance_root = dataset_roots
    lance_ds = LanceDBDataset(root=lance_root)
    restored = pickle.loads(pickle.dumps(lance_ds))
    assert_items_equal(restored[7], lance_ds[7])

    loader = torch.utils.data.DataLoader(
        lance_ds,
        batch_size=8,
        num_workers=2,
        multiprocessing_context="spawn",
        shuffle=True,
    )
    batch = next(iter(loader))
    assert batch["state"].shape[0] == 8
    assert len(batch["task"]) == 8

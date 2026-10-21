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
"""mcap storage backend: LeRobotDataset over an mcap root must return the same
items as over the default parquet/mp4 layout, through the same public class."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("mcap")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.mcap_backend import McapDatasetReader
from lerobot.datasets.mcap_convert import convert_lerobot_dataset_to_mcap
from lerobot.datasets.mcap_utils import EPISODE_START_TIME_COLUMN
from lerobot.datasets.video_utils import FrameTimestampError

REPO_ID = "dummy/mcap-test"
FPS = 10


def _build_ground_truth(root: Path, total_episodes: int = 3, use_videos: bool = True) -> LeRobotDataset:
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
        "action": {"dtype": "float32", "shape": (6,), "names": None},
    }
    if use_videos:
        features["observation.images.cam"] = {
            "dtype": "video",
            "shape": (3, 32, 32),
            "names": ["channels", "height", "width"],
        }
    ds = LeRobotDataset.create(repo_id=REPO_ID, fps=FPS, features=features, root=root, use_videos=use_videos)
    rng = np.random.default_rng(0)
    for ep in range(total_episodes):
        n_frames = 12 + ep
        for _ in range(n_frames):
            frame = {
                "observation.state": torch.from_numpy(rng.random(6, dtype=np.float32)),
                "action": torch.from_numpy(rng.random(6, dtype=np.float32)),
                "task": f"task_{ep % 2}",
            }
            if use_videos:
                frame["observation.images.cam"] = torch.from_numpy(
                    rng.integers(0, 255, size=(3, 32, 32), dtype=np.uint8)
                )
            ds.add_frame(frame)
        ds.save_episode()
    ds.finalize()
    return LeRobotDataset(REPO_ID, root=root)


@pytest.fixture
def dataset_roots(tmp_path) -> tuple[Path, Path]:
    src_root = tmp_path / "src"
    upstream = _build_ground_truth(src_root, use_videos=False)
    mcap_root = tmp_path / "mcap"
    convert_lerobot_dataset_to_mcap(upstream, mcap_root)
    return src_root, mcap_root


@pytest.fixture
def video_dataset_roots(tmp_path) -> tuple[Path, Path]:
    src_root = tmp_path / "src_video"
    upstream = _build_ground_truth(src_root, total_episodes=2, use_videos=True)
    mcap_root = tmp_path / "mcap_video"
    convert_lerobot_dataset_to_mcap(upstream, mcap_root)
    return src_root, mcap_root


def assert_items_equal(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected)
    for key, expected_val in expected.items():
        if isinstance(expected_val, str):
            assert actual[key] == expected_val, key
        else:
            assert actual[key].dtype == expected_val.dtype, key
            assert actual[key].shape == expected_val.shape, key
            torch.testing.assert_close(actual[key], expected_val, rtol=0, atol=0, msg=key)


def test_tabular_parity(dataset_roots):
    src_root, mcap_root = dataset_roots
    upstream = LeRobotDataset(REPO_ID, root=src_root)
    mcap_ds = LeRobotDataset(REPO_ID, root=mcap_root)
    assert isinstance(mcap_ds.reader, McapDatasetReader)
    assert mcap_ds.meta.storage_format == "mcap"
    assert len(mcap_ds) == len(upstream)
    for idx in [0, len(upstream) // 2, len(upstream) - 1]:
        assert_items_equal(mcap_ds[idx], upstream[idx])

    # delta windows: first/last frames of an episode exercise clamping + pads
    fps = upstream.meta.fps
    delta_timestamps = {
        "observation.state": [-2 / fps, -1 / fps, 0.0],
        "action": [0.0, 1 / fps, 2 / fps, 3 / fps],
    }
    upstream_d = LeRobotDataset(REPO_ID, root=src_root, delta_timestamps=delta_timestamps)
    mcap_d = LeRobotDataset(REPO_ID, root=mcap_root, delta_timestamps=delta_timestamps)
    ep_start = int(upstream_d.meta.episodes[1]["dataset_from_index"])
    ep_end = int(upstream_d.meta.episodes[1]["dataset_to_index"])
    for idx in [ep_start, ep_start + 5, ep_end - 1]:
        expected = upstream_d[idx]
        actual = mcap_d[idx]
        assert actual["observation.state_is_pad"].tolist() == expected["observation.state_is_pad"].tolist()
        assert_items_equal(actual, expected)

    # batched fetch must equal singles (duplicates and overlapping windows included)
    indices = [3, 3, 17, ep_start, len(upstream) - 1]
    for idx, item in zip(indices, mcap_d.__getitems__(indices), strict=True):
        assert_items_equal(item, mcap_d[idx])

    # negative indexing mirrors upstream; out-of-range raises
    assert_items_equal(mcap_ds[-1], upstream[-1])
    with pytest.raises(IndexError):
        mcap_ds[len(mcap_ds)]

    # episode subset: relative/absolute mapping mirrors upstream
    upstream_s = LeRobotDataset(REPO_ID, root=src_root, episodes=[1])
    mcap_s = LeRobotDataset(REPO_ID, root=mcap_root, episodes=[1])
    assert len(mcap_s) == len(upstream_s)
    assert mcap_s.num_episodes == upstream_s.num_episodes
    assert mcap_s.absolute_to_relative_idx == upstream_s.reader._absolute_to_relative_idx
    for idx in [0, len(upstream_s) - 1]:
        assert_items_equal(mcap_s[idx], upstream_s[idx])


def test_video_parity(video_dataset_roots):
    src_root, mcap_root = video_dataset_roots
    upstream = LeRobotDataset(REPO_ID, root=src_root, video_backend="torchcodec")
    mcap_ds = LeRobotDataset(REPO_ID, root=mcap_root)
    for idx in [0, len(upstream) // 2, len(upstream) - 1]:
        assert_items_equal(mcap_ds[idx], upstream[idx])

    fps = upstream.meta.fps
    video_key = upstream.meta.video_keys[0]
    delta_timestamps = {video_key: [-1 / fps, 0.0, 1 / fps], "action": [0.0, 1 / fps]}
    upstream_d = LeRobotDataset(
        REPO_ID, root=src_root, delta_timestamps=delta_timestamps, video_backend="torchcodec"
    )
    mcap_d = LeRobotDataset(REPO_ID, root=mcap_root, delta_timestamps=delta_timestamps)
    ep_start = int(upstream_d.meta.episodes[1]["dataset_from_index"])
    for idx in [0, ep_start - 1, ep_start]:
        expected = upstream_d[idx]
        actual = mcap_d[idx]
        assert actual[f"{video_key}_is_pad"].tolist() == expected[f"{video_key}_is_pad"].tolist()
        assert_items_equal(actual, expected)

    # one-element window: squeeze semantics must match upstream's shape
    single = {video_key: [0.0]}
    upstream_1 = LeRobotDataset(REPO_ID, root=src_root, delta_timestamps=single, video_backend="torchcodec")
    mcap_1 = LeRobotDataset(REPO_ID, root=mcap_root, delta_timestamps=single)
    for idx in [0, len(upstream_1) - 1]:
        assert_items_equal(mcap_1[idx], upstream_1[idx])

    # return_uint8: raw frames, no normalization
    mcap_u8 = LeRobotDataset(REPO_ID, root=mcap_root, return_uint8=True)
    item = mcap_u8[0]
    assert item[video_key].dtype == torch.uint8


def test_tolerance_violation(video_dataset_roots):
    """Shifting an episode's sync-grid origin away from its real message times
    must raise, not silently return a mismatched frame."""
    _, mcap_root = video_dataset_roots
    info_path = mcap_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    assert info["storage_format"] == "mcap"

    episodes_path = next((mcap_root / "meta" / "episodes").rglob("*.parquet"))
    import pyarrow.parquet as pq

    table = pq.read_table(episodes_path)
    col = table.column(EPISODE_START_TIME_COLUMN).to_pylist()
    # +37ms: off-grid (not a multiple of 1/fps=100ms) so it can't coincidentally
    # land back on a real message a whole number of frames away.
    col[0] += 37_000_000
    table = table.set_column(
        table.schema.get_field_index(EPISODE_START_TIME_COLUMN),
        EPISODE_START_TIME_COLUMN,
        [col],
    )
    pq.write_table(table, episodes_path)

    mcap_ds = LeRobotDataset(REPO_ID, root=mcap_root)
    with pytest.raises(FrameTimestampError):
        mcap_ds[0]


def test_pickle_and_dataloader(dataset_roots):
    _, mcap_root = dataset_roots
    mcap_ds = LeRobotDataset(REPO_ID, root=mcap_root)
    restored = pickle.loads(pickle.dumps(mcap_ds))
    assert_items_equal(restored[7], mcap_ds[7])

    loader = torch.utils.data.DataLoader(mcap_ds, batch_size=8, num_workers=0, shuffle=True)
    batch = next(iter(loader))
    assert batch["observation.state"].shape[0] == 8
    assert len(batch["task"]) == 8

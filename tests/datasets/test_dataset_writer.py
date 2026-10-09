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
"""Contract tests for DatasetWriter."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.configs import VideoEncoderConfig
from lerobot.datasets.dataset_writer import _encode_video_worker
from lerobot.datasets.io_utils import load_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
from tests.fixtures.constants import DEFAULT_FPS, DUMMY_REPO_ID

SIMPLE_FEATURES = {
    "state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
}

ZERO_WIDTH_CASES = [
    (dtype, shape, 0)
    for dtype in ("float32", "float64", "int64")
    for shape in ((0,), (0, 3), (5, 0), (2, 0, 3))
] + [
    ("float32", (2, 0, 3, 1), 0),
    ("float32", (1, 2, 0, 3, 1), 0),
    ("float64", (0,), 1),
    ("int64", (2, 0, 3), 1),
]


def _assert_empty_tensor(value: torch.Tensor, dtype: str, shape: tuple[int, ...]) -> None:
    assert value.dtype == getattr(torch, dtype)
    assert tuple(value.shape) == shape
    assert value.numel() == 0


def _make_frame(features: dict, task: str = "Dummy task") -> dict:
    """Build a valid frame dict for the given features."""
    frame = {"task": task}
    for key, ft in features.items():
        if ft["dtype"] in ("image", "video"):
            frame[key] = np.random.randint(0, 256, size=ft["shape"], dtype=np.uint8)
        elif ft["dtype"] in ("float32", "float64"):
            frame[key] = torch.randn(ft["shape"])
        elif ft["dtype"] == "int64":
            frame[key] = torch.zeros(ft["shape"], dtype=torch.int64)
    return frame


# ── Existing encode_video_worker tests ───────────────────────────────


def test_encode_video_worker_forwards_video_encoder(tmp_path):
    """_encode_video_worker forwards video_encoder to encode_video_frames."""
    video_key = "observation.images.laptop"
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=0, frame_index=0)
    img_dir = tmp_path / Path(fpath).parent
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="red").save(img_dir / "frame-000000.png")

    captured_kwargs = {}

    def mock_encode(imgs_dir, video_path, fps, **kwargs):
        captured_kwargs.update(kwargs)
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).touch()

    with patch("lerobot.datasets.dataset_writer.encode_video_frames", side_effect=mock_encode):
        _encode_video_worker(
            video_key,
            0,
            tmp_path,
            fps=30,
            video_encoder=VideoEncoderConfig(vcodec="h264", preset=None),
            encoder_threads=4,
        )

    assert captured_kwargs["video_encoder"].vcodec == "h264"
    assert captured_kwargs["encoder_threads"] == 4


def test_encode_video_worker_default_video_encoder(tmp_path):
    """_encode_video_worker passes None video_encoder which encode_video_frames defaults."""
    video_key = "observation.images.laptop"
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=0, frame_index=0)
    img_dir = tmp_path / Path(fpath).parent
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="red").save(img_dir / "frame-000000.png")

    captured_kwargs = {}

    def mock_encode(imgs_dir, video_path, fps, **kwargs):
        captured_kwargs.update(kwargs)
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).touch()

    with patch("lerobot.datasets.dataset_writer.encode_video_frames", side_effect=mock_encode):
        _encode_video_worker(video_key, 0, tmp_path, fps=30)

    assert captured_kwargs["video_encoder"] is None
    assert captured_kwargs["encoder_threads"] is None


# ── add_frame contracts ──────────────────────────────────────────────


def test_add_frame_increments_buffer_size(tmp_path):
    """Each add_frame() call increases episode_buffer['size'] by 1."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    assert dataset.writer.episode_buffer["size"] == 0

    dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    assert dataset.writer.episode_buffer["size"] == 1

    dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    assert dataset.writer.episode_buffer["size"] == 2


def test_add_frame_rejects_missing_feature(tmp_path):
    """add_frame() raises ValueError when a required feature is missing."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    with pytest.raises(ValueError, match="Missing features"):
        dataset.add_frame({"task": "Dummy task", "state": torch.randn(6)})
        # missing 'action'


# ── save_episode contracts ───────────────────────────────────────────


def test_save_episode_writes_parquet(tmp_path):
    """After save_episode(), at least one .parquet file exists under data/."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    parquet_files = list((tmp_path / "ds" / "data").rglob("*.parquet"))
    assert len(parquet_files) > 0


def test_save_episode_updates_counters(tmp_path):
    """After save_episode(), metadata counters are updated."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(5):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 5


def test_save_episode_resets_buffer(tmp_path):
    """After save_episode(), the episode buffer is reset."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    assert dataset.writer.episode_buffer["size"] == 0


def test_save_multiple_episodes(tmp_path):
    """Recording 3 episodes results in correct total counts."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    total_frames = 0
    for ep in range(3):
        n_frames = ep + 2  # 2, 3, 4
        for _ in range(n_frames):
            dataset.add_frame(_make_frame(SIMPLE_FEATURES))
        dataset.save_episode()
        total_frames += n_frames

    assert dataset.meta.total_episodes == 3
    assert dataset.meta.total_frames == total_frames


# ── clear / lifecycle ────────────────────────────────────────────────


def test_clear_resets_buffer(tmp_path):
    """clear_episode_buffer() resets the buffer size to 0."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    assert dataset.writer.episode_buffer["size"] == 1

    dataset.clear_episode_buffer()
    assert dataset.writer.episode_buffer["size"] == 0


def test_clear_removes_video_frame_staging_dir(tmp_path):
    """clear_episode_buffer() removes PNG staging dirs for video features."""
    video_key = "observation.images.cam"
    features = {
        video_key: {
            "dtype": "video",
            "shape": (64, 96, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=DEFAULT_FPS,
        features=features,
        root=tmp_path / "ds",
        use_videos=True,
    )

    dataset.add_frame(_make_frame(features))
    video_staging_dir = (
        dataset.root
        / Path(DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=0, frame_index=0)).parent
    )
    assert video_staging_dir.is_dir()

    dataset.clear_episode_buffer()

    assert dataset.writer.episode_buffer["size"] == 0
    assert not video_staging_dir.exists()


def test_batched_encoding_staging_survives_save(tmp_path):
    """The post-save clear must NOT delete video staging frames.

    With ``batch_encoding_size > 1`` the frames of already-saved episodes stay
    on disk until the batch encode runs; the encoder deletes them afterwards.
    A blanket switch of the post-save cleanup to ``camera_keys`` (as done in the
    discard path) would silently break batched encoding.
    """
    video_key = "observation.images.cam"
    features = {
        video_key: {
            "dtype": "video",
            "shape": (64, 96, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=DEFAULT_FPS,
        features=features,
        root=tmp_path / "ds",
        use_videos=True,
        batch_encoding_size=2,
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(features))

    staging_dir = dataset.writer._get_image_file_dir(0, video_key)
    assert staging_dir.is_dir()

    dataset.save_episode()  # first of a batch of 2: no encoding yet

    assert staging_dir.is_dir() and any(staging_dir.iterdir())


def test_finalize_is_idempotent(tmp_path):
    """Calling finalize() twice does not raise."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    dataset.finalize()
    dataset.finalize()  # second call should not raise


def test_finalize_then_read_roundtrip(tmp_path):
    """Write data, finalize, re-open, and verify data matches."""
    root = tmp_path / "roundtrip"
    features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
    dataset = LeRobotDataset.create(repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=features, root=root)

    # Record known values
    known_states = []
    for i in range(5):
        state = torch.tensor([float(i), float(i * 10)])
        known_states.append(state)
        dataset.add_frame({"task": "Test task", "state": state})
    dataset.save_episode()
    dataset.finalize()

    # Read back
    for i in range(5):
        item = dataset[i]
        assert torch.allclose(item["state"], known_states[i], atol=1e-5)


@pytest.mark.parametrize(("dtype", "shape", "num_workers"), ZERO_WIDTH_CASES)
def test_zero_width_features_roundtrip_multiple_episodes(
    tmp_path: Path, dtype: str, shape: tuple[int, ...], num_workers: int
) -> None:
    """Empty numeric columns survive storage and single, batch, and delta reads."""
    root = tmp_path / "zero_width"
    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "target": {"dtype": dtype, "shape": shape, "names": None},
    }
    dataset = LeRobotDataset.create(repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=features, root=root)
    actions = np.arange(10, dtype=np.float32).reshape(5, 2)
    index = 0
    for episode_length in (2, 3):
        for _ in range(episode_length):
            dataset.add_frame(
                {
                    "task": "Test task",
                    "action": actions[index].copy(),
                    "target": np.empty(shape, dtype=dtype),
                }
            )
            index += 1
        dataset.save_episode()
    dataset.finalize()

    # Inspect the stored column, independently of the Torch reading transform.
    table = pq.read_table(sorted((root / "data").rglob("*.parquet")))
    column = table.column("target").combine_chunks()
    storage = column.storage if isinstance(column, pa.ExtensionArray) else column
    assert len(storage) == len(actions)
    assert all(np.asarray(row).size == 0 for row in storage.to_pylist())
    arrow_dtype = column.type
    while (
        isinstance(arrow_dtype, pa.ExtensionType)
        or pa.types.is_list(arrow_dtype)
        or pa.types.is_large_list(arrow_dtype)
        or pa.types.is_fixed_size_list(arrow_dtype)
    ):
        if isinstance(arrow_dtype, pa.ExtensionType):
            arrow_dtype = arrow_dtype.storage_type
        else:
            arrow_dtype = arrow_dtype.value_type
    assert arrow_dtype == pa.from_numpy_dtype(np.dtype(dtype))

    reloaded = LeRobotDataset(repo_id=DUMMY_REPO_ID, root=root, download_videos=False)
    assert reloaded.num_episodes == 2
    assert len(reloaded) == 5
    assert reloaded.meta.features["target"]["dtype"] == dtype
    assert tuple(reloaded.meta.features["target"]["shape"]) == shape
    assert "target" not in reloaded.meta.stats
    assert "target" not in load_stats(root)
    for stat, expected in {
        "min": actions.min(axis=0),
        "max": actions.max(axis=0),
        "mean": actions.mean(axis=0),
        "std": actions.std(axis=0),
        "count": np.array([len(actions)]),
    }.items():
        np.testing.assert_allclose(reloaded.meta.stats["action"][stat], expected, rtol=1e-6)

    for index in range(len(reloaded)):
        item = reloaded[index]
        _assert_empty_tensor(item["target"], dtype, shape)
        torch.testing.assert_close(item["action"], torch.from_numpy(actions[index]))
        assert item["episode_index"].item() == (0 if index < 2 else 1)
        assert item["task"] == "Test task"

    # Include repeated and out-of-order indices to exercise batched gathering.
    indices = [4, 0, 2, 4]
    for index, item in zip(indices, reloaded.__getitems__(indices), strict=True):
        _assert_empty_tensor(item["target"], dtype, shape)
        torch.testing.assert_close(item["action"], torch.from_numpy(actions[index]))

    deltas = [-1 / DEFAULT_FPS, 0.0, 1 / DEFAULT_FPS]
    windowed = LeRobotDataset(
        repo_id=DUMMY_REPO_ID,
        root=root,
        download_videos=False,
        delta_timestamps={"action": deltas, "target": deltas},
    )
    for index, item in zip(indices, windowed.__getitems__(indices), strict=True):
        start, stop = (0, 2) if index < 2 else (2, 5)
        window = [min(max(index + offset, start), stop - 1) for offset in (-1, 0, 1)]
        padding = torch.tensor([index - 1 < start, False, index + 1 >= stop])
        _assert_empty_tensor(item["target"], dtype, (3, *shape))
        torch.testing.assert_close(item["action"], torch.from_numpy(actions[window]))
        assert torch.equal(item["target_is_pad"], padding)
        assert torch.equal(item["action_is_pad"], padding)
        torch.testing.assert_close(item["target"], windowed[index]["target"])

    loader = torch.utils.data.DataLoader(
        windowed,
        batch_size=2,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers else None,
    )
    batch = next(iter(loader))
    _assert_empty_tensor(batch["target"], dtype, (2, 3, *shape))
    expected_actions = torch.from_numpy(actions[np.array([[0, 0, 1], [0, 1, 1]])])
    torch.testing.assert_close(batch["action"], expected_actions)


@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
def test_add_frame_rejects_nonempty_zero_width_feature(tmp_path: Path, shape: tuple[int, ...]) -> None:
    """A variable-length storage column must not relax the declared frame shape."""
    features = {"target": {"dtype": "float32", "shape": shape, "names": None}}
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=features, root=tmp_path / "ds"
    )
    nonempty_shape = tuple(max(size, 1) for size in shape)
    with pytest.raises(ValueError, match="feature 'target'.*expected shape"):
        dataset.add_frame({"task": "Test task", "target": np.zeros(nonempty_shape, dtype=np.float32)})
    assert dataset.writer.episode_buffer["size"] == 0

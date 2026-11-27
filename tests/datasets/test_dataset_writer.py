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
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs import RGBEncoderConfig, VideoEncoderConfig
from lerobot.datasets.dataset_writer import _encode_video_worker
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
from tests.fixtures.constants import DEFAULT_FPS, DUMMY_REPO_ID

pytestmark = pytest.mark.dataset

SIMPLE_FEATURES = {
    "state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
}


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


@pytest.mark.parametrize("streaming", [False, True])
def test_interrupted_video_save_does_not_write_episode(tmp_path, streaming):
    """An encoding interruption must leave no orphaned data rows or episode metadata."""
    features = {
        **SIMPLE_FEATURES,
        "observation.images.camera": {
            "dtype": "video",
            "shape": (32, 32, 3),
            "names": ["height", "width", "channels"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=features, root=tmp_path / "ds"
    )
    writer = dataset.writer
    encoder = Mock()
    encoder.finish_episode.side_effect = KeyboardInterrupt("encoding interrupted")
    if streaming:
        writer._streaming_encoder = encoder
    for _ in range(3):
        dataset.add_frame(_make_frame(features))
    try:
        with (
            patch.object(writer, "_save_episode_data", wraps=writer._save_episode_data) as save_data,
            patch.object(dataset.meta, "save_episode", wraps=dataset.meta.save_episode) as save_meta,
            patch.object(
                writer,
                "_encode_temporary_episode_video",
                side_effect=KeyboardInterrupt("encoding interrupted"),
            ),
        ):
            with pytest.raises(KeyboardInterrupt, match="encoding interrupted"):
                writer.save_episode(parallel_encoding=False)
            save_data.assert_not_called()
            save_meta.assert_not_called()
            assert dataset.meta.total_episodes == 0
            assert dataset.meta.total_frames == 0
            assert not list((tmp_path / "ds" / "data").rglob("*.parquet"))
    finally:
        writer._streaming_encoder = None
        writer.finalize()


@pytest.mark.parametrize("saved_episodes", [0, 1])
def test_interrupted_second_camera_preserves_saved_videos(tmp_path, saved_episodes):
    """Failed encoding must not create or extend any committed camera video."""
    camera_keys = ["observation.images.front", "observation.images.wrist"]
    features = {
        **SIMPLE_FEATURES,
        **{
            key: {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]}
            for key in camera_keys
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=DEFAULT_FPS,
        features=features,
        root=tmp_path / "ds",
        rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast"),
        encoder_threads=1,
    )
    writer = dataset.writer

    def add_episode(value: int) -> None:
        for _ in range(3):
            frame = _make_frame(features)
            frame["action"] = torch.full((6,), float(value))
            for key in camera_keys:
                frame[key] = np.full((32, 32, 3), value, dtype=np.uint8)
            dataset.add_frame(frame)

    try:
        for _ in range(saved_episodes):
            add_episode(30)
            writer.save_episode(parallel_encoding=False)
        video_dir = dataset.root / "videos"
        saved_videos = {p: p.read_bytes() for p in video_dir.rglob("*.mp4")}
        encode = writer._encode_temporary_episode_video

        def interrupt_second_camera(video_key: str, episode_index: int) -> Path:
            if video_key == camera_keys[1]:
                raise KeyboardInterrupt("second camera interrupted")
            return encode(video_key, episode_index)

        add_episode(120)
        with (
            patch.object(writer, "_encode_temporary_episode_video", side_effect=interrupt_second_camera),
            pytest.raises(KeyboardInterrupt, match="second camera interrupted"),
        ):
            writer.save_episode(parallel_encoding=False)

        assert {p: p.read_bytes() for p in video_dir.rglob("*.mp4")} == saved_videos
        assert dataset.meta.total_episodes == saved_episodes
        assert dataset.meta.total_frames == 3 * saved_episodes

        writer.clear_episode_buffer(delete_images=False)
        add_episode(220)
        writer.save_episode(parallel_encoding=False)
        dataset.finalize()
        assert len(dataset) == 3 * (saved_episodes + 1)
        for index in range(len(dataset)):
            expected = 30 if index < 3 * saved_episodes else 220
            item = dataset[index]
            assert torch.all(item["action"] == expected)
            for key in camera_keys:
                assert item[key].mean().item() == pytest.approx(expected / 255, abs=2 / 255)
    finally:
        writer.finalize()

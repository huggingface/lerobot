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

from lerobot.configs import VideoEncoderConfig
from lerobot.datasets.dataset_writer import _encode_video_worker
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
from tests.fixtures.constants import DEFAULT_FPS, DUMMY_REPO_ID

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


def _assert_video_metadata_consistent(dataset: LeRobotDataset) -> None:
    """Every episode row carries its own index and a video segment exactly as
    long as the episode. Catches orphan rows (video metadata attached to the
    wrong file) and misaligned merges, which decoding a single frame may not.
    """
    from lerobot.datasets.io_utils import load_episodes

    episodes = load_episodes(dataset.root)
    assert len(episodes) == dataset.meta.total_episodes
    for row in episodes:
        assert row["episode_index"] is not None
        expected = row["length"] / dataset.meta.fps
        for key in dataset.meta.video_keys:
            duration = row[f"videos/{key}/to_timestamp"] - row[f"videos/{key}/from_timestamp"]
            assert abs(duration - expected) < 1e-3, (row["episode_index"], key, duration, expected)


def test_batched_encoding_end_to_end(tmp_path):
    """Recording with ``batch_encoding_size > 1`` produces a loadable dataset.

    Regression test: the batch encoder indexed ``meta.episodes`` — a view of
    the on-disk state that lags the session (episodes live in the metadata
    buffer until flushed) — so any fresh recording with a batch size above 1
    crashed on its first batch (#2404, #2509).
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
    for _ in range(5):  # two full batches + one remainder encoded at finalize
        for _ in range(3):
            dataset.add_frame(_make_frame(features))
        dataset.save_episode()
    dataset.finalize()

    reloaded = LeRobotDataset(DUMMY_REPO_ID, root=tmp_path / "ds")
    assert reloaded.meta.total_episodes == 5
    assert reloaded.num_frames == 15
    assert reloaded[reloaded.num_frames - 1][video_key].shape[-2:] == (64, 96)
    _assert_video_metadata_consistent(reloaded)


def test_batched_encoding_on_resumed_dataset(tmp_path):
    """Batch encoding works when appending to an existing dataset via resume().

    Regression test: on a resumed dataset, ``meta.episodes`` only covered the
    episodes recorded in previous sessions, so the first batch of a resumed
    session crashed with an IndexError. The video-metadata merge also aligned
    on the dataframe's positional index, corrupting any episodes file that
    does not start at episode 0.
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
    )
    for _ in range(3):
        for _ in range(3):
            dataset.add_frame(_make_frame(features))
        dataset.save_episode()
    dataset.finalize()

    dataset = LeRobotDataset.resume(
        repo_id=DUMMY_REPO_ID,
        root=tmp_path / "ds",
        batch_encoding_size=2,
    )
    for _ in range(4):
        for _ in range(3):
            dataset.add_frame(_make_frame(features))
        dataset.save_episode()
    dataset.finalize()

    reloaded = LeRobotDataset(DUMMY_REPO_ID, root=tmp_path / "ds")
    assert reloaded.meta.total_episodes == 7
    assert reloaded.num_frames == 21
    assert reloaded[0][video_key].shape[-2:] == (64, 96)
    assert reloaded[reloaded.num_frames - 1][video_key].shape[-2:] == (64, 96)
    _assert_video_metadata_consistent(reloaded)


def test_batched_encoding_across_two_resumed_sessions(tmp_path):
    """Metadata files roll over per batch while data files roll over per
    session, so their indices diverge after the first batch of a resumed
    session. The episodes parquet must be addressed by the metadata indices.
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
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=features, root=tmp_path / "ds", use_videos=True
    )
    for _ in range(2):
        for _ in range(3):
            dataset.add_frame(_make_frame(features))
        dataset.save_episode()
    dataset.finalize()

    for _session in range(2):
        dataset = LeRobotDataset.resume(repo_id=DUMMY_REPO_ID, root=tmp_path / "ds", batch_encoding_size=2)
        for _ in range(5):  # two full batches + one remainder per session
            for _ in range(3):
                dataset.add_frame(_make_frame(features))
            dataset.save_episode()
        dataset.finalize()

    reloaded = LeRobotDataset(DUMMY_REPO_ID, root=tmp_path / "ds")
    assert reloaded.meta.total_episodes == 12
    _assert_video_metadata_consistent(reloaded)


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

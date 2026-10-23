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
import json
import logging

import numpy as np
import pytest

pytest.importorskip("soundfile", reason="soundfile is required (install lerobot[audio])")

import soundfile as sf

from lerobot.configs import FeatureType
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.audio_utils import concatenate_audio_files, write_audio_file
from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.utils.feature_utils import dataset_to_policy_features
from tests.fixtures.constants import DUMMY_REPO_ID

AUDIO_KEY = "observation.audio.mic"
SAMPLE_RATE = 16_000
FPS = 10
NUM_CHANNELS = 2
# Lossless up to the 24-bit quantization of the FLAC files
ATOL = 2**-23

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    AUDIO_KEY: {
        "dtype": "audio",
        "shape": (NUM_CHANNELS,),
        "names": ["left", "right"],
        "info": {"audio.sample_rate": SAMPLE_RATE},
    },
}


def make_audio(num_frames: int, seed: int, extra_s: float = 0.2) -> np.ndarray:
    num_samples = round((num_frames / FPS + extra_s) * SAMPLE_RATE)
    return np.random.default_rng(seed).uniform(-0.9, 0.9, (num_samples, NUM_CHANNELS)).astype(np.float32)


def add_episode(dataset: LeRobotDataset, num_frames: int, audio, start_time_s: float = 0.0) -> None:
    for _ in range(num_frames):
        dataset.add_frame({"observation.state": np.zeros(2, dtype=np.float32), "task": "listen"})
    dataset.add_episode_audio(AUDIO_KEY, audio, start_time_s=start_time_s)
    dataset.save_episode()


def expected_window(audio: np.ndarray, timestamp: float, start_time_s: float, window_s: float) -> np.ndarray:
    window_size = round(window_s * SAMPLE_RATE)
    end = round((timestamp + start_time_s) * SAMPLE_RATE)
    indices = np.arange(end - window_size, end)
    valid = (indices >= 0) & (indices < len(audio))
    window = np.zeros((window_size, audio.shape[1]), dtype=np.float32)
    window[valid] = audio[indices[valid]]
    return window.T


@pytest.fixture
def audio_dataset(tmp_path):
    """A finalized dataset with two episodes, and the audio of each episode with its start time."""
    root = tmp_path / "audio_dataset"
    dataset = LeRobotDataset.create(DUMMY_REPO_ID, fps=FPS, features=FEATURES, root=root, use_videos=False)
    episodes = []
    for seed, (num_frames, start_time_s) in enumerate([(20, 0.05), (15, -0.03)]):
        audio = make_audio(num_frames, seed=seed)
        add_episode(dataset, num_frames, audio, start_time_s)
        episodes.append((audio, start_time_s))
    dataset.finalize()
    return root, episodes


def test_round_trip(audio_dataset):
    root, episodes = audio_dataset
    window_s = 0.25
    dataset = LeRobotDataset(DUMMY_REPO_ID, root=root, audio_window_s=window_s)

    for idx in range(len(dataset)):
        item = dataset[idx]
        audio, start_time_s = episodes[int(item["episode_index"])]
        window = item[AUDIO_KEY]
        assert window.shape == (NUM_CHANNELS, round(window_s * SAMPLE_RATE))
        np.testing.assert_allclose(
            window.numpy(),
            expected_window(audio, float(item["timestamp"]), start_time_s, window_s),
            atol=ATOL,
        )


def test_episodes_share_a_file(audio_dataset):
    root, episodes = audio_dataset
    meta = LeRobotDataset(DUMMY_REPO_ID, root=root).meta

    assert meta.get_audio_file_path(0, AUDIO_KEY) == meta.get_audio_file_path(1, AUDIO_KEY)
    first, second = meta.episodes[0], meta.episodes[1]
    assert first[f"audio/{AUDIO_KEY}/from_timestamp"] == 0.0
    assert second[f"audio/{AUDIO_KEY}/from_timestamp"] == first[f"audio/{AUDIO_KEY}/to_timestamp"]
    assert second[f"audio/{AUDIO_KEY}/start_time_s"] == episodes[1][1]
    stored, sample_rate = sf.read(root / meta.get_audio_file_path(0, AUDIO_KEY), dtype="float32")
    assert sample_rate == SAMPLE_RATE
    np.testing.assert_allclose(stored, np.concatenate([audio for audio, _ in episodes]), atol=ATOL)


def test_delta_timestamps(audio_dataset):
    root, episodes = audio_dataset
    window_s = 0.1
    deltas = [-0.2, -0.1, 0.0]
    dataset = LeRobotDataset(
        DUMMY_REPO_ID, root=root, audio_window_s=window_s, delta_timestamps={AUDIO_KEY: deltas}
    )

    item = dataset[1]
    audio, start_time_s = episodes[0]
    assert item[AUDIO_KEY].shape == (len(deltas), NUM_CHANNELS, round(window_s * SAMPLE_RATE))
    assert item[f"{AUDIO_KEY}_is_pad"].tolist() == [True, False, False]
    # A padded delta repeats the first frame of the episode, like other features
    for window, timestamp in zip(item[AUDIO_KEY], [0.0, 0.0, 0.1], strict=True):
        np.testing.assert_allclose(
            window.numpy(), expected_window(audio, timestamp, start_time_s, window_s), atol=ATOL
        )


def test_stats(audio_dataset):
    root, episodes = audio_dataset
    stats = LeRobotDataset(DUMMY_REPO_ID, root=root).meta.stats[AUDIO_KEY]

    all_audio = np.concatenate([audio for audio, _ in episodes])
    for key in ["min", "max", "mean", "std", "q01", "q50", "q99"]:
        assert stats[key].shape == (NUM_CHANNELS, 1)
    assert np.all(stats["min"][:, 0] >= all_audio.min(axis=0) - ATOL)
    assert np.all(stats["max"][:, 0] <= all_audio.max(axis=0) + ATOL)
    np.testing.assert_allclose(stats["std"][:, 0], all_audio.std(axis=0), rtol=0.1)
    # Episodes are weighted by their number of samples, not by the size of their subsample
    assert stats["count"].tolist() == [len(all_audio)]


def test_info(audio_dataset, tmp_path):
    root, _ = audio_dataset
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["audio_path"] == "audio/{audio_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.flac"
    assert info["audio_files_size_in_mb"] == 100
    assert info["features"][AUDIO_KEY]["info"] == {"audio.sample_rate": SAMPLE_RATE, "audio.codec": "flac"}

    features = {"observation.state": FEATURES["observation.state"]}
    no_audio_root = tmp_path / "no_audio"
    LeRobotDataset.create(DUMMY_REPO_ID, fps=FPS, features=features, root=no_audio_root, use_videos=False)
    no_audio_info = json.loads((no_audio_root / "meta" / "info.json").read_text())
    assert "audio_path" not in no_audio_info
    assert "audio_files_size_in_mb" not in no_audio_info


def test_audio_from_file(tmp_path):
    dataset = LeRobotDataset.create(
        DUMMY_REPO_ID, fps=FPS, features=FEATURES, root=tmp_path / "dataset", use_videos=False
    )
    audio = make_audio(10, seed=2)
    wav_path = tmp_path / "recording.wav"
    sf.write(wav_path, audio, SAMPLE_RATE, subtype="FLOAT")

    add_episode(dataset, 10, wav_path)
    dataset.finalize()

    assert wav_path.exists()
    item = LeRobotDataset(DUMMY_REPO_ID, root=tmp_path / "dataset", audio_window_s=0.3)[9]
    np.testing.assert_allclose(item[AUDIO_KEY].numpy(), expected_window(audio, 0.9, 0.0, 0.3), atol=ATOL)


def test_files_roll_over(tmp_path):
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        DUMMY_REPO_ID, fps=FPS, features=FEATURES, root=root, use_videos=False, audio_files_size_in_mb=1
    )
    # White noise barely compresses: about 1.1 MB of FLAC per episode
    episodes = [make_audio(110, seed=seed, extra_s=0.0) for seed in range(3)]
    for audio in episodes:
        add_episode(dataset, 110, audio)
    dataset.finalize()

    dataset = LeRobotDataset(DUMMY_REPO_ID, root=root, audio_window_s=0.1)
    assert [dataset.meta.episodes[i][f"audio/{AUDIO_KEY}/file_index"] for i in range(3)] == [0, 1, 2]
    assert all(dataset.meta.episodes[i][f"audio/{AUDIO_KEY}/from_timestamp"] == 0.0 for i in range(3))
    item = dataset[2 * 110 + 50]
    np.testing.assert_allclose(
        item[AUDIO_KEY].numpy(), expected_window(episodes[2], 5.0, 0.0, 0.1), atol=ATOL
    )


def test_resume(audio_dataset):
    root, episodes = audio_dataset
    dataset = LeRobotDataset.resume(DUMMY_REPO_ID, root=root)
    audio = make_audio(10, seed=3)
    add_episode(dataset, 10, audio, start_time_s=0.02)
    dataset.finalize()

    dataset = LeRobotDataset(DUMMY_REPO_ID, root=root, audio_window_s=0.2)
    assert dataset.meta.total_episodes == 3
    # Like videos, resuming starts a new file
    assert dataset.meta.episodes[2][f"audio/{AUDIO_KEY}/file_index"] == 1
    item = dataset[len(dataset) - 1]
    np.testing.assert_allclose(item[AUDIO_KEY].numpy(), expected_window(audio, 0.9, 0.02, 0.2), atol=ATOL)
    item = dataset[0]
    np.testing.assert_allclose(
        item[AUDIO_KEY].numpy(), expected_window(episodes[0][0], 0.0, episodes[0][1], 0.2), atol=ATOL
    )


def test_save_episode_requires_audio(tmp_path):
    dataset = LeRobotDataset.create(
        DUMMY_REPO_ID, fps=FPS, features=FEATURES, root=tmp_path / "dataset", use_videos=False
    )
    dataset.add_frame({"observation.state": np.zeros(2, dtype=np.float32), "task": "listen"})

    with pytest.raises(ValueError, match="Missing audio"):
        dataset.save_episode()

    dataset.add_episode_audio(AUDIO_KEY, make_audio(1, seed=0))
    dataset.save_episode()
    assert dataset.meta.total_episodes == 1


def test_clear_episode_buffer_discards_audio(tmp_path):
    dataset = LeRobotDataset.create(
        DUMMY_REPO_ID, fps=FPS, features=FEATURES, root=tmp_path / "dataset", use_videos=False
    )
    dataset.add_frame({"observation.state": np.zeros(2, dtype=np.float32), "task": "listen"})
    dataset.add_episode_audio(AUDIO_KEY, make_audio(1, seed=0))
    dataset.clear_episode_buffer()

    dataset.add_frame({"observation.state": np.zeros(2, dtype=np.float32), "task": "listen"})
    with pytest.raises(ValueError, match="Missing audio"):
        dataset.save_episode()


def test_add_frame_rejects_audio(tmp_path):
    dataset = LeRobotDataset.create(
        DUMMY_REPO_ID, fps=FPS, features=FEATURES, root=tmp_path / "dataset", use_videos=False
    )
    with pytest.raises(ValueError, match="Extra features"):
        dataset.add_frame(
            {
                "observation.state": np.zeros(2, dtype=np.float32),
                AUDIO_KEY: np.zeros((160, NUM_CHANNELS), dtype=np.float32),
                "task": "listen",
            }
        )


@pytest.mark.parametrize(
    "audio_key, audio, start_time_s, error, match",
    [
        ("observation.state", np.zeros((10, 2), dtype=np.float32), 0.0, ValueError, "not an audio feature"),
        (AUDIO_KEY, np.zeros((10, 3), dtype=np.float32), 0.0, ValueError, "channel"),
        (AUDIO_KEY, np.zeros((0, 2), dtype=np.float32), 0.0, ValueError, "empty"),
        (AUDIO_KEY, np.zeros((10, 2), dtype=np.int16), 0.0, TypeError, "floating point"),
        (AUDIO_KEY, np.zeros((10, 2, 1), dtype=np.float32), 0.0, ValueError, "shape"),
        (AUDIO_KEY, np.zeros((10, 2), dtype=np.float32), float("nan"), ValueError, "finite"),
    ],
)
def test_add_episode_audio_invalid(tmp_path, audio_key, audio, start_time_s, error, match):
    dataset = LeRobotDataset.create(
        DUMMY_REPO_ID, fps=FPS, features=FEATURES, root=tmp_path / "dataset", use_videos=False
    )
    with pytest.raises(error, match=match):
        dataset.add_episode_audio(audio_key, audio, start_time_s=start_time_s)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("from_file", [False, True])
def test_add_episode_audio_not_finite(tmp_path, value, from_file):
    dataset = LeRobotDataset.create(
        DUMMY_REPO_ID, fps=FPS, features=FEATURES, root=tmp_path / "dataset", use_videos=False
    )
    audio = np.zeros((10, NUM_CHANNELS), dtype=np.float32)
    audio[3, 1] = value
    if from_file:
        sf.write(tmp_path / "recording.wav", audio, SAMPLE_RATE, subtype="FLOAT")
        audio = tmp_path / "recording.wav"

    with pytest.raises(ValueError, match="finite"):
        dataset.add_episode_audio(AUDIO_KEY, audio)


@pytest.mark.parametrize("from_file", [False, True])
def test_out_of_range_audio_is_clipped(tmp_path, caplog, from_file):
    root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(DUMMY_REPO_ID, fps=FPS, features=FEATURES, root=root, use_videos=False)
    audio = make_audio(10, seed=0)
    audio[100] = [2.0, -1.5]
    expected = np.clip(audio, -1.0, 1.0)
    if from_file:
        sf.write(tmp_path / "recording.wav", audio, SAMPLE_RATE, subtype="FLOAT")
        audio = tmp_path / "recording.wav"

    with caplog.at_level(logging.WARNING):
        add_episode(dataset, 10, audio)
    dataset.finalize()

    assert "Clipping 2 audio sample(s)" in caplog.text
    stored, _ = sf.read(root / dataset.meta.get_audio_file_path(0, AUDIO_KEY), dtype="float32")
    np.testing.assert_allclose(stored, expected, atol=ATOL)


def test_add_episode_audio_wrong_sample_rate(tmp_path):
    dataset = LeRobotDataset.create(
        DUMMY_REPO_ID, fps=FPS, features=FEATURES, root=tmp_path / "dataset", use_videos=False
    )
    wav_path = tmp_path / "recording.wav"
    sf.write(wav_path, np.zeros((100, NUM_CHANNELS), dtype=np.float32), SAMPLE_RATE * 2)

    with pytest.raises(ValueError, match="Hz"):
        dataset.add_episode_audio(AUDIO_KEY, wav_path)


@pytest.mark.parametrize(
    "feature",
    [
        {"dtype": "audio", "shape": (2,), "names": None, "info": {}},
        {"dtype": "audio", "shape": (2,), "names": None, "info": {"audio.sample_rate": 0}},
        {"dtype": "audio", "shape": (2, 160), "names": None, "info": {"audio.sample_rate": SAMPLE_RATE}},
    ],
)
def test_create_invalid_audio_feature(tmp_path, feature):
    with pytest.raises(ValueError):
        LeRobotDataset.create(
            DUMMY_REPO_ID, fps=FPS, features={AUDIO_KEY: feature}, root=tmp_path / "dataset"
        )


def test_invalid_audio_window(audio_dataset):
    root, _ = audio_dataset
    with pytest.raises(ValueError, match="audio_window_s"):
        LeRobotDataset(DUMMY_REPO_ID, root=root, audio_window_s=0.0)


def test_policy_features():
    features = dataset_to_policy_features(FEATURES)
    assert features[AUDIO_KEY].type == FeatureType.AUDIO
    assert features[AUDIO_KEY].shape == (NUM_CHANNELS,)
    assert features["observation.state"].type == FeatureType.STATE


def test_unsupported_operations(audio_dataset, tmp_path):
    root, _ = audio_dataset
    dataset = LeRobotDataset(DUMMY_REPO_ID, root=root)

    with pytest.raises(NotImplementedError, match="audio"):
        delete_episodes(dataset, [0], output_dir=tmp_path / "deleted")
    with pytest.raises(NotImplementedError, match="audio"):
        aggregate_datasets([DUMMY_REPO_ID], "dummy/aggregated", roots=[root], aggr_root=tmp_path / "aggr")
    with pytest.raises(NotImplementedError, match="audio"):
        StreamingLeRobotDataset(DUMMY_REPO_ID, root=root)


def test_concatenate_audio_files(tmp_path):
    first, second = make_audio(5, seed=0), make_audio(5, seed=1)
    write_audio_file(tmp_path / "a.flac", first, SAMPLE_RATE)
    write_audio_file(tmp_path / "b.flac", second, SAMPLE_RATE)

    concatenate_audio_files([tmp_path / "a.flac", tmp_path / "b.flac"], tmp_path / "a.flac")

    stored, _ = sf.read(tmp_path / "a.flac", dtype="float32")
    np.testing.assert_allclose(stored, np.concatenate([first, second]), atol=ATOL)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.flac", "b.flac"]

    write_audio_file(tmp_path / "c.flac", second, SAMPLE_RATE * 2)
    with pytest.raises(ValueError, match="Cannot concatenate"):
        concatenate_audio_files([tmp_path / "a.flac", tmp_path / "c.flac"], tmp_path / "d.flac")
    assert not (tmp_path / "d.flac").exists()
    assert not (tmp_path / ".d.flac.tmp").exists()

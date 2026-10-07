#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import lerobot.datasets.streaming_dataset as streaming_dataset_module
from lerobot.datasets.io_utils import write_episodes
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.utils.constants import ACTION
from tests.fixtures.constants import DUMMY_REPO_ID


@pytest.mark.parametrize("token", ["hf_test_token", True, False])
@pytest.mark.parametrize("from_local", [False, True])
def test_streaming_dataset_forwards_token_to_metadata_and_remote_worker_io(
    tmp_path, monkeypatch, token, from_local
):
    requested_root = tmp_path / "local" if from_local else None
    metadata = SimpleNamespace(
        repo_id=DUMMY_REPO_ID,
        root=requested_root or tmp_path / "snapshots" / ("a" * 40),
        revision=streaming_dataset_module.CODEBASE_VERSION,
        _version=streaming_dataset_module.CODEBASE_VERSION,
        features={},
        total_episodes=0,
        video_keys=[],
        depth_keys=[],
        image_keys=[],
        rescale_depth_stats=Mock(),
    )
    metadata_cls = Mock(return_value=metadata)
    ensure_sidecar = Mock(return_value=None)
    monkeypatch.setattr(streaming_dataset_module, "LeRobotDatasetMetadata", metadata_cls)
    monkeypatch.setattr(streaming_dataset_module, "ensure_dataset_mp4_sidecar", ensure_sidecar)

    dataset = StreamingLeRobotDataset(DUMMY_REPO_ID, root=requested_root, token=token)

    metadata_cls.assert_called_once_with(
        DUMMY_REPO_ID,
        requested_root,
        streaming_dataset_module.CODEBASE_VERSION,
        force_cache_sync=False,
        repo_type="dataset",
        token=token,
    )
    assert ensure_sidecar.call_args.kwargs["token"] is (None if from_local else token)
    assert dataset._streaming_io_token is (None if from_local else token)
    assert dataset._data_root == (
        str(requested_root) if from_local else f"hf://datasets/{DUMMY_REPO_ID}@{'a' * 40}"
    )


def test_single_frame_consistency(tmp_path, lerobot_dataset_factory):
    """Test if are correctly accessed"""
    ds_num_frames = 400
    ds_num_episodes = 10
    buffer_size = 100

    local_path = tmp_path / "test"
    repo_id = f"{DUMMY_REPO_ID}"

    ds = lerobot_dataset_factory(
        root=local_path,
        repo_id=repo_id,
        total_episodes=ds_num_episodes,
        total_frames=ds_num_frames,
    )

    streaming_ds = iter(StreamingLeRobotDataset(repo_id=repo_id, root=local_path, buffer_size=buffer_size))

    key_checks = []
    for _ in range(ds_num_frames):
        streaming_frame = next(streaming_ds)
        frame_idx = int(streaming_frame["index"])
        target_frame = ds[frame_idx]

        for key in streaming_frame:
            left = streaming_frame[key]
            right = target_frame[key]

            if isinstance(left, str):
                check = left == right

            elif isinstance(left, torch.Tensor):
                check = torch.allclose(left, right) and left.shape == right.shape

            elif isinstance(left, float):
                check = left == right.item()  # right is a torch.Tensor

            key_checks.append((key, check))

        assert all(t[1] for t in key_checks), (
            f"Checking {list(filter(lambda t: not t[1], key_checks))[0][0]} left and right were found different (frame_idx: {frame_idx})"
        )


@pytest.mark.parametrize(
    "shuffle",
    [False, True],
)
def test_frames_order_over_epochs(tmp_path, lerobot_dataset_factory, shuffle):
    """Test if streamed frames correspond to shuffling operations over in-memory dataset."""
    ds_num_frames = 400
    ds_num_episodes = 10
    buffer_size = 100
    seed = 42
    n_epochs = 3

    local_path = tmp_path / "test"
    repo_id = f"{DUMMY_REPO_ID}"

    lerobot_dataset_factory(
        root=local_path,
        repo_id=repo_id,
        total_episodes=ds_num_episodes,
        total_frames=ds_num_frames,
    )

    streaming_ds = StreamingLeRobotDataset(
        repo_id=repo_id, root=local_path, buffer_size=buffer_size, seed=seed, shuffle=shuffle
    )

    first_epoch_indices = [int(frame["index"]) for frame in streaming_ds]
    assert sorted(first_epoch_indices) == list(range(ds_num_frames))
    for _ in range(n_epochs):
        streaming_indices = [int(frame["index"]) for frame in streaming_ds]
        assert sorted(streaming_indices) == list(range(ds_num_frames))

        if shuffle:
            assert streaming_indices != first_epoch_indices
        else:
            assert streaming_indices == first_epoch_indices


@pytest.mark.parametrize(
    "shuffle",
    [False, True],
)
def test_frames_order_with_shards(tmp_path, lerobot_dataset_factory, shuffle):
    """Test if streamed frames correspond to shuffling operations over in-memory dataset with multiple shards."""
    ds_num_frames = 100
    ds_num_episodes = 10
    buffer_size = 10

    seed = 42
    n_epochs = 3
    data_file_size_mb = 0.001

    chunks_size = 1

    local_path = tmp_path / "test"
    repo_id = f"{DUMMY_REPO_ID}-ciao"

    lerobot_dataset_factory(
        root=local_path,
        repo_id=repo_id,
        total_episodes=ds_num_episodes,
        total_frames=ds_num_frames,
        data_files_size_in_mb=data_file_size_mb,
        chunks_size=chunks_size,
    )

    streaming_ds = StreamingLeRobotDataset(
        repo_id=repo_id,
        root=local_path,
        buffer_size=buffer_size,
        seed=seed,
        shuffle=shuffle,
        max_num_shards=4,
    )

    first_epoch_indices = [int(frame["index"]) for frame in streaming_ds]
    assert sorted(first_epoch_indices) == list(range(ds_num_frames))

    for _ in range(n_epochs):
        streaming_indices = [int(frame["index"]) for frame in streaming_ds]
        assert sorted(streaming_indices) == list(range(ds_num_frames))
        if shuffle:
            assert streaming_indices != first_epoch_indices
        else:
            assert streaming_indices == first_epoch_indices


@pytest.mark.parametrize("delta_mode", ["none", "action", "cameras"])
def test_make_frame_uses_timestamps_relative_to_each_video_file(
    tmp_path, lerobot_dataset_factory, create_videos, monkeypatch, delta_mode
):
    """Iterate real rows with camera-specific offsets, including a padded video query."""
    local_path = tmp_path / "dataset"
    dataset = lerobot_dataset_factory(
        root=local_path,
        total_episodes=1,
        total_frames=5,
        download_videos=False,
    )
    offsets = {"laptop": 5.0, "phone": 1.0}
    delta_timestamps = None
    if delta_mode == "action":
        delta_timestamps = {ACTION: [0, 1 / dataset.fps]}
    elif delta_mode == "cameras":
        delta_timestamps = {key: [-1 / dataset.fps, 0, 1 / dataset.fps] for key in offsets}

    # Model an episode beginning at different positions within its camera files.
    episodes = dataset.meta.episodes.map(
        lambda episode: {
            f"videos/{key}/{boundary}_timestamp": episode[f"videos/{key}/{boundary}_timestamp"] + offset
            for key, offset in offsets.items()
            for boundary in ("from", "to")
        }
    )

    write_episodes(episodes, local_path)
    # The byte manifest needs actual files that cover both camera offsets.
    video_info = replace(dataset.meta.info, total_frames=6 * dataset.fps)
    create_videos(root=local_path, info=video_info)
    streaming_ds = StreamingLeRobotDataset(
        repo_id=DUMMY_REPO_ID,
        root=local_path,
        buffer_size=1,
        shuffle=False,
        delta_timestamps=delta_timestamps,
    )

    def decode_timestamps(self, episode_index, video_key, timestamps):
        # Use float markers so normalization returns the requested timestamps.
        # The stand-in exposes query errors in the returned samples.
        return (255 * torch.tensor(timestamps)).reshape(-1, 1, 1, 1).expand(-1, 3, 64, 96)

    monkeypatch.setattr(streaming_dataset_module.EpisodeByteCache, "get_frames", decode_timestamps)
    samples = list(streaming_ds)

    assert len(samples) == 5
    assert {int(sample["frame_index"]) for sample in samples} == set(range(5))
    for sample in samples:
        frame_index = int(sample["frame_index"])
        for key, offset in offsets.items():
            current_timestamp = offset + frame_index / dataset.fps
            if delta_mode == "cameras":
                expected = [
                    max(offset, current_timestamp - 1 / dataset.fps),
                    current_timestamp,
                    min(offset + 4 / dataset.fps, current_timestamp + 1 / dataset.fps),
                ]
                assert sample[key][:, 0, 0, 0].tolist() == pytest.approx(expected)
                # Past and future queries cross the episode boundaries at its ends.
                assert sample[f"{key}_is_pad"].tolist() == [frame_index == 0, False, frame_index == 4]
            else:
                assert sample[key][0, 0, 0].item() == pytest.approx(current_timestamp)
                assert f"{key}_is_pad" not in sample


@pytest.mark.parametrize(
    "state_deltas, action_deltas",
    [
        ([-1, -0.5, -0.20, 0], [0, 1, 2, 3]),
        ([-1, -0.5, -0.20, 0], [-1.5, -1, -0.5, -0.20, -0.10, 0]),
        ([-2, -1, -0.5, 0], [0, 1, 2, 3]),
        ([-2, -1, -0.5, 0], [-1.5, -1, -0.5, -0.20, -0.10, 0]),
    ],
)
def test_frames_with_delta_consistency(tmp_path, lerobot_dataset_factory, state_deltas, action_deltas):
    ds_num_frames = 500
    ds_num_episodes = 10
    buffer_size = 100

    seed = 42

    local_path = tmp_path / "test"
    repo_id = f"{DUMMY_REPO_ID}-ciao"
    camera_key = "phone"

    delta_timestamps = {
        camera_key: state_deltas,
        "state": state_deltas,
        ACTION: action_deltas,
    }

    ds = lerobot_dataset_factory(
        root=local_path,
        repo_id=repo_id,
        total_episodes=ds_num_episodes,
        total_frames=ds_num_frames,
        delta_timestamps=delta_timestamps,
    )

    streaming_ds = iter(
        StreamingLeRobotDataset(
            repo_id=repo_id,
            root=local_path,
            buffer_size=buffer_size,
            seed=seed,
            shuffle=False,
            delta_timestamps=delta_timestamps,
        )
    )

    for i in range(ds_num_frames):
        streaming_frame = next(streaming_ds)
        frame_idx = int(streaming_frame["index"])
        target_frame = ds[frame_idx]

        assert set(streaming_frame.keys()) == set(target_frame.keys()), (
            f"Keys differ between streaming frame and target one. Differ at: {set(streaming_frame.keys()) - set(target_frame.keys())}"
        )

        key_checks = []
        for key in streaming_frame:
            left = streaming_frame[key]
            right = target_frame[key]

            if isinstance(left, str):
                check = left == right

            elif isinstance(left, torch.Tensor):
                if (
                    key not in ds.meta.camera_keys
                    and "is_pad" not in key
                    and f"{key}_is_pad" in streaming_frame
                ):
                    # comparing frames only on non-padded regions. Padding is applied to last-valid broadcasting
                    left = left[~streaming_frame[f"{key}_is_pad"]]
                    right = right[~target_frame[f"{key}_is_pad"]]

                check = torch.allclose(left, right) and left.shape == right.shape

            key_checks.append((key, check))

        assert all(t[1] for t in key_checks), (
            f"Checking {list(filter(lambda t: not t[1], key_checks))[0][0]} left and right were found different (i: {i}, frame_idx: {frame_idx})"
        )


@pytest.mark.parametrize(
    "state_deltas, action_deltas",
    [
        ([-1, -0.5, -0.20, 0], [0, 1, 2, 3, 10, 20]),
        ([-1, -0.5, -0.20, 0], [-20, -1.5, -1, -0.5, -0.20, -0.10, 0]),
        ([-2, -1, -0.5, 0], [0, 1, 2, 3, 10, 20]),
        ([-2, -1, -0.5, 0], [-20, -1.5, -1, -0.5, -0.20, -0.10, 0]),
    ],
)
def test_frames_with_delta_consistency_with_shards(
    tmp_path, lerobot_dataset_factory, state_deltas, action_deltas
):
    ds_num_frames = 100
    ds_num_episodes = 10
    buffer_size = 10
    data_file_size_mb = 0.001
    chunks_size = 1

    seed = 42

    local_path = tmp_path / "test"
    repo_id = f"{DUMMY_REPO_ID}-ciao"
    camera_key = "phone"

    delta_timestamps = {
        camera_key: state_deltas,
        "state": state_deltas,
        ACTION: action_deltas,
    }

    ds = lerobot_dataset_factory(
        root=local_path,
        repo_id=repo_id,
        total_episodes=ds_num_episodes,
        total_frames=ds_num_frames,
        delta_timestamps=delta_timestamps,
        data_files_size_in_mb=data_file_size_mb,
        chunks_size=chunks_size,
    )
    streaming_ds = StreamingLeRobotDataset(
        repo_id=repo_id,
        root=local_path,
        buffer_size=buffer_size,
        seed=seed,
        shuffle=False,
        delta_timestamps=delta_timestamps,
        max_num_shards=4,
    )

    streaming_ds = iter(streaming_ds)

    for i in range(ds_num_frames):
        streaming_frame = next(streaming_ds)
        frame_idx = int(streaming_frame["index"])
        target_frame = ds[frame_idx]

        assert set(streaming_frame.keys()) == set(target_frame.keys()), (
            f"Keys differ between streaming frame and target one. Differ at: {set(streaming_frame.keys()) - set(target_frame.keys())}"
        )

        key_checks = []
        for key in streaming_frame:
            left = streaming_frame[key]
            right = target_frame[key]

            if isinstance(left, str):
                check = left == right

            elif isinstance(left, torch.Tensor):
                if (
                    key not in ds.meta.camera_keys
                    and "is_pad" not in key
                    and f"{key}_is_pad" in streaming_frame
                ):
                    # comparing frames only on non-padded regions. Padding is applied to last-valid broadcasting
                    left = left[~streaming_frame[f"{key}_is_pad"]]
                    right = right[~target_frame[f"{key}_is_pad"]]

                check = torch.allclose(left, right) and left.shape == right.shape

            elif isinstance(left, float):
                check = left == right.item()  # right is a torch.Tensor

            key_checks.append((key, check))

        assert all(t[1] for t in key_checks), (
            f"Checking {list(filter(lambda t: not t[1], key_checks))[0][0]} left and right were found different (i: {i}, frame_idx: {frame_idx})"
        )

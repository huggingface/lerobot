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

"""Tests for the HF-native large-scale streaming additions: distributed (per-rank) sharding,
DataLoader worker splitting, the episode pool (randomness, coverage, exact deltas), video
prefetching, deterministic fast-forward resume, and schema parity."""

import pytest
import torch
from torch.utils.data import DataLoader

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.utils.constants import ACTION
from tests.fixtures.constants import DUMMY_REPO_ID


def _make_local_dataset(factory, root, repo_id, *, total_episodes, total_frames, use_videos=False, **kw):
    factory(
        root=root,
        repo_id=repo_id,
        total_episodes=total_episodes,
        total_frames=total_frames,
        use_videos=use_videos,
        data_files_size_in_mb=0.001,
        chunks_size=1,
        **kw,
    )


def _stream_indices(ds: StreamingLeRobotDataset) -> list[int]:
    return [int(frame["index"]) for frame in ds]


def test_resolve_distributed_prefers_explicit_then_env(monkeypatch):
    assert StreamingLeRobotDataset._resolve_distributed(2, 8) == (2, 8)

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    # No accelerate state, no env -> single process.
    assert StreamingLeRobotDataset._resolve_distributed(None, None) == (0, 1)

    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert StreamingLeRobotDataset._resolve_distributed(None, None) == (3, 4)


def test_split_by_node_disjoint_across_ranks(tmp_path, lerobot_dataset_factory):
    """Each rank must stream a disjoint set of frames, and the ranks together must cover every frame."""
    repo_id = f"{DUMMY_REPO_ID}-ranks"
    total_frames, total_episodes = 200, 8
    _make_local_dataset(
        lerobot_dataset_factory,
        tmp_path / "ds",
        repo_id,
        total_episodes=total_episodes,
        total_frames=total_frames,
    )

    world_size = 2
    per_rank = []
    for rank in range(world_size):
        ds = StreamingLeRobotDataset(
            repo_id=repo_id,
            root=tmp_path / "ds",
            shuffle=False,
            episode_pool_size=8,
            max_num_shards=8,
            rank=rank,
            world_size=world_size,
        )
        per_rank.append(set(_stream_indices(ds)))

    assert per_rank[0].isdisjoint(per_rank[1]), (
        "ranks streamed overlapping frames (duplicate data across GPUs)"
    )
    assert per_rank[0] | per_rank[1] == set(range(total_frames)), "ranks did not jointly cover all frames"


def test_dataloader_workers_no_duplicates_within_rank(tmp_path, lerobot_dataset_factory):
    """DataLoader workers within a rank must split shards so no frame is yielded twice."""
    repo_id = f"{DUMMY_REPO_ID}-workers"
    total_frames, total_episodes = 120, 8
    _make_local_dataset(
        lerobot_dataset_factory,
        tmp_path / "ds",
        repo_id,
        total_episodes=total_episodes,
        total_frames=total_frames,
    )

    ds = StreamingLeRobotDataset(
        repo_id=repo_id, root=tmp_path / "ds", shuffle=False, episode_pool_size=4, max_num_shards=4
    )
    loader = DataLoader(ds, batch_size=None, num_workers=2)
    indices = [int(batch["index"]) for batch in loader]

    assert len(indices) == len(set(indices)), "DataLoader workers yielded duplicate frames within a rank"


def test_sarm_window_covers_long_horizon_without_padding(tmp_path, lerobot_dataset_factory):
    """A delta window longer than the old 100-frame ceiling must fetch real frames, not pad them.

    SARM uses a window of 8 steps spaced 1s (~160 frames @ fps20). Here fps=30, so +5s = 150 frames > 100.
    """
    repo_id = f"{DUMMY_REPO_ID}-sarm"
    # A single long episode so a +150-frame lookahead is unambiguously inside the episode (the fixture
    # gives episodes variable lengths, so multi-episode boundaries can't be assumed).
    episode_frames = 300
    _make_local_dataset(
        lerobot_dataset_factory, tmp_path / "ds", repo_id, total_episodes=1, total_frames=episode_frames
    )

    horizon_s = 5.0  # 150 frames @ fps30, well beyond LOOKAHEAD_BACKTRACKTABLE=100
    delta_timestamps = {ACTION: [0.0, horizon_s]}
    ds = StreamingLeRobotDataset(
        repo_id=repo_id,
        root=tmp_path / "ds",
        shuffle=False,
        episode_pool_size=1,
        max_num_shards=1,
        delta_timestamps=delta_timestamps,
    )

    horizon_frames = int(round(horizon_s * ds.fps))
    assert horizon_frames > 100, "test must exceed the old LOOKAHEAD_BACKTRACKTABLE ceiling"
    checked = 0
    for frame in ds:
        idx = int(frame["index"])
        # The +horizon target is inside the single episode -> it must be a real frame, not padding.
        if idx + horizon_frames < episode_frames:
            assert not bool(frame[f"{ACTION}_is_pad"][-1]), (
                f"frame {idx}: +{horizon_frames} target was padded; long delta window did not reach it"
            )
            checked += 1
    assert checked > 0, "test did not exercise any in-episode long-horizon frame"


def test_pool_order_is_deterministic_per_seed(tmp_path, lerobot_dataset_factory):
    repo_id = f"{DUMMY_REPO_ID}-seeds"
    _make_local_dataset(lerobot_dataset_factory, tmp_path / "ds", repo_id, total_episodes=6, total_frames=120)

    def order(seed):
        return _stream_indices(
            StreamingLeRobotDataset(
                repo_id=repo_id,
                root=tmp_path / "ds",
                shuffle=True,
                seed=seed,
                episode_pool_size=4,
                max_num_shards=2,
            )
        )

    assert order(0) == order(0), "same seed must reproduce the same order"
    assert order(0) != order(1), "different seeds should give different orders"


def test_pool_epochs_reshuffle_and_cover(tmp_path, lerobot_dataset_factory):
    """Consecutive passes over the same dataset object reshuffle (epoch advances) but keep coverage."""
    repo_id = f"{DUMMY_REPO_ID}-epochs"
    total_frames = 120
    _make_local_dataset(
        lerobot_dataset_factory, tmp_path / "ds", repo_id, total_episodes=6, total_frames=total_frames
    )
    ds = StreamingLeRobotDataset(
        repo_id=repo_id, root=tmp_path / "ds", shuffle=True, seed=3, episode_pool_size=4, max_num_shards=2
    )
    epoch_0 = _stream_indices(ds)
    epoch_1 = _stream_indices(ds)
    assert sorted(epoch_0) == sorted(epoch_1) == list(range(total_frames))
    assert epoch_0 != epoch_1, "epoch did not reshuffle"


def test_pool_mixes_episodes(tmp_path, lerobot_dataset_factory):
    """Early samples should already come from several distinct episodes (the pool's purpose)."""
    repo_id = f"{DUMMY_REPO_ID}-mix"
    _make_local_dataset(lerobot_dataset_factory, tmp_path / "ds", repo_id, total_episodes=8, total_frames=200)
    ds = StreamingLeRobotDataset(
        repo_id=repo_id, root=tmp_path / "ds", shuffle=True, seed=0, episode_pool_size=8, max_num_shards=4
    )
    episodes_in_head = {int(frame["episode_index"]) for _, frame in zip(range(20), ds, strict=False)}
    assert len(episodes_in_head) >= 3, f"pool did not mix episodes: {episodes_in_head}"


def test_schema_parity_with_map_style(tmp_path, lerobot_dataset_factory):
    """Streamed samples must have the same keys / shapes / dtypes as map-style LeRobotDataset."""
    repo_id = f"{DUMMY_REPO_ID}-parity"
    map_ds = lerobot_dataset_factory(
        root=tmp_path / "ds", repo_id=repo_id, total_episodes=4, total_frames=80, use_videos=True
    )
    stream_ds = StreamingLeRobotDataset(
        repo_id=repo_id, root=tmp_path / "ds", shuffle=False, episode_pool_size=4, max_num_shards=2
    )

    map_frame = map_ds[0]
    stream_frame = next(iter(stream_ds))

    assert set(stream_frame) == set(map_frame), set(stream_frame) ^ set(map_frame)
    for key, value in stream_frame.items():
        ref = map_frame[key]
        if isinstance(value, torch.Tensor):
            assert isinstance(ref, torch.Tensor) and value.shape == ref.shape and value.dtype == ref.dtype, (
                f"{key}: stream {tuple(value.shape)}/{value.dtype} vs map {tuple(ref.shape)}/{ref.dtype}"
            )
        elif isinstance(value, str):
            assert isinstance(ref, str), f"{key}: {type(value)} vs {type(ref)}"
        else:
            # Scalar numerics: streaming yields python floats where map-style yields 0-dim tensors
            # (a long-standing, accepted difference). Compare by value rather than exact type.
            assert float(value) == float(ref), f"{key}: {value} vs {ref}"


def test_video_path_resolution_local(tmp_path, lerobot_dataset_factory, monkeypatch):
    """For a local (prewarmed) root, video decode must be issued against the local path, not hf://."""
    import lerobot.datasets.streaming_dataset as sd

    repo_id = f"{DUMMY_REPO_ID}-vpath"
    lerobot_dataset_factory(
        root=tmp_path / "ds", repo_id=repo_id, total_episodes=2, total_frames=40, use_videos=True
    )
    ds = StreamingLeRobotDataset(
        repo_id=repo_id, root=tmp_path / "ds", shuffle=False, episode_pool_size=1, max_num_shards=1
    )

    seen_paths = []

    def fake_decode(video_path, query_ts, *args, **kwargs):
        seen_paths.append(str(video_path))
        return torch.zeros(len(query_ts), 3, 64, 96)

    monkeypatch.setattr(sd, "decode_video_frames_torchcodec", fake_decode)
    next(iter(ds))

    assert seen_paths, "no video decode was issued"
    assert all(str(ds.root) in p and not p.startswith("hf://") for p in seen_paths), seen_paths


def test_shuffle_decorrelates_output_order(tmp_path, lerobot_dataset_factory):
    """With shuffle on, streamed frame order must differ from the underlying sequential order."""
    repo_id = f"{DUMMY_REPO_ID}-shuf"
    _make_local_dataset(lerobot_dataset_factory, tmp_path / "ds", repo_id, total_episodes=8, total_frames=200)
    ordered = _stream_indices(
        StreamingLeRobotDataset(
            repo_id=repo_id, root=tmp_path / "ds", shuffle=False, episode_pool_size=1, max_num_shards=1
        )
    )
    shuffled = _stream_indices(
        StreamingLeRobotDataset(
            repo_id=repo_id, root=tmp_path / "ds", shuffle=True, episode_pool_size=8, max_num_shards=4, seed=0
        )
    )
    assert sorted(shuffled) == sorted(ordered), "shuffling changed the set of frames"
    assert shuffled != ordered, "shuffle did not decorrelate output order"


def test_native_resume_never_repeats_and_loss_is_bounded(tmp_path, lerobot_dataset_factory):
    """Native state_dict resume: no sample is re-yielded; loss is bounded by the shuffle buffers."""
    repo_id = f"{DUMMY_REPO_ID}-native-resume"
    total_frames = 100
    _make_local_dataset(
        lerobot_dataset_factory, tmp_path / "ds", repo_id, total_episodes=5, total_frames=total_frames
    )

    def fresh_ds():
        return StreamingLeRobotDataset(
            repo_id=repo_id,
            root=tmp_path / "ds",
            shuffle=True,
            seed=7,
            episode_pool_size=2,
            frame_shuffle_buffer_size=8,
        )

    ds = fresh_ds()
    it = iter(ds)
    consumed = [int(next(it)["index"]) for _ in range(30)]
    state = ds.state_dict()

    resumed_ds = fresh_ds()
    resumed_ds.load_state_dict(state)
    rest = [int(frame["index"]) for frame in resumed_ds]

    assert not set(consumed) & set(rest), "resume re-yielded already-seen frames"
    # in-flight buffer contents are skipped on resume (documented datasets behavior):
    # bounded by the episode pool (2 episodes of <= ~30 frames here) + frame buffer (8)
    covered = len(set(consumed) | set(rest))
    max_in_flight = 2 * 30 + 8
    assert covered >= total_frames - max_in_flight
    assert covered + len(consumed) >= total_frames - max_in_flight


def test_pipeline_uses_native_primitives(tmp_path, lerobot_dataset_factory):
    """The tabular pipeline is pure datasets: batch(by_column) + shuffle + map + shuffle."""
    repo_id = f"{DUMMY_REPO_ID}-native-pipe"
    _make_local_dataset(lerobot_dataset_factory, tmp_path / "ds", repo_id, total_episodes=4, total_frames=80)
    ds = StreamingLeRobotDataset(repo_id=repo_id, root=tmp_path / "ds", shuffle=True, episode_pool_size=2)
    import datasets as hf_datasets

    assert isinstance(ds._pipeline, hf_datasets.IterableDataset)
    state = ds._pipeline.state_dict()  # the native resume protocol is available end-to-end
    assert state is not None

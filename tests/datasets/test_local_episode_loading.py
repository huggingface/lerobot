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

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets import dataset_reader as dataset_reader_module
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler, balanced_episode_shards
from lerobot.datasets.streaming_sidecar import ensure_dataset_mp4_sidecar, make_sidecar_spec
from tests.fixtures.constants import DUMMY_CAMERA_FEATURES_WITH_DEPTH, DUMMY_REPO_ID


def _redirect_sidecar_cache(monkeypatch, cache_root: Path) -> None:
    real_ensure = ensure_dataset_mp4_sidecar
    monkeypatch.setattr(
        dataset_reader_module,
        "ensure_dataset_mp4_sidecar",
        lambda meta, data_root, **kwargs: real_ensure(meta, data_root, cache_root=cache_root, **kwargs),
    )


def test_mutating_local_video_invalidates_sidecar_identity(tmp_path: Path) -> None:
    relative_path = Path("videos/camera/chunk-000/file-000.mp4")
    video_path = tmp_path / relative_path
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"first-payload")
    meta = SimpleNamespace(
        repo_id=DUMMY_REPO_ID,
        revision="v3.0",
        total_episodes=1,
        video_keys=["camera"],
        get_video_file_path=lambda _episode, _key: relative_path,
    )
    original_stat = video_path.stat()
    first = make_sidecar_spec(meta, str(tmp_path))

    video_path.write_bytes(b"other-payload")
    os.utime(video_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second = make_sidecar_spec(meta, str(tmp_path))

    assert first.source_files == second.source_files
    assert first.source_fingerprint != second.source_fingerprint


@pytest.mark.parametrize("video_backend", ["pyav", "torchcodec"])
def test_indexed_local_video_reader_matches_map_style(
    tmp_path: Path,
    lerobot_dataset_factory,
    monkeypatch,
    video_backend: str,
) -> None:
    root = tmp_path / "dataset"
    reference = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=20,
        video_backend=video_backend,
    )
    _redirect_sidecar_cache(monkeypatch, tmp_path / "sidecars")
    indexed = LeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        episodes=[1],
        video_backend=video_backend,
        local_episode_loading=True,
    )

    for relative_index in (0, len(indexed) - 1):
        actual = indexed[relative_index]
        expected = reference[int(actual["index"])]
        for key in indexed.meta.camera_keys:
            assert torch.equal(actual[key], expected[key]), key
        assert actual["task"] == expected["task"]

    assert indexed.reader.local_video_cache is not None
    indexed.close()
    assert indexed.reader.local_video_cache is None


def test_indexed_reader_preserves_depth(tmp_path: Path, lerobot_dataset_factory, monkeypatch) -> None:
    root = tmp_path / "dataset"
    reference = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=20,
        camera_features=DUMMY_CAMERA_FEATURES_WITH_DEPTH,
        video_backend="pyav",
    )
    _redirect_sidecar_cache(monkeypatch, tmp_path / "sidecars")
    indexed = LeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        episodes=[1],
        video_backend="pyav",
        local_episode_loading=True,
    )

    actual = indexed[0]
    expected = reference[int(actual["index"])]
    for key in indexed.meta.camera_keys:
        assert torch.equal(actual[key], expected[key]), key
    indexed.close()


def test_indexed_reader_preserves_temporal_padding_transforms_and_uint8(
    tmp_path: Path, lerobot_dataset_factory, monkeypatch
) -> None:
    root = tmp_path / "dataset"
    created = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=20,
        video_backend="pyav",
    )
    delta_timestamps = {key: [-1 / created.fps, 0] for key in created.meta.video_keys}

    def flip_width(image: torch.Tensor) -> torch.Tensor:
        return image.flip(-1)

    reference = LeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        episodes=[1],
        video_backend="pyav",
        delta_timestamps=delta_timestamps,
        image_transforms=flip_width,
        return_uint8=True,
    )
    _redirect_sidecar_cache(monkeypatch, tmp_path / "sidecars")
    indexed = LeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        episodes=[1],
        video_backend="pyav",
        delta_timestamps=delta_timestamps,
        image_transforms=flip_width,
        return_uint8=True,
        local_episode_loading=True,
    )

    for index in (0, len(indexed) - 1):
        actual, expected = indexed[index], reference[index]
        for key in indexed.meta.video_keys:
            assert actual[key].dtype == torch.uint8
            assert torch.equal(actual[key], expected[key]), key
            assert torch.equal(actual[f"{key}_is_pad"], expected[f"{key}_is_pad"]), key
    indexed.close()


def test_indexed_reader_works_with_spawned_persistent_worker(
    tmp_path: Path, lerobot_dataset_factory, monkeypatch
) -> None:
    root = tmp_path / "dataset"
    lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=2,
        total_frames=20,
        video_backend="pyav",
    )
    _redirect_sidecar_cache(monkeypatch, tmp_path / "sidecars")
    indexed = LeRobotDataset(
        DUMMY_REPO_ID,
        root=root,
        episodes=[1],
        video_backend="pyav",
        local_episode_loading=True,
    )
    loader = torch.utils.data.DataLoader(
        indexed,
        batch_size=2,
        num_workers=1,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    iterator = iter(loader)
    try:
        first, second = next(iterator), next(iterator)
    finally:
        iterator._shutdown_workers()

    assert first["index"].shape == second["index"].shape == (2,)
    assert indexed.reader.local_video_cache is None


def test_rank_episode_shards_have_exact_coverage_before_padding(
    tmp_path: Path, lerobot_dataset_factory
) -> None:
    root = tmp_path / "dataset"
    created = lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        total_episodes=5,
        total_frames=47,
        use_videos=False,
    )
    episodes = list(range(created.meta.total_episodes))
    counts = {
        episode: int(
            created.meta.episodes[episode]["dataset_to_index"]
            - created.meta.episodes[episode]["dataset_from_index"]
        )
        for episode in episodes
    }
    shards = balanced_episode_shards(episodes, counts, world_size=2)
    padded_frames = max(sum(counts[episode] for episode in shard) for shard in shards)
    coverage = []
    lengths = []

    for shard in shards:
        dataset = LeRobotDataset(
            DUMMY_REPO_ID,
            root=root,
            episodes=shard,
            local_episode_loading=True,
        )
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=shard,
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
            pad_to_num_frames=padded_frames,
        )
        sampled = list(sampler)
        true_frames = sum(counts[episode] for episode in shard)
        coverage.extend(int(dataset[index]["index"]) for index in sampled[:true_frames])
        lengths.append(len(sampled))

    assert sorted(coverage) == list(range(created.meta.total_frames))
    assert len(coverage) == len(set(coverage))
    assert lengths == [padded_frames, padded_frames]

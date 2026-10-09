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
"""Several DataLoader workers for each rank: coverage, shards, resume, determinism and limits."""

from __future__ import annotations

from collections import Counter
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import datasets  # noqa: E402

from lerobot.datasets.streaming_dataset import (  # noqa: E402
    StreamingLeRobotDataset,
    _balanced_episode_shards,
    _batches_per_worker,
)
from lerobot.utils.utils import cycle  # noqa: E402
from tests.fixtures.constants import DUMMY_REPO_ID  # noqa: E402

# Fixed, uneven episode lengths: random multinomial lengths can contain empty episodes.
LENGTHS = (5, 9, 3, 11, 6, 8, 4, 10)


@pytest.fixture
def root(tmp_path: Path, info_factory, tasks_factory, episodes_factory, lerobot_dataset_factory) -> Path:
    root = tmp_path / "dataset"
    total = sum(LENGTHS)
    info = info_factory(total_episodes=len(LENGTHS), total_frames=total, total_tasks=1, use_videos=False)
    tasks = tasks_factory(total_tasks=1)
    episodes = episodes_factory(
        features=info.features, total_episodes=len(LENGTHS), total_frames=total, tasks=tasks
    ).to_list()
    start = 0
    for episode, length in zip(episodes, LENGTHS, strict=True):
        episode.update(length=length, dataset_from_index=start, dataset_to_index=start + length)
        start += length
    lerobot_dataset_factory(
        root=root,
        repo_id=DUMMY_REPO_ID,
        info=info,
        tasks=tasks,
        episodes_metadata=datasets.Dataset.from_list(episodes),
    )
    return root


def _dataset(root: Path, **kwargs) -> StreamingLeRobotDataset:
    kwargs.setdefault("seed", 7)
    return StreamingLeRobotDataset(DUMMY_REPO_ID, root=root, **kwargs)


def _batches(
    dataset: StreamingLeRobotDataset, num_workers: int, batch_size: int | None, count: int | None = None
) -> list[list[int]]:
    """Read anchor indices batch by batch; ``count`` batches from an endless (repeat) stream."""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=count is not None,
    )
    batches = loader if count is None else islice(cycle(loader), count)
    try:
        if batch_size is None:
            return [[int(item["index"])] for item in batches]
        return [[int(index) for index in batch["index"]] for batch in batches]
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def _as_worker(monkeypatch, worker_id: int, num_workers: int) -> None:
    info = SimpleNamespace(id=worker_id, num_workers=num_workers)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: info)


@pytest.mark.parametrize("num_workers", [2, 3])
@pytest.mark.parametrize("batch_size", [None, 4])
@pytest.mark.parametrize("sampling_strategy", ["remaining", "round_robin"])
def test_workers_yield_every_frame_once(root: Path, num_workers, batch_size, sampling_strategy) -> None:
    dataset = _dataset(root, sampling_strategy=sampling_strategy)

    seen = Counter(index for batch in _batches(dataset, num_workers, batch_size) for index in batch)

    assert seen == Counter(range(sum(LENGTHS)))


@pytest.mark.parametrize("num_workers", [2, 3])
def test_worker_shares_split_the_rank_shard(root: Path, monkeypatch, num_workers: int) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    dataset = _dataset(root)
    rank_episodes, _, _ = dataset._rank_episodes()
    shares = []
    for worker_id in range(num_workers):
        _as_worker(monkeypatch, worker_id, num_workers)
        shares.append(dataset._worker_share(rank_episodes)[0].episodes)

    assert sorted(episode for share in shares for episode in share) == sorted(rank_episodes)
    assert sum(len(share) for share in shares) == len(set().union(*shares))
    counts = {episode: LENGTHS[episode] for episode in rank_episodes}
    assert shares == _balanced_episode_shards(rank_episodes, counts, world_size=num_workers)


@pytest.mark.parametrize("num_workers", [2, 3])
@pytest.mark.parametrize("resume_batches", [3, 17])  # inside the first epoch, and after epoch boundaries
def test_workers_resume_from_a_step_count(root: Path, num_workers: int, resume_batches: int) -> None:
    batch_size, more = 4, 12
    full = _batches(_dataset(root, repeat=True), num_workers, batch_size, count=resume_batches + more)
    resumed = _dataset(root, repeat=True)
    resumed.load_state_dict({"epoch": 0, "offset": resume_batches * batch_size, "batch_size": batch_size})

    assert _batches(resumed, num_workers, batch_size, count=more) == full[resume_batches:]


@pytest.mark.parametrize("resume_batches", [2, 6, 14, 15])  # before and after workers end; 15 is the end
def test_workers_resume_within_one_epoch_without_repeat(root: Path, resume_batches: int) -> None:
    batch_size, num_workers = 4, 3
    full = _batches(_dataset(root), num_workers, batch_size)
    resumed = _dataset(root)
    resumed.load_state_dict({"epoch": 0, "offset": resume_batches * batch_size, "batch_size": batch_size})

    assert _batches(resumed, num_workers, batch_size) == full[resume_batches:]


def test_workers_order_is_set_by_the_seed(root: Path) -> None:
    first = _batches(_dataset(root, seed=3), 2, 4)

    assert _batches(_dataset(root, seed=3), 2, 4) == first
    assert _batches(_dataset(root, seed=4), 2, 4) != first


def test_worker_limits_split_the_rank_limits(root: Path, monkeypatch) -> None:
    default = _dataset(root, byte_budget_gb=3.0)
    explicit = _dataset(root, episode_pool_size=5, byte_budget_gb=3.0)
    rank_episodes = list(range(len(LENGTHS)))
    _as_worker(monkeypatch, 0, 3)

    share, _ = default._worker_share(rank_episodes)
    assert share.pool_size == round(32 / 3)
    assert share.byte_budget == int(3.0 * 1024**3) // 3
    assert explicit._worker_share(rank_episodes)[0].pool_size == 5

    _as_worker(monkeypatch, 0, 1)
    assert default._worker_share(rank_episodes)[0].pool_size == 32


def test_workers_need_one_episode_each(root: Path, monkeypatch) -> None:
    dataset = _dataset(root)
    _as_worker(monkeypatch, 0, 3)

    with pytest.raises(ValueError, match="fewer than its 3 DataLoader workers"):
        dataset._worker_share([0, 1])
    with pytest.raises(ValueError, match="fewer than its 9 DataLoader workers"):
        dataset.num_frames_for_rank(0, 1, 9)
    assert dataset.num_frames_for_rank(0, 1, 8) == sum(LENGTHS)
    assert dataset.num_episodes_for_rank(0, 2) == len(LENGTHS) // 2


def test_workers_resume_needs_whole_batches(root: Path) -> None:
    dataset = _dataset(root, repeat=True)
    dataset.load_state_dict({"epoch": 0, "offset": 6, "batch_size": 4})

    with pytest.raises(ValueError, match="whole number of batches"):
        _batches(dataset, 2, 4, count=1)


def test_batches_per_worker_follows_the_dataloader_turns() -> None:
    # (batches of each worker, worker of the next batch)
    assert _batches_per_worker(7, [None, None, None]) == ([3, 2, 2], 1)
    assert _batches_per_worker(5, [1, None]) == ([1, 4], 1)
    assert _batches_per_worker(4, [0, None, 2]) == ([0, 2, 2], 1)
    assert _batches_per_worker(3, [2, None, None]) == ([1, 1, 1], 0)
    assert _batches_per_worker(10, [2, 3]) == ([2, 3], 0)

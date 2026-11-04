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

"""Tests for recycling DataLoader workers mid-run (--worker_recycle_every)."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")

from accelerate import Accelerator  # noqa: E402

from lerobot.datasets.sampler import EpisodeAwareSampler  # noqa: E402
from lerobot.scripts import lerobot_train  # noqa: E402
from lerobot.scripts.lerobot_train import _recycle_dataloader_iterator  # noqa: E402
from lerobot.utils.utils import cycle  # noqa: E402

pytestmark = pytest.mark.dataset

NUM_FRAMES = 23  # not a multiple of the batch size, so the last batch of each epoch is partial
BATCH_SIZE = 4


def _make_dataloader(num_workers: int, prepare: bool = False) -> torch.utils.data.DataLoader:
    sampler = EpisodeAwareSampler([0, 10], [10, NUM_FRAMES], shuffle=True, seed=7)
    dataloader = torch.utils.data.DataLoader(
        list(range(NUM_FRAMES)),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    return Accelerator(cpu=True).prepare(dataloader) if prepare else dataloader


@pytest.mark.parametrize("prepare", [False, True], ids=["torch", "accelerate"])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("recycle_every", [3, 6])  # mid-epoch, and exactly on the epoch boundary
def test_recycle_preserves_sample_order(prepare, num_workers, recycle_every):
    num_steps = 20  # spans three epochs of 6 batches
    reference_iter = cycle(_make_dataloader(0, prepare))
    reference = [next(reference_iter).tolist() for _ in range(num_steps)]

    dataloader = _make_dataloader(num_workers, prepare)
    cfg = SimpleNamespace(batch_size=BATCH_SIZE)
    parallel_dims = SimpleNamespace(dp_world_size=1)
    dl_iter = iter(dataloader)
    batches = []
    for step in range(num_steps):
        if step > 0 and step % recycle_every == 0:
            dl_iter = _recycle_dataloader_iterator(dl_iter, dataloader, step, cfg, parallel_dims)
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)
        batches.append(batch.tolist())

    assert batches == reference


class WorkerIdentityDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int) -> tuple[int, int]:
        return index, os.getpid()


@pytest.mark.parametrize("prepare", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
def test_recycle_replaces_worker(prepare: bool, persistent: bool) -> None:
    loader = torch.utils.data.DataLoader(
        WorkerIdentityDataset(),
        batch_size=4,
        sampler=EpisodeAwareSampler([0], [16]),
        num_workers=1,
        persistent_workers=persistent,
        multiprocessing_context="spawn",
    )
    if prepare:
        loader = Accelerator(cpu=True).prepare(loader)
    iterator = iter(loader)
    try:
        first_indices, first_pids = next(iterator)
        iterator = _recycle_dataloader_iterator(
            iterator, loader, 1, SimpleNamespace(batch_size=4), SimpleNamespace(dp_world_size=1)
        )
        next_indices, next_pids = next(iterator)
        assert first_indices.tolist() == [0, 1, 2, 3]
        assert next_indices.tolist() == [4, 5, 6, 7]
        assert len(set(first_pids.tolist())) == len(set(next_pids.tolist())) == 1
        assert set(first_pids.tolist()).isdisjoint(next_pids.tolist())
        assert os.getpid() not in first_pids.tolist() + next_pids.tolist()
    finally:
        if hasattr(iterator, "close"):
            iterator.close()
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        base = getattr(loader, "base_dataloader", loader)
        if getattr(base, "_iterator", None) is not None:
            base._iterator._shutdown_workers()
            base._iterator = None


@pytest.mark.parametrize("saved_batch_size,saved_dp_world", [(4, 1), (2, 2), (2, 1), (None, None)])
@pytest.mark.parametrize("recycle_every", [0, 3])
def test_resume_changed_topology(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    saved_batch_size: int | None,
    saved_dp_world: int | None,
    recycle_every: int,
) -> None:
    monkeypatch.setattr(
        lerobot_train,
        "load_training_metadata",
        lambda _: {"batch_size": saved_batch_size, "dp_world_size": saved_dp_world},
    )
    dataset = WorkerIdentityDataset()
    dataset.meta = SimpleNamespace(
        episodes={"dataset_from_index": [0], "dataset_to_index": [16]}, has_language_columns=False
    )
    dataset.episodes = None
    dataset.absolute_to_relative_idx = None
    cfg = SimpleNamespace(
        trainable_config=SimpleNamespace(),
        dataset=SimpleNamespace(streaming=False),
        resume=True,
        checkpoint_path=tmp_path,
        batch_size=2,
        worker_recycle_every=recycle_every,
        seed=7,
        num_workers=0,
        persistent_workers=False,
        dataloader_multiprocessing_context="spawn",
    )
    dims = SimpleNamespace(dp_world_size=1, device_type="cpu")
    mismatch = saved_batch_size not in (None, 2) or saved_dp_world not in (None, 1)
    if recycle_every and mismatch:
        with pytest.raises(ValueError, match="Worker recycling requires"):
            lerobot_train.make_dataloaders(cfg, dataset, None, 4, dims)
    else:
        loader, _ = lerobot_train.make_dataloaders(cfg, dataset, None, 4, dims)
        expected = lerobot_train.compute_sampler_state(4, 16, saved_batch_size or 2, saved_dp_world or 1)
        assert loader.sampler.state_dict() == expected

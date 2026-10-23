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

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")

from accelerate import Accelerator  # noqa: E402

from lerobot.datasets.sampler import EpisodeAwareSampler  # noqa: E402
from lerobot.scripts.lerobot_train import _recycle_dataloader_iterator  # noqa: E402
from lerobot.utils.utils import cycle  # noqa: E402

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

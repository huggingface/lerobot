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
"""Resume through the production loader builders and real Accelerate sharding on CPU."""

import itertools
import json
import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("accelerate")

from accelerate import Accelerator
from accelerate.data_loader import prepare_data_loader

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.distributed import ParallelDims
from lerobot.scripts.lerobot_train import make_dataloaders, make_training_iterator
from lerobot.utils.constants import TRAINING_STATE_DIR, TRAINING_STEP
from tests.fixtures.dummy_checkpoint_policy import DummyCheckpointConfig


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size
        self.meta = SimpleNamespace(
            episodes={"dataset_from_index": [0], "dataset_to_index": [size]},
            has_language_columns=False,
        )
        self.episodes = None
        self.absolute_to_relative_idx = None
        self.read_indices = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.read_indices.append(index)
        return index


def make_cfg(tmp_path, batch_size, world_size, metadata="current"):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="dummy/indexed"),
        policy=DummyCheckpointConfig(device="cpu"),
        batch_size=batch_size,
        num_workers=0,
        seed=42,
    )
    cfg.checkpoint_path = tmp_path
    state_dir = tmp_path / TRAINING_STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    state = {"step": 0}
    if metadata != "missing":
        state["num_processes" if metadata == "legacy" else "dp_world_size"] = world_size
        state["batch_size"] = batch_size
    (state_dir / TRAINING_STEP).write_text(json.dumps(state))
    return cfg


def make_iterator(cfg, dataset, world_size, rank, step, accelerator, *, prepare_with_accelerator=False):
    dims = ParallelDims(world_size, 1, 1, 1, world_size, "cpu")
    loader, _ = make_dataloaders(cfg, dataset, None, dims)
    sampler = loader.sampler
    if prepare_with_accelerator:
        loader = accelerator.prepare(loader)
    else:
        loader = prepare_data_loader(
            loader,
            device=torch.device("cpu"),
            num_processes=world_size,
            process_index=rank,
            even_batches=True,
            split_batches=False,
            rng_types=[],
        )
    return make_training_iterator(cfg, loader, sampler, step, dims, accelerator)


def take(iterator, count):
    return [batch.tolist() for batch in itertools.islice(iterator, count)]


@pytest.mark.parametrize(
    "size,batch_size,world_size", [(3, 2, 2), (10, 2, 2), (20, 2, 1), (37, 4, 3), (8, 2, 2)]
)
@pytest.mark.parametrize("metadata", ["current", "legacy", "missing"])
@pytest.mark.parametrize("drop_last", [0, 1])
def test_resume_matches_uninterrupted_across_epochs(
    tmp_path, size, batch_size, world_size, metadata, drop_last
):
    accelerator = Accelerator(cpu=True)
    cfg = make_cfg(tmp_path, batch_size, world_size, metadata)
    cfg.policy.drop_n_last_frames = drop_last
    batches_per_epoch = math.ceil(math.ceil((size - drop_last) / batch_size) / world_size)
    total_batches = 4 * batches_per_epoch

    for rank in range(world_size):
        cfg.resume = False
        reference = take(
            make_iterator(cfg, IndexedDataset(size), world_size, rank, 0, accelerator), total_batches
        )
        for step in sorted({0, 1, batches_per_epoch, batches_per_epoch + 1, 2 * batches_per_epoch}):
            cfg.resume = True
            resumed = make_iterator(cfg, IndexedDataset(size), world_size, rank, step, accelerator)
            assert take(resumed, total_batches - step) == reference[step:]


def test_resume_through_accelerator_prepare(tmp_path):
    accelerator = Accelerator(cpu=True)
    cfg = make_cfg(tmp_path, 2, 1)
    reference = take(
        make_iterator(cfg, IndexedDataset(10), 1, 0, 0, accelerator, prepare_with_accelerator=True), 15
    )
    cfg.resume = True
    # Step is a micro-batch count, independent of the configured accumulation factor.
    cfg.accelerator.gradient_accumulation.steps = 3
    resumed = make_iterator(cfg, IndexedDataset(10), 1, 0, 7, accelerator, prepare_with_accelerator=True)
    assert take(resumed, 8) == reference[7:]


def test_skipping_does_not_decode_consumed_samples(tmp_path):
    accelerator = Accelerator(cpu=True)
    cfg = make_cfg(tmp_path, 2, 2)
    cfg.resume = True
    dataset = IndexedDataset(10)
    resumed = make_iterator(cfg, dataset, 2, 1, 1, accelerator)
    # The partial epoch consists of two batches; the padded batch must still use
    # the original prefix. No samples from consumed batches should be loaded.
    suffix = take(resumed, 2)
    assert dataset.read_indices == list(itertools.chain.from_iterable(suffix))


@pytest.mark.parametrize("batch_size,world_size", [(3, 2), (2, 3)])
def test_changed_topology_retains_saved_offset_and_warns(tmp_path, caplog, batch_size, world_size):
    accelerator = Accelerator(cpu=True)
    cfg = make_cfg(tmp_path, 2, 2)
    cfg.batch_size = batch_size
    cfg.resume = True
    dims = ParallelDims(world_size, 1, 1, 1, world_size, "cpu")
    loader, _ = make_dataloaders(cfg, IndexedDataset(20), None, dims)
    sampler = loader.sampler
    loader = prepare_data_loader(loader, num_processes=world_size, process_index=0, rng_types=[])
    # Old layout has 5 batches/epoch. Step 6 means epoch 1, offset 4.
    iterator = make_training_iterator(cfg, loader, sampler, 6, dims, accelerator)
    assert sampler.state_dict() == {"epoch": 1, "start_index": 4}
    assert "per-rank sample-exactness" in caplog.text
    assert next(iterator).numel() == batch_size


def test_streaming_keeps_existing_cycle_behavior(tmp_path):
    cfg = make_cfg(tmp_path, 2, 1)
    cfg.resume = True
    cfg.dataset.streaming = True
    dims = ParallelDims(1, 1, 1, 1, 1, "cpu")
    iterator = make_training_iterator(cfg, [1, 2], None, 100, dims, None)
    assert list(itertools.islice(iterator, 5)) == [1, 2, 1, 2, 1]

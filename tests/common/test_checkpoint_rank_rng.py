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
"""Rank RNG persistence through the native training-state entry points.

The rank facade keeps these file/validation tests usable without a distributed
launcher. Native two-process training is a separate integration qualification.
"""

import random

import numpy as np
import pytest
import torch

from lerobot.common import train_utils
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import RNG_STATE, TRAINING_STATE_DIR, TRAINING_STEP
from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.random_utils import serialize_rng_state


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def snapshot():
    return {key: value.clone() for key, value in serialize_rng_state().items()}


def assert_state(expected):
    actual = snapshot()
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], value) for key, value in expected.items())


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    runtime = {"rank": 0, "world": 2}
    monkeypatch.setattr(train_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(train_utils.dist, "get_rank", lambda: runtime["rank"])
    monkeypatch.setattr(train_utils.dist, "get_world_size", lambda: runtime["world"])
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"))
    cfg.parallelism.resolve(2)
    cfg.checkpoint_path = tmp_path
    states = {}
    for rank in (0, 1):
        runtime["rank"] = rank
        seed(100 + rank)
        random.gauss(0, 1)
        np.random.standard_normal(3)
        torch.rand(5 + rank)
        states[rank] = snapshot()
        train_utils.save_training_state(tmp_path, 4, cfg)
    return cfg, runtime, states


def test_each_rank_restores_its_own_stream(checkpoint):
    cfg, runtime, states = checkpoint
    assert not torch.equal(states[0]["torch_rng_state"], states[1]["torch_rng_state"])
    for rank in (0, 1):
        runtime["rank"] = rank
        seed(999)
        assert train_utils.resume_before_prepare(cfg) == 4
        assert_state(states[rank])


@pytest.mark.parametrize("missing_rank", [0, 1])
@pytest.mark.parametrize("resuming_rank", [0, 1])
def test_missing_peer_state_rejects_before_rng_change(checkpoint, missing_rank, resuming_rank):
    cfg, runtime, _ = checkpoint
    (cfg.checkpoint_path / TRAINING_STATE_DIR / "rng_states" / f"rank_{missing_rank}" / RNG_STATE).unlink()
    runtime["rank"] = resuming_rank
    seed(999)
    before = snapshot()
    with pytest.raises(FileNotFoundError, match="per-rank RNG"):
        train_utils.resume_before_prepare(cfg)
    assert_state(before)


@pytest.mark.parametrize("invalid", [True, 0, 1, "2", 3])
def test_invalid_rng_topology_rejects_before_rng_change(checkpoint, invalid):
    cfg, _, _ = checkpoint
    path = cfg.checkpoint_path / TRAINING_STATE_DIR / TRAINING_STEP
    metadata = load_json(path)
    metadata["rng_state_world_size"] = invalid
    write_json(metadata, path)
    seed(999)
    before = snapshot()
    with pytest.raises(ValueError, match="(rng_state_world_size|RNG world size)"):
        train_utils.resume_before_prepare(cfg)
    assert_state(before)


def test_missing_rank_metadata_cannot_masquerade_as_legacy(checkpoint):
    cfg, _, _ = checkpoint
    path = cfg.checkpoint_path / TRAINING_STATE_DIR / TRAINING_STEP
    metadata = load_json(path)
    del metadata["rng_state_world_size"]
    write_json(metadata, path)
    seed(999)
    before = snapshot()
    with pytest.raises(ValueError, match="no rng_state_world_size"):
        train_utils.resume_before_prepare(cfg)
    assert_state(before)


def test_world_size_change_keeps_warned_legacy_fallback(checkpoint, caplog):
    cfg, runtime, states = checkpoint
    runtime.update(rank=0, world=1)
    cfg.parallelism.dp_replicate = 1
    seed(999)
    with caplog.at_level("WARNING"):
        assert train_utils.resume_before_prepare(cfg) == 4
    assert "RNG world size changed from 2 to 1" in caplog.text
    assert_state(states[0])


def test_legacy_distributed_rng_fallback_is_explicit(checkpoint, caplog):
    import shutil

    cfg, runtime, states = checkpoint
    path = cfg.checkpoint_path / TRAINING_STATE_DIR / TRAINING_STEP
    metadata = load_json(path)
    del metadata["rng_state_world_size"]
    write_json(metadata, path)
    shutil.rmtree(cfg.checkpoint_path / TRAINING_STATE_DIR / "rng_states")
    runtime["rank"] = 1
    seed(999)
    with caplog.at_level("WARNING"):
        assert train_utils.resume_before_prepare(cfg) == 4
    assert "Legacy checkpoint has no per-rank RNG states" in caplog.text
    assert_state(states[0])

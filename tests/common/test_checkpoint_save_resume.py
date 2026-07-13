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
"""Checkpoint save/resume round-trips on the non-sharded paths."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from lerobot.common.train_utils import (
    load_training_batch_size,
    load_training_dp_world_size,
    load_training_grad_accum_steps,
    resume_after_prepare,
    resume_before_prepare,
    save_checkpoint,
)
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import CheckpointFormat, TrainPipelineConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR, TRAINING_STEP
from lerobot.utils.io_utils import write_json
from tests.fixtures.dummy_checkpoint_policy import make_dummy_policy


def make_cfg(**overrides) -> TrainPipelineConfig:
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"), batch_size=3)
    cfg.parallelism.resolve(1)
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def passthrough_accelerator() -> SimpleNamespace:
    """The accelerator surface save/resume touches on non-sharded runs."""
    return SimpleNamespace(unwrap_model=lambda m: m, wait_for_everyone=lambda: None)


class TestSaveCheckpoint:
    def test_non_sharded_layout(self, tmp_path):
        policy = make_dummy_policy()
        optimizer = torch.optim.Adam(policy.parameters())
        save_checkpoint(
            tmp_path,
            step=7,
            cfg=make_cfg(),
            policy=policy,
            optimizer=optimizer,
            accelerator=passthrough_accelerator(),
        )
        pretrained = tmp_path / PRETRAINED_MODEL_DIR
        state = tmp_path / TRAINING_STATE_DIR
        assert (pretrained / "model.safetensors").is_file()
        assert (pretrained / "config.json").is_file()
        assert (pretrained / "train_config.json").is_file()
        assert (state / TRAINING_STEP).is_file()
        assert (state / "rng_state.safetensors").is_file()
        assert (state / "optimizer_state.safetensors").is_file()
        # single-file artifact, no index, weights intact
        weights = load_file(pretrained / "model.safetensors")
        assert torch.allclose(weights["net.weight"], torch.full_like(weights["net.weight"], 0.5))
        assert not list(pretrained.glob("*.index.json"))

    def test_training_step_records_topology(self, tmp_path):
        cfg = make_cfg()
        cfg.accelerator.gradient_accumulation.steps = 4
        policy = make_dummy_policy()
        save_checkpoint(
            tmp_path,
            step=11,
            cfg=cfg,
            policy=policy,
            optimizer=torch.optim.Adam(policy.parameters()),
            accelerator=passthrough_accelerator(),
        )
        assert load_training_dp_world_size(tmp_path) == 1
        assert load_training_batch_size(tmp_path) == 3
        assert load_training_grad_accum_steps(tmp_path) == 4

    def test_dp_world_size_legacy_fallback(self, tmp_path):
        """Pre-v0.7 checkpoints recorded num_processes; the reader falls back to it."""
        state_dir = tmp_path / TRAINING_STATE_DIR
        state_dir.mkdir(parents=True)
        write_json({"step": 5, "num_processes": 4}, state_dir / TRAINING_STEP)
        assert load_training_dp_world_size(tmp_path) == 4
        assert load_training_batch_size(tmp_path) is None


class TestResume:
    def _checkpointed_run(self, tmp_path):
        policy = make_dummy_policy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.123)
        # give the optimizer real state
        policy.forward({"observation.state": torch.randn(2, 4)})[0].backward()
        optimizer.step()
        cfg = make_cfg()
        save_checkpoint(
            tmp_path,
            step=42,
            cfg=cfg,
            policy=policy,
            optimizer=optimizer,
            accelerator=passthrough_accelerator(),
        )
        cfg.checkpoint_path = tmp_path
        return cfg, policy, optimizer

    def test_two_phase_resume_round_trip(self, tmp_path):
        cfg, _, optimizer = self._checkpointed_run(tmp_path)
        assert resume_before_prepare(cfg) == 42

        fresh_policy = make_dummy_policy()
        fresh_optimizer = torch.optim.Adam(fresh_policy.parameters(), lr=0.999)
        resume_after_prepare(cfg, passthrough_accelerator(), fresh_policy, fresh_optimizer, None)
        restored = fresh_optimizer.state_dict()
        original = optimizer.state_dict()
        assert restored["param_groups"][0]["lr"] == original["param_groups"][0]["lr"]
        for key, tensor in original["state"][0].items():
            assert torch.equal(restored["state"][0][key], tensor), key

    def test_resume_warns_on_changed_cadence_and_topology(self, tmp_path, caplog):
        """The recorded grad-accum factor and parallelism snapshot must be compared on
        resume, with one warning naming the diff."""
        import logging

        cfg, _, _ = self._checkpointed_run(tmp_path)
        cfg.accelerator.gradient_accumulation.steps = 4
        cfg.parallelism.dp_replicate = 2  # same dp_world_size story is irrelevant here
        with caplog.at_level(logging.WARNING):
            assert resume_before_prepare(cfg) == 42
        warning = next(m for m in caplog.messages if "differ from the checkpoint" in m)
        assert "grad_accum_steps: 1 -> 4" in warning
        assert "dp_replicate: 1 -> 2" in warning

    def test_resume_unchanged_settings_stay_silent(self, tmp_path, caplog):
        import logging

        cfg, _, _ = self._checkpointed_run(tmp_path)
        with caplog.at_level(logging.WARNING):
            resume_before_prepare(cfg)
        assert not [m for m in caplog.messages if "differ from the checkpoint" in m]

    def test_resume_before_prepare_requires_training_state(self, tmp_path):
        cfg = make_cfg()
        cfg.checkpoint_path = tmp_path
        with pytest.raises(NotADirectoryError):
            resume_before_prepare(cfg)

    def test_dcp_format_integrity_preflight(self, tmp_path):
        """A checkpoint declaring DCP shards without the shard dir fails with the converter hint."""
        pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")
        cfg, policy, optimizer = self._checkpointed_run(tmp_path)
        cfg.parallelism.dp_shard = 2  # pretend the recorded run was sharded
        cfg.checkpoint_format = CheckpointFormat.DCP
        with pytest.raises(FileNotFoundError, match="lerobot-convert-dcp"):
            resume_after_prepare(cfg, passthrough_accelerator(), policy, optimizer, None)

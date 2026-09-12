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
"""TrainPipelineConfig integration for the distributed fields: fail-fasts + config compat."""

import draccus
import pytest

from lerobot.configs.accelerator import ActivationCheckpointingMode
from lerobot.configs.default import DatasetConfig, PeftConfig
from lerobot.configs.parallelism import ContextParallelConfig, ParallelismConfig
from lerobot.configs.train import CheckpointFormat, TrainPipelineConfig
from lerobot.optim.optimizers import AdamConfig, MultiAdamConfig


def make_cfg(**overrides) -> TrainPipelineConfig:
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"))
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def sharded() -> ParallelismConfig:
    return ParallelismConfig(dp_shard=-1)


class TestDistributedFailFasts:
    def test_defaults_pass(self):
        make_cfg()._validate_distributed()

    def test_cp_reserved(self):
        cfg = make_cfg(parallelism=ParallelismConfig(context_parallel=ContextParallelConfig(ring_degree=2)))
        with pytest.raises(ValueError, match="not implemented"):
            cfg._validate_distributed()

    def test_cfg_parallel_training_rejected(self):
        cfg = make_cfg(parallelism=ParallelismConfig(cfg_parallel=2))
        with pytest.raises(ValueError, match="inference-only"):
            cfg._validate_distributed()

    def test_compile_placeholder(self):
        cfg = make_cfg()
        cfg.accelerator.compile.enabled = True
        with pytest.raises(ValueError, match="compile"):
            cfg._validate_distributed()

    def test_activation_checkpointing_placeholder(self):
        cfg = make_cfg()
        cfg.accelerator.activation_checkpointing.mode = ActivationCheckpointingMode.FULL
        with pytest.raises(ValueError, match="activation_checkpointing"):
            cfg._validate_distributed()

    def test_dcp_format_requires_sharding(self):
        cfg = make_cfg(checkpoint_format=CheckpointFormat.DCP)
        with pytest.raises(ValueError, match="sharded"):
            cfg._validate_distributed()
        cfg.parallelism = sharded()
        cfg._validate_distributed()

    def test_fp16_rejected_when_sharded(self):
        cfg = make_cfg(parallelism=sharded())
        cfg.accelerator.mixed_precision = "fp16"
        with pytest.raises(ValueError, match="fp16"):
            cfg._validate_distributed()
        cfg.accelerator.mixed_precision = "bf16"
        cfg._validate_distributed()

    def test_peft_rejected_when_sharded(self):
        cfg = make_cfg(parallelism=sharded(), peft=PeftConfig())
        with pytest.raises(ValueError, match="PEFT"):
            cfg._validate_distributed()

    def test_env_eval_rejected_when_sharded(self):
        cfg = make_cfg(parallelism=sharded(), env_eval_freq=1000)
        cfg.env = object()  # any configured env triggers the check
        with pytest.raises(ValueError, match="environment evaluation"):
            cfg._validate_distributed()

    def test_multi_optimizer_rejected_when_sharded(self):
        cfg = make_cfg(parallelism=sharded(), optimizer=MultiAdamConfig())
        with pytest.raises(ValueError, match="Multi-optimizer"):
            cfg._validate_distributed()
        cfg.optimizer = AdamConfig()
        cfg._validate_distributed()


class TestConfigCompat:
    def test_checkpoint_format_round_trip(self):
        for fmt in CheckpointFormat:
            assert draccus.decode(CheckpointFormat, draccus.encode(fmt)) is fmt

    def test_wants_predicates(self):
        assert CheckpointFormat.SAFETENSORS.wants_safetensors
        assert not CheckpointFormat.SAFETENSORS.wants_dcp
        assert CheckpointFormat.DCP.wants_dcp and not CheckpointFormat.DCP.wants_safetensors
        both = CheckpointFormat.SAFETENSORS_AND_DCP
        assert both.wants_safetensors and both.wants_dcp


def test_reward_model_rejected_when_sharded():
    """Sharded reward runs previously failed late (missing wrap
    units, DTensor serialization at the first checkpoint) instead of at validation."""
    cfg = make_cfg(parallelism=sharded())
    cfg.reward_model = object()  # any configured reward model triggers the check
    with pytest.raises(ValueError, match="Reward-model"):
        cfg._validate_distributed()

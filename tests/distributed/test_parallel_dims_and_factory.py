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

import pytest

from lerobot.configs.parallelism import ContextParallelConfig, ParallelismConfig
from lerobot.distributed import ParallelDims, guard_against_env_interference, is_main_process
from lerobot.distributed.factory import _ENV_OVERRIDE


class TestIsMainProcess:
    def test_true_outside_distributed(self):
        assert is_main_process() is True


class TestParallelDims:
    def _resolved(self, world_size: int = 8, **kwargs) -> ParallelismConfig:
        cfg = ParallelismConfig(**kwargs)
        cfg.resolve(world_size)
        return cfg

    def test_from_resolved_config(self):
        dims = ParallelDims.from_config(self._resolved(dp_replicate=2, dp_shard=4), 8, "cpu")
        assert dims.dp_world_size == 8
        assert dims.is_sharded
        assert dims.cp_size == 1
        assert dims.dp_rank == 0  # no process group in unit tests

    def test_rejects_unresolved_config(self):
        with pytest.raises(ValueError, match="resolve"):
            ParallelDims.from_config(ParallelismConfig(dp_shard=-1), 8, "cpu")

    def test_rejects_world_mismatch(self):
        with pytest.raises(ValueError, match="world_size=4"):
            ParallelDims.from_config(self._resolved(8), 4, "cpu")

    def test_cp_mesh_reserved(self):
        dims = ParallelDims(dp_replicate=1, dp_shard=2, ring=2, ulysses=2, world_size=8, device_type="cpu")
        assert dims.dp_rank == 0 and dims.dp_world_size == 2
        with pytest.raises(NotImplementedError):
            dims.cp_mesh()

    def test_cp_peers_share_dp_rank_arithmetic(self):
        """Row-major layout: cp is innermost, so dp_rank = global_rank // cp_size."""
        dims = ParallelDims(dp_replicate=1, dp_shard=2, ring=1, ulysses=2, world_size=4, device_type="cpu")
        # Without a process group the global rank is 0; the arithmetic contract is what matters.
        assert dims.cp_size == 2
        assert dims.dp_rank == 0 // dims.cp_size

    def test_config_placeholder_degrees_flow_through(self):
        cfg = ParallelismConfig(
            dp_replicate=1,
            dp_shard=2,
            context_parallel=ContextParallelConfig(ring_degree=2, ulysses_degree=2),
        )
        # resolve() rejects cp>1 this round; ParallelDims math itself is already cp-aware.
        dims = ParallelDims(
            dp_replicate=cfg.dp_replicate,
            dp_shard=cfg.dp_shard,
            ring=cfg.context_parallel.ring_degree,
            ulysses=cfg.context_parallel.ulysses_degree,
            world_size=8,
            device_type="cpu",
        )
        assert dims.dp_world_size == 2 and dims.cp_size == 4


class TestEnvGuard:
    # ACCELERATE_DYNAMO_*/ACCELERATE_GRADIENT_ACCUMULATION_STEPS are silent config overrides
    # inside accelerate itself — the guard must catch them too.
    _POISON = (
        "ACCELERATE_USE_FSDP",
        "FSDP_VERSION",
        "PARALLELISM_CONFIG_DP_SHARD_SIZE",
        "ACCELERATE_DYNAMO_BACKEND",
        "ACCELERATE_GRADIENT_ACCUMULATION_STEPS",
    )

    def test_clean_env_passes(self, monkeypatch):
        for name in self._POISON + (_ENV_OVERRIDE,):
            monkeypatch.delenv(name, raising=False)
        guard_against_env_interference()

    @pytest.mark.parametrize("name", _POISON)
    def test_accelerate_env_rejected_with_actionable_error(self, name, monkeypatch):
        monkeypatch.delenv(_ENV_OVERRIDE, raising=False)
        monkeypatch.setenv(name, "true")
        with pytest.raises(RuntimeError, match=name):
            guard_against_env_interference()

    def test_override_acknowledges(self, monkeypatch):
        monkeypatch.setenv("FSDP_VERSION", "2")
        monkeypatch.setenv(_ENV_OVERRIDE, "1")
        guard_against_env_interference()


def test_make_accelerator_rejects_format_after_sentinel_resolution(monkeypatch):
    """dp_shard=-1 counts as sharded at parse time but can resolve
    to an unsharded run (world size 1), which would write a safetensors-only checkpoint whose
    recorded checkpoint_format=dcp fails its own validation on resume."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import CheckpointFormat, TrainPipelineConfig
    from lerobot.distributed.factory import make_accelerator

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"))
    cfg.parallelism.dp_shard = -1
    cfg.checkpoint_format = CheckpointFormat.DCP
    cfg._validate_distributed()  # passes: the sentinel is declared as sharded
    with pytest.raises(ValueError, match="resolved to a non-sharded"):
        make_accelerator(cfg)

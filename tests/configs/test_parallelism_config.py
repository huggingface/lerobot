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

import json

import draccus
import pytest

from lerobot.configs.parallelism import ContextParallelConfig, ParallelismConfig


class TestResolve:
    def test_single_process_defaults(self):
        cfg = ParallelismConfig()
        cfg.resolve(1)
        assert (cfg.dp_replicate, cfg.dp_shard) == (1, 1)
        assert not cfg.is_sharded and not cfg.is_replicated_only
        assert cfg.dp_world_size == 1

    def test_untouched_config_fills_ddp(self):
        """Plain `torchrun --nproc-per-node=8` with a default config resolves to DDP."""
        cfg = ParallelismConfig()
        cfg.resolve(8)
        assert cfg.dp_replicate == 8
        assert cfg.is_replicated_only and not cfg.is_sharded
        assert cfg.dp_world_size == 8

    def test_full_shard_sentinel(self):
        cfg = ParallelismConfig(dp_shard=-1)
        assert cfg.is_sharded  # sharded even before resolve: -1 is an explicit opt-in
        cfg.resolve(8)
        assert cfg.dp_shard == 8 and cfg.dp_replicate == 1

    def test_hsdp_sentinel_infers_shard(self):
        cfg = ParallelismConfig(dp_replicate=2, dp_shard=-1)
        cfg.resolve(8)
        assert (cfg.dp_replicate, cfg.dp_shard) == (2, 4)
        assert cfg.dp_world_size == 8

    def test_explicit_hsdp(self):
        cfg = ParallelismConfig(dp_replicate=2, dp_shard=4)
        cfg.resolve(8)
        assert cfg.is_sharded and not cfg.is_replicated_only

    def test_product_mismatch_lists_all_degrees(self):
        cfg = ParallelismConfig(dp_replicate=2, dp_shard=2)
        with pytest.raises(ValueError, match=r"dp_replicate=2 \* dp_shard=2.*WORLD_SIZE=8"):
            cfg.resolve(8)

    def test_explicit_replicate_must_match_world(self):
        cfg = ParallelismConfig(dp_replicate=4)
        with pytest.raises(ValueError, match="WORLD_SIZE=8"):
            cfg.resolve(8)

    def test_sentinel_indivisible_world(self):
        cfg = ParallelismConfig(dp_replicate=3, dp_shard=-1)
        with pytest.raises(ValueError, match="not divisible"):
            cfg.resolve(8)

    def test_cp_fails_fast(self):
        cfg = ParallelismConfig(dp_shard=-1, context_parallel=ContextParallelConfig(ulysses_degree=2))
        with pytest.raises(ValueError, match="not implemented"):
            cfg.resolve(8)


class TestFieldValidation:
    @pytest.mark.parametrize("kwargs", [{"dp_replicate": 0}, {"dp_shard": 0}, {"dp_shard": -2}])
    def test_bad_dp_degrees(self, kwargs):
        with pytest.raises(ValueError):
            ParallelismConfig(**kwargs)

    def test_cfg_parallel_capped_at_two(self):
        ParallelismConfig(cfg_parallel=2)  # reserved but representable
        with pytest.raises(ValueError, match="cfg_parallel"):
            ParallelismConfig(cfg_parallel=3)

    @pytest.mark.parametrize("kwargs", [{"ring_degree": 0}, {"ulysses_degree": -1}])
    def test_bad_cp_degrees(self, kwargs):
        with pytest.raises(ValueError):
            ContextParallelConfig(**kwargs)

    def test_dp_world_size_undefined_before_resolve(self):
        with pytest.raises(RuntimeError, match="resolve"):
            _ = ParallelismConfig(dp_shard=-1).dp_world_size


class TestDraccusRoundTrip:
    @pytest.mark.parametrize(
        "cfg",
        [
            ParallelismConfig(),
            ParallelismConfig(dp_replicate=2, dp_shard=4, cfg_parallel=2),
            ParallelismConfig(
                dp_shard=-1,
                context_parallel=ContextParallelConfig(ring_degree=2, ulysses_degree=4),
            ),
        ],
    )
    def test_encode_json_decode_identity(self, cfg):
        payload = json.loads(json.dumps(draccus.encode(cfg)))
        assert draccus.decode(ParallelismConfig, payload) == cfg

    def test_pre_existing_config_without_fields_gets_defaults(self):
        """Checkpoints written before this feature parse with default topology."""
        assert draccus.decode(ParallelismConfig, {}) == ParallelismConfig()

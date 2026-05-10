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

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.rl.learner import compute_online_replay_ratio, compute_replay_batch_sizes


@pytest.fixture
def rl_cfg():
    cfg = TrainRLServerPipelineConfig()
    cfg.policy = SACConfig()
    return cfg


def test_compute_online_replay_ratio_fixed_mode(rl_cfg):
    rl_cfg.policy.replay_mixing.mode = "fixed"
    rl_cfg.policy.replay_mixing.online_ratio = 0.35

    assert compute_online_replay_ratio(rl_cfg, optimization_step=0) == pytest.approx(0.35)
    assert compute_online_replay_ratio(rl_cfg, optimization_step=10_000) == pytest.approx(0.35)


def test_compute_online_replay_ratio_linear_mode(rl_cfg):
    rl_cfg.policy.replay_mixing.mode = "linear"
    rl_cfg.policy.replay_mixing.online_ratio = 0.2
    rl_cfg.policy.replay_mixing.final_online_ratio = 0.8
    rl_cfg.policy.replay_mixing.schedule_steps = 100

    assert compute_online_replay_ratio(rl_cfg, optimization_step=0) == pytest.approx(0.2)
    assert compute_online_replay_ratio(rl_cfg, optimization_step=50) == pytest.approx(0.5)
    assert compute_online_replay_ratio(rl_cfg, optimization_step=100) == pytest.approx(0.8)
    assert compute_online_replay_ratio(rl_cfg, optimization_step=200) == pytest.approx(0.8)


def test_compute_replay_batch_sizes_without_offline_buffer():
    online_bs, offline_bs = compute_replay_batch_sizes(
        total_batch_size=128,
        online_ratio=0.25,
        use_offline_buffer=False,
    )
    assert online_bs == 128
    assert offline_bs == 0


def test_compute_replay_batch_sizes_preserves_both_sources_when_enabled():
    online_bs, offline_bs = compute_replay_batch_sizes(
        total_batch_size=32,
        online_ratio=1.0,
        use_offline_buffer=True,
        keep_minimum_offline_samples=True,
    )
    assert online_bs == 31
    assert offline_bs == 1


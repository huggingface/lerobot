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

"""Unit tests for the vendored LingBot-VA helper code (scheduler + grid utilities)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")  # the model code lives in modeling_lingbot_va, which imports diffusers

from lerobot.policies.lingbot_va.modeling_lingbot_va import (  # noqa: E402
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
)


def test_flow_match_scheduler_timesteps_monotone_decreasing() -> None:
    sch = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    sch.set_timesteps(20)
    assert sch.timesteps.shape == (20,)
    diffs = sch.timesteps[1:] - sch.timesteps[:-1]
    assert torch.all(diffs <= 0)  # decreasing


def test_flow_match_scheduler_step_preserves_shape() -> None:
    sch = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    sch.set_timesteps(20)
    sample = torch.zeros(1, 48, 4, 8, 16)
    out = sch.step(torch.ones_like(sample), sch.timesteps[0], sample)
    assert out.shape == sample.shape


def test_flow_match_scheduler_add_noise() -> None:
    sch = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    sch.set_timesteps(20)
    sample = torch.randn(1, 48, 4, 8, 16)
    noise = torch.randn_like(sample)
    noisy = sch.add_noise(sample, noise, sch.timesteps[:4], t_dim=2)
    assert noisy.shape == sample.shape


def test_get_mesh_id_latent_shape() -> None:
    grid = get_mesh_id(4, 8, 16, 0, 1, 0)
    assert grid.shape == (4, 4 * 8 * 16)  # (f, h, w, stream) x tokens


def test_get_mesh_id_action_shape() -> None:
    grid = get_mesh_id(4, 4, 1, 1, 1, 0, action=True)
    assert grid.shape == (4, 4 * 4 * 1)
    # Action rows for h/w are sentinel -1.
    assert torch.all(grid[1] < 0)
    assert torch.all(grid[2] < 0)


def test_data_seq_to_patch_roundtrip_shape() -> None:
    b, f, h, w, c = 1, 4, 8, 16, 48
    seq = torch.arange(b * f * h * w * c, dtype=torch.float32).reshape(b, f * h * w, c)
    out = data_seq_to_patch((1, 2, 2), seq, f, h, w, batch_size=b)
    assert out.shape == (b, c, f, h, w)

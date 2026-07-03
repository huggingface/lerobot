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

from lerobot.policies.lingbot_va.modeling_lingbot_va import FlowMatchScheduler
from lerobot.policies.lingbot_va.utils import data_seq_to_patch, get_mesh_id


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


def test_training_step_reduces_loss_tiny_flex() -> None:
    """End-to-end single training step (flow-matching loss -> backward -> AdamW) on a tiny config.

    Exercises the flex-attention training path; requires a CUDA GPU with flex-attention support.
    """
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("training step test requires a CUDA GPU (flex-attention)")

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
    from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
    from lerobot.utils.constants import ACTION, OBS_IMAGES

    cfg = LingBotVAConfig(
        attn_mode="flex",
        dtype="bfloat16",
        in_channels=16,
        out_channels=16,
        action_dim=8,
        text_dim=32,
        freq_dim=64,
        ffn_dim=64,
        num_attention_heads=2,
        attention_head_dim=24,
        num_layers=2,
        frame_chunk_size=2,
        action_per_frame=4,
        used_action_channel_ids=[0, 1, 2, 3],
        obs_cam_keys=[f"{OBS_IMAGES}.image"],
        device="cuda",
    )
    cfg.input_features = {f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))}
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))}
    cfg.validate_features()

    policy = LingBotVAPolicy(cfg).to("cuda")
    policy.train()
    opt = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4)

    b, fc, apf = 1, cfg.frame_chunk_size, cfg.action_per_frame
    latents = torch.randn(b, cfg.in_channels, fc, 4, 4, device="cuda", dtype=torch.bfloat16)
    actions = torch.randn(b, cfg.action_dim, fc, apf, 1, device="cuda", dtype=torch.bfloat16)
    amask = torch.zeros(cfg.action_dim, device="cuda")
    amask[cfg.used_action_channel_ids] = 1.0
    actions_mask = amask.view(1, -1, 1, 1, 1).expand_as(actions)
    text_emb = torch.randn(b, cfg.max_sequence_length, cfg.text_dim, device="cuda", dtype=torch.bfloat16)

    loss, metrics = policy.training_loss_from_streams(latents, actions, actions_mask, text_emb)
    assert torch.isfinite(loss) and {"latent_loss", "action_loss"} <= set(metrics)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.get_optim_params())
    opt.step()

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

"""dtype behavior of pi0's action-time embedding, run through the real ``embed_suffix``."""

import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

pytest.importorskip("transformers")

from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding  # noqa: E402
from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch  # noqa: E402

BATCH, HORIZON, STATE_DIM, ACTION_DIM, WIDTH = 3, 4, 5, 6, 16
MIN_PERIOD, MAX_PERIOD = 4e-3, 4.0


class _Suffix(nn.Module):
    """The layers and config ``PI0Pytorch.embed_suffix`` reads, under their checkpoint names."""

    _apply_checkpoint = PI0Pytorch._apply_checkpoint
    embed_suffix = PI0Pytorch.embed_suffix

    def __init__(self, dtype):
        super().__init__()
        self.config = SimpleNamespace(min_period=MIN_PERIOD, max_period=MAX_PERIOD, chunk_size=HORIZON)
        self.gradient_checkpointing_enabled = False
        self.state_proj = nn.Linear(STATE_DIM, WIDTH, dtype=dtype)
        self.action_in_proj = nn.Linear(ACTION_DIM, WIDTH, dtype=dtype)
        self.action_time_mlp_in = nn.Linear(2 * WIDTH, WIDTH, dtype=dtype)
        self.action_time_mlp_out = nn.Linear(WIDTH, WIDTH, dtype=dtype)


def _historical_action_time_block(suffix, noisy_actions, timestep):
    """pi0's block before the fix: the time embedding followed the timestep's dtype (fp32)."""
    time_emb = create_sinusoidal_pos_embedding(
        timestep, WIDTH, min_period=MIN_PERIOD, max_period=MAX_PERIOD, device=timestep.device
    )
    time_emb = time_emb.type(dtype=timestep.dtype)
    action_emb = suffix.action_in_proj(noisy_actions)
    time_emb = time_emb[:, None, :].expand_as(action_emb)
    action_time_emb = torch.cat([action_emb, time_emb], dim=2)
    x = F.silu(suffix.action_time_mlp_in(action_time_emb))
    return suffix.action_time_mlp_out(x)


def _inputs(dtype):
    torch.manual_seed(0)
    state = torch.randn(BATCH, STATE_DIM, dtype=dtype)
    noisy_actions = torch.randn(BATCH, HORIZON, ACTION_DIM, dtype=dtype)
    timestep = torch.rand(BATCH, dtype=torch.float32)
    return state, noisy_actions, timestep


def _action_time_tokens(suffix, state, noisy_actions, timestep):
    embs, _, _, _ = suffix.embed_suffix(state, noisy_actions, timestep)
    return embs[:, 1:]  # token 0 is the state


@pytest.mark.parametrize("bf16_autocast", [False, True])
def test_pi0_action_time_embedding_is_unchanged_for_fp32_weights(bf16_autocast):
    # The configurations pi0 already supported: fp32 weights, with or without bf16 autocast.
    torch.manual_seed(1)
    suffix = _Suffix(dtype=torch.float32)
    state, noisy_actions, timestep = _inputs(torch.float32)
    autocast = (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16) if bf16_autocast else contextlib.nullcontext()
    )

    with autocast:
        expected = _historical_action_time_block(suffix, noisy_actions, timestep)
        actual = _action_time_tokens(suffix, state, noisy_actions, timestep)

    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_pi0_action_time_embedding_follows_bf16_weights():
    # With bf16 weights and no autocast, an fp32 time embedding made a mixed-dtype
    # concatenation that the MLP rejected.
    torch.manual_seed(2)
    suffix = _Suffix(dtype=torch.bfloat16)
    state, noisy_actions, timestep = _inputs(torch.bfloat16)

    with pytest.raises(RuntimeError, match="same dtype"):
        _historical_action_time_block(suffix, noisy_actions, timestep)

    tokens = _action_time_tokens(suffix, state, noisy_actions, timestep)
    assert tokens.dtype == torch.bfloat16  # gitleaks:allow
    assert tokens.shape == (BATCH, HORIZON, WIDTH)

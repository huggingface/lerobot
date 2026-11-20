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

"""pi0.5's ``embed_suffix`` given the time conditioning, run through the real method."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.policies.pi05 import modeling_pi05  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch  # noqa: E402

BATCH, HORIZON, ACTION_DIM, WIDTH = 2, 4, 6, 16


class _Suffix(nn.Module):
    """The layers and config ``PI05Pytorch.embed_suffix`` reads, under their checkpoint names."""

    _apply_checkpoint = PI05Pytorch._apply_checkpoint
    embed_suffix = PI05Pytorch.embed_suffix
    use_on_device_suffix_mask = False

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(min_period=4e-3, max_period=4.0, chunk_size=HORIZON)
        self.gradient_checkpointing_enabled = False
        self.action_in_proj = nn.Linear(ACTION_DIM, WIDTH)
        self.time_mlp_in = nn.Linear(WIDTH, WIDTH)
        self.time_mlp_out = nn.Linear(WIDTH, WIDTH)


def test_given_time_conditioning_replaces_the_time_mlp(monkeypatch):
    torch.manual_seed(0)
    suffix = _Suffix()
    noisy_actions, timestep = torch.randn(BATCH, HORIZON, ACTION_DIM), torch.rand(BATCH)
    expected = suffix.embed_suffix(noisy_actions, timestep)

    # Neither the time embedding nor the time MLP may run when the conditioning is given.
    monkeypatch.setattr(modeling_pi05, "create_sinusoidal_pos_embedding", None)
    suffix.time_mlp_in = None
    actual = suffix.embed_suffix(noisy_actions, timestep, expected[3])

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


def test_denoise_step_hands_its_time_conditioning_to_embed_suffix():
    class HandedError(Exception):
        pass

    def embed_suffix(noisy_actions, timestep, adarms_cond=None):
        raise HandedError(adarms_cond)

    conditioning = torch.randn(BATCH, WIDTH)
    with pytest.raises(HandedError) as handed:
        PI05Pytorch.denoise_step(
            SimpleNamespace(embed_suffix=embed_suffix),
            prefix_pad_masks=None,
            past_key_values=None,
            x_t=torch.randn(BATCH, HORIZON, ACTION_DIM),
            timestep=torch.rand(BATCH),
            adarms_cond=conditioning,
        )
    assert handed.value.args[0] is conditioning

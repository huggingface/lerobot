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

import torch

from lerobot.processor import ChunkSafetyProcessorStep, TransitionKey
from lerobot.processor.chunk_safety_processor import clamp_action_chunk
from lerobot.processor.converters import create_transition


def test_clamp_action_chunk_no_limits_is_noop():
    actions = torch.randn(1, 5, 3)
    out, was_clamped = clamp_action_chunk(actions)
    assert not was_clamped
    assert torch.equal(out, actions)


def test_clamp_action_chunk_discontinuity():
    # Step 2 jumps far away from step 1 — should get pulled back within max_relative_target.
    actions = torch.tensor([[[0.0], [0.0], [10.0], [10.1]]])  # (1, 4, 1)
    out, was_clamped = clamp_action_chunk(actions, max_relative_target=1.0)
    assert was_clamped
    deltas = (out[:, 1:, :] - out[:, :-1, :]).abs()
    assert torch.all(deltas <= 1.0 + 1e-6)


def test_clamp_action_chunk_jerk():
    # Delta grows smoothly then spikes — jerk check should smooth the spike.
    actions = torch.tensor([[[0.0], [1.0], [2.0], [10.0]]])  # deltas: 1, 1, 8
    out, was_clamped = clamp_action_chunk(actions, max_jerk=0.5)
    assert was_clamped
    deltas = out[:, 1:, :] - out[:, :-1, :]
    jerks = (deltas[:, 1:, :] - deltas[:, :-1, :]).abs()
    assert torch.all(jerks <= 0.5 + 1e-6)


def test_clamp_action_chunk_discontinuity_reconstruction_respects_absolute_bounds():
    # Regression test for a real bug found via PR #4241's hardware/eval demo
    # (chunk_safety_eval.py against a real ACT checkpoint): reconstructing each step from the
    # previous *reconstructed* value plus a rel/jerk-limited delta can walk the trajectory back
    # outside [min_action, max_action] over several steps, since absolute bounds were only
    # checked once, on the original tensor, before the sequential discontinuity/jerk pass ran.
    # Without the fix, step 4 here ends up at -1.464, well past min_action=-1.0.
    vals = [
        0.8905413911078446,
        0.8028549152229671,
        -0.9388200339328929,
        -0.9491082780130784,
        0.08282494558699316,
        0.8782983255570211,
        -0.23759152462357513,
        -0.5668012057387732,
        5.285957629296512,  # fault
        -0.9419184248502641,
        -0.5566166674539299,
        -0.12422481269885588,
    ]
    actions = torch.tensor(vals).view(1, len(vals), 1)
    out, was_clamped = clamp_action_chunk(
        actions,
        max_relative_target=1.165848038808939,
        max_jerk=0.5745146655245432,
        min_action=-1.0,
        max_action=1.0,
    )
    assert was_clamped
    assert torch.all(out <= 1.0 + 1e-6)
    assert torch.all(out >= -1.0 - 1e-6)


def test_clamp_action_chunk_absolute_bounds():
    actions = torch.tensor([[[-5.0, 5.0]]])  # (1, 1, 2), no time dimension to check
    out, was_clamped = clamp_action_chunk(actions, min_action=-1.0, max_action=1.0)
    assert was_clamped
    assert torch.all(out <= 1.0 + 1e-6)
    assert torch.all(out >= -1.0 - 1e-6)


def test_clamp_action_chunk_single_action_skips_sequential_checks():
    # (batch, action_dim), no time dimension — discontinuity/jerk checks are a no-op.
    actions = torch.tensor([[100.0, -100.0]])
    out, was_clamped = clamp_action_chunk(actions, max_relative_target=1.0, max_jerk=0.1)
    assert not was_clamped
    assert torch.equal(out, actions)


def test_chunk_safety_processor_step_disabled_is_noop():
    step = ChunkSafetyProcessorStep(enabled=False, max_relative_target=0.1)
    action = torch.tensor([[[0.0], [10.0]]])
    transition = create_transition(action=action)
    result = step(transition)
    assert torch.equal(result[TransitionKey.ACTION], action)


def test_chunk_safety_processor_step_no_limits_is_noop():
    step = ChunkSafetyProcessorStep()
    action = torch.tensor([[[0.0], [10.0]]])
    transition = create_transition(action=action)
    result = step(transition)
    assert torch.equal(result[TransitionKey.ACTION], action)


def test_chunk_safety_processor_step_clamps_chunk():
    step = ChunkSafetyProcessorStep(max_relative_target=1.0)
    action = torch.tensor([[[0.0], [0.0], [10.0]]])
    transition = create_transition(action=action)
    result = step(transition)
    clamped = result[TransitionKey.ACTION]
    deltas = (clamped[:, 1:, :] - clamped[:, :-1, :]).abs()
    assert torch.all(deltas <= 1.0 + 1e-6)


def test_chunk_safety_processor_step_get_config_roundtrip():
    step = ChunkSafetyProcessorStep(max_relative_target=0.5, max_jerk=0.2, min_action=-1.0, max_action=1.0)
    config = step.get_config()
    assert config == {
        "enabled": True,
        "max_relative_target": 0.5,
        "max_jerk": 0.2,
        "min_action": -1.0,
        "max_action": 1.0,
    }

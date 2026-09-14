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

"""Tests for the piR2 latency-adaptive staircase schedule (arXiv 2607.26055)."""

import pytest
import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import (
    _build_flow_matching_inputs,
    _build_staircase_schedule,
)


def _expected_staircase(delay: int, horizon: int) -> torch.Tensor:
    """Eq. 3 written out literally, converted to this file's t=0-clean convention."""
    values = []
    for position in range(horizon):
        if position < delay:
            tau = 1.0
        elif position < horizon - delay:
            tau = 1.0 - (position - delay) / (horizon - 2 * delay)
        else:
            tau = 0.0
        values.append(1.0 - tau)
    return torch.tensor(values)


@pytest.mark.parametrize("horizon", [16, 50])
def test_staircase_matches_paper_equation(horizon):
    torch.manual_seed(0)
    prefix_mask, position_time = _build_staircase_schedule(
        batch_size=64, action_horizon=horizon, max_delay=5, shared_time=torch.rand(64)
    )

    delays = prefix_mask.sum(dim=1)
    assert delays.max() <= 5, "sampled delay must respect max_delay"
    for row, delay in enumerate(delays.tolist()):
        expected = _expected_staircase(delay, horizon)
        torch.testing.assert_close(position_time[row], expected, atol=1e-6, rtol=0)


def test_staircase_regions_are_clean_front_ramp_and_noise_tail():
    torch.manual_seed(0)
    horizon, max_delay = 16, 4
    prefix_mask, position_time = _build_staircase_schedule(
        batch_size=32, action_horizon=horizon, max_delay=max_delay, shared_time=torch.rand(32)
    )

    # Monotone non-decreasing from clean to noise, which is what makes one denoising step
    # able to finish the front while the tail is still pure noise.
    assert (position_time.diff(dim=1) >= 0).all()

    for row, delay in enumerate(prefix_mask.sum(dim=1).tolist()):
        assert (position_time[row, :delay] == 0.0).all(), "front must be clean"
        if delay > 0:
            assert (position_time[row, horizon - delay :] == 1.0).all(), "tail must be pure noise"


def test_prefix_mask_marks_exactly_the_clamped_front():
    torch.manual_seed(0)
    horizon = 16
    prefix_mask, position_time = _build_staircase_schedule(
        batch_size=32, action_horizon=horizon, max_delay=4, shared_time=torch.rand(32)
    )

    # The loss mask (Alg. 1 line 14) must select a contiguous front, otherwise the model is
    # scored on positions it was handed as ground truth.
    assert (prefix_mask.int().diff(dim=1) <= 0).all()
    assert (position_time[prefix_mask] == 0.0).all()

    # Eq. 3 evaluates the ramp to tau=1 at its left endpoint, so the first *supervised*
    # position is also fully clean. Pinned deliberately: it means one position per sample is
    # trained at exactly t=0, where the velocity target a - eps is pure noise in expectation.
    for row, delay in enumerate(prefix_mask.sum(dim=1).tolist()):
        if delay < horizon:
            assert position_time[row, delay] == 0.0


def test_front_is_exactly_ground_truth_even_when_jittered():
    torch.manual_seed(0)
    batch, horizon, dim = 8, 16, 4
    actions = torch.randn(batch, horizon, dim)
    noise = torch.randn(batch, horizon, dim)
    prefix_mask, position_time = _build_staircase_schedule(
        batch_size=batch,
        action_horizon=horizon,
        max_delay=4,
        shared_time=torch.rand(batch),
        time_jitter=0.1,
    )

    x_t, model_time = _build_flow_matching_inputs(
        actions, noise, torch.rand(batch), prefix_mask, position_time
    )

    assert (x_t[prefix_mask] == actions[prefix_mask]).all()
    torch.testing.assert_close(model_time, position_time)


def test_warmup_branch_reproduces_the_standard_flow_path():
    torch.manual_seed(0)
    batch, horizon, dim = 8, 16, 4
    actions = torch.randn(batch, horizon, dim)
    noise = torch.randn(batch, horizon, dim)
    time = torch.rand(batch)

    prefix_mask, position_time = _build_staircase_schedule(
        batch_size=batch,
        action_horizon=horizon,
        max_delay=4,
        shared_time=time,
        warmup_prob=1.0,
    )

    assert not prefix_mask.any(), "the warm-up branch must not clamp a front"
    staircase_x_t, _ = _build_flow_matching_inputs(actions, noise, time, prefix_mask, position_time)
    standard_x_t, _ = _build_flow_matching_inputs(actions, noise, time, None)
    torch.testing.assert_close(staircase_x_t, standard_x_t)


def test_jitter_stays_within_bounds_and_is_off_by_default():
    torch.manual_seed(0)
    kwargs = {
        "batch_size": 32,
        "action_horizon": 16,
        "max_delay": 4,
        "shared_time": torch.rand(32),
    }
    _, plain = _build_staircase_schedule(**kwargs)
    torch.manual_seed(0)
    _, jittered = _build_staircase_schedule(**kwargs, time_jitter=0.2)

    assert not torch.equal(plain, jittered)
    assert jittered.min() >= 0.0 and jittered.max() <= 1.0


def test_config_rejects_staircase_without_a_delay_budget():
    with pytest.raises(ValueError, match="requires rtc_training_max_delay > 0"):
        PI05Config(rtc_training_schedule="staircase", rtc_training_max_delay=0)


def test_config_rejects_a_delay_that_leaves_no_ramp():
    # chunk_size - 2 * max_delay < 1 would collapse the interior the ramp is defined on.
    with pytest.raises(ValueError, match="chunk_size - 2 \\* rtc_training_max_delay"):
        PI05Config(
            chunk_size=8, n_action_steps=8, rtc_training_schedule="staircase", rtc_training_max_delay=4
        )


def test_config_rejects_unknown_schedule():
    with pytest.raises(ValueError, match="must be 'prefix' or 'staircase'"):
        PI05Config(rtc_training_schedule="linear")


def test_default_config_keeps_the_prefix_schedule():
    assert PI05Config().rtc_training_schedule == "prefix"

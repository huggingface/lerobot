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

"""Regression tests for watchdog tracking discrepancy with separate normalization statistics."""

import pytest
import torch

from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from tests.policies.test_act_watchdog import _create_test_policy as _create_act_test_policy
from tests.policies.test_diffusion_watchdog import _create_test_policy as _create_diffusion_test_policy


@pytest.mark.parametrize("state_mean", [0.0, 5.0])
def test_watchdog_perfect_tracking_with_separate_statistics(monkeypatch, state_mean):
    """When state and action have differing normalization statistics, perfect physical tracking
    must not trigger false watchdog discrepancy alarms."""
    policy = _create_act_test_policy(watchdog_tracking_threshold=0.25)
    stats = {
        OBS_STATE: {"mean": torch.full((4,), state_mean), "std": torch.ones(4)},
        ACTION: {"mean": torch.full((4,), 5.0), "std": torch.ones(4)},
    }
    pre, post = make_act_pre_post_processors(policy.config, stats)
    monkeypatch.setattr(policy, "predict_action_chunk", lambda batch: torch.zeros(1, 8, 4))
    batch = pre({OBS_STATE: torch.full((4,), 5.0), OBS_ENV_STATE: torch.zeros(4)})
    command = post(policy.select_action(batch))
    torch.testing.assert_close(command, torch.full_like(command, 5.0))
    tracked = pre({OBS_STATE: command.squeeze(0), OBS_ENV_STATE: torch.zeros(4)})
    policy.select_action(tracked)
    assert policy.watchdog_stats["triggers"] == 0


@pytest.mark.parametrize("state_mean", [0.0, 5.0])
def test_watchdog_drift_triggers_with_separate_statistics(monkeypatch, state_mean):
    """Physical disturbance beyond tracking threshold triggers watchdog even with differing statistics."""
    policy = _create_act_test_policy(watchdog_tracking_threshold=0.25)
    stats = {
        OBS_STATE: {"mean": torch.full((4,), state_mean), "std": torch.ones(4)},
        ACTION: {"mean": torch.full((4,), 5.0), "std": torch.ones(4)},
    }
    pre, post = make_act_pre_post_processors(policy.config, stats)
    monkeypatch.setattr(policy, "predict_action_chunk", lambda batch: torch.zeros(1, 8, 4))
    batch = pre({OBS_STATE: torch.full((4,), 5.0), OBS_ENV_STATE: torch.zeros(4)})
    command = post(policy.select_action(batch))
    torch.testing.assert_close(command, torch.full_like(command, 5.0))

    # Introduce physical disturbance of 1.5 (threshold is 0.25)
    disturbed_state = command.squeeze(0) + 1.5
    tracked = pre({OBS_STATE: disturbed_state, OBS_ENV_STATE: torch.zeros(4)})
    policy.select_action(tracked)
    assert policy.watchdog_stats["triggers"] == 1
    assert policy.watchdog_stats["last_discrepancy"] > 0.25


def test_diffusion_watchdog_separate_statistics(monkeypatch):
    """Verify Diffusion policy watchdog coordinate alignment under MIN_MAX normalization."""
    policy = _create_diffusion_test_policy(watchdog_tracking_threshold=0.25)
    stats = {
        OBS_STATE: {"min": torch.zeros(4), "max": torch.full((4,), 10.0)},
        ACTION: {"min": torch.full((4,), 2.0), "max": torch.full((4,), 8.0)},
    }
    pre, post = make_diffusion_pre_post_processors(policy.config, stats)
    # Return normalized action 0.0 -> maps to raw physical action: (0 + 1) / 2 * (8 - 2) + 2 = 5.0
    monkeypatch.setattr(policy, "predict_action_chunk", lambda batch, noise=None: torch.zeros(1, 16, 4))
    batch = pre({OBS_STATE: torch.full((4,), 5.0), OBS_ENV_STATE: torch.zeros(4)})
    command = post(policy.select_action(batch))
    torch.testing.assert_close(command, torch.full_like(command, 5.0))

    # Perfect tracking: robot reaches 5.0 -> no watchdog trigger
    tracked = pre({OBS_STATE: command.squeeze(0), OBS_ENV_STATE: torch.zeros(4)})
    policy.select_action(tracked)
    assert policy.watchdog_stats["triggers"] == 0

    # Disturbed tracking: robot pushed to 8.0 -> triggers watchdog
    disturbed = pre({OBS_STATE: torch.full((4,), 8.0), OBS_ENV_STATE: torch.zeros(4)})
    policy.select_action(disturbed)
    assert policy.watchdog_stats["triggers"] == 1
    assert policy.watchdog_stats["last_discrepancy"] > 0.25

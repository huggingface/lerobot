#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for ActionInterpolator."""

import importlib.util
from pathlib import Path

import pytest
import torch

# Direct import to avoid triggering the full lerobot.policies init chain,
# which may fail on branches where some processor modules are absent.
_spec = importlib.util.spec_from_file_location(
    "action_interpolator",
    Path(__file__).resolve().parents[3] / "src" / "lerobot" / "policies" / "rtc" / "action_interpolator.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ActionInterpolator = _mod.ActionInterpolator


# Fixtures


@pytest.fixture
def interp2():
    return ActionInterpolator(multiplier=2)


@pytest.fixture
def interp3():
    return ActionInterpolator(multiplier=3)


# Constructor and properties


class TestInit:
    def test_multiplier_1_no_interpolation(self):
        interp = ActionInterpolator(multiplier=1)
        assert interp.multiplier == 1
        assert not interp.enabled

    def test_multiplier_2_enabled(self):
        interp = ActionInterpolator(multiplier=2)
        assert interp.multiplier == 2
        assert interp.enabled

    def test_multiplier_0_raises(self):
        with pytest.raises(ValueError, match="multiplier must be >= 1"):
            ActionInterpolator(multiplier=0)

    def test_negative_multiplier_raises(self):
        with pytest.raises(ValueError, match="multiplier must be >= 1"):
            ActionInterpolator(multiplier=-1)

    def test_default_multiplier_is_1(self):
        interp = ActionInterpolator()
        assert interp.multiplier == 1
        assert not interp.enabled


# needs_new_action


class TestNeedsNewAction:
    def test_true_initially(self, interp2):
        assert interp2.needs_new_action()

    def test_false_after_add(self, interp2):
        interp2.add(torch.tensor([1.0, 2.0]))
        assert not interp2.needs_new_action()

    def test_true_after_buffer_exhausted(self, interp2):
        interp2.add(torch.tensor([1.0, 2.0]))
        interp2.get()  # first action (no prev, so buffer=1)
        assert interp2.needs_new_action()

    def test_true_after_all_interpolated_consumed(self, interp2):
        interp2.add(torch.tensor([0.0, 0.0]))
        interp2.get()  # consume first (buffer=1, no prev yet)
        assert interp2.needs_new_action()

        interp2.add(torch.tensor([2.0, 4.0]))
        interp2.get()  # consume 1st interpolated
        assert not interp2.needs_new_action()
        interp2.get()  # consume 2nd interpolated
        assert interp2.needs_new_action()


# Passthrough (multiplier=1)


class TestPassthrough:
    def test_single_action_returned_as_is(self):
        interp = ActionInterpolator(multiplier=1)
        action = torch.tensor([3.0, 5.0])
        interp.add(action)

        result = interp.get()
        assert result is not None
        torch.testing.assert_close(result, action)

    def test_none_after_single_get(self):
        interp = ActionInterpolator(multiplier=1)
        interp.add(torch.tensor([1.0]))
        interp.get()
        assert interp.get() is None

    def test_sequential_actions(self):
        interp = ActionInterpolator(multiplier=1)
        for val in [1.0, 2.0, 3.0]:
            action = torch.tensor([val])
            interp.add(action)
            result = interp.get()
            torch.testing.assert_close(result, action)
            assert interp.get() is None


# Interpolation (multiplier=2)


class TestInterpolation2x:
    def test_first_action_no_interpolation(self, interp2):
        """First action has no previous — buffer is just [action]."""
        interp2.add(torch.tensor([0.0, 0.0]))
        result = interp2.get()
        torch.testing.assert_close(result, torch.tensor([0.0, 0.0]))
        assert interp2.get() is None

    def test_second_action_produces_two_steps(self, interp2):
        interp2.add(torch.tensor([0.0, 0.0]))
        interp2.get()  # consume first

        interp2.add(torch.tensor([2.0, 4.0]))
        step1 = interp2.get()
        step2 = interp2.get()

        # t=0.5: midpoint, t=1.0: target
        torch.testing.assert_close(step1, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(step2, torch.tensor([2.0, 4.0]))
        assert interp2.get() is None

    def test_three_consecutive_actions(self, interp2):
        a0 = torch.tensor([0.0])
        a1 = torch.tensor([4.0])
        a2 = torch.tensor([10.0])

        interp2.add(a0)
        torch.testing.assert_close(interp2.get(), a0)

        interp2.add(a1)
        torch.testing.assert_close(interp2.get(), torch.tensor([2.0]))  # midpoint(0, 4)
        torch.testing.assert_close(interp2.get(), torch.tensor([4.0]))  # target

        interp2.add(a2)
        torch.testing.assert_close(interp2.get(), torch.tensor([7.0]))  # midpoint(4, 10)
        torch.testing.assert_close(interp2.get(), torch.tensor([10.0]))  # target


# Interpolation (multiplier=3)


class TestInterpolation3x:
    def test_produces_three_steps(self, interp3):
        interp3.add(torch.tensor([0.0, 0.0]))
        interp3.get()  # consume first (no prev)

        interp3.add(torch.tensor([3.0, 6.0]))
        s1 = interp3.get()
        s2 = interp3.get()
        s3 = interp3.get()

        # t=1/3, t=2/3, t=1
        torch.testing.assert_close(s1, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(s2, torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(s3, torch.tensor([3.0, 6.0]))
        assert interp3.get() is None

    def test_last_step_equals_target(self, interp3):
        interp3.add(torch.tensor([10.0]))
        interp3.get()

        target = torch.tensor([100.0])
        interp3.add(target)
        interp3.get()  # skip 1/3
        interp3.get()  # skip 2/3
        last = interp3.get()
        torch.testing.assert_close(last, target)


# Reset


class TestReset:
    def test_reset_clears_buffer(self, interp2):
        interp2.add(torch.tensor([1.0]))
        interp2.reset()
        assert interp2.needs_new_action()
        assert interp2.get() is None

    def test_reset_clears_prev(self, interp2):
        """After reset, next add should produce single-element buffer (no prev)."""
        interp2.add(torch.tensor([0.0]))
        interp2.get()
        interp2.add(torch.tensor([10.0]))  # would normally interpolate
        interp2.get()
        interp2.get()

        interp2.reset()
        interp2.add(torch.tensor([5.0]))
        result = interp2.get()
        # No interpolation (no prev after reset), just the raw action
        torch.testing.assert_close(result, torch.tensor([5.0]))
        assert interp2.get() is None

    def test_episode_boundary(self, interp2):
        """Simulate two episodes with reset between them."""
        interp2.add(torch.tensor([0.0]))
        interp2.get()
        interp2.add(torch.tensor([10.0]))
        interp2.get()
        interp2.get()

        interp2.reset()

        interp2.add(torch.tensor([100.0]))
        result = interp2.get()
        torch.testing.assert_close(result, torch.tensor([100.0]))
        assert interp2.get() is None


# get_control_interval


class TestControlInterval:
    def test_30fps_multiplier_1(self):
        interp = ActionInterpolator(multiplier=1)
        assert interp.get_control_interval(30.0) == pytest.approx(1.0 / 30.0)

    def test_30fps_multiplier_2(self, interp2):
        assert interp2.get_control_interval(30.0) == pytest.approx(1.0 / 60.0)

    def test_30fps_multiplier_3(self, interp3):
        assert interp3.get_control_interval(30.0) == pytest.approx(1.0 / 90.0)

    def test_60fps_multiplier_2(self, interp2):
        assert interp2.get_control_interval(60.0) == pytest.approx(1.0 / 120.0)


# get() on empty


class TestGetEmpty:
    def test_get_before_any_add(self):
        interp = ActionInterpolator(multiplier=2)
        assert interp.get() is None

    def test_get_after_reset(self, interp2):
        interp2.add(torch.tensor([1.0]))
        interp2.reset()
        assert interp2.get() is None


# Multi-dimensional actions


class TestMultiDim:
    def test_6dof_interpolation(self, interp2):
        prev = torch.zeros(6)
        target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        interp2.add(prev)
        interp2.get()

        interp2.add(target)
        mid = interp2.get()
        end = interp2.get()

        torch.testing.assert_close(mid, target / 2)
        torch.testing.assert_close(end, target)


# Simulated control loop


class TestControlLoop:
    def test_loop_produces_correct_action_count(self):
        """Simulate a control loop: N policy actions with multiplier M should yield
        1 + (N-1)*M total robot commands (first add has no prev so only 1 action)."""
        multiplier = 3
        n_policy_actions = 5
        interp = ActionInterpolator(multiplier=multiplier)

        robot_commands = 0
        for i in range(n_policy_actions):
            action = torch.tensor([float(i)])
            if interp.needs_new_action():
                interp.add(action)
            while True:
                a = interp.get()
                if a is None:
                    break
                robot_commands += 1

        expected = 1 + (n_policy_actions - 1) * multiplier
        assert robot_commands == expected

    def test_loop_monotonic_increase(self):
        """Actions [0, 1, 2, 3] with multiplier=2 should produce monotonically
        increasing interpolated values."""
        interp = ActionInterpolator(multiplier=2)
        all_values = []

        for i in range(4):
            interp.add(torch.tensor([float(i)]))
            while True:
                a = interp.get()
                if a is None:
                    break
                all_values.append(a.item())

        for i in range(1, len(all_values)):
            assert all_values[i] >= all_values[i - 1]

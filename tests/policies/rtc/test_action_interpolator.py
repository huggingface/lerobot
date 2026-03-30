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

"""Tests for ActionInterpolator and its interaction with ActionQueue (RTC)."""

import pytest
import torch

from lerobot.policies.rtc.action_interpolator import ActionInterpolator
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig

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


# ActionQueue + ActionInterpolator integration


def _make_chunk(n_steps: int, action_dim: int = 2, offset: float = 0.0) -> torch.Tensor:
    """Create a simple action chunk: each row is [offset + step_idx, offset + step_idx]."""
    return torch.arange(n_steps, dtype=torch.float32).unsqueeze(1).expand(-1, action_dim) + offset


class TestQueueInterpolatorIntegration:
    """Verify the interpolator doesn't interfere with ActionQueue leftover tracking."""

    def test_queue_consumption_rate_matches_base_fps(self):
        """With multiplier=3, the interpolator calls queue.get() once per 3 control
        ticks, so the queue consumption rate equals base fps, not multiplied fps."""
        cfg = RTCConfig(enabled=True, execution_horizon=10)
        queue = ActionQueue(cfg)
        interp = ActionInterpolator(multiplier=3)

        chunk = _make_chunk(10)
        queue.merge(chunk, chunk.clone(), real_delay=0)

        queue_gets = 0
        control_ticks = 0

        # Drain both queue AND remaining interpolator buffer
        while True:
            if interp.needs_new_action():
                if queue.empty():
                    break
                action = queue.get()
                if action is None:
                    break
                interp.add(action)
                queue_gets += 1

            result = interp.get()
            if result is not None:
                control_ticks += 1

        assert queue_gets == 10
        # First get produces 1 tick (no prev), remaining 9 produce 3 ticks each
        assert control_ticks == 1 + 9 * 3

    def test_leftover_decreases_only_on_queue_get(self):
        """get_left_over() should shrink only when queue.get() is called,
        not when the interpolator produces intermediate actions."""
        cfg = RTCConfig(enabled=True, execution_horizon=10)
        queue = ActionQueue(cfg)
        interp = ActionInterpolator(multiplier=3)

        chunk = _make_chunk(6)
        queue.merge(chunk, chunk.clone(), real_delay=0)

        # Pull first action into interpolator
        assert interp.needs_new_action()
        interp.add(queue.get())
        leftover_after_first_get = queue.get_left_over()
        assert leftover_after_first_get is not None
        assert len(leftover_after_first_get) == 5  # 6 - 1

        # Consume interpolated sub-steps — queue leftover must NOT change
        interp.get()  # only 1 sub-step (no prev)
        assert len(queue.get_left_over()) == 5  # unchanged

        # Pull second action
        interp.add(queue.get())
        assert len(queue.get_left_over()) == 4  # 6 - 2

        # Consume all 3 sub-steps — leftover stays at 4
        for _ in range(3):
            assert interp.get() is not None
        assert len(queue.get_left_over()) == 4

    def test_processed_leftover_tracks_queue_index(self):
        """get_processed_left_over() must reflect the queue's last_index,
        independent of interpolator state."""
        cfg = RTCConfig(enabled=True, execution_horizon=10)
        queue = ActionQueue(cfg)
        interp = ActionInterpolator(multiplier=2)

        original = _make_chunk(8, offset=0.0)
        processed = _make_chunk(8, offset=100.0)
        queue.merge(original, processed, real_delay=0)

        # Before any get
        left = queue.get_processed_left_over()
        assert len(left) == 8

        # Get 3 actions from queue through interpolator
        for _ in range(3):
            if interp.needs_new_action():
                action = queue.get()
                if action is not None:
                    interp.add(action)
            interp.get()

        # Queue has consumed 2 items (first get + second get after 2 interp sub-steps)
        # But let's verify by checking the actual leftover
        proc_left = queue.get_processed_left_over()
        orig_left = queue.get_left_over()
        assert proc_left is not None and orig_left is not None
        assert len(proc_left) == len(orig_left)
        # Processed leftovers start at 100+idx, original at 0+idx — different offsets
        assert proc_left[0, 0].item() >= 100.0
        assert orig_left[0, 0].item() < 100.0

    def test_merge_resets_queue_but_interpolator_keeps_prev(self):
        """After a queue merge (new chunk), the interpolator retains its prev action
        so the first get from the new chunk still interpolates smoothly."""
        cfg = RTCConfig(enabled=True, execution_horizon=10)
        queue = ActionQueue(cfg)
        interp = ActionInterpolator(multiplier=2)

        # Chunk 1: [0, 2, 4, 6, 8]
        chunk1 = torch.tensor([[0.0], [2.0], [4.0], [6.0], [8.0]])
        queue.merge(chunk1, chunk1.clone(), real_delay=0)

        # Consume exactly 3 queue actions (5 control ticks: 1 + 2 + 2)
        consumed = []
        for _ in range(5):
            if interp.needs_new_action():
                a = queue.get()
                if a is not None:
                    interp.add(a)
            r = interp.get()
            if r is not None:
                consumed.append(r.item())

        # After 5 ticks: queue.get() called 3 times → consumed [0], [2], [4]
        # Interpolator prev = [4], buffer exhausted
        assert interp.needs_new_action()
        assert consumed[-1] == pytest.approx(4.0)

        # Simulate RTC: capture index before inference (as the real code does)
        idx_before = queue.get_action_index()

        # Merge new chunk — pass idx_before so delay = last_index - idx_before = 0
        chunk2 = torch.tensor([[10.0], [12.0], [14.0]])
        queue.merge(chunk2, chunk2.clone(), real_delay=0, action_index_before_inference=idx_before)

        # Interpolator still has prev=[4] from chunk1
        first_action = queue.get()
        assert first_action is not None
        interp.add(first_action)
        first_from_new = interp.get()
        assert first_from_new is not None
        # Midpoint between prev=4 and new=10 → 7.0
        assert first_from_new.item() == pytest.approx(7.0)

    def test_interpolator_reset_does_not_affect_queue(self):
        """Resetting the interpolator (e.g. on episode boundary or resume) should
        not touch the queue state."""
        cfg = RTCConfig(enabled=True, execution_horizon=10)
        queue = ActionQueue(cfg)
        interp = ActionInterpolator(multiplier=2)

        chunk = _make_chunk(5)
        queue.merge(chunk, chunk.clone(), real_delay=0)

        # Consume 2 actions from queue
        interp.add(queue.get())
        interp.get()
        interp.add(queue.get())
        interp.get()
        interp.get()

        assert queue.qsize() == 3

        # Reset interpolator
        interp.reset()

        # Queue is untouched
        assert queue.qsize() == 3
        assert len(queue.get_left_over()) == 3

        # Can still pull from queue after reset
        interp.add(queue.get())
        result = interp.get()
        assert result is not None
        assert queue.qsize() == 2

    def test_no_interpolation_queue_consumption_is_1_to_1(self):
        """With multiplier=1, each queue.get() produces exactly 1 robot command."""
        cfg = RTCConfig(enabled=True, execution_horizon=10)
        queue = ActionQueue(cfg)
        interp = ActionInterpolator(multiplier=1)

        chunk = _make_chunk(5)
        queue.merge(chunk, chunk.clone(), real_delay=0)

        robot_commands = 0
        while not queue.empty():
            if interp.needs_new_action():
                action = queue.get()
                if action is not None:
                    interp.add(action)
            result = interp.get()
            if result is not None:
                robot_commands += 1

        assert robot_commands == 5

    def test_queue_delay_with_interpolation(self):
        """Verify merge with real_delay correctly skips stale actions,
        and the interpolator picks up from the right point."""
        cfg = RTCConfig(enabled=True, execution_horizon=10)
        queue = ActionQueue(cfg)
        interp = ActionInterpolator(multiplier=2)

        # Initial chunk
        chunk1 = _make_chunk(10)
        queue.merge(chunk1, chunk1.clone(), real_delay=0)

        # Consume 3 actions while "inference" runs
        for _ in range(5):  # 1 + 2 + 2 = 5 control ticks = 3 queue gets
            if interp.needs_new_action():
                a = queue.get()
                if a is not None:
                    interp.add(a)
            interp.get()

        assert queue.get_action_index() == 3

        # New chunk arrives, delay=3 (3 actions consumed during inference)
        chunk2 = _make_chunk(10, offset=100.0)
        queue.merge(chunk2, chunk2.clone(), real_delay=0, action_index_before_inference=0)

        # Queue should have skipped delay actions
        first_action = queue.get()
        assert first_action is not None
        # After merge with index-based delay of 3, first available = chunk2[3]
        torch.testing.assert_close(first_action, torch.tensor([103.0, 103.0]))

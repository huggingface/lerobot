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

import math

import pytest
import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.action_interpolator import ActionInterpolator

# ====================== Fixtures ======================


@pytest.fixture
def interp2():
    """Create an ActionInterpolator with multiplier=2."""
    return ActionInterpolator(multiplier=2)


@pytest.fixture
def interp3():
    """Create an ActionInterpolator with multiplier=3."""
    return ActionInterpolator(multiplier=3)


# ====================== Initialization Tests ======================


def test_interpolator_multiplier_1_no_interpolation():
    """Test multiplier=1 creates a disabled interpolator."""
    interp = ActionInterpolator(multiplier=1)
    assert interp.multiplier == 1
    assert not interp.enabled


def test_interpolator_multiplier_2_enabled():
    """Test multiplier=2 creates an enabled interpolator."""
    interp = ActionInterpolator(multiplier=2)
    assert interp.multiplier == 2
    assert interp.enabled


def test_interpolator_multiplier_0_raises():
    """Test multiplier=0 raises ValueError."""
    with pytest.raises(ValueError, match="multiplier must be >= 1"):
        ActionInterpolator(multiplier=0)


def test_interpolator_negative_multiplier_raises():
    """Test negative multiplier raises ValueError."""
    with pytest.raises(ValueError, match="multiplier must be >= 1"):
        ActionInterpolator(multiplier=-1)


def test_interpolator_default_multiplier_is_1():
    """Test default multiplier is 1 (disabled)."""
    interp = ActionInterpolator()
    assert interp.multiplier == 1
    assert not interp.enabled


# ====================== needs_new_action Tests ======================


def test_needs_new_action_true_initially(interp2):
    """Test needs_new_action() returns True before any action is added."""
    assert interp2.needs_new_action()


def test_needs_new_action_false_after_add(interp2):
    """Test needs_new_action() returns False right after add()."""
    interp2.add(torch.tensor([1.0, 2.0]))
    assert not interp2.needs_new_action()


def test_needs_new_action_true_after_buffer_exhausted(interp2):
    """Test needs_new_action() returns True after consuming all buffered actions."""
    interp2.add(torch.tensor([1.0, 2.0]))
    interp2.get()
    assert interp2.needs_new_action()


def test_needs_new_action_true_after_all_interpolated_consumed(interp2):
    """Test needs_new_action() tracks interpolated sub-steps correctly."""
    interp2.add(torch.tensor([0.0, 0.0]))
    interp2.get()
    assert interp2.needs_new_action()

    interp2.add(torch.tensor([2.0, 4.0]))
    interp2.get()
    assert not interp2.needs_new_action()
    interp2.get()
    assert interp2.needs_new_action()


# ====================== Passthrough Tests (multiplier=1) ======================


def test_passthrough_single_action_returned_as_is():
    """Test multiplier=1 returns the action unchanged."""
    interp = ActionInterpolator(multiplier=1)
    action = torch.tensor([3.0, 5.0])
    interp.add(action)

    result = interp.get()
    assert result is not None
    torch.testing.assert_close(result, action)


def test_passthrough_none_after_single_get():
    """Test multiplier=1 returns None after consuming the single action."""
    interp = ActionInterpolator(multiplier=1)
    interp.add(torch.tensor([1.0]))
    interp.get()
    assert interp.get() is None


def test_passthrough_sequential_actions():
    """Test multiplier=1 passes through consecutive actions one at a time."""
    interp = ActionInterpolator(multiplier=1)
    for val in [1.0, 2.0, 3.0]:
        action = torch.tensor([val])
        interp.add(action)
        result = interp.get()
        torch.testing.assert_close(result, action)
        assert interp.get() is None


# ====================== Interpolation Tests (multiplier=2) ======================


def test_interpolation_2x_first_action_no_interpolation(interp2):
    """Test first action has no previous, so buffer is just [action]."""
    interp2.add(torch.tensor([0.0, 0.0]))
    result = interp2.get()
    torch.testing.assert_close(result, torch.tensor([0.0, 0.0]))
    assert interp2.get() is None


def test_interpolation_2x_second_action_produces_two_steps(interp2):
    """Test second action produces 2 interpolated sub-steps."""
    interp2.add(torch.tensor([0.0, 0.0]))
    interp2.get()

    interp2.add(torch.tensor([2.0, 4.0]))
    step1 = interp2.get()
    step2 = interp2.get()

    torch.testing.assert_close(step1, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(step2, torch.tensor([2.0, 4.0]))
    assert interp2.get() is None


def test_interpolation_2x_three_consecutive_actions(interp2):
    """Test interpolation across three consecutive actions."""
    a0 = torch.tensor([0.0])
    a1 = torch.tensor([4.0])
    a2 = torch.tensor([10.0])

    interp2.add(a0)
    torch.testing.assert_close(interp2.get(), a0)

    interp2.add(a1)
    torch.testing.assert_close(interp2.get(), torch.tensor([2.0]))
    torch.testing.assert_close(interp2.get(), torch.tensor([4.0]))

    interp2.add(a2)
    torch.testing.assert_close(interp2.get(), torch.tensor([7.0]))
    torch.testing.assert_close(interp2.get(), torch.tensor([10.0]))


# ====================== Interpolation Tests (multiplier=3) ======================


def test_interpolation_3x_produces_three_steps(interp3):
    """Test multiplier=3 produces 3 interpolated sub-steps."""
    interp3.add(torch.tensor([0.0, 0.0]))
    interp3.get()

    interp3.add(torch.tensor([3.0, 6.0]))
    s1 = interp3.get()
    s2 = interp3.get()
    s3 = interp3.get()

    torch.testing.assert_close(s1, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(s2, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(s3, torch.tensor([3.0, 6.0]))
    assert interp3.get() is None


def test_interpolation_3x_last_step_equals_target(interp3):
    """Test last interpolated step equals the target action exactly."""
    interp3.add(torch.tensor([10.0]))
    interp3.get()

    target = torch.tensor([100.0])
    interp3.add(target)
    interp3.get()
    interp3.get()
    last = interp3.get()
    torch.testing.assert_close(last, target)


# ====================== Reset Tests ======================


def test_reset_clears_buffer(interp2):
    """Test reset() clears the action buffer."""
    interp2.add(torch.tensor([1.0]))
    interp2.reset()
    assert interp2.needs_new_action()
    assert interp2.get() is None


def test_reset_clears_prev(interp2):
    """Test after reset, next add produces single-element buffer (no prev)."""
    interp2.add(torch.tensor([0.0]))
    interp2.get()
    interp2.add(torch.tensor([10.0]))
    interp2.get()
    interp2.get()

    interp2.reset()
    interp2.add(torch.tensor([5.0]))
    result = interp2.get()
    torch.testing.assert_close(result, torch.tensor([5.0]))
    assert interp2.get() is None


def test_reset_episode_boundary(interp2):
    """Test reset between two simulated episodes."""
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


# ====================== get_control_interval Tests ======================


def test_control_interval_30fps_multiplier_1():
    """Test control interval at 30fps with no interpolation."""
    interp = ActionInterpolator(multiplier=1)
    assert interp.get_control_interval(30.0) == pytest.approx(1.0 / 30.0)


def test_control_interval_30fps_multiplier_2(interp2):
    """Test control interval at 30fps with 2x interpolation."""
    assert interp2.get_control_interval(30.0) == pytest.approx(1.0 / 60.0)


def test_control_interval_30fps_multiplier_3(interp3):
    """Test control interval at 30fps with 3x interpolation."""
    assert interp3.get_control_interval(30.0) == pytest.approx(1.0 / 90.0)


def test_control_interval_60fps_multiplier_2(interp2):
    """Test control interval at 60fps with 2x interpolation."""
    assert interp2.get_control_interval(60.0) == pytest.approx(1.0 / 120.0)


# ====================== get() on Empty Tests ======================


def test_get_returns_none_before_any_add():
    """Test get() returns None when no action has been added."""
    interp = ActionInterpolator(multiplier=2)
    assert interp.get() is None


def test_get_returns_none_after_reset(interp2):
    """Test get() returns None after reset."""
    interp2.add(torch.tensor([1.0]))
    interp2.reset()
    assert interp2.get() is None


# ====================== Multi-Dimensional Action Tests ======================


def test_6dof_interpolation(interp2):
    """Test interpolation works correctly with 6-dimensional actions."""
    prev = torch.zeros(6)
    target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    interp2.add(prev)
    interp2.get()

    interp2.add(target)
    mid = interp2.get()
    end = interp2.get()

    torch.testing.assert_close(mid, target / 2)
    torch.testing.assert_close(end, target)


# ====================== Simulated Control Loop Tests ======================


def test_control_loop_produces_correct_action_count():
    """Test N policy actions with multiplier M yields 1 + (N-1)*M robot commands."""
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


def test_control_loop_monotonic_increase():
    """Test actions [0, 1, 2, 3] with multiplier=2 produce monotonically increasing values."""
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


# ====================== ActionQueue + ActionInterpolator Integration Tests ======================


def _make_chunk(n_steps: int, action_dim: int = 2, offset: float = 0.0) -> torch.Tensor:
    """Create a simple action chunk: each row is [offset + step_idx, offset + step_idx]."""
    return torch.arange(n_steps, dtype=torch.float32).unsqueeze(1).expand(-1, action_dim) + offset


def test_queue_interpolator_consumption_rate_matches_base_fps():
    """Test queue.get() is called at base fps rate, not multiplied fps."""
    cfg = RTCConfig(enabled=True, execution_horizon=10)
    queue = ActionQueue(cfg)
    interp = ActionInterpolator(multiplier=3)

    chunk = _make_chunk(10)
    queue.merge(chunk, chunk.clone(), real_delay=0)

    queue_gets = 0
    control_ticks = 0

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
    assert control_ticks == 1 + 9 * 3


def test_queue_interpolator_leftover_decreases_only_on_queue_get():
    """Test get_left_over() shrinks only on queue.get(), not on interpolator sub-steps."""
    cfg = RTCConfig(enabled=True, execution_horizon=10)
    queue = ActionQueue(cfg)
    interp = ActionInterpolator(multiplier=3)

    chunk = _make_chunk(6)
    queue.merge(chunk, chunk.clone(), real_delay=0)

    assert interp.needs_new_action()
    interp.add(queue.get())
    leftover_after_first_get = queue.get_left_over()
    assert leftover_after_first_get is not None
    assert len(leftover_after_first_get) == 5

    interp.get()
    assert len(queue.get_left_over()) == 5

    interp.add(queue.get())
    assert len(queue.get_left_over()) == 4

    for _ in range(3):
        assert interp.get() is not None
    assert len(queue.get_left_over()) == 4


def test_queue_interpolator_processed_leftover_tracks_queue_index():
    """Test get_processed_left_over() reflects queue's last_index, not interpolator state."""
    cfg = RTCConfig(enabled=True, execution_horizon=10)
    queue = ActionQueue(cfg)
    interp = ActionInterpolator(multiplier=2)

    original = _make_chunk(8, offset=0.0)
    processed = _make_chunk(8, offset=100.0)
    queue.merge(original, processed, real_delay=0)

    left = queue.get_processed_left_over()
    assert len(left) == 8

    for _ in range(3):
        if interp.needs_new_action():
            action = queue.get()
            if action is not None:
                interp.add(action)
        interp.get()

    proc_left = queue.get_processed_left_over()
    orig_left = queue.get_left_over()
    assert proc_left is not None and orig_left is not None
    assert len(proc_left) == len(orig_left)
    assert proc_left[0, 0].item() >= 100.0
    assert orig_left[0, 0].item() < 100.0


def test_queue_interpolator_merge_resets_queue_but_interpolator_keeps_prev():
    """Test queue merge doesn't affect interpolator's prev, enabling smooth transitions."""
    cfg = RTCConfig(enabled=True, execution_horizon=10)
    queue = ActionQueue(cfg)
    interp = ActionInterpolator(multiplier=2)

    chunk1 = torch.tensor([[0.0], [2.0], [4.0], [6.0], [8.0]])
    queue.merge(chunk1, chunk1.clone(), real_delay=0)

    consumed = []
    for _ in range(5):
        if interp.needs_new_action():
            a = queue.get()
            if a is not None:
                interp.add(a)
        r = interp.get()
        if r is not None:
            consumed.append(r.item())

    assert interp.needs_new_action()
    assert consumed[-1] == pytest.approx(4.0)

    idx_before = queue.get_action_index()

    chunk2 = torch.tensor([[10.0], [12.0], [14.0]])
    queue.merge(chunk2, chunk2.clone(), real_delay=0, action_index_before_inference=idx_before)

    first_action = queue.get()
    assert first_action is not None
    interp.add(first_action)
    first_from_new = interp.get()
    assert first_from_new is not None
    assert first_from_new.item() == pytest.approx(7.0)


def test_queue_interpolator_reset_does_not_affect_queue():
    """Test interpolator reset leaves queue state untouched."""
    cfg = RTCConfig(enabled=True, execution_horizon=10)
    queue = ActionQueue(cfg)
    interp = ActionInterpolator(multiplier=2)

    chunk = _make_chunk(5)
    queue.merge(chunk, chunk.clone(), real_delay=0)

    interp.add(queue.get())
    interp.get()
    interp.add(queue.get())
    interp.get()
    interp.get()

    assert queue.qsize() == 3

    interp.reset()

    assert queue.qsize() == 3
    assert len(queue.get_left_over()) == 3

    interp.add(queue.get())
    result = interp.get()
    assert result is not None
    assert queue.qsize() == 2


def test_queue_interpolator_no_interpolation_1_to_1():
    """Test multiplier=1 produces exactly 1 robot command per queue.get()."""
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


def test_queue_interpolator_delay_skips_stale_actions():
    """Test merge with delay correctly skips stale actions for the interpolator."""
    cfg = RTCConfig(enabled=True, execution_horizon=10)
    queue = ActionQueue(cfg)
    interp = ActionInterpolator(multiplier=2)

    chunk1 = _make_chunk(10)
    queue.merge(chunk1, chunk1.clone(), real_delay=0)

    for _ in range(5):
        if interp.needs_new_action():
            a = queue.get()
            if a is not None:
                interp.add(a)
        interp.get()

    assert queue.get_action_index() == 3

    chunk2 = _make_chunk(10, offset=100.0)
    queue.merge(chunk2, chunk2.clone(), real_delay=3, action_index_before_inference=0)

    first_action = queue.get()
    assert first_action is not None
    torch.testing.assert_close(first_action, torch.tensor([103.0, 103.0]))


# ====================== Rotation-vector canonicalization tests (#3691) ======================


def _rotmat(rv: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula: rotation vector -> rotation matrix."""
    theta = torch.linalg.vector_norm(rv)
    if theta < 1e-12:
        return torch.eye(3, dtype=rv.dtype)
    k = rv / theta
    kx = torch.tensor([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=rv.dtype)
    return torch.eye(3, dtype=rv.dtype) + torch.sin(theta) * kx + (1 - torch.cos(theta)) * (kx @ kx)


def _geodesic_deg(rv_a: torch.Tensor, rv_b: torch.Tensor) -> float:
    """Rotation angle in degrees between the rotations encoded by two rotvecs."""
    cos = (torch.trace(_rotmat(rv_a).T @ _rotmat(rv_b)) - 1) / 2
    return float(torch.rad2deg(torch.arccos(torch.clamp(cos, -1.0, 1.0))))


def _antipodal_twin_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Two rotvecs encoding the same physical rotation, ~2*pi apart in vector space.

    179 deg about +x and 181 deg about -x are the same rotation (geodesic
    distance 0) but their rotvec encodings differ by ~2*pi.
    """
    prev = torch.tensor([math.radians(179.0), 0.0, 0.0], dtype=torch.float64)
    cur = torch.tensor([-math.radians(181.0), 0.0, 0.0], dtype=torch.float64)
    return prev, cur


def test_rotation_dims_antipodal_twins_no_identity_sweep():
    """Interpolating between antipodal-twin rotvecs must not sweep through identity."""
    prev, cur = _antipodal_twin_pair()
    assert _geodesic_deg(prev, cur) < 1e-6  # same physical rotation

    # Without rotation_dims the midpoint collapses to ~identity (the bug).
    interp_bug = ActionInterpolator(multiplier=4)
    interp_bug.add(prev)
    interp_bug.get()
    interp_bug.add(cur)
    mid = [interp_bug.get() for _ in range(4)][1]  # t = 0.5
    assert _geodesic_deg(mid, prev) > 170.0

    # With rotation_dims every interpolated rotvec stays at the endpoint pose.
    interp = ActionInterpolator(multiplier=4, rotation_dims=[0, 1, 2])
    interp.add(prev)
    interp.get()
    interp.add(cur)
    for _ in range(4):
        step = interp.get()
        assert step is not None
        assert _geodesic_deg(step, prev) < 1e-6


def test_rotation_dims_none_matches_previous_behavior():
    """rotation_dims=None must reproduce plain linear interpolation exactly."""
    prev, cur = _antipodal_twin_pair()
    interp_default = ActionInterpolator(multiplier=3)
    interp_explicit = ActionInterpolator(multiplier=3, rotation_dims=None)
    for i in (prev, cur):
        interp_default.add(i)
        interp_explicit.add(i)
    for _ in range(3):
        torch.testing.assert_close(interp_default.get(), interp_explicit.get(), rtol=0, atol=0)


def test_rotation_dims_close_rotvecs_identical_to_plain_lerp():
    """When no twin is closer, canonicalization is a no-op and output equals plain lerp."""
    prev = torch.tensor([0.1, 0.2, 0.3])
    cur = torch.tensor([0.15, 0.25, 0.35])
    plain = ActionInterpolator(multiplier=2)
    canon = ActionInterpolator(multiplier=2, rotation_dims=[0, 1, 2])
    for i in (prev, cur):
        plain.add(i)
        canon.add(i)
    for _ in range(2):
        torch.testing.assert_close(plain.get(), canon.get(), rtol=0, atol=0)


def test_rotation_dims_other_dims_unaffected():
    """Position and gripper dims must interpolate linearly even with rotation_dims set."""
    prev_rot, cur_rot = _antipodal_twin_pair()
    prev = torch.cat(
        [
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
            prev_rot,
            torch.tensor([0.0], dtype=torch.float64),
        ]
    )
    cur = torch.cat(
        [
            torch.tensor([0.3, 0.6, 0.9], dtype=torch.float64),
            cur_rot,
            torch.tensor([1.0], dtype=torch.float64),
        ]
    )
    interp = ActionInterpolator(multiplier=3, rotation_dims=[3, 4, 5])
    interp.add(prev)
    interp.get()
    interp.add(cur)
    for i in range(1, 4):
        t = i / 3
        step = interp.get()
        assert step is not None
        expected_pos = prev[:3] + t * (cur[:3] - prev[:3])
        expected_grip = prev[6:] + t * (cur[6:] - prev[6:])
        torch.testing.assert_close(step[:3], expected_pos, rtol=0, atol=0)
        torch.testing.assert_close(step[6:], expected_grip, rtol=0, atol=0)


def test_rotation_dims_zero_rotation_no_nan():
    """A zero rotvec (identity rotation) must not produce NaNs."""
    interp = ActionInterpolator(multiplier=2, rotation_dims=[0, 1, 2])
    interp.add(torch.tensor([0.5, 0.0, 0.0]))
    interp.get()
    interp.add(torch.zeros(3))
    for _ in range(2):
        step = interp.get()
        assert step is not None
        assert torch.isfinite(step).all()


def test_rotation_dims_bimanual_triplets():
    """Independent triplets: one arm needing a twin flip must not disturb the other."""
    prev_rot, cur_rot = _antipodal_twin_pair()
    near_prev = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64)
    near_cur = torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64)
    prev = torch.cat([prev_rot, near_prev])
    cur = torch.cat([cur_rot, near_cur])
    interp = ActionInterpolator(multiplier=2, rotation_dims=[0, 1, 2, 3, 4, 5])
    interp.add(prev)
    interp.get()
    interp.add(cur)
    mid = interp.get()
    assert mid is not None
    assert _geodesic_deg(mid[:3], prev_rot) < 1e-6  # arm 1: stays at pose (twins)
    torch.testing.assert_close(mid[3:], (near_prev + near_cur) / 2, rtol=0, atol=1e-12)  # arm 2: plain lerp


def test_rotation_dims_invalid_length_raises():
    """rotation_dims not grouped in triplets must raise ValueError."""
    with pytest.raises(ValueError, match="triplets"):
        ActionInterpolator(multiplier=2, rotation_dims=[3, 4])

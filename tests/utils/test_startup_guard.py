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

"""Tests for the startup joint-mismatch guard (pure logic, no hardware)."""

import pytest

from lerobot.utils.startup_guard import StartupJointGuard, StartupMismatchError


def test_pass_through_when_agreeing():
    guard = StartupJointGuard(threshold=10.0)
    action = {"shoulder_pan.pos": 1.0, "gripper.pos": -5.0}
    obs = {"shoulder_pan.pos": 0.5, "gripper.pos": -4.0}
    out = guard.process(action, obs, now=0.0)
    assert out is action
    assert not guard.is_ramping
    # Subsequent frames stay pass-through even with huge deltas (guard is disarmed).
    wild = {"shoulder_pan.pos": 500.0}
    assert guard.process(wild, obs, now=0.1) is wild


def test_first_frame_mismatch_triggers_ramp():
    guard = StartupJointGuard(threshold=10.0, ramp_duration_s=1.0)
    # The verified real-world case: a 2*pi-wrapped multi-turn gripper. Leader streams
    # +356.8 deg while the follower measures ~0 (physically closed).
    action = {"gripper.pos": 356.8, "shoulder_pan.pos": 0.2}
    obs = {"gripper.pos": -0.5, "shoulder_pan.pos": 0.0}
    out = guard.process(action, obs, now=0.0)
    assert guard.is_ramping
    # At t=0 the command must equal the measured position, not the target.
    assert out["gripper.pos"] == pytest.approx(-0.5)
    # Joints within threshold ramp too (base captured for all pairs), starting at measured.
    assert out["shoulder_pan.pos"] == pytest.approx(0.0)

    # Halfway through the ramp: halfway between measured and the live target.
    out = guard.process(action, obs, now=0.5)
    assert out["gripper.pos"] == pytest.approx(-0.5 + 0.5 * (356.8 - (-0.5)))

    # After the ramp: exact pass-through again.
    out = guard.process(action, obs, now=1.5)
    assert out is action
    assert not guard.is_ramping


def test_ramp_tracks_moving_target():
    guard = StartupJointGuard(threshold=1.0, ramp_duration_s=1.0)
    obs = {"j1.pos": 0.0}
    guard.process({"j1.pos": 100.0}, obs, now=0.0)
    # Target moved mid-ramp: blend goes toward the NEW target (leader keeps authority).
    out = guard.process({"j1.pos": 50.0}, obs, now=0.5)
    assert out["j1.pos"] == pytest.approx(0.0 + 0.5 * 50.0)


def test_abort_mode_raises_with_offending_joints():
    guard = StartupJointGuard(threshold=10.0, mode="abort")
    action = {"gripper.pos": 356.8, "shoulder_pan.pos": 0.0}
    obs = {"gripper.pos": -0.5, "shoulder_pan.pos": 0.0}
    with pytest.raises(StartupMismatchError, match="gripper.pos"):
        guard.process(action, obs, now=0.0)


def test_ignores_non_pos_and_missing_keys():
    guard = StartupJointGuard(threshold=1.0)
    action = {"j1.vel": 999.0, "j2.pos": 5.0, "camera": object()}
    obs = {"j1.vel": 0.0}  # j2.pos missing from obs -> not comparable -> ignored
    out = guard.process(action, obs, now=0.0)
    assert out is action
    assert not guard.is_ramping


def test_reset_rearms():
    guard = StartupJointGuard(threshold=1.0, ramp_duration_s=1.0)
    obs = {"j1.pos": 0.0}
    assert guard.process({"j1.pos": 0.5}, obs, now=0.0) is not None
    assert not guard.is_ramping  # agreed -> disarmed
    guard.reset()
    guard.process({"j1.pos": 100.0}, obs, now=10.0)
    assert guard.is_ramping  # re-armed guard caught the new mismatch


def test_disabled_guard_is_pass_through():
    guard = StartupJointGuard(threshold=0.1, enabled=False)
    action = {"j1.pos": 1000.0}
    obs = {"j1.pos": 0.0}
    assert guard.process(action, obs, now=0.0) is action
    assert not guard.is_ramping


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        StartupJointGuard(mode="explode")
    with pytest.raises(ValueError):
        StartupJointGuard(threshold=0.0)
    with pytest.raises(ValueError):
        StartupJointGuard(ramp_duration_s=-1.0)

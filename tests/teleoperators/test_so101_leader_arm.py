#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Pure-numpy unit tests for the Isaac Teleop SO-101 leader-arm unit conversion.

These tests deliberately do NOT import ``isaacteleop`` -- :func:`leader_joints_to_robot_action`
is the pure-math bridge from the leader's streamed joint angles [rad] to the follower's native
``{joint}.pos`` units (arm degrees, gripper RANGE_0_100), and must be testable without the XR
runtime (mirroring ``test_xr_controller_processor.py``).
"""

import numpy as np
import pytest

from lerobot.teleoperators.isaac_teleop.config_isaac_teleop import SO101LeaderArmConfig
from lerobot.teleoperators.isaac_teleop.teleop_so101_leader_arm import (
    SO101_LEADER_JOINTS,
    leader_joints_to_robot_action,
)

# Convenience: convert with the default gripper endpoints unless overridden.
_GRIPPER_OPEN = -0.074
_GRIPPER_CLOSE = 1.460


def _convert(joints_rad, gripper_open=_GRIPPER_OPEN, gripper_close=_GRIPPER_CLOSE):
    return leader_joints_to_robot_action(
        joints_rad,
        gripper_joint="gripper",
        gripper_open_rad=gripper_open,
        gripper_close_rad=gripper_close,
    )


def _all_zero_joints():
    return dict.fromkeys(SO101_LEADER_JOINTS, 0.0)


# ----------------------------------------------------------------------------
# Arm joints: rad -> deg
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rad", "expected_deg"),
    [(0.0, 0.0), (np.pi, 180.0), (-np.pi / 2, -90.0), (np.pi / 4, 45.0)],
)
def test_arm_joint_rad2deg(rad, expected_deg):
    out = _convert({**_all_zero_joints(), "shoulder_pan": float(rad)})
    assert out["shoulder_pan.pos"] == pytest.approx(expected_deg)


def test_all_arm_joints_converted_to_degrees():
    # Every non-gripper DOF is converted rad2deg independently.
    joints = dict.fromkeys(SO101_LEADER_JOINTS, np.pi)
    out = _convert(joints)
    for name in SO101_LEADER_JOINTS:
        if name != "gripper":
            assert out[f"{name}.pos"] == pytest.approx(180.0)


def test_emits_pos_key_per_joint_in_order():
    out = _convert(_all_zero_joints())
    assert list(out.keys()) == [f"{name}.pos" for name in SO101_LEADER_JOINTS]


# ----------------------------------------------------------------------------
# Gripper: rad -> RANGE_0_100 (100 = open, 0 = closed), clipped
# ----------------------------------------------------------------------------


def test_gripper_open_maps_to_100():
    # At the open endpoint, the follower jaw is fully open (100).
    out = _convert({**_all_zero_joints(), "gripper": _GRIPPER_OPEN})
    assert out["gripper.pos"] == pytest.approx(100.0)


def test_gripper_close_maps_to_0():
    # At the close endpoint, the follower jaw is fully closed (0).
    out = _convert({**_all_zero_joints(), "gripper": _GRIPPER_CLOSE})
    assert out["gripper.pos"] == pytest.approx(0.0)


def test_gripper_midpoint_maps_to_50():
    mid = (_GRIPPER_OPEN + _GRIPPER_CLOSE) / 2.0
    out = _convert({**_all_zero_joints(), "gripper": mid})
    assert out["gripper.pos"] == pytest.approx(50.0)


@pytest.mark.parametrize("rad", [_GRIPPER_OPEN - 1.0, _GRIPPER_CLOSE + 1.0])
def test_gripper_clipped_to_range(rad):
    # Out-of-range leader angles clip to [0, 100] rather than over/undershoot.
    out = _convert({**_all_zero_joints(), "gripper": rad})
    assert 0.0 <= out["gripper.pos"] <= 100.0


def test_gripper_polarity_can_be_flipped_by_swapping_endpoints():
    # Swapping open/close inverts the mapping (the documented fix for a backwards jaw).
    rad = _GRIPPER_OPEN  # would be 100 with normal endpoints
    out = _convert(
        {**_all_zero_joints(), "gripper": rad}, gripper_open=_GRIPPER_CLOSE, gripper_close=_GRIPPER_OPEN
    )
    assert out["gripper.pos"] == pytest.approx(0.0)


def test_degenerate_gripper_range_does_not_divide_by_zero():
    # open == close: span is 0; the mapping must not raise (defaults to fully open).
    out = _convert({**_all_zero_joints(), "gripper": 0.5}, gripper_open=0.3, gripper_close=0.3)
    assert out["gripper.pos"] == pytest.approx(100.0)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


def test_config_defaults_register_and_round_trip():
    cfg = SO101LeaderArmConfig()
    assert cfg.type == "so101_leader"
    assert cfg.collection_id == "so101_leader"
    # Provisional defaults: open < close so the normalized mapping is monotonic.
    assert cfg.gripper_open_rad < cfg.gripper_close_rad

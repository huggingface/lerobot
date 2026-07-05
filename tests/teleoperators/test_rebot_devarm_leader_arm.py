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

"""Pure unit tests for the Isaac Teleop reBot DevArm leader-arm unit conversion.

These tests deliberately do NOT import ``isaacteleop`` --
:func:`rebot_leader_joints_to_robot_action` is the pure-math bridge from the leader's
streamed joint angles [rad] to the follower's native ``{joint}.pos`` degrees, and must be
testable without the XR runtime (mirroring ``test_so101_leader_arm.py``).
"""

import numpy as np
import pytest

from lerobot.teleoperators.isaac_teleop.config_isaac_teleop import RebotDevArmLeaderArmConfig
from lerobot.teleoperators.isaac_teleop.teleop_rebot_devarm_leader_arm import (
    REBOT_DEVARM_LEADER_JOINTS,
    rebot_leader_joints_to_robot_action,
)

# The default stream-name -> follower-motor-name map (rebot_b601_follower naming).
_NAME_MAP = RebotDevArmLeaderArmConfig().joint_name_map


def _convert(joints_rad, name_map=None):
    return rebot_leader_joints_to_robot_action(
        joints_rad, joint_name_map=_NAME_MAP if name_map is None else name_map
    )


def _all_zero_joints():
    return dict.fromkeys(REBOT_DEVARM_LEADER_JOINTS, 0.0)


# ----------------------------------------------------------------------------
# All joints (gripper included): rad -> deg, 1:1 mirror
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rad", "expected_deg"),
    [(0.0, 0.0), (np.pi, 180.0), (-np.pi / 2, -90.0), (np.pi / 4, 45.0)],
)
def test_joint_rad2deg(rad, expected_deg):
    out = _convert({**_all_zero_joints(), "joint1": float(rad)})
    assert out["shoulder_pan.pos"] == pytest.approx(expected_deg)


def test_all_joints_converted_to_degrees_including_gripper():
    # Leader and follower are the same hardware: EVERY DOF (gripper too) is a plain
    # rad2deg mirror, unlike the SO-101 device which normalizes its gripper.
    joints = dict.fromkeys(REBOT_DEVARM_LEADER_JOINTS, np.pi)
    out = _convert(joints)
    assert len(out) == len(REBOT_DEVARM_LEADER_JOINTS)
    for value in out.values():
        assert value == pytest.approx(180.0)


def test_emits_follower_motor_names_in_stream_order():
    out = _convert(_all_zero_joints())
    assert list(out.keys()) == [f"{_NAME_MAP[name]}.pos" for name in REBOT_DEVARM_LEADER_JOINTS]


def test_default_map_targets_rebot_b601_follower_motors():
    # The default action layout must be follower-ready for rebot_b601_follower.
    out = _convert(_all_zero_joints())
    assert set(out.keys()) == {
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_yaw.pos",
        "wrist_roll.pos",
        "gripper.pos",
    }


def test_gripper_multi_turn_angle_passes_through_unclipped():
    # The B601 gripper is multi-turn (several rad of travel); the leader mirrors the raw
    # angle and leaves clipping to the follower's soft joint_limits.
    out = _convert({**_all_zero_joints(), "gripper": -6.8})
    assert out["gripper.pos"] == pytest.approx(float(np.rad2deg(-6.8)))


# ----------------------------------------------------------------------------
# joint_name_map behavior
# ----------------------------------------------------------------------------


def test_unmapped_stream_joints_are_dropped():
    # A stream joint absent from the map is dropped, not guessed at.
    partial_map = {"joint1": "shoulder_pan"}
    out = _convert(_all_zero_joints(), name_map=partial_map)
    assert list(out.keys()) == ["shoulder_pan.pos"]


def test_custom_name_map_renames_targets():
    custom = {name: f"motor_{i}" for i, name in enumerate(REBOT_DEVARM_LEADER_JOINTS)}
    out = _convert(_all_zero_joints(), name_map=custom)
    assert list(out.keys()) == [f"motor_{i}.pos" for i in range(len(REBOT_DEVARM_LEADER_JOINTS))]


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


def test_config_defaults_register_and_round_trip():
    cfg = RebotDevArmLeaderArmConfig()
    assert cfg.type == "rebot_devarm_leader"
    assert cfg.collection_id == "rebot_devarm_leader"
    assert cfg.device == ""
    # The default map covers the full stream, one target per stream joint (no fan-in).
    assert set(cfg.joint_name_map.keys()) == set(REBOT_DEVARM_LEADER_JOINTS)
    assert len(set(cfg.joint_name_map.values())) == len(REBOT_DEVARM_LEADER_JOINTS)


def test_config_joint_name_map_is_not_shared_between_instances():
    # default_factory must give each config its own dict (no mutable default aliasing).
    a = RebotDevArmLeaderArmConfig()
    b = RebotDevArmLeaderArmConfig()
    a.joint_name_map["joint1"] = "mutated"
    assert b.joint_name_map["joint1"] == "shoulder_pan"

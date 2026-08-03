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

from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.roarm_m3_follower.roarm_m3_common import (
    ANGLES_MAX as F_ANGLES_MAX,
    ANGLES_MIN as F_ANGLES_MIN,
    JOINT_NAMES as F_JOINT_NAMES,
)
from lerobot.teleoperators.roarm_m3_leader import RoarmM3Leader, RoarmM3LeaderConfig
from lerobot.teleoperators.roarm_m3_leader.roarm_m3_leader import (
    ANGLES_MAX as L_ANGLES_MAX,
    ANGLES_MIN as L_ANGLES_MIN,
    JOINT_NAMES as L_JOINT_NAMES,
)

_MODULE = "lerobot.teleoperators.roarm_m3_leader.roarm_m3_leader"


def _make_arm_mock(positions) -> MagicMock:
    arm = MagicMock(name="RoarmMock")
    arm.joints_angle_get.return_value = list(positions)
    return arm


def _make_leader(arm, **cfg_kwargs):
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.roarm", return_value=arm),
    ):
        leader = RoarmM3Leader(RoarmM3LeaderConfig(id="test", port="/dev/null", **cfg_kwargs))
        leader.connect(calibrate=False)
    return leader


def test_constants_in_sync_with_follower():
    """The leader keeps its own copy of the shared schema; it must match the follower."""
    assert L_JOINT_NAMES == F_JOINT_NAMES
    assert L_ANGLES_MIN == F_ANGLES_MIN
    assert L_ANGLES_MAX == F_ANGLES_MAX


def test_features_and_feedback():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        leader = RoarmM3Leader(RoarmM3LeaderConfig(id="test", port="/dev/null"))
    assert set(leader.action_features) == {f"{n}.pos" for n in L_JOINT_NAMES}
    assert leader.feedback_features == {}
    with pytest.raises(NotImplementedError):
        leader.send_feedback({})


def test_get_action_gripper_passthrough_by_default():
    """Default: gripper passes through (stock/identical grippers). 50 -> 50, floored."""
    arm = _make_arm_mock([3.4, -98.2, 182.7, -2.1, 4.9, 50.0])
    leader = _make_leader(arm)  # gripper_remap defaults to False
    try:
        action = leader.get_action()
        assert action["base.pos"] == 3.0  # floor quantized, EMA first step == raw
        assert action["shoulder.pos"] == -98.0
        assert action["gripper.pos"] == 50.0  # pass-through
    finally:
        leader.disconnect()


def test_get_action_gripper_remap_opt_in():
    """gripper_remap=True maps the leader range to the follower's Gripper B range."""
    arm = _make_arm_mock([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
    leader = _make_leader(arm, gripper_remap=True)
    try:
        action = leader.get_action()
        # ratio = (50-100)/(0-100) = 0.5 ; 0.5^2 = 0.25 ; 115 + 0.25*(73-115) = 104.5 -> 104
        assert action["gripper.pos"] == 104.0
    finally:
        leader.disconnect()

    arm_open = _make_arm_mock([0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    leader_open = _make_leader(arm_open, gripper_remap=True)
    try:
        assert leader_open.get_action()["gripper.pos"] == 115.0  # leader open -> follower open
    finally:
        leader_open.disconnect()

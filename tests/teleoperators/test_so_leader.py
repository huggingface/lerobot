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

from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig, SOLeaderConfig

# Present_Position as the bus returns it, in calibrated units (degrees for the body, 0-100 for the gripper).
POSITIONS = {
    "shoulder_pan": 30.0,
    "shoulder_lift": -100.0,
    "elbow_flex": 95.0,
    "wrist_flex": 60.0,
    "wrist_roll": -90.0,
    "gripper": 20.0,
}
MIRRORED = POSITIONS | {"shoulder_pan": -30.0, "wrist_roll": 90.0}


def _make_bus_mock(*_args, **kwargs) -> MagicMock:
    """Return a bus mock with just the attributes used by the leader."""
    bus = MagicMock(name="FeetechBusMock")
    bus.motors = kwargs["motors"]
    bus.is_connected = False
    bus.is_calibrated = True

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    bus.sync_read.side_effect = lambda *_a, **_kw: dict(POSITIONS)
    return bus


@pytest.fixture
def make_leader(tmp_path):
    leaders = []

    def _make(**kwargs):
        cfg = SO101LeaderConfig(port="/dev/null", calibration_dir=tmp_path, **kwargs)
        leader = SO101Leader(cfg)
        leader.connect()
        leaders.append(leader)
        return leader

    with (
        patch("lerobot.teleoperators.so_leader.so_leader.FeetechMotorsBus", side_effect=_make_bus_mock),
        patch.object(SO101Leader, "configure", lambda self: None),
    ):
        yield _make
        for leader in leaders:
            if leader.is_connected:
                leader.disconnect()


def test_get_action_not_mirrored_by_default(make_leader):
    leader = make_leader()
    assert leader.get_action() == {f"{m}.pos": v for m, v in POSITIONS.items()}


def test_get_action_mirrored(make_leader):
    leader = make_leader(mirror=True)
    assert leader.get_action() == {f"{m}.pos": v for m, v in MIRRORED.items()}


def test_send_feedback_mirrored(make_leader):
    # Feedback comes in the follower's frame; the mirror maps it back to the leader's own joints.
    leader = make_leader(mirror=True)
    leader.send_feedback({f"{m}.pos": v for m, v in MIRRORED.items()})
    leader.bus.sync_write.assert_called_once_with("Goal_Position", POSITIONS)


def test_bimanual_leader_passes_mirror_to_each_arm(tmp_path):
    cfg = BiSOLeaderConfig(
        left_arm_config=SOLeaderConfig(port="/dev/null", mirror=True),
        right_arm_config=SOLeaderConfig(port="/dev/null"),
        calibration_dir=tmp_path,
    )
    with patch("lerobot.teleoperators.so_leader.so_leader.FeetechMotorsBus", side_effect=_make_bus_mock):
        leader = BiSOLeader(cfg)
    assert leader.left_arm.config.mirror
    assert not leader.right_arm.config.mirror

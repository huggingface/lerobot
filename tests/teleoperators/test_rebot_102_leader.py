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

from lerobot.teleoperators.bi_rebot_102_leader import BiRebot102Leader, BiRebot102LeaderConfig
from lerobot.teleoperators.rebot_102_leader import (
    RebotArm102Leader,
    RebotArm102LeaderConfig,
    RebotArm102LeaderTeleopConfig,
)

_MODULE = "lerobot.teleoperators.rebot_102_leader.rebot_102_leader"


def _make_bus_mock(joint_ids: dict[str, int]) -> MagicMock:
    bus = MagicMock(name="FashionStarServoMock")
    bus.ping.return_value = True

    def _sync_monitor(ids):
        # Report each servo at 5 degrees raw.
        monitors = {}
        for servo_id in ids:
            monitor = MagicMock()
            monitor.angle_deg = 5.0
            monitors[servo_id] = monitor
        return monitors

    bus.sync_monitor.side_effect = _sync_monitor
    return bus


@pytest.fixture
def leader():
    cfg = RebotArm102LeaderTeleopConfig(port="/dev/null")
    bus_mock = _make_bus_mock(cfg.joint_ids)
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.FashionStarServo", return_value=bus_mock),
    ):
        teleop = RebotArm102Leader(cfg)
        teleop.connect(calibrate=False)
        yield teleop
        if teleop.is_connected:
            teleop.disconnect()


def test_action_features_match_joints():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        teleop = RebotArm102Leader(RebotArm102LeaderTeleopConfig(port="/dev/null"))
    assert set(teleop.action_features) == {f"{m}.pos" for m in teleop.motor_names}
    assert teleop.feedback_features == {}


def test_connect_disconnect(leader):
    assert leader.is_connected
    leader.disconnect()
    assert not leader.is_connected


def test_get_action_applies_direction_and_clamp(leader):
    action = leader.get_action()
    assert set(action) == {f"{m}.pos" for m in leader.motor_names}
    # shoulder_pan has direction -1, so a +5deg raw reading flips to -5deg.
    assert action["shoulder_pan.pos"] == pytest.approx(-5.0)
    # Every joint stays within its configured range.
    for motor, value in action.items():
        lo, hi = leader.config.joint_ranges[motor.removesuffix(".pos")]
        assert lo <= value <= hi


def test_send_feedback_not_implemented(leader):
    with pytest.raises(NotImplementedError):
        leader.send_feedback({})


def _make_leader(raw_positions_deg: dict[str, float], **overrides) -> RebotArm102Leader:
    cfg = RebotArm102LeaderTeleopConfig(port="/dev/null", **overrides)
    bus = _make_bus_mock(cfg.joint_ids)
    id_to_raw = {cfg.joint_ids[name]: deg for name, deg in raw_positions_deg.items()}

    def _sync_monitor(ids):
        monitors = {}
        for servo_id in ids:
            monitor = MagicMock()
            monitor.angle_deg = id_to_raw.get(servo_id, 0.0)
            monitors[servo_id] = monitor
        return monitors

    bus.sync_monitor.side_effect = _sync_monitor
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.FashionStarServo", return_value=bus),
    ):
        teleop = RebotArm102Leader(cfg)
        teleop.connect(calibrate=False)
    return teleop


def _elbow_overrides(direction: float) -> dict:
    cfg = RebotArm102LeaderTeleopConfig(port="/dev/null")
    return {
        "joint_directions": {**cfg.joint_directions, "elbow_flex": direction},
        "joint_ranges": {**cfg.joint_ranges, "elbow_flex": [0, 180]},
    }


def test_fractional_direction_tracks_full_leader_travel():
    """The unwrap window is centred on the raw angle that maps to the middle of the range, so the extremes of
    a scaled joint saturate at the range limit instead of unwrapping a full turn onto the opposite limit."""
    for raw, expected in [(0.0, 0.0), (-135.0, 90.0), (-269.9, 180.0), (-271.0, 180.0), (-280.0, 180.0)]:
        leader = _make_leader({"elbow_flex": raw}, **_elbow_overrides(-2 / 3))
        assert leader.get_action()["elbow_flex.pos"] == pytest.approx(expected, abs=0.5)


def test_fractional_direction_removes_whole_turns():
    for extra_turns in (-1, 1, 2):
        leader = _make_leader({"elbow_flex": -135.0 + extra_turns * 360.0}, **_elbow_overrides(-2 / 3))
        assert leader.get_action()["elbow_flex.pos"] == pytest.approx(90.0, abs=0.5)


def test_default_gripper_mapping_is_unchanged():
    """The default -6 gripper maps its raw travel, and whole turns of it, onto [-270, 0] as before."""
    for raw in (0.0, 15.0, 30.0, 45.0):
        for extra_turns in (-1, 0, 1):
            leader = _make_leader({"gripper": raw + extra_turns * 360.0})
            assert leader.get_action()["gripper.pos"] == pytest.approx(-6.0 * raw)


def test_zero_direction_is_rejected():
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        pytest.raises(ValueError, match="nonzero"),
    ):
        RebotArm102Leader(RebotArm102LeaderTeleopConfig(port="/dev/null", **_elbow_overrides(0)))


def test_bimanual_prefixes_features():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        cfg = BiRebot102LeaderConfig(
            left_arm_config=RebotArm102LeaderConfig(port="/dev/null0"),
            right_arm_config=RebotArm102LeaderConfig(port="/dev/null1"),
        )
        teleop = BiRebot102Leader(cfg)
    assert any(k.startswith("left_") for k in teleop.action_features)
    assert any(k.startswith("right_") for k in teleop.action_features)
    assert "left_gripper.pos" in teleop.action_features
    assert "right_gripper.pos" in teleop.action_features

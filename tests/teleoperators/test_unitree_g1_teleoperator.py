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

"""Tests for Unitree G1 teleoperator. Meant to be run in an environment where the Unitree SDK is installed."""

from unittest.mock import MagicMock

import pytest

from lerobot.utils.import_utils import _unitree_sdk_available

if not _unitree_sdk_available:
    pytest.skip("Unitree SDK not available", allow_module_level=True)

from lerobot.robots.unitree_g1.g1_utils import REMOTE_AXES
from lerobot.teleoperators.unitree_g1.config_unitree_g1 import (
    ExoskeletonArmPortConfig,
    UnitreeG1TeleoperatorConfig,
)
from lerobot.teleoperators.unitree_g1.unitree_g1 import RemoteController, UnitreeG1Teleoperator

# ---------------------------------------------------------------------------
# Tests for RemoteController
# ---------------------------------------------------------------------------


def _make_joystick_mock():
    """Create a mock Joystick class matching the SDK interface."""
    joystick = MagicMock()
    # Axes are Axis objects with .data attribute
    joystick.lx = MagicMock(data=0.0, smooth=0.03, deadzone=0.01)
    joystick.ly = MagicMock(data=0.0, smooth=0.03, deadzone=0.01)
    joystick.rx = MagicMock(data=0.0, smooth=0.03, deadzone=0.01)
    joystick.ry = MagicMock(data=0.0, smooth=0.03, deadzone=0.01)
    # Buttons are Button objects with .data attribute
    for name in ["RB", "LB", "start", "back", "RT", "LT", "A", "B", "X", "Y", "up", "right", "down", "left"]:
        setattr(joystick, name, MagicMock(data=0))
    return joystick


@pytest.fixture
def remote_controller():
    """Create a RemoteController with a mocked Joystick."""
    mock_joystick = _make_joystick_mock()

    rc = RemoteController()
    rc._joystick = mock_joystick
    yield rc, mock_joystick


def test_remote_controller_init(remote_controller):
    rc, _ = remote_controller
    assert rc.lx == 0.0
    assert rc.ly == 0.0
    assert rc.rx == 0.0
    assert rc.ry == 0.0
    assert len(rc.button) == 16
    assert all(b == 0 for b in rc.button)


def test_sync_remote_action(remote_controller):
    rc, _ = remote_controller
    rc.lx = 0.5
    rc.ly = -0.3
    rc.rx = 0.1
    rc.ry = 0.0
    rc._sync_remote_action()

    assert rc.remote_action["remote.lx"] == 0.5
    assert rc.remote_action["remote.ly"] == -0.3
    assert rc.remote_action["remote.rx"] == 0.1
    assert rc.remote_action["remote.ry"] == 0.0


def test_set_from_wireless_calls_extract(remote_controller):
    rc, mock_joystick = remote_controller
    # Set up the mock to populate data after extract
    mock_joystick.lx.data = 0.5
    mock_joystick.ly.data = -0.3
    mock_joystick.rx.data = 0.1
    mock_joystick.ry.data = 0.0

    wireless_data = b"\x00" * 40
    rc.set_from_wireless(wireless_data)

    mock_joystick.extract.assert_called_once_with(wireless_data)
    assert rc.lx == 0.5
    assert rc.ly == -0.3


def test_set_from_wireless_short_data(remote_controller):
    rc, mock_joystick = remote_controller
    rc.set_from_wireless(b"\x00" * 10)  # Too short
    mock_joystick.extract.assert_not_called()


def test_set_from_wireless_buttons(remote_controller):
    rc, mock_joystick = remote_controller
    # Simulate RB pressed
    mock_joystick.RB.data = 1
    mock_joystick.lx.data = 0.0
    mock_joystick.ly.data = 0.0
    mock_joystick.rx.data = 0.0
    mock_joystick.ry.data = 0.0

    rc.set_from_wireless(b"\x00" * 40)
    assert rc.button[0] == 1  # RB maps to button[0]


def test_set_from_exo_left(remote_controller):
    rc, _ = remote_controller
    rc.use_left_exo_joystick = True
    rc.left_center_x = 2048
    rc.left_center_y = 2048

    raw16 = [0] * 16
    raw16[11] = 3048  # X axis: (3048 - 2048) / 2047.5 ≈ 0.488
    raw16[13] = 1048  # Y axis: (1048 - 2048) / 2047.5 ≈ -0.488
    raw16[12] = 0  # Button pressed (below ADC_HALF)

    rc.set_from_exo(raw16, "left")
    assert rc.lx == pytest.approx((3048 - 2048) / 2047.5, abs=1e-3)
    assert rc.ly == pytest.approx((1048 - 2048) / 2047.5, abs=1e-3)
    assert rc.button[4] == 1  # Left button maps to button[4]


def test_set_from_exo_clears_button(remote_controller):
    rc, _ = remote_controller
    rc.use_left_exo_joystick = True
    rc.button[4] = 1  # Pre-set

    raw16 = [0] * 16
    raw16[12] = 4000  # Button NOT pressed (above ADC_HALF)

    rc.set_from_exo(raw16, "left")
    assert rc.button[4] == 0  # Should be cleared


def test_set_from_exo_ignored_when_not_enabled(remote_controller):
    rc, _ = remote_controller
    rc.use_left_exo_joystick = False
    raw16 = [0] * 16
    raw16[11] = 3000

    rc.set_from_exo(raw16, "left")
    assert rc.lx == 0.0  # Unchanged


# ---------------------------------------------------------------------------
# Tests for UnitreeG1TeleoperatorConfig (no SDK needed)
# ---------------------------------------------------------------------------


class TestTeleoperatorConfig:
    def test_default_config(self):
        cfg = UnitreeG1TeleoperatorConfig()
        assert cfg.left_arm_config.port == ""
        assert cfg.right_arm_config.port == ""
        assert cfg.frozen_joints == ""

    def test_config_with_ports(self):
        cfg = UnitreeG1TeleoperatorConfig(
            left_arm_config=ExoskeletonArmPortConfig(port="/dev/ttyACM0"),
            right_arm_config=ExoskeletonArmPortConfig(port="/dev/ttyACM1"),
        )
        assert cfg.left_arm_config.port == "/dev/ttyACM0"
        assert cfg.right_arm_config.port == "/dev/ttyACM1"


# ---------------------------------------------------------------------------
# Tests for UnitreeG1Teleoperator
# ---------------------------------------------------------------------------


@pytest.fixture
def teleop_remote_only():
    """Create a UnitreeG1Teleoperator in remote-only mode (no exo arms)."""
    cfg = UnitreeG1TeleoperatorConfig()  # No ports = remote-only mode
    teleop = UnitreeG1Teleoperator(cfg)
    yield teleop


def test_remote_only_connect(teleop_remote_only):
    """Remote-only mode should connect immediately without serial ports."""
    teleop = teleop_remote_only
    teleop.connect()
    assert teleop.is_connected
    assert not teleop._arm_control_enabled


def test_remote_only_action_features(teleop_remote_only):
    teleop = teleop_remote_only
    features = teleop.action_features
    # Remote-only: just the 4 remote axes
    assert set(features.keys()) == set(REMOTE_AXES)


def test_feedback_features(teleop_remote_only):
    teleop = teleop_remote_only
    features = teleop.feedback_features
    assert "wireless_remote" in features
    assert features["wireless_remote"] is bytes


def test_remote_only_get_action(teleop_remote_only):
    teleop = teleop_remote_only
    teleop.connect()
    action = teleop.get_action()
    assert set(action.keys()) == set(REMOTE_AXES)
    assert all(isinstance(v, float) for v in action.values())


def test_send_feedback(teleop_remote_only):
    teleop = teleop_remote_only
    teleop.connect()
    # Should not raise
    teleop.send_feedback({"wireless_remote": b"\x00" * 40})


def test_send_feedback_missing_key(teleop_remote_only):
    teleop = teleop_remote_only
    teleop.connect()
    # Should not raise even with missing key
    teleop.send_feedback({"other_key": 42})


def test_asymmetric_exo_ports_raises():
    """Configuring only one exo port should raise ValueError."""
    cfg = UnitreeG1TeleoperatorConfig(
        left_arm_config=ExoskeletonArmPortConfig(port="/dev/ttyACM0"),
        # right_arm_config left empty
    )
    with pytest.raises(ValueError, match="set both left/right"):
        UnitreeG1Teleoperator(cfg)


# ---------------------------------------------------------------------------
# Tests for ExoskeletonArm (needs serial mock)
# ---------------------------------------------------------------------------


class TestExoskeletonArm:
    def test_parse_raw16_valid(self):
        from lerobot.teleoperators.unitree_g1.exo_serial import parse_raw16

        line = b"100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600\n"
        result = parse_raw16(line)
        assert result is not None
        assert len(result) == 16
        assert result[0] == 100
        assert result[15] == 1600

    def test_parse_raw16_too_short(self):
        from lerobot.teleoperators.unitree_g1.exo_serial import parse_raw16

        line = b"100 200 300\n"
        assert parse_raw16(line) is None

    def test_parse_raw16_garbage(self):
        from lerobot.teleoperators.unitree_g1.exo_serial import parse_raw16

        assert parse_raw16(b"not numbers at all\n") is None
        assert parse_raw16(b"\xff\xfe\xfd\n") is None
        assert parse_raw16(b"") is None

    def test_calibrate_requires_connection(self):
        from lerobot.teleoperators.unitree_g1.exo_serial import ExoskeletonArm

        arm = ExoskeletonArm(
            port="/dev/null",
            calibration_fpath=MagicMock(is_file=MagicMock(return_value=False)),
            side="left",
        )
        with pytest.raises(RuntimeError, match="not connected"):
            arm.calibrate()

    def test_is_connected_false_by_default(self):
        from lerobot.teleoperators.unitree_g1.exo_serial import ExoskeletonArm

        arm = ExoskeletonArm(
            port="/dev/null",
            calibration_fpath=MagicMock(is_file=MagicMock(return_value=False)),
            side="left",
        )
        assert not arm.is_connected
        assert not arm.is_calibrated

    def test_read_raw_when_disconnected(self):
        from lerobot.teleoperators.unitree_g1.exo_serial import ExoskeletonArm

        arm = ExoskeletonArm(
            port="/dev/null",
            calibration_fpath=MagicMock(is_file=MagicMock(return_value=False)),
            side="left",
        )
        assert arm.read_raw() is None

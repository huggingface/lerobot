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

import struct
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
from lerobot.teleoperators.unitree_g1.exo_serial import ExoJoystickState
from lerobot.teleoperators.unitree_g1.unitree_g1 import JoystickState, UnitreeG1Teleoperator, WirelessRemote


def _wireless_bytes(lx=0.0, ly=0.0, rx=0.0, ry=0.0, buttons=0) -> bytes:
    """Builds a 40-byte wireless_remote payload: button bits in bytes 2-3, stick floats after."""
    data = bytearray(40)
    data[2:4] = buttons.to_bytes(2, "little")
    data[4:8] = struct.pack("f", lx)
    data[8:12] = struct.pack("f", rx)
    data[12:16] = struct.pack("f", ry)
    data[20:24] = struct.pack("f", ly)
    return bytes(data)


# ---------------------------------------------------------------------------
# Tests for JoystickState
# ---------------------------------------------------------------------------


def test_joystick_state_defaults():
    remote = JoystickState()
    assert not remote.is_active
    assert len(remote.buttons) == 16
    assert remote.to_action() == dict.fromkeys(REMOTE_AXES, 0.0)


def test_joystick_state_to_action():
    remote = JoystickState(lx=0.5, ly=-0.3, rx=0.1, ry=0.0)
    assert remote.to_action() == {
        "remote.lx": 0.5,
        "remote.ly": -0.3,
        "remote.rx": 0.1,
        "remote.ry": 0.0,
    }


def test_joystick_state_is_active():
    assert not JoystickState(lx=5e-4).is_active  # below the noise threshold
    assert JoystickState(ry=-0.2).is_active
    buttons = [0] * 16
    buttons[8] = 1
    assert JoystickState(buttons=buttons).is_active


# ---------------------------------------------------------------------------
# Tests for WirelessRemote
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
def wireless_remote():
    """Create a WirelessRemote with a mocked Joystick."""
    mock_joystick = _make_joystick_mock()

    remote = WirelessRemote()
    remote._joystick = mock_joystick
    yield remote, mock_joystick


def test_parse_calls_extract(wireless_remote):
    remote, mock_joystick = wireless_remote
    mock_joystick.lx.data = 0.5
    mock_joystick.ly.data = -0.3
    mock_joystick.rx.data = 0.1
    mock_joystick.ry.data = 0.0

    wireless_data = b"\x00" * 40
    parsed = remote.parse(wireless_data)

    mock_joystick.extract.assert_called_once_with(wireless_data)
    assert parsed is not None
    assert parsed.lx == 0.5
    assert parsed.ly == -0.3


def test_parse_short_data(wireless_remote):
    remote, mock_joystick = wireless_remote
    assert remote.parse(b"\x00" * 10) is None  # Too short
    mock_joystick.extract.assert_not_called()


def test_parse_buttons(wireless_remote):
    remote, mock_joystick = wireless_remote
    mock_joystick.RB.data = 1
    mock_joystick.A.data = 1

    parsed = remote.parse(b"\x00" * 40)
    assert parsed is not None
    assert parsed.buttons[0] == 1  # RB maps to button[0]
    assert parsed.buttons[8] == 1  # A maps to button[8]
    assert sum(parsed.buttons) == 2


def test_parse_raw_bytes():
    """Decodes a real payload with the SDK parser, smoothing and deadzone disabled."""
    parsed = WirelessRemote().parse(_wireless_bytes(lx=0.25, ly=-0.5, rx=0.75, ry=-1.0, buttons=0b1 | 1 << 8))
    assert parsed is not None
    assert parsed.to_action() == {
        "remote.lx": 0.25,
        "remote.ly": -0.5,
        "remote.rx": 0.75,
        "remote.ry": -1.0,
    }
    assert parsed.buttons[0] == 1  # RB
    assert parsed.buttons[8] == 1  # A


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


def test_remote_only_action_follows_wireless(teleop_remote_only):
    teleop = teleop_remote_only
    teleop.connect()
    teleop.send_feedback({"wireless_remote": _wireless_bytes(lx=0.5, ly=-0.25)})
    assert teleop.get_action() == {"remote.lx": 0.5, "remote.ly": -0.25, "remote.rx": 0.0, "remote.ry": 0.0}


def test_send_feedback_ignores_short_payload(teleop_remote_only):
    teleop = teleop_remote_only
    teleop.connect()
    teleop.send_feedback({"wireless_remote": _wireless_bytes(lx=0.5)})
    teleop.send_feedback({"wireless_remote": b"\x00" * 10})
    assert teleop.get_action()["remote.lx"] == 0.5


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


@pytest.fixture
def teleop_with_exo(tmp_path):
    """A UnitreeG1Teleoperator with exo arms whose serial, IK and thumbsticks are mocked."""
    cfg = UnitreeG1TeleoperatorConfig(
        left_arm_config=ExoskeletonArmPortConfig(port="/dev/ttyACM0"),
        right_arm_config=ExoskeletonArmPortConfig(port="/dev/ttyACM1"),
        calibration_dir=tmp_path,
    )
    teleop = UnitreeG1Teleoperator(cfg)
    teleop.ik_helper = MagicMock(compute_g1_joints_from_exo=MagicMock(return_value={"left_shoulder.q": 0.1}))
    teleop.left_arm = MagicMock(read_raw=MagicMock(return_value=[0] * 16), angles_from_raw=MagicMock())
    teleop.right_arm = MagicMock(read_raw=MagicMock(return_value=[0] * 16), angles_from_raw=MagicMock())
    teleop.left_arm.joystick_from_raw.return_value = ExoJoystickState(x=0.0, y=0.0, pressed=False)
    teleop.right_arm.joystick_from_raw.return_value = ExoJoystickState(x=0.0, y=0.0, pressed=False)
    yield teleop


def test_exo_joystick_drives_when_wireless_idle(teleop_with_exo):
    teleop = teleop_with_exo
    teleop.left_arm.joystick_from_raw.return_value = ExoJoystickState(x=0.4, y=-0.2, pressed=False)
    teleop.send_feedback({"wireless_remote": _wireless_bytes()})

    action = teleop.get_action()
    assert action["left_shoulder.q"] == 0.1
    assert action["remote.lx"] == pytest.approx(0.4)
    assert action["remote.ly"] == pytest.approx(-0.2)


def test_wireless_overrides_exo_joystick(teleop_with_exo):
    teleop = teleop_with_exo
    teleop.left_arm.joystick_from_raw.return_value = ExoJoystickState(x=0.4, y=-0.2, pressed=False)
    teleop.send_feedback({"wireless_remote": _wireless_bytes(rx=0.6)})

    action = teleop.get_action()
    assert action["remote.lx"] == 0.0
    assert action["remote.rx"] == pytest.approx(0.6)


def test_exo_joystick_returns_to_rest_without_wireless_data(teleop_with_exo):
    """Without wireless_remote in the observation, a released exo stick must read zero again."""
    teleop = teleop_with_exo
    teleop.left_arm.joystick_from_raw.return_value = ExoJoystickState(x=0.4, y=0.0, pressed=False)
    assert teleop.get_action()["remote.lx"] == pytest.approx(0.4)

    teleop.left_arm.joystick_from_raw.return_value = ExoJoystickState(x=0.0, y=0.0, pressed=False)
    assert teleop.get_action()["remote.lx"] == 0.0


def test_missing_exo_sample_keeps_last_joystick_reading(teleop_with_exo):
    teleop = teleop_with_exo
    teleop.left_arm.joystick_from_raw.return_value = ExoJoystickState(x=0.4, y=0.0, pressed=False)
    teleop.get_action()

    teleop.left_arm.joystick_from_raw.return_value = None
    assert teleop.get_action()["remote.lx"] == pytest.approx(0.4)


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

    def _arm_reading(self, raw):
        from lerobot.teleoperators.unitree_g1.exo_serial import ExoskeletonArm

        arm = ExoskeletonArm(
            port="/dev/null",
            calibration_fpath=MagicMock(is_file=MagicMock(return_value=False)),
            side="left",
        )
        arm.read_raw = MagicMock(return_value=raw)
        return arm

    def test_detect_joystick(self):
        raw = [0] * 16
        raw[11], raw[12], raw[13] = 2000, 4000, 2100  # X, button (released = high), Y
        arm = self._arm_reading(raw)
        assert arm.detect_joystick()

        raw[11] = 3048  # X: (3048 - 2000) / 2047.5
        raw[13] = 1100  # Y: (1100 - 2100) / 2047.5
        raw[12] = 0  # Button pressed (below ADC_HALF)
        stick = arm.joystick_from_raw(raw)
        assert stick is not None
        assert stick.x == pytest.approx((3048 - 2000) / 2047.5, abs=1e-3)
        assert stick.y == pytest.approx((1100 - 2100) / 2047.5, abs=1e-3)
        assert stick.pressed

    def test_detect_joystick_absent(self):
        raw = [0] * 16
        raw[12] = 100  # No stick fitted: the button channel reads low
        arm = self._arm_reading(raw)
        assert not arm.detect_joystick()
        assert arm.joystick_from_raw(raw) is None

    def test_detect_joystick_no_data(self):
        arm = self._arm_reading(None)
        assert not arm.detect_joystick()
        assert arm.joystick_from_raw(None) is None

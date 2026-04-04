"""Tests for gamepad controller HID parsing and input handling.

These tests validate the Xbox controller GIP report parsing and the
gamepad axis-to-delta mapping without requiring actual hardware.
"""

import struct

import pytest

from lerobot.teleoperators.gamepad.gamepad_utils import (
    GamepadControllerHID,
    InputController,
)

# ---------------------------------------------------------------------------
# InputController base class
# ---------------------------------------------------------------------------


class TestInputController:
    def test_default_attributes(self):
        ctrl = InputController()
        assert ctrl.wrist_roll_command == 0.0
        assert ctrl.right_x == 0.0
        assert ctrl.open_gripper_command is False
        assert ctrl.close_gripper_command is False

    def test_gripper_command_stay(self):
        ctrl = InputController()
        assert ctrl.gripper_command() == "stay"

    def test_gripper_command_open(self):
        ctrl = InputController()
        ctrl.open_gripper_command = True
        assert ctrl.gripper_command() == "open"

    def test_gripper_command_close(self):
        ctrl = InputController()
        ctrl.close_gripper_command = True
        assert ctrl.gripper_command() == "close"


# ---------------------------------------------------------------------------
# Xbox GIP report helpers
# ---------------------------------------------------------------------------


def _build_xbox_report(
    *,
    lx: int = 0,
    ly: int = 0,
    rx: int = 0,
    ry: int = 0,
    lt: int = 0,
    rt: int = 0,
    buttons0: int = 0,
    buttons1: int = 0,
) -> list[int]:
    """Build a minimal 18-byte Xbox One GIP HID report.

    Args:
        lx, ly, rx, ry: Signed 16-bit stick values (-32768..32767).
        lt, rt: Unsigned 16-bit trigger values (0..1023).
        buttons0: Byte 4 bitmask (A=bit4, B=bit5, X=bit6, Y=bit7).
        buttons1: Byte 5 bitmask (LB=bit4, RB=bit5).
    """
    buf = bytearray(18)
    buf[0] = 0x20  # GIP packet type
    buf[1] = 0x00
    struct.pack_into("<H", buf, 2, 0)  # counter
    buf[4] = buttons0
    buf[5] = buttons1
    struct.pack_into("<H", buf, 6, lt)
    struct.pack_into("<H", buf, 8, rt)
    struct.pack_into("<h", buf, 10, lx)
    struct.pack_into("<h", buf, 12, ly)
    struct.pack_into("<h", buf, 14, rx)
    struct.pack_into("<h", buf, 16, ry)
    return list(buf)


# ---------------------------------------------------------------------------
# Xbox HID parsing tests
# ---------------------------------------------------------------------------


class TestXboxHIDParsing:
    """Test _update_xbox() by feeding raw byte data to the controller."""

    def _make_controller(self, deadzone: float = 0.1) -> GamepadControllerHID:
        ctrl = GamepadControllerHID(deadzone=deadzone)
        # Don't call start() — we'll feed data directly via _update_xbox()
        return ctrl

    def test_sticks_at_rest(self):
        ctrl = self._make_controller()
        report = _build_xbox_report()
        ctrl._update_xbox(report)

        assert ctrl.left_x == 0.0
        assert ctrl.left_y == 0.0
        assert ctrl.right_x == 0.0
        assert ctrl.right_y == 0.0

    def test_left_stick_full_right(self):
        ctrl = self._make_controller()
        report = _build_xbox_report(lx=32767)
        ctrl._update_xbox(report)

        assert ctrl.left_x == pytest.approx(32767 / 32768.0, abs=0.01)
        assert ctrl.left_y == 0.0

    def test_left_stick_full_up(self):
        """Xbox HID: stick up = positive ly. After negation, left_y should be negative."""
        ctrl = self._make_controller()
        report = _build_xbox_report(ly=32767)
        ctrl._update_xbox(report)

        # Y is negated to match SDL convention (up = negative)
        assert ctrl.left_y == pytest.approx(-32767 / 32768.0, abs=0.01)

    def test_right_stick_full_up(self):
        """right_y should also be negated (up = negative)."""
        ctrl = self._make_controller()
        report = _build_xbox_report(ry=32767)
        ctrl._update_xbox(report)

        assert ctrl.right_y == pytest.approx(-32767 / 32768.0, abs=0.01)

    def test_right_stick_x_not_negated(self):
        """right_x should NOT be negated."""
        ctrl = self._make_controller()
        report = _build_xbox_report(rx=32767)
        ctrl._update_xbox(report)

        assert ctrl.right_x == pytest.approx(32767 / 32768.0, abs=0.01)

    def test_deadzone_filters_small_values(self):
        ctrl = self._make_controller(deadzone=0.1)
        # ~5% deflection → should be zeroed by deadzone
        small_val = int(0.05 * 32768)
        report = _build_xbox_report(lx=small_val, ly=small_val)
        ctrl._update_xbox(report)

        assert ctrl.left_x == 0.0
        assert ctrl.left_y == 0.0

    def test_deadzone_passes_large_values(self):
        ctrl = self._make_controller(deadzone=0.1)
        # ~50% deflection → should pass through
        large_val = int(0.5 * 32768)
        report = _build_xbox_report(lx=large_val)
        ctrl._update_xbox(report)

        assert ctrl.left_x != 0.0

    def test_triggers_gripper(self):
        ctrl = self._make_controller()

        # LT pressed → close gripper
        report = _build_xbox_report(lt=500)
        ctrl._update_xbox(report)
        assert ctrl.close_gripper_command is True
        assert ctrl.open_gripper_command is False

        # RT pressed → open gripper
        report = _build_xbox_report(rt=500)
        ctrl._update_xbox(report)
        assert ctrl.open_gripper_command is True

    def test_triggers_below_threshold_no_gripper(self):
        ctrl = self._make_controller()
        report = _build_xbox_report(lt=50, rt=50)
        ctrl._update_xbox(report)

        assert ctrl.close_gripper_command is False
        assert ctrl.open_gripper_command is False

    def test_lb_wrist_roll_left(self):
        ctrl = self._make_controller()
        report = _build_xbox_report(buttons1=(1 << 4))  # LB
        ctrl._update_xbox(report)

        assert ctrl.wrist_roll_command == -1.0

    def test_rb_wrist_roll_right(self):
        ctrl = self._make_controller()
        report = _build_xbox_report(buttons1=(1 << 5))  # RB
        ctrl._update_xbox(report)

        assert ctrl.wrist_roll_command == 1.0

    def test_both_bumpers_no_roll(self):
        ctrl = self._make_controller()
        report = _build_xbox_report(buttons1=(1 << 4) | (1 << 5))  # LB+RB
        ctrl._update_xbox(report)

        assert ctrl.wrist_roll_command == 0.0

    def test_no_bumpers_no_roll(self):
        ctrl = self._make_controller()
        report = _build_xbox_report()
        ctrl._update_xbox(report)

        assert ctrl.wrist_roll_command == 0.0

    def test_y_button_success(self):
        from lerobot.teleoperators.utils import TeleopEvents

        ctrl = self._make_controller()
        report = _build_xbox_report(buttons0=(1 << 7))  # Y
        ctrl._update_xbox(report)

        assert ctrl.episode_end_status == TeleopEvents.SUCCESS

    def test_x_button_failure(self):
        from lerobot.teleoperators.utils import TeleopEvents

        ctrl = self._make_controller()
        report = _build_xbox_report(buttons0=(1 << 6))  # X
        ctrl._update_xbox(report)

        assert ctrl.episode_end_status == TeleopEvents.FAILURE

    def test_a_button_rerecord(self):
        from lerobot.teleoperators.utils import TeleopEvents

        ctrl = self._make_controller()
        report = _build_xbox_report(buttons0=(1 << 4))  # A
        ctrl._update_xbox(report)

        assert ctrl.episode_end_status == TeleopEvents.RERECORD_EPISODE

    def test_no_buttons_clears_status(self):
        ctrl = self._make_controller()
        report = _build_xbox_report()
        ctrl._update_xbox(report)

        assert ctrl.episode_end_status is None


# ---------------------------------------------------------------------------
# get_deltas() axis mapping tests
# ---------------------------------------------------------------------------


class TestGetDeltas:
    """Test the axis-to-delta mapping in get_deltas()."""

    def _make_controller(self) -> GamepadControllerHID:
        ctrl = GamepadControllerHID(x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.0)
        return ctrl

    def test_left_stick_up_gives_positive_delta_x(self):
        """Left stick up (negative left_y in SDL convention) → positive delta_x (forward)."""
        ctrl = self._make_controller()
        ctrl.left_y = -1.0  # SDL: up = negative
        dx, dy, dz = ctrl.get_deltas()
        assert dx == pytest.approx(1.0)

    def test_left_stick_right_gives_negative_delta_y(self):
        """Left stick right (positive left_x) → negative delta_y."""
        ctrl = self._make_controller()
        ctrl.left_x = 1.0
        dx, dy, dz = ctrl.get_deltas()
        assert dy == pytest.approx(-1.0)

    def test_right_stick_up_gives_positive_delta_z(self):
        """Right stick up (negative right_y in SDL convention) → positive delta_z (up)."""
        ctrl = self._make_controller()
        ctrl.right_y = -1.0
        dx, dy, dz = ctrl.get_deltas()
        assert dz == pytest.approx(1.0)

    def test_idle_gives_zero_deltas(self):
        ctrl = self._make_controller()
        dx, dy, dz = ctrl.get_deltas()
        assert dx == 0.0
        assert dy == 0.0
        assert dz == 0.0

    def test_step_size_scaling(self):
        ctrl = GamepadControllerHID(x_step_size=2.0, y_step_size=3.0, z_step_size=4.0, deadzone=0.0)
        ctrl.left_y = -1.0
        ctrl.left_x = 1.0
        ctrl.right_y = -1.0

        dx, dy, dz = ctrl.get_deltas()

        assert dx == pytest.approx(2.0)
        assert dy == pytest.approx(-3.0)
        assert dz == pytest.approx(4.0)

    def test_xbox_stick_up_through_full_pipeline(self):
        """Simulate Xbox stick up → _update_xbox → get_deltas → positive delta_x."""
        ctrl = GamepadControllerHID(x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.05)
        report = _build_xbox_report(ly=32767)  # Xbox HID: stick up = positive ly
        ctrl._update_xbox(report)

        dx, dy, dz = ctrl.get_deltas()

        # Should be positive (forward)
        assert dx > 0.9

    def test_xbox_right_stick_up_through_full_pipeline(self):
        """Simulate Xbox right stick up → _update_xbox → get_deltas → positive delta_z."""
        ctrl = GamepadControllerHID(x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.05)
        report = _build_xbox_report(ry=32767)  # Xbox HID: stick up = positive ry
        ctrl._update_xbox(report)

        dx, dy, dz = ctrl.get_deltas()

        # Should be positive (up)
        assert dz > 0.9

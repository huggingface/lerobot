#!/usr/bin/env python
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

"""Tests for the NexArm motor bus driver.

Cover frame building/parsing, the standard ``sync_read``/``sync_write`` API,
torque control, LeRobot bridge-mode entry/exit, and the calibration flow
(``set_half_turn_homings`` + ``record_ranges_of_motion``). All tests use mock
serial I/O — no physical hardware is required.
"""

import struct
from unittest.mock import MagicMock, patch

import pytest

serial = pytest.importorskip("serial", reason="pyserial is required for NexArm tests")

from lerobot.motors import MotorCalibration, MotorNormMode  # noqa: E402
from lerobot.motors.nexarm.nexarm import (  # noqa: E402
    CMD_LEROBOT_MODE,
    CMD_READ_POS,
    CMD_TORQUE,
    CMD_WRITE_POS,
    DEFAULT_BAUDRATE,
    HALF_TURN,
    JOINT_COUNT,
    JOINT_NAMES,
    POSITION_MAX,
    POSITION_MIN,
    SYSTEM_ID,
    NexArmMotorsBus,
    build_frame,
    parse_frame,
)

# ── Frame building / parsing ─────────────────────────────────────────


class TestBuildFrame:
    def test_header_bytes(self):
        frame = build_frame(0xFF, CMD_READ_POS)
        assert frame[:2] == b"\xff\xff"

    def test_length_no_args(self):
        frame = build_frame(0xFF, CMD_READ_POS)
        assert frame[3] == 2

    def test_length_with_args(self):
        frame = build_frame(0xFF, CMD_READ_POS, bytes([1, 2, 3]))
        assert frame[3] == 5

    def test_cmd_byte(self):
        assert build_frame(0xFF, CMD_READ_POS)[4] == CMD_READ_POS

    def test_checksum_correct(self):
        frame = build_frame(0xFF, CMD_READ_POS)
        expected = (~sum(frame[2:-1])) & 0xFF
        assert frame[-1] == expected

    def test_round_trip(self):
        payload = bytes([0x01, 0x02, 0x03])
        frame = build_frame(SYSTEM_ID, CMD_TORQUE, payload)
        result = parse_frame(frame)
        assert result == (SYSTEM_ID, CMD_TORQUE, payload)

    @pytest.mark.parametrize("cmd", [CMD_READ_POS, CMD_WRITE_POS, CMD_TORQUE, CMD_LEROBOT_MODE])
    def test_all_commands_round_trip(self, cmd):
        frame = build_frame(SYSTEM_ID, cmd)
        result = parse_frame(frame)
        assert result is not None
        assert result[1] == cmd


class TestParseFrame:
    def test_too_short(self):
        assert parse_frame(b"\xff\xff\x01") is None

    def test_bad_header(self):
        frame = build_frame(SYSTEM_ID, CMD_READ_POS)
        assert parse_frame(b"\x00\x00" + frame[2:]) is None

    def test_truncated_data(self):
        frame = build_frame(SYSTEM_ID, CMD_READ_POS, bytes(12))
        assert parse_frame(frame[:-3]) is None

    def test_bad_checksum(self):
        frame = bytearray(build_frame(SYSTEM_ID, CMD_READ_POS))
        frame[-1] ^= 0xFF
        assert parse_frame(bytes(frame)) is None

    def test_firmware_buggy_checksum_accepted(self):
        """Leader firmware uses rx_packet.elements.length instead of tx_packet's."""
        args = bytes(12)
        length = len(args) + 2
        data_raw = bytes([SYSTEM_ID, length, CMD_READ_POS]) + args
        correct = (~sum(data_raw)) & 0xFF
        buggy = (~sum(data_raw[:3])) & 0xFF
        assert parse_frame(b"\xff\xff" + data_raw + bytes([correct])) is not None
        if buggy != correct:
            assert parse_frame(b"\xff\xff" + data_raw + bytes([buggy])) is not None

    def test_parse_position_reply(self):
        positions = [2048, 1024, 3072, 512, 4000, 2000]
        args = b"".join(struct.pack("<h", p) for p in positions)
        result = parse_frame(build_frame(SYSTEM_ID, CMD_READ_POS, args))
        assert result is not None
        _, cmd, parsed_args = result
        assert cmd == CMD_READ_POS
        parsed = [struct.unpack_from("<h", parsed_args, i * 2)[0] for i in range(JOINT_COUNT)]
        assert parsed == positions


# ── Mock helpers ─────────────────────────────────────────────────────


def _make_mock_serial(reply_bytes: bytes | None = None) -> MagicMock:
    mock = MagicMock()
    mock.is_open = True
    mock.port = "/dev/null"
    if reply_bytes is None:
        mock.in_waiting = 0
        mock.read.return_value = b""
    else:
        # The bus polls in_waiting then reads exactly that many bytes.
        # Each write() re-arms the reply so successive request/response cycles
        # (handshake → sync_read → sync_read → …) all see the canned bytes.
        state = {"sent": False}

        def _in_waiting():
            return 0 if state["sent"] else len(reply_bytes)

        def _read(n):
            if state["sent"]:
                return b""
            state["sent"] = True
            return reply_bytes

        def _write(data):
            state["sent"] = False
            return len(data)

        # ``in_waiting`` is a property on real Serial — emulate as PropertyMock.
        type(mock).in_waiting = property(lambda self: _in_waiting())
        mock.read.side_effect = _read
        mock.write.side_effect = _write
    return mock


def _reply_for_positions(positions: list[int]) -> bytes:
    args = b"".join(struct.pack("<h", p) for p in positions)
    return build_frame(SYSTEM_ID, CMD_READ_POS, args)


# ── Bus lifecycle ────────────────────────────────────────────────────


class TestNexArmMotorsBusLifecycle:
    def test_default_motors(self):
        bus = NexArmMotorsBus(port="/dev/null")
        assert set(bus.motors) == set(JOINT_NAMES)
        assert bus.motors["gripper"].norm_mode == MotorNormMode.RANGE_0_100
        assert bus.motors["shoulder_pan"].norm_mode == MotorNormMode.RANGE_M100_100

    def test_init_defaults(self):
        bus = NexArmMotorsBus(port="/dev/null")
        assert bus.port == "/dev/null"
        assert bus.baudrate == DEFAULT_BAUDRATE
        assert not bus.is_connected
        assert not bus.is_calibrated

    def test_connect_disconnect(self):
        positions = [2048] * JOINT_COUNT
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as serial_cls:
            mock_ser = _make_mock_serial(_reply_for_positions(positions))
            serial_cls.return_value = mock_ser
            bus = NexArmMotorsBus(port="/dev/null")
            bus.connect()
            assert bus.is_connected
            bus.disconnect(disable_torque=False)
            assert not bus.is_connected
            mock_ser.close.assert_called_once()

    def test_connect_idempotent(self):
        positions = [2048] * JOINT_COUNT
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as serial_cls:
            serial_cls.return_value = _make_mock_serial(_reply_for_positions(positions))
            bus = NexArmMotorsBus(port="/dev/null")
            bus.connect()
            bus.connect()
            assert serial_cls.call_count == 1

    def test_handshake_failure_raises(self):
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as serial_cls:
            serial_cls.return_value = _make_mock_serial(None)
            bus = NexArmMotorsBus(port="/dev/null")
            with pytest.raises(ConnectionError, match="handshake failed"):
                bus.connect()


# ── sync_read / sync_write ──────────────────────────────────────────


class TestSyncRead:
    def test_returns_raw_when_no_calibration(self):
        positions = [2048, 1024, 3072, 512, 4000, 2000]
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as serial_cls:
            serial_cls.return_value = _make_mock_serial(_reply_for_positions(positions))
            bus = NexArmMotorsBus(port="/dev/null")
            bus.connect()
            result = bus.sync_read("Present_Position")
            for i, name in enumerate(JOINT_NAMES):
                assert result[name] == float(positions[i])

    def test_normalizes_when_calibrated(self):
        positions = [2048] * JOINT_COUNT
        calibration = {
            name: MotorCalibration(
                id=i + 1, drive_mode=0, homing_offset=0, range_min=0, range_max=4095
            )
            for i, name in enumerate(JOINT_NAMES)
        }
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as serial_cls:
            serial_cls.return_value = _make_mock_serial(_reply_for_positions(positions))
            bus = NexArmMotorsBus(port="/dev/null", calibration=calibration)
            bus.connect()
            result = bus.sync_read("Present_Position")
            # 2048 is the midpoint -> ~0 in RANGE_M100_100, ~50 in RANGE_0_100.
            assert abs(result["shoulder_pan"]) < 0.05
            assert abs(result["gripper"] - 50.0) < 0.05

    def test_rejects_unknown_data_name(self):
        bus = NexArmMotorsBus(port="/dev/null")
        bus._serial = MagicMock()
        bus._serial.is_open = True
        with pytest.raises(NotImplementedError):
            bus.sync_read("Present_Voltage")


class TestSyncWrite:
    def _make_bus(self, mock_ser):
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as serial_cls:
            serial_cls.return_value = mock_ser
            bus = NexArmMotorsBus(port="/dev/null")
            bus.connect()
            mock_ser.write.reset_mock()
            return bus

    def test_writes_cmd97_with_six_int16(self):
        # _reply_for_positions handshake then ignore subsequent reads
        mock_ser = _make_mock_serial(_reply_for_positions([2048] * 6))
        bus = self._make_bus(mock_ser)

        calibration = {
            name: MotorCalibration(
                id=i + 1, drive_mode=0, homing_offset=0, range_min=0, range_max=4095
            )
            for i, name in enumerate(JOINT_NAMES)
        }
        bus.calibration = calibration

        values = dict.fromkeys(JOINT_NAMES, 0.0)
        values["gripper"] = 50.0
        bus.sync_write("Goal_Position", values)

        written = mock_ser.write.call_args[0][0]
        result = parse_frame(written)
        assert result is not None
        _, cmd, args = result
        assert cmd == CMD_WRITE_POS
        assert len(args) == JOINT_COUNT * 2
        parsed = [struct.unpack_from("<h", args, i * 2)[0] for i in range(JOINT_COUNT)]
        # 0% in RANGE_M100_100 -> midpoint 2047; 50% in RANGE_0_100 -> midpoint 2047
        for v in parsed:
            assert 2046 <= v <= 2049

    def test_clamps_to_range_min_max(self):
        mock_ser = _make_mock_serial(_reply_for_positions([2048] * 6))
        bus = self._make_bus(mock_ser)

        # Tighten elbow_flex range to [968, 3200].
        bus.calibration = {
            name: MotorCalibration(
                id=i + 1,
                drive_mode=0,
                homing_offset=0,
                range_min=968 if name == "elbow_flex" else 0,
                range_max=3200 if name == "elbow_flex" else 4095,
            )
            for i, name in enumerate(JOINT_NAMES)
        }

        # Ask for elbow_flex at +200 (RANGE_M100_100 -> outside max) → clamp to 3200.
        values = dict.fromkeys(JOINT_NAMES, 0.0)
        values["elbow_flex"] = 200.0
        bus.sync_write("Goal_Position", values)

        written = mock_ser.write.call_args[0][0]
        _, _, args = parse_frame(written)
        parsed = [struct.unpack_from("<h", args, i * 2)[0] for i in range(JOINT_COUNT)]
        assert parsed[2] == 3200

    def test_rejects_unknown_data_name(self):
        bus = NexArmMotorsBus(port="/dev/null")
        bus._serial = MagicMock()
        bus._serial.is_open = True
        with pytest.raises(NotImplementedError):
            bus.sync_write("Goal_Velocity", {"shoulder_pan": 0.0})


# ── Torque / LeRobot bridge mode ────────────────────────────────────


class TestTorqueAndBridgeMode:
    def _bus_with_mock(self, mock_ser):
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as serial_cls:
            serial_cls.return_value = mock_ser
            bus = NexArmMotorsBus(port="/dev/null")
            bus.connect()
            mock_ser.write.reset_mock()
            return bus

    def test_enable_torque_sends_cmd98_one(self):
        mock_ser = _make_mock_serial(_reply_for_positions([2048] * 6))
        bus = self._bus_with_mock(mock_ser)
        bus.enable_torque()
        written = mock_ser.write.call_args[0][0]
        _, cmd, args = parse_frame(written)
        assert cmd == CMD_TORQUE
        assert args == bytes([1])

    def test_disable_torque_sends_cmd98_zero(self):
        mock_ser = _make_mock_serial(_reply_for_positions([2048] * 6))
        bus = self._bus_with_mock(mock_ser)
        bus.disable_torque()
        written = mock_ser.write.call_args[0][0]
        _, cmd, args = parse_frame(written)
        assert cmd == CMD_TORQUE
        assert args == bytes([0])

    def test_enter_lerobot_mode(self):
        mock_ser = _make_mock_serial(_reply_for_positions([2048] * 6))
        bus = self._bus_with_mock(mock_ser)
        bus.enter_lerobot_mode()
        written = mock_ser.write.call_args[0][0]
        _, cmd, args = parse_frame(written)
        assert cmd == CMD_LEROBOT_MODE
        assert args == bytes([1])

    def test_exit_lerobot_mode(self):
        mock_ser = _make_mock_serial(_reply_for_positions([2048] * 6))
        bus = self._bus_with_mock(mock_ser)
        bus.exit_lerobot_mode()
        written = mock_ser.write.call_args[0][0]
        _, cmd, args = parse_frame(written)
        assert cmd == CMD_LEROBOT_MODE
        assert args == bytes([0])


# ── Calibration helpers ─────────────────────────────────────────────


class TestCalibrationHelpers:
    def test_set_half_turn_homings_stores_offset(self):
        positions = [2200, 1900, 2050, 2100, 2000, 1800]
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as serial_cls:
            serial_cls.return_value = _make_mock_serial(_reply_for_positions(positions))
            bus = NexArmMotorsBus(port="/dev/null")
            bus.connect()
            offsets = bus.set_half_turn_homings()
        for i, name in enumerate(JOINT_NAMES):
            assert offsets[name] == positions[i] - HALF_TURN
            assert bus.calibration[name].homing_offset == positions[i] - HALF_TURN
            assert bus.calibration[name].range_min == POSITION_MIN
            assert bus.calibration[name].range_max == POSITION_MAX

    def test_is_calibrated_requires_distinct_range(self):
        bus = NexArmMotorsBus(port="/dev/null")
        # range_min == range_max should be rejected.
        bus.calibration = {
            name: MotorCalibration(id=i + 1, drive_mode=0, homing_offset=0, range_min=0, range_max=0)
            for i, name in enumerate(JOINT_NAMES)
        }
        assert not bus.is_calibrated
        bus.calibration["shoulder_pan"] = MotorCalibration(
            id=1, drive_mode=0, homing_offset=0, range_min=100, range_max=3900
        )
        # Still false because other motors have invalid ranges.
        assert not bus.is_calibrated


# ── Constants ───────────────────────────────────────────────────────


class TestConstants:
    def test_joint_count_matches_names(self):
        assert len(JOINT_NAMES) == JOINT_COUNT

    def test_position_range(self):
        assert POSITION_MIN == 0
        assert POSITION_MAX == 4095

    def test_joint_names(self):
        assert JOINT_NAMES == (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        )

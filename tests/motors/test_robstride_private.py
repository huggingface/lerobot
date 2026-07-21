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

"""Unit tests for RobstridePrivateMotorsBus against a mocked python-can bus (no hardware)."""

import logging
import math
import struct
import time
from collections import deque
from collections.abc import Callable, Iterable
from unittest.mock import patch

import pytest

from lerobot.utils.import_utils import _can_available

if not _can_available:
    pytest.skip("python-can not available", allow_module_level=True)

import can

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.robstride import RobstridePrivateMotorsBus
from lerobot.motors.robstride.tables import (
    DEFAULT_PRIVATE_HOST_ID,
    PRIVATE_MODE_SWITCH_RETRIES,
    PRIVATE_PARAMS,
    RS_MODEL_LIMITS,
    PrivateCommType,
    PrivateControlMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

HOST_ID = DEFAULT_PRIVATE_HOST_ID
LOGGER = "lerobot.motors.robstride.robstride_private"

RS00 = RS_MODEL_LIMITS["rs00"]
RS06 = RS_MODEL_LIMITS["rs06"]

RUN_MODE_IDX = PRIVATE_PARAMS["run_mode"].index
MECH_POS_IDX = PRIVATE_PARAMS["mech_pos"].index
MECH_VEL_IDX = PRIVATE_PARAMS["mech_vel"].index
LOC_REF_IDX = PRIVATE_PARAMS["loc_ref"].index
SPD_REF_IDX = PRIVATE_PARAMS["spd_ref"].index
IQ_REF_IDX = PRIVATE_PARAMS["iq_ref"].index
LIMIT_SPD_IDX = PRIVATE_PARAMS["limit_spd"].index
ZERO_STA_IDX = PRIVATE_PARAMS["zero_sta"].index
VBUS_IDX = PRIVATE_PARAMS["vbus"].index
_FMT_BY_INDEX = {entry.index: entry.fmt for entry in PRIVATE_PARAMS.values()}


def make_motors() -> dict[str, Motor]:
    return {
        "shoulder": Motor(id=1, model="robstride", norm_mode=MotorNormMode.DEGREES, motor_type_str="rs06"),
        "elbow": Motor(id=2, model="robstride", norm_mode=MotorNormMode.DEGREES, motor_type_str="RS-06"),
        "wrist": Motor(id=3, model="robstride", norm_mode=MotorNormMode.DEGREES, motor_type_str="rs00"),
    }


def ext_id(comm_type: int, data16: int, target_id: int) -> int:
    """Reference (spec) packing of a private-protocol 29-bit extended CAN ID."""
    return ((comm_type & 0x1F) << 24) | ((data16 & 0xFFFF) << 8) | (target_id & 0xFF)


def u16_phys(value: float, half_range: float) -> int:
    """Reference (spec) encoding of a physical value into the symmetric u16 window."""
    clamped = min(max(value, -half_range), half_range)
    return int(round((clamped / half_range + 1.0) * 32767.0))


def param_reply(
    motor_id: int, index: int, value: float | int, host_id: int = HOST_ID, error_flag: int = 0
) -> can.Message:
    """Motor -> host type-0x11 reply: index echoed LE in bytes 0-1, value LE in bytes 4-7.

    ``error_flag`` sets data16's high byte, which real firmware uses to answer a read of an
    unsupported index (with a zeroed value payload; pass ``value=0`` to mimic it).
    """
    raw = struct.pack("<" + _FMT_BY_INDEX.get(index, "f"), value)
    data = struct.pack("<H", index) + b"\x00\x00" + raw + bytes(4 - len(raw))
    return can.Message(
        arbitration_id=ext_id(
            PrivateCommType.PARAM_READ, ((error_flag & 0xFF) << 8) | (motor_id & 0xFF), host_id
        ),
        data=data,
        is_extended_id=True,
    )


def feedback_frame(
    motor_id: int,
    pos_u16: int = 32767,
    vel_u16: int = 32767,
    torque_u16: int = 32767,
    temp_u16: int = 250,
    fault: int = 0,
    status: int = 2,
    host_id: int = HOST_ID,
) -> can.Message:
    """Motor -> host type-0x02 feedback: 4 BE u16 payload, fault/status bits in the ext ID."""
    data16 = ((status & 0x3) << 14) | ((fault & 0x3F) << 8) | (motor_id & 0xFF)
    data = struct.pack(">HHHH", pos_u16, vel_u16, torque_u16, temp_u16)
    return can.Message(
        arbitration_id=ext_id(PrivateCommType.FEEDBACK, data16, host_id),
        data=data,
        is_extended_id=True,
    )


def report_frame(motor_id: int, host_id: int = HOST_ID) -> can.Message:
    """Type-0x18 compact report frame streamed by RS firmware by default.

    The payload deliberately mimics a valid ``mech_pos`` reply so a decoder that ignores the
    comm type would mistake it for a parameter read result.
    """
    data = struct.pack("<H", MECH_POS_IDX) + b"\x00\x00" + struct.pack("<f", 999.0)
    return can.Message(
        arbitration_id=ext_id(PrivateCommType.ACTIVE_REPORT, motor_id, host_id),
        data=data,
        is_extended_id=True,
    )


class FakeCanBus:
    """Duck-typed ``can.BusABC``: records sent frames, serves scripted or responder replies."""

    def __init__(self) -> None:
        self.sent: list[can.Message] = []
        self.rx: deque[can.Message] = deque()
        self.responder: Callable[[can.Message], Iterable[can.Message]] | None = None
        self.shut_down = False

    def send(self, msg: can.Message) -> None:
        self.sent.append(msg)
        if self.responder is not None:
            self.rx.extend(self.responder(msg))

    def recv(self, timeout: float | None = None) -> can.Message | None:
        if self.rx:
            return self.rx.popleft()
        if timeout:
            time.sleep(timeout)
        return None

    def shutdown(self) -> None:
        self.shut_down = True


class MotorSim:
    """Responder simulating RS motors: acks commands with feedback frames, answers param reads."""

    def __init__(self, motor_ids: Iterable[int], host_id: int = HOST_ID) -> None:
        self.host_id = host_id
        self.params: dict[int, dict[int, float | int]] = {
            motor_id: {RUN_MODE_IDX: int(PrivateControlMode.POSITION)} for motor_id in motor_ids
        }
        self.accept_run_mode_writes = True

    def __call__(self, msg: can.Message) -> list[can.Message]:
        arb = msg.arbitration_id
        comm_type = (arb >> 24) & 0x1F
        target = arb & 0xFF
        if target not in self.params:
            return []
        if comm_type in (
            PrivateCommType.ENABLE,
            PrivateCommType.STOP,
            PrivateCommType.SET_ZERO,
            PrivateCommType.SAVE_PARAMS,
        ):
            return [feedback_frame(target, host_id=self.host_id)]
        if comm_type == PrivateCommType.PARAM_WRITE:
            index = struct.unpack("<H", bytes(msg.data[:2]))[0]
            if index != RUN_MODE_IDX or self.accept_run_mode_writes:
                fmt = _FMT_BY_INDEX.get(index, "f")
                size = struct.calcsize(fmt)
                self.params[target][index] = struct.unpack("<" + fmt, bytes(msg.data[4 : 4 + size]))[0]
            return [feedback_frame(target, host_id=self.host_id)]
        if comm_type == PrivateCommType.PARAM_READ:
            index = struct.unpack("<H", bytes(msg.data[:2]))[0]
            fmt = _FMT_BY_INDEX.get(index, "f")
            value = self.params[target].get(index, 0.0 if fmt == "f" else 0)
            return [param_reply(target, index, value, host_id=self.host_id)]
        if comm_type == PrivateCommType.PING:
            reply_id = ext_id(PrivateCommType.PING, target, self.host_id)
            return [can.Message(arbitration_id=reply_id, data=bytes(8), is_extended_id=True)]
        return []


def sent(fake: FakeCanBus, comm_type: int | None = None, motor_id: int | None = None) -> list[can.Message]:
    frames = []
    for msg in fake.sent:
        if comm_type is not None and (msg.arbitration_id >> 24) & 0x1F != comm_type:
            continue
        if motor_id is not None and msg.arbitration_id & 0xFF != motor_id:
            continue
        frames.append(msg)
    return frames


def unpack_mit(msg: can.Message) -> tuple[int, int, int, int, int]:
    """Return (pos_u16, vel_u16, kp_u16, kd_u16, torque_u16) of a type-0x01 frame."""
    pos_u16, vel_u16, kp_u16, kd_u16 = struct.unpack(">HHHH", bytes(msg.data))
    return pos_u16, vel_u16, kp_u16, kd_u16, (msg.arbitration_id >> 8) & 0xFFFF


@pytest.fixture
def fake() -> FakeCanBus:
    return FakeCanBus()


@pytest.fixture
def bus(fake: FakeCanBus) -> RobstridePrivateMotorsBus:
    with patch("can.interface.Bus", return_value=fake):
        bus = RobstridePrivateMotorsBus(port="can0", motors=make_motors())
        bus.connect(handshake=False)
    return bus


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_constructor_rejects_missing_motor_type():
    motors = {"joint": Motor(id=1, model="robstride", norm_mode=MotorNormMode.DEGREES)}
    with pytest.raises(ValueError, match="no motor_type_str"):
        RobstridePrivateMotorsBus(port="can0", motors=motors)


def test_constructor_rejects_unknown_motor_type():
    motors = {"joint": Motor(id=1, model="robstride", norm_mode=MotorNormMode.DEGREES, motor_type_str="rs99")}
    with pytest.raises(ValueError, match="unknown motor_type_str 'rs99'"):
        RobstridePrivateMotorsBus(port="can0", motors=motors)


def test_constructor_rejects_duplicate_ids():
    motors = {
        "a": Motor(id=1, model="robstride", norm_mode=MotorNormMode.DEGREES, motor_type_str="rs00"),
        "b": Motor(id=1, model="robstride", norm_mode=MotorNormMode.DEGREES, motor_type_str="rs00"),
    }
    with pytest.raises(ValueError, match="Duplicate motor CAN ids"):
        RobstridePrivateMotorsBus(port="can0", motors=motors)


def test_constructor_normalizes_model_names():
    motors = {
        "a": Motor(id=1, model="robstride", norm_mode=MotorNormMode.DEGREES, motor_type_str="RS-06"),
        "b": Motor(id=2, model="robstride", norm_mode=MotorNormMode.DEGREES, motor_type_str="rs_00"),
        "c": Motor(id=3, model="robstride", norm_mode=MotorNormMode.DEGREES, motor_type_str="RS02"),
    }
    bus = RobstridePrivateMotorsBus(port="can0", motors=motors)
    assert bus._model_limits["a"] is RS_MODEL_LIMITS["rs06"]
    assert bus._model_limits["b"] is RS_MODEL_LIMITS["rs00"]
    assert bus._model_limits["c"] is RS_MODEL_LIMITS["rs02"]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def test_telemetry_param_indices_match_manual():
    # Manual order: 0x7019 mechPos, 0x701A iqf (filtered current!), 0x701B mechVel, 0x701C VBUS.
    assert PRIVATE_PARAMS["mech_pos"].index == 0x7019
    assert PRIVATE_PARAMS["iqf"].index == 0x701A
    assert PRIVATE_PARAMS["mech_vel"].index == 0x701B
    assert PRIVATE_PARAMS["vbus"].index == 0x701C


def test_rs05_limits_match_manual():
    rs05 = RS_MODEL_LIMITS["rs05"]
    assert rs05.v_max == 50.0
    assert rs05.t_max == 5.5


# ---------------------------------------------------------------------------
# Connect / handshake / disconnect
# ---------------------------------------------------------------------------


def test_connect_handshake_success():
    fake = FakeCanBus()
    fake.responder = MotorSim([1, 2, 3])
    with patch("can.interface.Bus", return_value=fake) as bus_factory:
        bus = RobstridePrivateMotorsBus(port="can0", motors=make_motors())
        bus.connect()
    assert bus.is_connected
    bus_factory.assert_called_once_with(channel="can0", bitrate=1_000_000, interface="socketcan")
    requests = sent(fake, PrivateCommType.PARAM_READ)
    expected_ids = [ext_id(PrivateCommType.PARAM_READ, HOST_ID, motor_id) for motor_id in (1, 2, 3)]
    assert [msg.arbitration_id for msg in requests] == expected_ids
    for msg in requests:
        assert msg.is_extended_id
        assert bytes(msg.data) == struct.pack("<H", MECH_POS_IDX) + bytes(6)


def test_connect_auto_detects_slcan():
    fake = FakeCanBus()
    with patch("can.interface.Bus", return_value=fake) as bus_factory:
        bus = RobstridePrivateMotorsBus(port="/dev/ttyACM0", motors=make_motors())
        bus.connect(handshake=False)
    assert bus.is_connected
    bus_factory.assert_called_once_with(channel="/dev/ttyACM0", bitrate=1_000_000, interface="slcan")


def test_connect_missing_motor_raises_and_rolls_back():
    fake = FakeCanBus()
    fake.responder = MotorSim([1, 2])  # wrist (id 3) never answers
    with patch("can.interface.Bus", return_value=fake):
        bus = RobstridePrivateMotorsBus(port="can0", motors=make_motors())
        with pytest.raises(ConnectionError, match="wrist"):
            bus.connect()
    assert not bus.is_connected
    assert bus.canbus is None
    assert fake.shut_down


def test_connect_twice_raises(bus):
    with pytest.raises(DeviceAlreadyConnectedError):
        bus.connect(handshake=False)


def test_methods_require_connection():
    bus = RobstridePrivateMotorsBus(port="can0", motors=make_motors())
    with pytest.raises(DeviceNotConnectedError):
        bus.read("Present_Position", "shoulder")
    with pytest.raises(DeviceNotConnectedError):
        bus.write("Goal_Position", "shoulder", 0.0)
    with pytest.raises(DeviceNotConnectedError):
        bus.enable_torque()


def test_disconnect_disables_torque_and_shuts_down(bus, fake):
    fake.responder = MotorSim([1, 2, 3])
    bus.disconnect()
    stops = sent(fake, PrivateCommType.STOP)
    assert [msg.arbitration_id & 0xFF for msg in stops] == [1, 2, 3]
    assert fake.shut_down
    assert not bus.is_connected
    with pytest.raises(DeviceNotConnectedError):
        bus.read("Present_Position", "shoulder")


def test_disconnect_without_disable_torque_sends_nothing(bus, fake):
    bus.disconnect(disable_torque=False)
    assert fake.sent == []
    assert fake.shut_down


# ---------------------------------------------------------------------------
# Wire format: parameter access
# ---------------------------------------------------------------------------


def test_param_read_request_frame(bus, fake):
    assert bus.read_param("shoulder", "mech_pos") is None  # no reply scripted
    (msg,) = sent(fake, PrivateCommType.PARAM_READ)
    assert msg.is_extended_id
    assert msg.arbitration_id == ext_id(PrivateCommType.PARAM_READ, HOST_ID, 1)
    assert bytes(msg.data) == struct.pack("<H", MECH_POS_IDX) + bytes(6)


def test_param_read_by_raw_index(bus, fake):
    fake.responder = MotorSim([1, 2, 3])
    assert bus.read_param("wrist", 0x7019) == 0.0
    (msg,) = sent(fake, PrivateCommType.PARAM_READ)
    assert msg.arbitration_id == ext_id(PrivateCommType.PARAM_READ, HOST_ID, 3)
    assert bytes(msg.data[:2]) == struct.pack("<H", 0x7019)


def test_param_write_frame_f32(bus, fake):
    bus.write_param("shoulder", "limit_spd", 2.5)
    (msg,) = sent(fake, PrivateCommType.PARAM_WRITE)
    assert msg.is_extended_id
    assert msg.arbitration_id == ext_id(PrivateCommType.PARAM_WRITE, HOST_ID, 1)
    assert bytes(msg.data) == struct.pack("<H", LIMIT_SPD_IDX) + b"\x00\x00" + struct.pack("<f", 2.5)


def test_param_write_frame_u8(bus, fake):
    bus.write_param("shoulder", "run_mode", 2)
    (msg,) = sent(fake, PrivateCommType.PARAM_WRITE)
    assert bytes(msg.data) == struct.pack("<H", RUN_MODE_IDX) + b"\x00\x00\x02\x00\x00\x00"


def test_write_param_rejects_read_only(bus, fake):
    for param in ("mech_pos", "mech_vel", "iqf", "vbus"):
        with pytest.raises(ValueError, match="read-only"):
            bus.write_param("shoulder", param, 1.0)
    assert fake.sent == []


def test_unknown_param_name_raises(bus):
    with pytest.raises(KeyError, match="Unknown parameter"):
        bus.read_param("shoulder", "no_such_param")


def test_unknown_motor_name_and_id_raise(bus):
    with pytest.raises(ValueError, match="Unknown motor: nope"):
        bus.read_param("nope", "mech_pos")
    with pytest.raises(ValueError, match="Unknown motor ID: 99"):
        bus.read_param(99, "mech_pos")


# ---------------------------------------------------------------------------
# Wire format: commands
# ---------------------------------------------------------------------------


def test_enable_torque_frame_and_ack(bus, fake):
    fake.responder = MotorSim([1, 2, 3])
    bus.enable_torque("shoulder")
    (msg,) = sent(fake, PrivateCommType.ENABLE)
    assert msg.is_extended_id
    assert msg.arbitration_id == ext_id(PrivateCommType.ENABLE, HOST_ID, 1)
    assert bytes(msg.data) == bytes(8)
    assert bus.enabled["shoulder"]


def test_disable_torque_frame(bus, fake):
    fake.responder = MotorSim([1, 2, 3])
    bus.enabled["wrist"] = True
    bus.disable_torque("wrist")
    (msg,) = sent(fake, PrivateCommType.STOP)
    assert msg.arbitration_id == ext_id(PrivateCommType.STOP, HOST_ID, 3)
    assert bytes(msg.data) == bytes(8)
    assert not bus.enabled["wrist"]


def test_clear_fault_frame(bus, fake):
    fake.responder = MotorSim([1, 2, 3])
    bus.clear_fault("shoulder")
    (msg,) = sent(fake, PrivateCommType.STOP)
    assert msg.arbitration_id == ext_id(PrivateCommType.STOP, HOST_ID, 1)
    assert bytes(msg.data) == b"\x01" + bytes(7)


def test_set_zero_position_sequence(bus, fake):
    fake.responder = MotorSim([1, 2, 3])
    bus.set_zero_position("wrist")
    (zero,) = sent(fake, PrivateCommType.SET_ZERO)
    assert zero.arbitration_id == ext_id(PrivateCommType.SET_ZERO, HOST_ID, 3)
    assert bytes(zero.data) == b"\x01" + bytes(7)
    (write,) = sent(fake, PrivateCommType.PARAM_WRITE)
    assert write.arbitration_id == ext_id(PrivateCommType.PARAM_WRITE, HOST_ID, 3)
    assert bytes(write.data) == struct.pack("<H", ZERO_STA_IDX) + b"\x00\x00\x01\x00\x00\x00"
    # The zero command must come before the zero_sta write.
    assert fake.sent.index(zero) < fake.sent.index(write)


def test_set_zero_position_refuses_enabled_motor(bus, fake):
    bus.enabled["wrist"] = True
    with pytest.raises(RuntimeError, match="torque-disabled"):
        bus.set_zero_position("wrist")
    assert fake.sent == []


def test_save_parameters_payload(bus, fake):
    fake.responder = MotorSim([1, 2, 3])
    bus.save_parameters("shoulder")
    (msg,) = sent(fake, PrivateCommType.SAVE_PARAMS)
    assert msg.arbitration_id == ext_id(PrivateCommType.SAVE_PARAMS, HOST_ID, 1)
    assert bytes(msg.data) == bytes([1, 2, 3, 4, 5, 6, 7, 8])


def test_ping(bus, fake):
    fake.responder = MotorSim([1, 2, 3])
    assert bus.ping("shoulder") is True
    assert bus.ping(2) is True  # by CAN id
    (first, second) = sent(fake, PrivateCommType.PING)
    assert first.arbitration_id == ext_id(PrivateCommType.PING, HOST_ID, 1)
    assert second.arbitration_id == ext_id(PrivateCommType.PING, HOST_ID, 2)
    assert bytes(first.data) == bytes(8)


def test_ping_no_answer_returns_false(bus):
    assert bus.ping("wrist") is False


# ---------------------------------------------------------------------------
# read()
# ---------------------------------------------------------------------------


def test_read_position_converts_rad_to_deg_exactly(bus, fake):
    fake.rx.append(param_reply(1, MECH_POS_IDX, 1.5))
    value = bus.read("Present_Position", "shoulder")
    assert value == pytest.approx(math.degrees(1.5), abs=1e-12)
    assert bus._last_known_states["shoulder"]["position"] == value


def test_read_velocity_converts_rad_to_deg(bus, fake):
    fake.rx.append(param_reply(1, MECH_VEL_IDX, -0.25))
    value = bus.read("Present_Velocity", "shoulder")
    assert value == pytest.approx(math.degrees(-0.25), abs=1e-12)


def test_read_vbus(bus, fake):
    fake.rx.append(param_reply(1, VBUS_IDX, 24.5))
    assert bus.read("VBUS", "shoulder") == pytest.approx(24.5)


def test_read_torque_and_temperature_come_from_cache(bus, fake):
    assert bus.read("Present_Torque", "shoulder") == 0.0
    assert bus.read("Temperature_MOS", "shoulder") == 0.0
    fake.rx.append(feedback_frame(1, torque_u16=65534, temp_u16=421))
    bus.flush_rx_queue()
    assert bus.read("Present_Torque", "shoulder") == pytest.approx(RS06.t_max)
    assert bus.read("Temperature_MOS", "shoulder") == pytest.approx(42.1)


def test_read_unsupported_data_name_raises(bus):
    with pytest.raises(ValueError, match="not supported"):
        bus.read("Temperature_Rotor", "shoulder")


def test_read_position_stale_fallback_warns(bus, fake, caplog):
    fake.rx.append(param_reply(1, MECH_POS_IDX, 0.5))
    first = bus.read("Present_Position", "shoulder")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        value = bus.read("Present_Position", "shoulder")  # no reply scripted
    assert value == first
    assert bus._consecutive_read_failures["shoulder"] == 1
    records = [r for r in caplog.records if "last known value" in r.message]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING


def test_read_failures_escalate_to_error(bus, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(3):
            bus.read("Present_Position", "shoulder")
    levels = [r.levelno for r in caplog.records if "consecutive" in r.message]
    assert levels == [logging.WARNING, logging.WARNING, logging.ERROR]


def test_successful_read_resets_failure_counter(bus, fake):
    bus.read("Present_Position", "shoulder")  # stale
    assert bus._consecutive_read_failures["shoulder"] == 1
    fake.rx.append(param_reply(1, MECH_POS_IDX, 0.5))
    bus.read("Present_Position", "shoulder")
    assert bus._consecutive_read_failures["shoulder"] == 0


def test_read_error_echo_falls_back_to_stale_and_warns(bus, fake, caplog):
    fake.rx.append(param_reply(1, MECH_POS_IDX, 1.5))
    first = bus.read("Present_Position", "shoulder")
    fake.rx.append(param_reply(1, MECH_POS_IDX, 0.0, error_flag=0x01))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        value = bus.read("Present_Position", "shoulder")
    assert value == first  # the zeroed error payload must not be decoded as 0.0
    assert bus._consecutive_read_failures["shoulder"] == 1
    assert any("error flag 0x01" in r.message for r in caplog.records)


def test_read_param_error_echo_returns_none(bus, fake, caplog):
    fake.rx.append(param_reply(1, 0x1003, 0.0, error_flag=0x01))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert bus.read_param("shoulder", 0x1003) is None
    assert any("0x1003" in r.message for r in caplog.records)


def test_read_ignores_reply_for_other_host(bus, fake):
    bus._last_known_states["shoulder"]["position"] = 42.0
    fake.rx.append(param_reply(1, MECH_POS_IDX, 5.0, host_id=0x42))
    assert bus.read("Present_Position", "shoulder") == 42.0


def test_read_skips_reply_with_wrong_index(bus, fake):
    fake.rx.append(param_reply(1, LIMIT_SPD_IDX, 3.3))
    fake.rx.append(param_reply(1, MECH_POS_IDX, 0.25))
    assert bus.read("Present_Position", "shoulder") == pytest.approx(math.degrees(0.25), abs=1e-12)


def test_read_skips_reply_from_wrong_motor(bus, fake):
    # A reply from motor 2 (elbow) must not satisfy or pollute a read of motor 1 (shoulder).
    fake.rx.append(param_reply(2, MECH_POS_IDX, 3.0))
    fake.rx.append(param_reply(1, MECH_POS_IDX, 0.25))
    assert bus.read("Present_Position", "shoulder") == pytest.approx(math.degrees(0.25), abs=1e-12)
    assert bus._last_known_states["elbow"]["position"] == 0.0


def test_set_position_speed_limit_wire_format(bus, fake):
    fake.responder = MotorSim([1, 2, 3])
    bus.set_position_speed_limit("shoulder", 150.0)
    (msg,) = sent(fake, PrivateCommType.PARAM_WRITE)
    assert msg.arbitration_id == ext_id(PrivateCommType.PARAM_WRITE, HOST_ID, 1)
    # limit_spd is 0x7017 (spec literal) and the value must be radians/second, float32 LE.
    assert bytes(msg.data) == struct.pack("<H", 0x7017) + b"\x00\x00" + struct.pack("<f", math.radians(150.0))


def test_set_zero_position_raises_without_ack(bus, fake):
    with pytest.raises(RuntimeError, match="did not acknowledge the set-zero"):
        bus.set_zero_position("shoulder")


def test_save_parameters_raises_without_ack(bus, fake):
    with pytest.raises(RuntimeError, match="did not acknowledge the parameter save"):
        bus.save_parameters("shoulder")


def test_read_harvests_feedback_seen_along_the_way(bus, fake):
    fake.rx.append(feedback_frame(1, torque_u16=0, temp_u16=385, status=2))
    fake.rx.append(param_reply(1, MECH_POS_IDX, 0.5))
    value = bus.read("Present_Position", "shoulder")
    assert value == pytest.approx(math.degrees(0.5), abs=1e-12)
    assert bus._last_known_states["shoulder"]["torque"] == pytest.approx(-RS06.t_max)
    assert bus._last_known_states["shoulder"]["temp_mos"] == pytest.approx(38.5)
    assert bus.status_code["shoulder"] == 2
    assert bus.last_feedback_time["shoulder"] is not None


# ---------------------------------------------------------------------------
# Type-0x18 report frames
# ---------------------------------------------------------------------------


def test_report_flood_is_skipped_not_decoded(bus, fake):
    for _ in range(100):
        fake.rx.append(report_frame(1))
    fake.rx.append(param_reply(1, MECH_POS_IDX, 1.0))
    value = bus.read("Present_Position", "shoulder")
    assert value == pytest.approx(math.degrees(1.0), abs=1e-12)  # not the 999.0 report payload
    assert not fake.rx


def test_endless_report_flood_cannot_starve_the_deadline(bus, fake):
    bus._last_known_states["shoulder"]["position"] = 7.5
    fake.recv = lambda timeout=None: report_frame(1)  # bus streams reports forever
    start = time.monotonic()
    value = bus.read("Present_Position", "shoulder")
    elapsed = time.monotonic() - start
    assert value == 7.5  # stale fallback
    assert elapsed < 1.0  # 2 attempts x 20 ms deadline, not an infinite spin


def test_ack_wait_skips_report_frames(bus, fake):
    fake.rx.extend([report_frame(1), report_frame(1), feedback_frame(1)])
    bus.enable_torque("shoulder")
    assert bus.enabled["shoulder"]


# ---------------------------------------------------------------------------
# sync_read()
# ---------------------------------------------------------------------------


def test_sync_read_batches_one_request_per_motor(bus, fake):
    sim = MotorSim([1, 2, 3])
    sim.params[1][MECH_POS_IDX] = 0.5
    sim.params[2][MECH_POS_IDX] = -0.25
    sim.params[3][MECH_POS_IDX] = 1.5
    fake.responder = sim
    result = bus.sync_read("Present_Position")
    assert result == {
        "shoulder": pytest.approx(math.degrees(0.5), abs=1e-12),
        "elbow": pytest.approx(math.degrees(-0.25), abs=1e-12),
        "wrist": pytest.approx(math.degrees(1.5), abs=1e-12),
    }
    requests = sent(fake, PrivateCommType.PARAM_READ)
    assert [msg.arbitration_id & 0xFF for msg in requests] == [1, 2, 3]  # no retries needed
    for msg in requests:
        assert bytes(msg.data[:2]) == struct.pack("<H", MECH_POS_IDX)


def test_sync_read_handles_out_of_order_replies(bus, fake):
    fake.rx.extend(
        [
            param_reply(3, MECH_POS_IDX, 0.75),
            param_reply(1, MECH_POS_IDX, 0.5),
            param_reply(2, MECH_POS_IDX, -0.25),
        ]
    )
    result = bus.sync_read("Present_Position")
    assert result == {
        "shoulder": pytest.approx(math.degrees(0.5), abs=1e-12),
        "elbow": pytest.approx(math.degrees(-0.25), abs=1e-12),
        "wrist": pytest.approx(math.degrees(0.75), abs=1e-12),
    }
    assert len(sent(fake, PrivateCommType.PARAM_READ)) == 3


def test_sync_read_retries_straggler(bus, fake):
    sim = MotorSim([1, 2, 3])
    sim.params[3][MECH_POS_IDX] = 1.0
    dropped: list[can.Message] = []

    def responder(msg: can.Message) -> list[can.Message]:
        comm_type = (msg.arbitration_id >> 24) & 0x1F
        if comm_type == PrivateCommType.PARAM_READ and msg.arbitration_id & 0xFF == 3 and not dropped:
            dropped.append(msg)  # drop the wrist's first reply
            return []
        return sim(msg)

    fake.responder = responder
    result = bus.sync_read("Present_Position")
    assert result["wrist"] == pytest.approx(math.degrees(1.0), abs=1e-12)
    assert len(sent(fake, PrivateCommType.PARAM_READ, motor_id=3)) == 2
    assert len(sent(fake, PrivateCommType.PARAM_READ, motor_id=1)) == 1


def test_sync_read_stale_fallback_for_missing_motor(bus, fake, caplog):
    sim = MotorSim([1, 2])  # wrist never answers
    sim.params[1][MECH_POS_IDX] = 1.0
    fake.responder = sim
    bus._last_known_states["wrist"]["position"] = 42.0
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = bus.sync_read("Present_Position")
    assert result["wrist"] == 42.0
    assert result["shoulder"] == pytest.approx(math.degrees(1.0), abs=1e-12)
    assert bus._consecutive_read_failures["wrist"] == 1
    assert bus._consecutive_read_failures["shoulder"] == 0
    assert any("wrist" in r.message for r in caplog.records)


def test_sync_read_error_echo_falls_back_to_stale(bus, fake, caplog):
    sim = MotorSim([1, 2])
    sim.params[1][MECH_POS_IDX] = 1.0

    def responder(msg: can.Message) -> list[can.Message]:
        is_read = (msg.arbitration_id >> 24) & 0x1F == PrivateCommType.PARAM_READ
        if is_read and msg.arbitration_id & 0xFF == 3:
            return [param_reply(3, MECH_POS_IDX, 0.0, error_flag=0x01)]
        return sim(msg)

    fake.responder = responder
    bus._last_known_states["wrist"]["position"] = 42.0
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = bus.sync_read("Present_Position")
    assert result["wrist"] == 42.0  # error echo not decoded as 0.0
    assert result["shoulder"] == pytest.approx(math.degrees(1.0), abs=1e-12)
    assert any("error flag 0x01" in r.message for r in caplog.records)
    # An errored motor is not pointlessly retried within the same sync read.
    assert len(sent(fake, PrivateCommType.PARAM_READ, motor_id=3)) == 1


def test_sync_read_skips_interleaved_report_frames(bus, fake):
    for motor_id in (1, 2, 3):
        fake.rx.append(report_frame(motor_id))
        fake.rx.append(param_reply(motor_id, MECH_POS_IDX, 0.5))
    result = bus.sync_read("Present_Position")
    assert all(value == pytest.approx(math.degrees(0.5), abs=1e-12) for value in result.values())


def test_sync_read_velocity(bus, fake):
    sim = MotorSim([1, 2, 3])
    sim.params[1][MECH_VEL_IDX] = 0.5
    sim.params[3][MECH_VEL_IDX] = -1.0
    fake.responder = sim
    result = bus.sync_read("Present_Velocity", ["shoulder", "wrist"])
    assert result == {
        "shoulder": pytest.approx(math.degrees(0.5), abs=1e-12),
        "wrist": pytest.approx(math.degrees(-1.0), abs=1e-12),
    }


def test_sync_read_torque_and_temperature_from_cache(bus, fake):
    fake.rx.append(feedback_frame(1, torque_u16=65534, temp_u16=421))
    bus.flush_rx_queue()
    assert bus.sync_read("Present_Torque", ["shoulder"]) == {"shoulder": pytest.approx(RS06.t_max)}
    assert bus.sync_read("Temperature_MOS", ["shoulder"]) == {"shoulder": pytest.approx(42.1)}


def test_sync_read_unsupported_data_name_raises(bus):
    with pytest.raises(ValueError, match="not supported"):
        bus.sync_read("Goal_Position")


def test_sync_read_all_states(bus, fake):
    sim = MotorSim([1, 2, 3])
    sim.params[1][MECH_POS_IDX] = 1.0
    fake.responder = sim
    fake.rx.append(feedback_frame(1, torque_u16=65534, temp_u16=300))
    bus.flush_rx_queue()
    states = bus.sync_read_all_states()
    assert set(states) == {"shoulder", "elbow", "wrist"}
    assert states["shoulder"]["position"] == pytest.approx(math.degrees(1.0), abs=1e-12)
    assert states["shoulder"]["torque"] == pytest.approx(RS06.t_max)
    assert states["shoulder"]["temp_mos"] == pytest.approx(30.0)
    assert states["elbow"]["position"] == pytest.approx(0.0)
    # The returned states are copies, not the live cache.
    states["shoulder"]["position"] = -1.0
    assert bus._last_known_states["shoulder"]["position"] == pytest.approx(math.degrees(1.0), abs=1e-12)


# ---------------------------------------------------------------------------
# write() dispatch per control mode
# ---------------------------------------------------------------------------


def test_write_kp_kd_stores_gains_without_sending(bus, fake):
    bus.write("Kp", "shoulder", 123.0)
    bus.write("Kd", "shoulder", 4.5)
    assert fake.sent == []
    assert bus._gains["shoulder"] == {"kp": 123.0, "kd": 4.5}


def test_write_goal_position_mit_mode_sends_mit_frame(bus, fake):
    bus.control_mode["shoulder"] = PrivateControlMode.MIT
    bus.write("Kp", "shoulder", 2500.0)
    bus.write("Kd", "shoulder", 25.0)
    bus.write("Goal_Position", "shoulder", 180.0)
    (msg,) = sent(fake, PrivateCommType.MIT_CONTROL)
    assert msg.is_extended_id
    assert msg.arbitration_id & 0xFF == 1
    pos_u16, vel_u16, kp_u16, kd_u16, torque_u16 = unpack_mit(msg)
    assert pos_u16 == u16_phys(math.pi, RS06.p_max) == 40959
    assert vel_u16 == 32767  # zero velocity is mid-scale
    assert kp_u16 == 32768  # 2500 / 5000 * 65535, rounded
    assert kd_u16 == 16384  # 25 / 100 * 65535, rounded
    assert torque_u16 == 32767  # zero torque is mid-scale, carried in the ID's data16
    assert not sent(fake, PrivateCommType.PARAM_WRITE)


def test_write_goal_position_position_mode_writes_loc_ref(bus, fake):
    bus.write("Goal_Position", "shoulder", 90.0)  # POSITION is the default mode
    (msg,) = sent(fake, PrivateCommType.PARAM_WRITE)
    assert msg.arbitration_id == ext_id(PrivateCommType.PARAM_WRITE, HOST_ID, 1)
    assert bytes(msg.data) == struct.pack("<H", LOC_REF_IDX) + b"\x00\x00" + struct.pack(
        "<f", math.radians(90.0)
    )
    assert not sent(fake, PrivateCommType.MIT_CONTROL)


def test_write_goal_position_velocity_mode_raises(bus, fake):
    bus.control_mode["shoulder"] = PrivateControlMode.VELOCITY
    with pytest.raises(ValueError, match="Goal_Position requires MIT or Position mode"):
        bus.write("Goal_Position", "shoulder", 10.0)
    assert fake.sent == []


def test_write_goal_velocity_requires_velocity_mode(bus, fake):
    with pytest.raises(ValueError, match="requires VELOCITY mode"):
        bus.write("Goal_Velocity", "shoulder", 10.0)
    assert fake.sent == []
    bus.control_mode["shoulder"] = PrivateControlMode.VELOCITY
    bus.write("Goal_Velocity", "shoulder", 90.0)
    (msg,) = sent(fake, PrivateCommType.PARAM_WRITE)
    assert bytes(msg.data) == struct.pack("<H", SPD_REF_IDX) + b"\x00\x00" + struct.pack(
        "<f", math.radians(90.0)
    )


def test_write_goal_current_requires_current_mode(bus, fake):
    with pytest.raises(ValueError, match="requires CURRENT mode"):
        bus.write("Goal_Current", "shoulder", 1.25)
    assert fake.sent == []
    bus.control_mode["shoulder"] = PrivateControlMode.CURRENT
    bus.write("Goal_Current", "shoulder", 1.25)
    (msg,) = sent(fake, PrivateCommType.PARAM_WRITE)
    # Amps go on the wire untouched (no deg->rad conversion).
    assert bytes(msg.data) == struct.pack("<H", IQ_REF_IDX) + b"\x00\x00" + struct.pack("<f", 1.25)


def test_write_unsupported_data_name_raises(bus):
    with pytest.raises(ValueError, match="not supported"):
        bus.write("Goal_Torque", "shoulder", 1.0)


# ---------------------------------------------------------------------------
# MIT frame packing math
# ---------------------------------------------------------------------------


def test_mit_packing_clamps_upper_extremes(bus, fake):
    bus._mit_control("shoulder", kp=1e9, kd=1e9, position=1e9, velocity=1e9, torque=1e9)
    (msg,) = sent(fake, PrivateCommType.MIT_CONTROL)
    assert unpack_mit(msg) == (65534, 65534, 65535, 65535, 65534)


def test_mit_packing_clamps_lower_extremes(bus, fake):
    bus._mit_control("shoulder", kp=-1.0, kd=-1.0, position=-1e9, velocity=-1e9, torque=-1e9)
    (msg,) = sent(fake, PrivateCommType.MIT_CONTROL)
    assert unpack_mit(msg) == (0, 0, 0, 0, 0)
    assert msg.arbitration_id == ext_id(PrivateCommType.MIT_CONTROL, 0, 1)


def test_mit_packing_velocity_is_deg_per_s(bus, fake):
    bus._mit_control("shoulder", kp=0.0, kd=0.0, position=0.0, velocity=math.degrees(5.0), torque=0.0)
    (msg,) = sent(fake, PrivateCommType.MIT_CONTROL)
    _, vel_u16, _, _, _ = unpack_mit(msg)
    assert vel_u16 == u16_phys(5.0, RS06.v_max) == 40959  # 5 rad/s is quarter-scale of 20 rad/s


def test_mit_packing_torque_uses_per_model_limits(bus, fake):
    bus._mit_control("wrist", kp=0.0, kd=0.0, position=0.0, velocity=0.0, torque=3.5)
    bus._mit_control("shoulder", kp=0.0, kd=0.0, position=0.0, velocity=0.0, torque=3.5)
    (wrist_msg,) = sent(fake, PrivateCommType.MIT_CONTROL, motor_id=3)
    (shoulder_msg,) = sent(fake, PrivateCommType.MIT_CONTROL, motor_id=1)
    wrist_torque = (wrist_msg.arbitration_id >> 8) & 0xFFFF
    shoulder_torque = (shoulder_msg.arbitration_id >> 8) & 0xFFFF
    assert wrist_torque == u16_phys(3.5, RS00.t_max) == 40959  # 3.5 N·m is quarter-scale on rs00
    assert shoulder_torque == u16_phys(3.5, RS06.t_max)
    assert wrist_torque != shoulder_torque


# ---------------------------------------------------------------------------
# sync_write()
# ---------------------------------------------------------------------------


def test_sync_write_kp_kd(bus, fake):
    bus.sync_write("Kp", {"shoulder": 100.0, "wrist": 50.0})
    bus.sync_write("Kd", {"shoulder": 2.0})
    assert fake.sent == []
    assert bus._gains["shoulder"] == {"kp": 100.0, "kd": 2.0}
    assert bus._gains["wrist"]["kp"] == 50.0


def test_sync_write_goal_position_mit_batch(bus, fake):
    bus.control_mode["shoulder"] = PrivateControlMode.MIT
    bus.control_mode["wrist"] = PrivateControlMode.MIT
    bus.sync_write("Goal_Position", {"shoulder": 180.0, "wrist": 180.0})
    frames = sent(fake, PrivateCommType.MIT_CONTROL)
    assert [msg.arbitration_id & 0xFF for msg in frames] == [1, 3]
    for msg in frames:
        pos_u16, _, _, _, torque_u16 = unpack_mit(msg)
        assert pos_u16 == u16_phys(math.pi, RS06.p_max)  # p_max is 4*pi on every RS model
        assert torque_u16 == 32767


def test_sync_write_goal_position_mixed_modes(bus, fake):
    bus.control_mode["shoulder"] = PrivateControlMode.MIT
    bus.sync_write("Goal_Position", {"shoulder": 0.0, "elbow": 90.0, "wrist": -90.0})
    mit_frames = sent(fake, PrivateCommType.MIT_CONTROL)
    assert [msg.arbitration_id & 0xFF for msg in mit_frames] == [1]
    writes = sent(fake, PrivateCommType.PARAM_WRITE)
    assert [msg.arbitration_id & 0xFF for msg in writes] == [2, 3]
    for msg, expected in zip(writes, (math.radians(90.0), math.radians(-90.0)), strict=True):
        assert bytes(msg.data) == struct.pack("<H", LOC_REF_IDX) + b"\x00\x00" + struct.pack("<f", expected)


def test_sync_write_goal_position_wrong_mode_raises(bus):
    bus.control_mode["elbow"] = PrivateControlMode.VELOCITY
    with pytest.raises(ValueError, match="Goal_Position requires MIT or Position mode"):
        bus.sync_write("Goal_Position", {"elbow": 10.0})


def test_sync_write_goal_velocity_mode_guard(bus):
    with pytest.raises(ValueError, match="requires VELOCITY mode"):
        bus.sync_write("Goal_Velocity", {"shoulder": 10.0})


def test_sync_write_unsupported_data_name_raises(bus):
    with pytest.raises(ValueError, match="not supported"):
        bus.sync_write("Goal_Torque", {"shoulder": 1.0})


# ---------------------------------------------------------------------------
# Torque enable/disable acknowledgment handling
# ---------------------------------------------------------------------------


def test_enable_torque_without_ack_raises(bus):
    with pytest.raises(RuntimeError, match="did not acknowledge torque enable"):
        bus.enable_torque("shoulder")
    assert not bus.enabled["shoulder"]


def test_enable_torque_retry_recovers(bus, fake):
    sim = MotorSim([1, 2, 3])
    dropped: list[can.Message] = []

    def responder(msg: can.Message) -> list[can.Message]:
        if (msg.arbitration_id >> 24) & 0x1F == PrivateCommType.ENABLE and not dropped:
            dropped.append(msg)  # swallow the first enable's ack
            return []
        return sim(msg)

    fake.responder = responder
    bus.enable_torque("shoulder", num_retry=1)
    assert len(sent(fake, PrivateCommType.ENABLE)) == 2
    assert bus.enabled["shoulder"]


def test_disable_torque_without_ack_warns_but_does_not_raise(bus, caplog):
    bus.enabled["shoulder"] = True
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        bus.disable_torque("shoulder")
    assert not bus.enabled["shoulder"]
    assert any("did not acknowledge torque disable" in r.message for r in caplog.records)


def test_torque_disabled_context_manager(bus, fake):
    fake.responder = MotorSim([1, 2, 3])
    bus.enable_torque("shoulder")
    with bus.torque_disabled("shoulder"):
        assert not bus.enabled["shoulder"]
    assert bus.enabled["shoulder"]
    comm_types = [(msg.arbitration_id >> 24) & 0x1F for msg in fake.sent]
    assert comm_types == [PrivateCommType.ENABLE, PrivateCommType.STOP, PrivateCommType.ENABLE]


# ---------------------------------------------------------------------------
# Control mode switching
# ---------------------------------------------------------------------------


def test_set_control_mode_stop_write_verify_sequence(bus, fake):
    fake.responder = MotorSim([1, 2, 3])
    bus.enabled["shoulder"] = True
    bus.set_control_mode(PrivateControlMode.VELOCITY, "shoulder")
    assert bus.control_mode["shoulder"] == PrivateControlMode.VELOCITY
    assert not bus.enabled["shoulder"]  # left torque-disabled
    comm_types = [(msg.arbitration_id >> 24) & 0x1F for msg in fake.sent]
    assert comm_types == [PrivateCommType.STOP, PrivateCommType.PARAM_WRITE, PrivateCommType.PARAM_READ]
    (write,) = sent(fake, PrivateCommType.PARAM_WRITE)
    assert bytes(write.data) == struct.pack("<H", RUN_MODE_IDX) + b"\x00\x00\x02\x00\x00\x00"
    (read,) = sent(fake, PrivateCommType.PARAM_READ)
    assert bytes(read.data[:2]) == struct.pack("<H", RUN_MODE_IDX)


def test_set_control_mode_raises_when_readback_never_matches(bus, fake):
    sim = MotorSim([1, 2, 3])
    sim.accept_run_mode_writes = False  # firmware acks the write but silently ignores it
    fake.responder = sim
    with pytest.raises(RuntimeError, match="did not switch to VELOCITY"):
        bus.set_control_mode(PrivateControlMode.VELOCITY, "shoulder")
    writes = sent(fake, PrivateCommType.PARAM_WRITE)
    assert len(writes) == PRIVATE_MODE_SWITCH_RETRIES
    assert bus.control_mode["shoulder"] == PrivateControlMode.POSITION  # unchanged


def test_configure_motors_puts_all_motors_in_mode(bus, fake):
    fake.responder = MotorSim([1, 2, 3])
    bus.configure_motors(PrivateControlMode.MIT)
    assert all(mode == PrivateControlMode.MIT for mode in bus.control_mode.values())
    writes = sent(fake, PrivateCommType.PARAM_WRITE)
    assert [msg.arbitration_id & 0xFF for msg in writes] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Feedback (type-0x02) harvesting
# ---------------------------------------------------------------------------


def test_feedback_frame_decodes_into_state_cache(bus, fake):
    fake.rx.append(feedback_frame(1, pos_u16=40959, vel_u16=49151, torque_u16=0, temp_u16=385, status=2))
    assert bus.flush_rx_queue() == 1
    state = bus._last_known_states["shoulder"]
    assert state["position"] == pytest.approx((40959 / 32767 - 1.0) * RS06.p_max * 180.0 / math.pi)
    assert state["velocity"] == pytest.approx((49151 / 32767 - 1.0) * RS06.v_max * 180.0 / math.pi)
    assert state["torque"] == pytest.approx(-RS06.t_max)
    assert state["temp_mos"] == pytest.approx(38.5)
    assert bus.status_code["shoulder"] == 2
    assert bus.fault_bits["shoulder"] == 0
    assert bus.last_feedback_time["shoulder"] is not None


def test_feedback_fault_bits_logged_once_per_change(bus, fake, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        fake.rx.append(feedback_frame(1, fault=0b000101))
        bus.flush_rx_queue()
    assert bus.fault_bits["shoulder"] == 0b000101
    assert any("0b000101" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        fake.rx.append(feedback_frame(1, fault=0b000101))  # same fault again
        bus.flush_rx_queue()
    assert not caplog.records
    fake.rx.append(feedback_frame(1, fault=0))
    bus.flush_rx_queue()
    assert bus.fault_bits["shoulder"] == 0


def test_feedback_from_unknown_motor_is_ignored(bus, fake):
    before = {name: dict(state) for name, state in bus._last_known_states.items()}
    fake.rx.append(feedback_frame(9))
    assert bus.flush_rx_queue() == 1
    assert bus._last_known_states == before


def test_standard_id_frames_are_ignored(bus, fake):
    fake.rx.append(can.Message(arbitration_id=0x123, data=bytes(8), is_extended_id=False))
    assert bus.flush_rx_queue() == 1
    assert all(t is None for t in bus.last_feedback_time.values())


# ---------------------------------------------------------------------------
# Calibration cache
# ---------------------------------------------------------------------------


def test_calibration_is_cached_in_memory_only(bus, fake):
    assert bus.read_calibration() == {}
    assert not bus.is_calibrated
    calibration = {
        name: MotorCalibration(id=motor.id, drive_mode=0, homing_offset=0, range_min=-90, range_max=90)
        for name, motor in make_motors().items()
    }
    bus.write_calibration(calibration)
    assert bus.read_calibration() == calibration
    assert bus.is_calibrated
    assert fake.sent == []  # nothing goes on the wire

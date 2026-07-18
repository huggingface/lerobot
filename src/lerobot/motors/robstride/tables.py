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

"""Configuration tables for Robstride motors (MIT-mode and private-protocol buses)."""

from dataclasses import dataclass
from enum import IntEnum


# Motor type definitions
class MotorType(IntEnum):
    O0 = 0
    O1 = 1
    O2 = 2
    O3 = 3
    O4 = 4
    O5 = 5
    ELO5 = 6
    O6 = 7


class CommMode(IntEnum):
    PrivateProtocole = 0
    CANopen = 1
    MIT = 2


# Control modes
class ControlMode(IntEnum):
    MIT = 0
    POS_VEL = 1
    VEL = 2


# Motor limit parameters [PMAX, VMAX, TMAX]
# PMAX: Maximum position (rad)
# VMAX: Maximum velocity (rad/s)
# TMAX: Maximum torque (N·m)
MOTOR_LIMIT_PARAMS: dict[MotorType, tuple[float, float, float]] = {
    MotorType.O0: (12.57, 33, 14),
    MotorType.O1: (12.57, 44, 17),
    MotorType.O2: (12.57, 33, 20),
    MotorType.O3: (12.57, 33, 60),
    MotorType.O4: (12.57, 33, 120),
    MotorType.O5: (12.57, 50, 5.5),
    MotorType.ELO5: (12.57, 50, 6),
    MotorType.O6: (112.5, 50, 36),
}

# Motor model names
MODEL_NAMES = {
    MotorType.O0: "O0",
    MotorType.O1: "O1",
    MotorType.O2: "O2",
    MotorType.O3: "O3",
    MotorType.O4: "O4",
    MotorType.O5: "O5",
    MotorType.ELO5: "ELO5",
    MotorType.O6: "O6",
}

# Motor resolution table (encoder counts per revolution)
MODEL_RESOLUTION = {
    "O0": 65536,
    "O1": 65536,
    "O2": 65536,
    "O3": 65536,
    "O4": 65536,
    "O5": 65536,
    "ELO5": 65536,
    "O6": 65536,
}

# CAN baudrates supported by Robstride motors
AVAILABLE_BAUDRATES = [
    1000000,  # 4: 1 mbps (default)
]
DEFAULT_BAUDRATE = 1000000

# Default timeout in milliseconds
DEFAULT_TIMEOUT_MS = 0  # disabled by default, otherwise 20000 is 1s


# Data that should be normalized
NORMALIZED_DATA = ["Present_Position", "Goal_Position"]


# MIT control parameter ranges
MIT_KP_RANGE = (0.0, 500.0)
MIT_KD_RANGE = (0.0, 5.0)

# CAN frame command IDs
CAN_CMD_ENABLE = 0xFC
CAN_CMD_DISABLE = 0xFD
CAN_CMD_SET_ZERO = 0xFE
CAN_CMD_CLEAR_FAULT = 0xFB


CAN_CMD_QUERY_PARAM = 0x33
CAN_CMD_WRITE_PARAM = 0x55
CAN_CMD_SAVE_PARAM = 0xAA

# CAN ID for parameter operations
CAN_PARAM_ID = 0x7FF


RUNNING_TIMEOUT = 0.003
HANDSHAKE_TIMEOUT_S = 0.05
PARAM_TIMEOUT = 0.01

STATE_CACHE_TTL_S = 0.02


# ---------------------------------------------------------------------------
# Robstride private (vendor/factory-default) protocol
# ---------------------------------------------------------------------------
# Unlike the MIT communication mode above (11-bit standard IDs), the private
# protocol uses 29-bit extended CAN IDs packed as:
#
#   ext_id[28:0] = comm_type[28:24] | data16[23:8] | target_id[7:0]
#
# Host -> motor: data16 carries the host ID (low byte), target_id the motor ID.
# Motor -> host replies swap the fields: target_id carries the host ID and
# data16's low byte carries the motor ID (type-0x02 feedback additionally packs
# fault/mode bits in data16's high byte).


class PrivateCommType(IntEnum):
    """Communication types (ext ID bits 28:24) of the Robstride private protocol."""

    PING = 0x00  # get device ID (replies with the 64-bit MCU unique ID)
    MIT_CONTROL = 0x01  # operation-mode control frame (pos/vel/kp/kd payload, torque in data16)
    FEEDBACK = 0x02  # feedback frame; also the ack to enable/stop/zero/param-write
    ENABLE = 0x03  # torque on
    STOP = 0x04  # torque off; with data[0]=1 clears faults instead
    SET_ZERO = 0x06  # set mechanical zero (data[0]=1)
    SET_CAN_ID = 0x07  # change the motor CAN ID
    PARAM_READ = 0x11  # single parameter read (index LE in bytes 0-1, value LE in bytes 4-7)
    PARAM_WRITE = 0x12  # single parameter write (same layout as the read reply)
    FAULT_REPORT = 0x15  # unsolicited fault/warning report
    SAVE_PARAMS = 0x16  # persist parameters to flash (payload 01..08)
    ACTIVE_REPORT = 0x18  # set active-report on/off; also the type of the streamed report frames


class PrivateControlMode(IntEnum):
    """Values of the ``run_mode`` parameter (0x7005) in the private protocol."""

    MIT = 0  # operation/MIT mode, driven by type-0x01 control frames
    POSITION = 1  # profile position mode, driven by loc_ref (0x7016) parameter writes
    VELOCITY = 2  # velocity mode, driven by spd_ref (0x700A) parameter writes
    CURRENT = 3  # current (Iq) mode, driven by iq_ref (0x7006) parameter writes


# Host ID used in outgoing frames. 0xFD is the convention of the vendor PC tools;
# it must be unique per bus master (never run two masters concurrently).
DEFAULT_PRIVATE_HOST_ID = 0xFD


@dataclass(frozen=True)
class PrivateParam:
    """One entry of the private-protocol runtime parameter table (0x7000 region)."""

    index: int
    fmt: str  # struct format of the value in bytes 4-7: "f", "b", "B", "H" or "I"
    writable: bool = True


# Runtime parameters (read with comm type 0x11, written with 0x12).
PRIVATE_PARAMS: dict[str, PrivateParam] = {
    "run_mode": PrivateParam(0x7005, "B"),
    "iq_ref": PrivateParam(0x7006, "f"),  # current-mode target [A]
    "spd_ref": PrivateParam(0x700A, "f"),  # velocity-mode target [rad/s]
    "limit_torque": PrivateParam(0x700B, "f"),  # torque limit [N.m]
    "cur_kp": PrivateParam(0x7010, "f"),
    "cur_ki": PrivateParam(0x7011, "f"),
    "cur_filter_gain": PrivateParam(0x7014, "f"),
    "loc_ref": PrivateParam(0x7016, "f"),  # position-mode target [rad]
    "limit_spd": PrivateParam(0x7017, "f"),  # position-mode speed limit [rad/s]
    "limit_cur": PrivateParam(0x7018, "f"),  # current limit [A]
    "mech_pos": PrivateParam(0x7019, "f", writable=False),  # mechanical position [rad], exact f32
    "mech_vel": PrivateParam(0x701A, "f", writable=False),  # mechanical velocity [rad/s]; see caveat
    "vbus": PrivateParam(0x701C, "f", writable=False),  # bus voltage [V]
    "loc_kp": PrivateParam(0x701E, "f"),  # position loop Kp
    "spd_kp": PrivateParam(0x701F, "f"),  # velocity loop Kp
    "spd_ki": PrivateParam(0x7020, "f"),  # velocity loop Ki
    "spd_filter_gain": PrivateParam(0x7021, "f"),
    "acc_rad": PrivateParam(0x7022, "f"),  # velocity-mode acceleration [rad/s^2]
    "vel_max": PrivateParam(0x7024, "f"),  # profile-position max velocity [rad/s]
    "acc_set": PrivateParam(0x7025, "f"),  # profile-position acceleration [rad/s^2]
    "zero_sta": PrivateParam(0x7029, "B"),
}


@dataclass(frozen=True)
class RSModelLimits:
    """Physical ranges of one RS-series model, used for MIT-frame packing and feedback decode."""

    p_max: float  # position half-range [rad] (feedback and MIT position window is +/- p_max)
    v_max: float  # velocity half-range [rad/s]
    t_max: float  # torque half-range [N.m]
    kp_max: float  # MIT kp full-scale
    kd_max: float  # MIT kd full-scale


_RS_P_MAX = 12.566370614359172  # 4*pi, identical for every RS model

# RS-series model table. kp/kd MIT full-scales are model-grouped:
# rs00/rs01/rs02/rs05 use 500/5, rs03/rs04/rs06 use 5000/100.
RS_MODEL_LIMITS: dict[str, RSModelLimits] = {
    "rs00": RSModelLimits(_RS_P_MAX, 50.0, 14.0, 500.0, 5.0),
    "rs01": RSModelLimits(_RS_P_MAX, 44.0, 17.0, 500.0, 5.0),
    "rs02": RSModelLimits(_RS_P_MAX, 44.0, 17.0, 500.0, 5.0),
    "rs03": RSModelLimits(_RS_P_MAX, 50.0, 60.0, 5000.0, 100.0),
    "rs04": RSModelLimits(_RS_P_MAX, 15.0, 120.0, 5000.0, 100.0),
    "rs05": RSModelLimits(_RS_P_MAX, 33.0, 17.0, 500.0, 5.0),
    "rs06": RSModelLimits(_RS_P_MAX, 20.0, 36.0, 5000.0, 100.0),
}

# Timeouts specific to the private protocol (seconds).
PRIVATE_RECV_POLL_S = 0.0005  # single recv() poll while draining/collecting
PRIVATE_PARAM_TIMEOUT_S = 0.02  # per-motor parameter read deadline
PRIVATE_ACK_TIMEOUT_S = 0.1  # enable/stop/zero acknowledgment deadline
PRIVATE_MODE_SWITCH_RETRIES = 3  # write run_mode + read-back-verify attempts

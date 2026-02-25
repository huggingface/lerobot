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

"""Configuration tables for Damiao motors."""

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


RUNNING_TIMEOUT = 0.001
PARAM_TIMEOUT = 0.01

STATE_CACHE_TTL_S = 0.02

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
    DM3507 = 0
    DM4310 = 1
    DM4310_48V = 2
    DM4340 = 3
    DM4340_48V = 4
    DM6006 = 5
    DM8006 = 6
    DM8009 = 7
    DM10010L = 8
    DM10010 = 9
    DMH3510 = 10
    DMH6215 = 11
    DMG6220 = 12


# Control modes
class ControlMode(IntEnum):
    MIT = 1
    POS_VEL = 2
    VEL = 3
    TORQUE_POS = 4


# Motor variable IDs (RID)
class MotorVariable(IntEnum):
    UV_VALUE = 0
    KT_VALUE = 1
    OT_VALUE = 2
    OC_VALUE = 3
    ACC = 4
    DEC = 5
    MAX_SPD = 6
    MST_ID = 7
    ESC_ID = 8
    TIMEOUT = 9
    CTRL_MODE = 10
    DAMP = 11
    INERTIA = 12
    HW_VER = 13
    SW_VER = 14
    SN = 15
    NPP = 16
    RS = 17
    LS = 18
    FLUX = 19
    GR = 20
    PMAX = 21
    VMAX = 22
    TMAX = 23
    I_BW = 24
    KP_ASR = 25
    KI_ASR = 26
    KP_APR = 27
    KI_APR = 28
    OV_VALUE = 29
    GREF = 30
    DETA = 31
    V_BW = 32
    IQ_C1 = 33
    VL_C1 = 34
    CAN_BR = 35
    SUB_VER = 36
    U_OFF = 50
    V_OFF = 51
    K1 = 52
    K2 = 53
    M_OFF = 54
    DIR = 55
    P_M = 80
    XOUT = 81


# Motor limit parameters [PMAX, VMAX, TMAX]
# PMAX: Maximum position (rad)
# VMAX: Maximum velocity (rad/s)
# TMAX: Maximum torque (N·m)
MOTOR_LIMIT_PARAMS = {
    MotorType.DM3507: (12.5, 30, 10),
    MotorType.DM4310: (12.5, 30, 10),
    MotorType.DM4310_48V: (12.5, 50, 10),
    MotorType.DM4340: (12.5, 8, 28),
    MotorType.DM4340_48V: (12.5, 10, 28),
    MotorType.DM6006: (12.5, 45, 20),
    MotorType.DM8006: (12.5, 45, 40),
    MotorType.DM8009: (12.5, 45, 54),
    MotorType.DM10010L: (12.5, 25, 200),
    MotorType.DM10010: (12.5, 20, 200),
    MotorType.DMH3510: (12.5, 280, 1),
    MotorType.DMH6215: (12.5, 45, 10),
    MotorType.DMG6220: (12.5, 45, 10),
}

# Motor model names
MODEL_NAMES = {
    MotorType.DM3507: "dm3507",
    MotorType.DM4310: "dm4310",
    MotorType.DM4310_48V: "dm4310_48v",
    MotorType.DM4340: "dm4340",
    MotorType.DM4340_48V: "dm4340_48v",
    MotorType.DM6006: "dm6006",
    MotorType.DM8006: "dm8006",
    MotorType.DM8009: "dm8009",
    MotorType.DM10010L: "dm10010l",
    MotorType.DM10010: "dm10010",
    MotorType.DMH3510: "dmh3510",
    MotorType.DMH6215: "dmh6215",
    MotorType.DMG6220: "dmg6220",
}

# Motor resolution table (encoder counts per revolution)
MODEL_RESOLUTION = {
    "dm3507": 65536,
    "dm4310": 65536,
    "dm4310_48v": 65536,
    "dm4340": 65536,
    "dm4340_48v": 65536,
    "dm6006": 65536,
    "dm8006": 65536,
    "dm8009": 65536,
    "dm10010l": 65536,
    "dm10010": 65536,
    "dmh3510": 65536,
    "dmh6215": 65536,
    "dmg6220": 65536,
}

# CAN baudrates supported by Damiao motors
AVAILABLE_BAUDRATES = [
    125000,  # 0: 125 kbps
    200000,  # 1: 200 kbps
    250000,  # 2: 250 kbps
    500000,  # 3: 500 kbps
    1000000,  # 4: 1 mbps (default for OpenArms)
    2000000,  # 5: 2 mbps
    2500000,  # 6: 2.5 mbps
    3200000,  # 7: 3.2 mbps
    4000000,  # 8: 4 mbps
    5000000,  # 9: 5 mbps
]
DEFAULT_BAUDRATE = 1000000  # 1 Mbps is standard for OpenArms

# Default timeout in milliseconds
DEFAULT_TIMEOUT_MS = 1000

# OpenArms specific configurations
# Based on: https://docs.openarm.dev/software/setup/configure-test
# OpenArms has 7 DOF per arm (14 total for dual arm)
OPENARMS_ARM_MOTOR_IDS = {
    "joint_1": {"send": 0x01, "recv": 0x11},  # J1 - Shoulder pan
    "joint_2": {"send": 0x02, "recv": 0x12},  # J2 - Shoulder lift
    "joint_3": {"send": 0x03, "recv": 0x13},  # J3 - Elbow flex
    "joint_4": {"send": 0x04, "recv": 0x14},  # J4 - Wrist flex
    "joint_5": {"send": 0x05, "recv": 0x15},  # J5 - Wrist roll
    "joint_6": {"send": 0x06, "recv": 0x16},  # J6 - Wrist pitch
    "joint_7": {"send": 0x07, "recv": 0x17},  # J7 - Wrist rotation
}

OPENARMS_GRIPPER_MOTOR_IDS = {
    "gripper": {"send": 0x08, "recv": 0x18},  # J8 - Gripper
}

# Default motor types for OpenArms
OPENARMS_DEFAULT_MOTOR_TYPES = {
    "joint_1": MotorType.DM8009,  # Shoulder pan - high torque
    "joint_2": MotorType.DM8009,  # Shoulder lift - high torque
    "joint_3": MotorType.DM4340,  # Shoulder rotation
    "joint_4": MotorType.DM4340,  # Elbow flex
    "joint_5": MotorType.DM4310,  # Wrist roll
    "joint_6": MotorType.DM4310,  # Wrist pitch
    "joint_7": MotorType.DM4310,  # Wrist rotation
    "gripper": MotorType.DM4310,  # Gripper
}

# MIT control parameter ranges
MIT_KP_RANGE = (0.0, 500.0)
MIT_KD_RANGE = (0.0, 5.0)

# CAN frame command IDs
CAN_CMD_ENABLE = 0xFC
CAN_CMD_DISABLE = 0xFD
CAN_CMD_SET_ZERO = 0xFE
CAN_CMD_REFRESH = 0xCC
CAN_CMD_QUERY_PARAM = 0x33
CAN_CMD_WRITE_PARAM = 0x55
CAN_CMD_SAVE_PARAM = 0xAA

# CAN ID for parameter operations
CAN_PARAM_ID = 0x7FF

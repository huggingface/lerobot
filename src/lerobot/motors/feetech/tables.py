# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

FIRMWARE_MAJOR_VERSION = (0, 1)
FIRMWARE_MINOR_VERSION = (1, 1)
MODEL_NUMBER = (3, 2)

# TODO(Steven): Consider doing the following:
# from enum import Enum
# class MyControlTableKey(Enum):
#   ID = "ID"
#   GOAL_SPEED = "Goal_Speed"
#   ...
#
# MY_CONTROL_TABLE ={
#   MyControlTableKey.ID.value: (5,1)
#   MyControlTableKey.GOAL_SPEED.value: (46, 2)
#   ...
# }
# This allows me do to:
# bus.write(MyControlTableKey.GOAL_SPEED, ...)
# Instead of:
# bus.write("Goal_Speed", ...)
# This is important for two reasons:
# 1. The linter will tell me if I'm trying to use an invalid key, instead of me realizing when I get the RunTimeError
# 2. We can change the value of the MyControlTableKey enums without impacting the client code

# data_name: (address, size_byte)
# http://doc.feetech.cn/#/prodinfodownload?srcType=FT-SMS-STS-emanual-229f4476422d4059abfb1cb0
STS_SMS_SERIES_CONTROL_TABLE = {
    # EPROM
    "Firmware_Major_Version": FIRMWARE_MAJOR_VERSION,  # read-only
    "Firmware_Minor_Version": FIRMWARE_MINOR_VERSION,  # read-only
    "Model_Number": MODEL_NUMBER,  # read-only
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay_Time": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1),
    "Min_Voltage_Limit": (15, 1),
    "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1),
    "Unloading_Condition": (19, 1),
    "LED_Alarm_Condition": (20, 1),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Minimum_Startup_Force": (24, 2),
    "CW_Dead_Zone": (26, 1),
    "CCW_Dead_Zone": (27, 1),
    "Protection_Current": (28, 2),
    "Angular_Resolution": (30, 1),
    "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1),
    "Protective_Torque": (34, 1),
    "Protection_Time": (35, 1),
    "Overload_Torque": (36, 1),
    "Velocity_closed_loop_P_proportional_coefficient": (37, 1),
    "Over_Current_Protection_Time": (38, 1),
    "Velocity_closed_loop_I_integral_coefficient": (39, 1),
    # SRAM
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Time": (44, 2),
    "Goal_Velocity": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),  # read-only
    "Present_Velocity": (58, 2),  # read-only
    "Present_Load": (60, 2),  # read-only
    "Present_Voltage": (62, 1),  # read-only
    "Present_Temperature": (63, 1),  # read-only
    "Status": (65, 1),  # read-only
    "Moving": (66, 1),  # read-only
    "Present_Current": (69, 2),  # read-only
    "Goal_Position_2": (71, 2),  # read-only
    # Factory
    "Moving_Velocity": (80, 1),
    "Moving_Velocity_Threshold": (80, 1),
    "DTs": (81, 1),  # (ms)
    "Velocity_Unit_factor": (82, 1),
    "Hts": (83, 1),  # (ns) valid for firmware >= 2.54, other versions keep 0
    "Maximum_Velocity_Limit": (84, 1),
    "Maximum_Acceleration": (85, 1),
    "Acceleration_Multiplier ": (86, 1),  # Acceleration multiplier in effect when acceleration is 0
}

# http://doc.feetech.cn/#/prodinfodownload?srcType=FT-SCSCL-emanual-cbcc8ab2e3384282a01d4bf3
SCS_SERIES_CONTROL_TABLE = {
    # EPROM
    "Firmware_Major_Version": FIRMWARE_MAJOR_VERSION,  # read-only
    "Firmware_Minor_Version": FIRMWARE_MINOR_VERSION,  # read-only
    "Model_Number": MODEL_NUMBER,  # read-only
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay_Time": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
    "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1),
    "Min_Voltage_Limit": (15, 1),
    "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1),
    "Unloading_Condition": (19, 1),
    "LED_Alarm_Condition": (20, 1),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Minimum_Startup_Force": (24, 2),
    "CW_Dead_Zone": (26, 1),
    "CCW_Dead_Zone": (27, 1),
    "Protective_Torque": (37, 1),
    "Protection_Time": (38, 1),
    # SRAM
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Running_Time": (44, 2),
    "Goal_Velocity": (46, 2),
    "Lock": (48, 1),
    "Present_Position": (56, 2),  # read-only
    "Present_Velocity": (58, 2),  # read-only
    "Present_Load": (60, 2),  # read-only
    "Present_Voltage": (62, 1),  # read-only
    "Present_Temperature": (63, 1),  # read-only
    "Sync_Write_Flag": (64, 1),  # read-only
    "Status": (65, 1),  # read-only
    "Moving": (66, 1),  # read-only
    # Factory
    "PWM_Maximum_Step": (78, 1),
    "Moving_Velocity_Threshold*50": (79, 1),
    "DTs": (80, 1),  # (ms)
    "Minimum_Velocity_Limit*50": (81, 1),
    "Maximum_Velocity_Limit*50": (82, 1),
    "Acceleration_2": (83, 1),  # don't know what that is
}

# http://doc.feetech.cn/#/prodinfodownload?srcType=FT-SMS-STS-emanual-229f4476422d4059abfb1cb0
HLS_SERIES_CONTROL_TABLE = {
    # Version Information (0-4) - read-only
    "Firmware_Major_Version": FIRMWARE_MAJOR_VERSION,  # (0, 1) read-only
    "Firmware_Minor_Version": FIRMWARE_MINOR_VERSION,  # (1, 1) read-only
    "End_Type": (2, 1),  # read-only - 0 represents little-endian storage
    "Model_Number": MODEL_NUMBER,  # (3, 2) read-only
    # EPROM configuration (5-39)
    "ID": (5, 1),  # Main ID - unique identifier on bus
    "Baud_Rate": (6, 1),  # 0-7 for different baud rates
    "Secondary_ID": (7, 1),  # Secondary ID for write instructions
    "Response_Status_Level": (8, 1),  # 0: limited response, 1: full response
    "Min_Position_Limit": (9, 2),  # 0-4094 (0.087 degrees per unit)
    "Max_Position_Limit": (11, 2),  # 1-4095 (0.087 degrees per unit)
    "Max_Temperature_Limit": (13, 1),  # 0-100 (°C)
    "Max_Voltage_Limit": (14, 1),  # 0-254 (0.1V per unit)
    "Min_Voltage_Limit": (15, 1),  # 0-254 (0.1V per unit)
    "Max_Torque_Limit": (16, 2),  # 0-1000 (0.1% per unit)
    "Phase": (18, 1),  # Special function byte for motor phase configuration
    "Unloading_Condition": (19, 1),  # Bit flags for protection conditions
    "LED_Alarm_Condition": (20, 1),  # Bit flags for LED alarm conditions
    "P_Coefficient": (21, 1),  # Position ring P proportional coefficient
    "D_Coefficient": (22, 1),  # Position ring D differential coefficient
    "I_Coefficient": (23, 1),  # Position ring I integral coefficient
    "Minimum_Startup_Force": (24, 1),  # 0-254 (0.1% per unit)
    "Point_Limit_Value": (25, 1),  # 0-254 - maximum point value = point_limit * 4
    "CW_Dead_Zone": (26, 1),  # 0-16 (0.087 degrees per unit)
    "CCW_Dead_Zone": (27, 1),  # 0-16 (0.087 degrees per unit)
    "Protection_Current": (28, 2),  # 0-2047 (6.5 mA per unit)
    "Angle_Resolution": (30, 1),  # 1-128 - amplification coefficient
    "Homing_Offset": (31, 2),  # -4095 to 4095 (0.087 degrees per unit)
    "Operating_Mode": (33, 1),  # 0: position, 1: speed, 2: current, 3: PWM
    "P_Coefficient_Curr": (34, 1),  # Current ring P proportional coefficient
    "I_Coefficient_Curr": (35, 1),  # Current ring I integral coefficient
    # Address 36 undefined
    "Speed_P_Coefficient": (37, 1),  # Speed closed-loop P proportional coefficient
    "Overcurrent_Protection_Time": (38, 1),  # 0-254 (10ms per unit)
    "Speed_I_Coefficient": (39, 1),  # Speed closed-loop I integral coefficient
    # SRAM control (40-55)
    "Torque_Enable": (40, 1),  # 0: off, 1: on, 2: damping
    "Acceleration": (41, 1),  # 0-254 (8.7 degrees/second² per unit)
    "Goal_Position": (42, 2),  # -32767 to 32767 (0.087 degrees per unit)
    "Target_Torque": (44, 2),  # -2047 to 2047 (6.5 mA per unit)
    "Goal_Velocity": (46, 2),  # -32767 to 32767 (0.732 RPM per unit)
    "Torque_Limit": (48, 2),  # 0-1000 (0.1% per unit)
    "P_Coefficient_Ring": (50, 1),  # Motor position ring proportional coefficient
    "D_Coefficient_Ring": (51, 1),  # Motor position ring differential coefficient
    "I_Coefficient_Ring": (52, 1),  # Motor position ring integral coefficient
    "km": (53, 1),  # 0: position+current dual loop, 1: position single loop
    # Address 54 undefined
    "Lock": (55, 1),  # 0: close write lock, 1: open write lock
    # SRAM feedback (56-73) - read-only
    "Present_Position": (56, 2),  # read-only - current absolute position
    "Present_Velocity": (58, 2),  # read-only - current motor rotation speed
    "Present_Load": (60, 2),  # read-only - current load (0.1% per unit)
    "Present_Voltage": (62, 1),  # read-only - current voltage (0.1V per unit)
    "Present_Temperature": (63, 1),  # read-only - current temperature (°C)
    "Async_Write_Flag": (64, 1),  # read-only - async write instruction flag
    "Status": (65, 1),  # read-only - servo status bit flags
    "Moving": (66, 1),  # read-only - movement status flags
    "Target_Position": (67, 2),  # read-only - current target position
    "Present_Current": (69, 2),  # read-only - current motor phase current (6.5 mA per unit)
    # Address 71 undefined
    "Present_Bias": (73, 2),  # read-only - current 0-point offset value
    # Factory parameters (77-86) - read-only
    "VFk_x10": (77, 1),  # read-only - factory parameter
    "vKgI": (78, 1),  # read-only - factory parameter
    "PFk_x10": (79, 1),  # read-only - factory parameter
    "Moving_Velocity_Threshold": (80, 1),  # read-only - factory parameter
    "DTs_ms": (81, 1),  # read-only - factory parameter
    "eFk_x10": (82, 1),  # read-only - factory parameter
    "Vk_ms": (83, 1),  # read-only - factory parameter
    "Maximum_Velocity_Limit": (84, 1),  # read-only - factory parameter
    "Maximum_Acceleration": (85, 1),  # read-only - factory parameter
    "Acceleration_Multiplier": (86, 1),  # read-only - factory parameter
}

# HLS series baud rate table (same as STS/SMS series)
HLS_SERIES_BAUDRATE_TABLE = {
    1_000_000: 0,
    500_000: 1,
    250_000: 2,
    128_000: 3,
    115_200: 4,
    76_800: 5,  # Note: HLS documentation mentions 76800 instead of 57600
    57_600: 6,
    38_400: 7,
}

STS_SMS_SERIES_BAUDRATE_TABLE = {
    1_000_000: 0,
    500_000: 1,
    250_000: 2,
    128_000: 3,
    115_200: 4,
    57_600: 5,
    38_400: 6,
    19_200: 7,
}

SCS_SERIES_BAUDRATE_TABLE = {
    1_000_000: 0,
    500_000: 1,
    250_000: 2,
    128_000: 3,
    115_200: 4,
    57_600: 5,
    38_400: 6,
    19_200: 7,
}

MODEL_CONTROL_TABLE = {
    "sts_series": STS_SMS_SERIES_CONTROL_TABLE,
    "scs_series": SCS_SERIES_CONTROL_TABLE,
    "sms_series": STS_SMS_SERIES_CONTROL_TABLE,
    "sts3215": STS_SMS_SERIES_CONTROL_TABLE,
    "sts3250": STS_SMS_SERIES_CONTROL_TABLE,
    "scs0009": SCS_SERIES_CONTROL_TABLE,
    "sm8512bl": STS_SMS_SERIES_CONTROL_TABLE,
    "hls3625": HLS_SERIES_CONTROL_TABLE,
}

MODEL_RESOLUTION = {
    "sts_series": 4096,
    "sms_series": 4096,
    "scs_series": 1024,
    "sts3215": 4096,
    "sts3250": 4096,
    "sm8512bl": 65536,
    "scs0009": 1024,
    "hls3625": 4096,
}

MODEL_BAUDRATE_TABLE = {
    "sts_series": STS_SMS_SERIES_BAUDRATE_TABLE,
    "sms_series": STS_SMS_SERIES_BAUDRATE_TABLE,
    "scs_series": SCS_SERIES_BAUDRATE_TABLE,
    "sm8512bl": STS_SMS_SERIES_BAUDRATE_TABLE,
    "sts3215": STS_SMS_SERIES_BAUDRATE_TABLE,
    "sts3250": STS_SMS_SERIES_BAUDRATE_TABLE,
    "scs0009": SCS_SERIES_BAUDRATE_TABLE,
    "hls3625": HLS_SERIES_BAUDRATE_TABLE,
}

# Sign-Magnitude encoding bits
STS_SMS_SERIES_ENCODINGS_TABLE = {
    "Homing_Offset": 11,
    "Goal_Velocity": 15,
    "Present_Velocity": 15,
}

# HLS series sign-magnitude encoding bits
HLS_SERIES_ENCODINGS_TABLE = {
    "Homing_Offset": 15,  # BIT15 represents positive/negative direction
    # "Goal_Position": 15,  # BIT15 represents positive/negative direction
    "Target_Torque": 15,  # BIT15 represents positive/negative direction in constant current mode
    "Goal_Velocity": 15,  # BIT15 represents positive/negative direction in constant speed mode
    # "Present_Position": 15,  # BIT15 represents positive/negative direction
    "Present_Velocity": 15,  # BIT15 represents positive/negative direction
    "Present_Current": 15,  # BIT15 represents positive/negative direction
    "Present_Load": 10,  # BIT10 represents positive/negative direction
}

MODEL_ENCODING_TABLE = {
    "sts_series": STS_SMS_SERIES_ENCODINGS_TABLE,
    "sms_series": STS_SMS_SERIES_ENCODINGS_TABLE,
    "scs_series": {},
    "sts3215": STS_SMS_SERIES_ENCODINGS_TABLE,
    "sts3250": STS_SMS_SERIES_ENCODINGS_TABLE,
    "sm8512bl": STS_SMS_SERIES_ENCODINGS_TABLE,
    "scs0009": {},
    "hls3625": HLS_SERIES_ENCODINGS_TABLE,
}

SCAN_BAUDRATES = [
    4_800,
    9_600,
    14_400,
    19_200,
    38_400,
    57_600,
    115_200,
    128_000,
    250_000,
    500_000,
    1_000_000,
]

MODEL_NUMBER_TABLE = {
    "sts3215": 777,
    "sts3250": 2825,
    "sm8512bl": 11272,
    "scs0009": 1284,
    "hls3625": 3338,
}

MODEL_PROTOCOL = {
    "sts_series": 0,
    "sms_series": 0,
    "scs_series": 1,
    "sts3215": 0,
    "sts3250": 0,
    "sm8512bl": 0,
    "scs0009": 1,
    "hls3625": 0,  # Uses FT-SCS protocol
}

# HLS series special byte bit flag definitions
# These are used to interpret the bit flags in Phase, Status, Unloading_Condition, and LED_Alarm_Condition

# Phase byte (address 18) bit meanings:
# BIT0 (1): Current driving direction phase (0: forward, 1: reverse) - invalid in firmware 3.42+
# BIT1 (2): Current feedback direction phase (0: forward, 1: reverse)
# BIT2 (4): Drive bridge direction phase (0: forward, 1: reverse)
# BIT3 (8): Speed feedback direction phase (0: forward, 1: reverse)
# BIT4 (16): Angle feedback mode (0: single-circle angle, 1: full angle)
# BIT5 (32): Driver bridge configuration (0: independent H bridge, 1: integrated H bridge)
# BIT6 (64): PWM frequency (0: 24kHz, 1: 16kHz)
# BIT7 (128): Position feedback direction phase (0: forward, 1: reverse)

# Status byte (address 65) bit meanings (0: normal, 1: abnormal):
# BIT0 (1): Voltage status
# BIT1 (2): Magnetic coding state
# BIT2 (4): Temperature status
# BIT3 (8): Current status
# BIT4 (16): Reserved
# BIT5 (32): Load status
# BIT6 (64): Reserved
# BIT7 (128): Reserved

# Unloading_Condition byte (address 19) bit meanings (0: off, 1: on):
# BIT0 (1): Voltage protection
# BIT1 (2): Magnetic coding protection
# BIT2 (4): Overheating protection
# BIT3 (8): Overcurrent protection
# BIT4 (16): Reserved
# BIT5 (32): Load overload protection
# BIT6 (64): Reserved
# BIT7 (128): Reserved

# LED_Alarm_Condition byte (address 20) bit meanings (0: off, 1: on):
# BIT0 (1): Voltage alarm
# BIT1 (2): Magnetic coding alarm
# BIT2 (4): Overheating alarm
# BIT3 (8): Overcurrent alarm
# BIT4 (16): Reserved
# BIT5 (32): Load overload alarm
# BIT6 (64): Reserved
# BIT7 (128): Reserved

# Moving byte (address 66) bit meanings:
# BIT0 (1): Motor movement status (1: moving, 0: stopped)
# BIT1 (2): Target position status (0: reached target, 1: not reached target)

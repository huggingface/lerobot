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
}

MODEL_RESOLUTION = {
    "sts_series": 4096,
    "sms_series": 4096,
    "scs_series": 1024,
    "sts3215": 4096,
    "sts3250": 4096,
    "sm8512bl": 4096,
    "scs0009": 1024,
}

MODEL_BAUDRATE_TABLE = {
    "sts_series": STS_SMS_SERIES_BAUDRATE_TABLE,
    "sms_series": STS_SMS_SERIES_BAUDRATE_TABLE,
    "scs_series": SCS_SERIES_BAUDRATE_TABLE,
    "sm8512bl": STS_SMS_SERIES_BAUDRATE_TABLE,
    "sts3215": STS_SMS_SERIES_BAUDRATE_TABLE,
    "sts3250": STS_SMS_SERIES_BAUDRATE_TABLE,
    "scs0009": SCS_SERIES_BAUDRATE_TABLE,
}

# Sign-Magnitude encoding bits
STS_SMS_SERIES_ENCODINGS_TABLE = {
    "Homing_Offset": 11,
    "Goal_Position": 15,
    "Goal_Velocity": 15,
    "Goal_Speed": 15,
    "Present_Position": 15,
    "Present_Velocity": 15,
    "Present_Speed": 15,
}

MODEL_ENCODING_TABLE = {
    "sts_series": STS_SMS_SERIES_ENCODINGS_TABLE,
    "sms_series": STS_SMS_SERIES_ENCODINGS_TABLE,
    "scs_series": {},
    "sts3215": STS_SMS_SERIES_ENCODINGS_TABLE,
    "sts3250": STS_SMS_SERIES_ENCODINGS_TABLE,
    "sm8512bl": STS_SMS_SERIES_ENCODINGS_TABLE,
    "scs0009": {},
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
}

MODEL_PROTOCOL = {
    "sts_series": 0,
    "sms_series": 0,
    "scs_series": 1,
    "sts3215": 0,
    "sts3250": 0,
    "sm8512bl": 0,
    "scs0009": 1,
}

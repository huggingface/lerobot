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


# {data_name: (address, size_byte)}
# https://emanual.robotis.com/docs/en/dxl/x/{MODEL}/#control-table
X_SERIES_CONTROL_TABLE = {
    "Model_Number": (0, 2),
    "Model_Information": (2, 4),
    "Firmware_Version": (6, 1),
    "ID": (7, 1),
    "Baud_Rate": (8, 1),
    "Return_Delay_Time": (9, 1),
    "Drive_Mode": (10, 1),
    "Operating_Mode": (11, 1),
    "Secondary_ID": (12, 1),
    "Protocol_Type": (13, 1),
    "Homing_Offset": (20, 4),
    "Moving_Threshold": (24, 4),
    "Temperature_Limit": (31, 1),
    "Max_Voltage_Limit": (32, 2),
    "Min_Voltage_Limit": (34, 2),
    "PWM_Limit": (36, 2),
    "Current_Limit": (38, 2),
    "Acceleration_Limit": (40, 4),
    "Velocity_Limit": (44, 4),
    "Max_Position_Limit": (48, 4),
    "Min_Position_Limit": (52, 4),
    "Shutdown": (63, 1),
    "Torque_Enable": (64, 1),
    "LED": (65, 1),
    "Status_Return_Level": (68, 1),
    "Registered_Instruction": (69, 1),
    "Hardware_Error_Status": (70, 1),
    "Velocity_I_Gain": (76, 2),
    "Velocity_P_Gain": (78, 2),
    "Position_D_Gain": (80, 2),
    "Position_I_Gain": (82, 2),
    "Position_P_Gain": (84, 2),
    "Feedforward_2nd_Gain": (88, 2),
    "Feedforward_1st_Gain": (90, 2),
    "Bus_Watchdog": (98, 1),
    "Goal_PWM": (100, 2),
    "Goal_Current": (102, 2),
    "Goal_Velocity": (104, 4),
    "Profile_Acceleration": (108, 4),
    "Profile_Velocity": (112, 4),
    "Goal_Position": (116, 4),
    "Realtime_Tick": (120, 2),
    "Moving": (122, 1),
    "Moving_Status": (123, 1),
    "Present_PWM": (124, 2),
    "Present_Current": (126, 2),
    "Present_Velocity": (128, 4),
    "Present_Position": (132, 4),
    "Velocity_Trajectory": (136, 4),
    "Position_Trajectory": (140, 4),
    "Present_Input_Voltage": (144, 2),
    "Present_Temperature": (146, 1),
}

# https://emanual.robotis.com/docs/en/dxl/x/{MODEL}/#baud-rate8
X_SERIES_BAUDRATE_TABLE = {
    9_600: 0,
    57_600: 1,
    115_200: 2,
    1_000_000: 3,
    2_000_000: 4,
    3_000_000: 5,
    4_000_000: 6,
}

# {data_name: size_byte}
X_SERIES_ENCODINGS_TABLE = {
    "Homing_Offset": X_SERIES_CONTROL_TABLE["Homing_Offset"][1],
    "Goal_PWM": X_SERIES_CONTROL_TABLE["Goal_PWM"][1],
    "Goal_Current": X_SERIES_CONTROL_TABLE["Goal_Current"][1],
    "Goal_Velocity": X_SERIES_CONTROL_TABLE["Goal_Velocity"][1],
    "Present_PWM": X_SERIES_CONTROL_TABLE["Present_PWM"][1],
    "Present_Current": X_SERIES_CONTROL_TABLE["Present_Current"][1],
    "Present_Velocity": X_SERIES_CONTROL_TABLE["Present_Velocity"][1],
}

MODEL_ENCODING_TABLE = {
    "x_series": X_SERIES_ENCODINGS_TABLE,
    "xl330-m077": X_SERIES_ENCODINGS_TABLE,
    "xl330-m288": X_SERIES_ENCODINGS_TABLE,
    "xl430-w250": X_SERIES_ENCODINGS_TABLE,
    "xm430-w350": X_SERIES_ENCODINGS_TABLE,
    "xm540-w270": X_SERIES_ENCODINGS_TABLE,
    "xc430-w150": X_SERIES_ENCODINGS_TABLE,
}

# {model: model_resolution}
# https://emanual.robotis.com/docs/en/dxl/x/{MODEL}/#specifications
MODEL_RESOLUTION = {
    "x_series": 4096,
    "xl330-m077": 4096,
    "xl330-m288": 4096,
    "xl430-w250": 4096,
    "xm430-w350": 4096,
    "xm540-w270": 4096,
    "xc430-w150": 4096,
}

# {model: model_number}
# https://emanual.robotis.com/docs/en/dxl/x/{MODEL}/#control-table-of-eeprom-area
MODEL_NUMBER_TABLE = {
    "xl330-m077": 1190,
    "xl330-m288": 1200,
    "xl430-w250": 1060,
    "xm430-w350": 1020,
    "xm540-w270": 1120,
    "xc430-w150": 1070,
}

# {model: available_operating_modes}
# https://emanual.robotis.com/docs/en/dxl/x/{MODEL}/#operating-mode11
MODEL_OPERATING_MODES = {
    "xl330-m077": [0, 1, 3, 4, 5, 16],
    "xl330-m288": [0, 1, 3, 4, 5, 16],
    "xl430-w250": [1, 3, 4, 16],
    "xm430-w350": [0, 1, 3, 4, 5, 16],
    "xm540-w270": [0, 1, 3, 4, 5, 16],
    "xc430-w150": [1, 3, 4, 16],
}

MODEL_CONTROL_TABLE = {
    "x_series": X_SERIES_CONTROL_TABLE,
    "xl330-m077": X_SERIES_CONTROL_TABLE,
    "xl330-m288": X_SERIES_CONTROL_TABLE,
    "xl430-w250": X_SERIES_CONTROL_TABLE,
    "xm430-w350": X_SERIES_CONTROL_TABLE,
    "xm540-w270": X_SERIES_CONTROL_TABLE,
    "xc430-w150": X_SERIES_CONTROL_TABLE,
}

MODEL_BAUDRATE_TABLE = {
    "x_series": X_SERIES_BAUDRATE_TABLE,
    "xl330-m077": X_SERIES_BAUDRATE_TABLE,
    "xl330-m288": X_SERIES_BAUDRATE_TABLE,
    "xl430-w250": X_SERIES_BAUDRATE_TABLE,
    "xm430-w350": X_SERIES_BAUDRATE_TABLE,
    "xm540-w270": X_SERIES_BAUDRATE_TABLE,
    "xc430-w150": X_SERIES_BAUDRATE_TABLE,
}

AVAILABLE_BAUDRATES = [
    9_600,
    19_200,
    38_400,
    57_600,
    115_200,
    230_400,
    460_800,
    500_000,
    576_000,
    921_600,
    1_000_000,
    1_152_000,
    2_000_000,
    2_500_000,
    3_000_000,
    3_500_000,
    4_000_000,
]

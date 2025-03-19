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

# TODO(aliberts): Should we implement FastSyncRead/Write?
# https://github.com/ROBOTIS-GIT/DynamixelSDK/pull/643
# https://github.com/ROBOTIS-GIT/DynamixelSDK/releases/tag/3.8.2
# https://emanual.robotis.com/docs/en/dxl/protocol2/#fast-sync-read-0x8a
# -> Need to check compatibility across models

from copy import deepcopy

import numpy as np

from ..motors_bus import MotorsBus

PROTOCOL_VERSION = 2.0
BAUDRATE = 1_000_000
DEFAULT_TIMEOUT_MS = 1000

MAX_ID_RANGE = 252

# The following bounds define the lower and upper joints range (after calibration).
# For joints in degree (i.e. revolute joints), their nominal range is [-180, 180] degrees
# which corresponds to a half rotation on the left and half rotation on the right.
# Some joints might require higher range, so we allow up to [-270, 270] degrees until
# an error is raised.
LOWER_BOUND_DEGREE = -270
UPPER_BOUND_DEGREE = 270

# For joints in percentage (i.e. joints that move linearly like the prismatic joint of a gripper),
# their nominal range is [0, 100] %. For instance, for Aloha gripper, 0% is fully
# closed, and 100% is fully open. To account for slight calibration issue, we allow up to
# [-10, 110] until an error is raised.
LOWER_BOUND_LINEAR = -10
UPPER_BOUND_LINEAR = 110

HALF_TURN_DEGREE = 180

# https://emanual.robotis.com/docs/en/dxl/x/xl330-m077
# https://emanual.robotis.com/docs/en/dxl/x/xl330-m288
# https://emanual.robotis.com/docs/en/dxl/x/xl430-w250
# https://emanual.robotis.com/docs/en/dxl/x/xm430-w350
# https://emanual.robotis.com/docs/en/dxl/x/xm540-w270
# https://emanual.robotis.com/docs/en/dxl/x/xc430-w150

# data_name: (address, size_byte)
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

X_SERIES_BAUDRATE_TABLE = {
    0: 9_600,
    1: 57_600,
    2: 115_200,
    3: 1_000_000,
    4: 2_000_000,
    5: 3_000_000,
    6: 4_000_000,
}

CALIBRATION_REQUIRED = ["Goal_Position", "Present_Position"]
CONVERT_UINT32_TO_INT32_REQUIRED = ["Goal_Position", "Present_Position"]

MODEL_CONTROL_TABLE = {
    "x_series": X_SERIES_CONTROL_TABLE,
    "xl330-m077": X_SERIES_CONTROL_TABLE,
    "xl330-m288": X_SERIES_CONTROL_TABLE,
    "xl430-w250": X_SERIES_CONTROL_TABLE,
    "xm430-w350": X_SERIES_CONTROL_TABLE,
    "xm540-w270": X_SERIES_CONTROL_TABLE,
    "xc430-w150": X_SERIES_CONTROL_TABLE,
}

MODEL_RESOLUTION = {
    "x_series": 4096,
    "xl330-m077": 4096,
    "xl330-m288": 4096,
    "xl430-w250": 4096,
    "xm430-w350": 4096,
    "xm540-w270": 4096,
    "xc430-w150": 4096,
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

NUM_READ_RETRY = 10
NUM_WRITE_RETRY = 10


def convert_degrees_to_steps(degrees: float | np.ndarray, models: str | list[str]) -> np.ndarray:
    """This function converts the degree range to the step range for indicating motors rotation.
    It assumes a motor achieves a full rotation by going from -180 degree position to +180.
    The motor resolution (e.g. 4096) corresponds to the number of steps needed to achieve a full rotation.
    """
    resolutions = [MODEL_RESOLUTION[model] for model in models]
    steps = degrees / 180 * np.array(resolutions) / 2
    steps = steps.astype(int)
    return steps


class DynamixelMotorsBus(MotorsBus):
    """
    The Dynamixel implementation for a MotorsBus. It relies on the python dynamixel sdk to communicate with
    the motors. For more info, see the Dynamixel SDK Documentation:
    https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python_read_write_protocol_2_0/#python-read-write-protocol-20
    """

    model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
    model_resolution_table = deepcopy(MODEL_RESOLUTION)
    model_baudrate_table = deepcopy(MODEL_BAUDRATE_TABLE)
    calibration_required = deepcopy(CALIBRATION_REQUIRED)
    default_timeout = DEFAULT_TIMEOUT_MS

    def __init__(
        self,
        port: str,
        motors: dict[str, tuple[int, str]],
    ):
        super().__init__(port, motors)
        import dynamixel_sdk as dxl

        self.port_handler = dxl.PortHandler(self.port)
        self.packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)
        self.reader = dxl.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
        self.writer = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)

    def broadcast_ping(self) -> tuple[list, int]:
        data_list, comm = self.packet_handler.broadcastPing(self.port_handler)
        # TODO(aliberts): translate data_list into {id: model}
        return data_list, comm

    def calibrate_values(self, ids_values: dict[int, int]) -> dict[int, float]:
        # TODO
        return ids_values

    def uncalibrate_values(self, ids_values: dict[int, float]) -> dict[int, int]:
        # TODO
        return ids_values

    def _is_comm_success(self, comm: int) -> bool:
        import dynamixel_sdk as dxl

        return comm == dxl.COMM_SUCCESS

    @staticmethod
    def split_int_bytes(value: int, n_bytes: int) -> list[int]:
        # Validate input
        if value < 0:
            raise ValueError(f"Negative values are not allowed: {value}")

        max_value = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}.get(n_bytes)
        if max_value is None:
            raise NotImplementedError(f"Unsupported byte size: {n_bytes}. Expected [1, 2, 4].")

        if value > max_value:
            raise ValueError(f"Value {value} exceeds the maximum for {n_bytes} bytes ({max_value}).")

        import dynamixel_sdk as dxl

        # Note: No need to convert back into unsigned int, since this byte preprocessing
        # already handles it for us.
        if n_bytes == 1:
            data = [
                dxl.DXL_LOBYTE(dxl.DXL_LOWORD(value)),
            ]
        elif n_bytes == 2:
            data = [
                dxl.DXL_LOBYTE(dxl.DXL_LOWORD(value)),
                dxl.DXL_HIBYTE(dxl.DXL_LOWORD(value)),
            ]
        elif n_bytes == 4:
            data = [
                dxl.DXL_LOBYTE(dxl.DXL_LOWORD(value)),
                dxl.DXL_HIBYTE(dxl.DXL_LOWORD(value)),
                dxl.DXL_LOBYTE(dxl.DXL_HIWORD(value)),
                dxl.DXL_HIBYTE(dxl.DXL_HIWORD(value)),
            ]
        return data

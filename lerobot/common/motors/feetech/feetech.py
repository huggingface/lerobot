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

from copy import deepcopy

import numpy as np

from ..motors_bus import (
    CalibrationMode,
    JointOutOfRangeError,
    MotorsBus,
    assert_same_address,
    get_group_sync_key,
)

PROTOCOL_VERSION = 0
BAUDRATE = 1_000_000
TIMEOUT_MS = 1000

MAX_ID_RANGE = 252

# For joints in percentage (i.e. joints that move linearly like the prismatic joint of a gripper),
# their nominal range is [0, 100] %. For instance, for Aloha gripper, 0% is fully
# closed, and 100% is fully open. To account for slight calibration issue, we allow up to
# [-10, 110] until an error is raised.
LOWER_BOUND_LINEAR = -10
UPPER_BOUND_LINEAR = 110

HALF_TURN_DEGREE = 180

# See this link for STS3215 Memory Table:
# https://docs.google.com/spreadsheets/d/1GVs7W1VS1PqdhA1nW-abeyAHhTUxKUdR/edit?usp=sharing&ouid=116566590112741600240&rtpof=true&sd=true
# data_name: (address, size_byte)
SCS_SERIES_CONTROL_TABLE = {
    "Model": (3, 2),
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Angle_Limit": (9, 2),
    "Max_Angle_Limit": (11, 2),
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
    "Offset": (31, 2),
    "Mode": (33, 1),
    "Protective_Torque": (34, 1),
    "Protection_Time": (35, 1),
    "Overload_Torque": (36, 1),
    "Speed_closed_loop_P_proportional_coefficient": (37, 1),
    "Over_Current_Protection_Time": (38, 1),
    "Velocity_closed_loop_I_integral_coefficient": (39, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Time": (44, 2),
    "Goal_Speed": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
    "Present_Speed": (58, 2),
    "Present_Load": (60, 2),
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Status": (65, 1),
    "Moving": (66, 1),
    "Present_Current": (69, 2),
    # Not in the Memory Table
    "Maximum_Acceleration": (85, 2),
}

SCS_SERIES_BAUDRATE_TABLE = {
    0: 1_000_000,
    1: 500_000,
    2: 250_000,
    3: 128_000,
    4: 115_200,
    5: 57_600,
    6: 38_400,
    7: 19_200,
}

CALIBRATION_REQUIRED = ["Goal_Position", "Present_Position"]

MODEL_CONTROL_TABLE = {
    "scs_series": SCS_SERIES_CONTROL_TABLE,
    "sts3215": SCS_SERIES_CONTROL_TABLE,
}

MODEL_RESOLUTION = {
    "scs_series": 4096,
    "sts3215": 4096,
}

MODEL_BAUDRATE_TABLE = {
    "scs_series": SCS_SERIES_BAUDRATE_TABLE,
    "sts3215": SCS_SERIES_BAUDRATE_TABLE,
}

# High number of retries is needed for feetech compared to dynamixel motors.
NUM_READ_RETRY = 20
NUM_WRITE_RETRY = 20


def convert_ticks_to_degrees(ticks, model):
    resolutions = MODEL_RESOLUTION[model]
    # Convert the ticks to degrees
    return ticks * (360.0 / resolutions)


def convert_degrees_to_ticks(degrees, model):
    resolutions = MODEL_RESOLUTION[model]
    # Convert degrees to motor ticks
    return int(degrees * (resolutions / 360.0))


def adjusted_to_homing_ticks(raw_motor_ticks: int, model: str, motorbus, motor_id: int) -> int:
    """
    Takes a raw reading [0..(res-1)] (e.g. 0..4095) and shifts it so that '2048'
    becomes 0 in the homed coordinate system ([-2048..+2047] for 4096 resolution).
    """
    resolutions = MODEL_RESOLUTION[model]

    # Shift raw ticks by half-resolution so 2048 -> 0, then wrap [0..res-1].
    ticks = (raw_motor_ticks - (resolutions // 2)) % resolutions

    # If above halfway, fold it into negative territory => [-2048..+2047].
    if ticks > (resolutions // 2):
        ticks -= resolutions

    # Flip sign if drive_mode is set.
    drive_mode = 0
    if motorbus.calibration is not None:
        drive_mode = motorbus.calibration["drive_mode"][motor_id - 1]

    if drive_mode:
        ticks *= -1

    return ticks


def adjusted_to_motor_ticks(adjusted_pos: int, model: str, motorbus, motor_id: int) -> int:
    """
    Inverse of adjusted_to_homing_ticks(). Takes a 'homed' position in [-2048..+2047]
    and recovers the raw [0..(res-1)] ticks with 2048 as midpoint.
    """
    # Flip sign if drive_mode was set.
    drive_mode = 0
    if motorbus.calibration is not None:
        drive_mode = motorbus.calibration["drive_mode"][motor_id - 1]

    if drive_mode:
        adjusted_pos *= -1

    resolutions = MODEL_RESOLUTION[model]

    # Shift by +half-resolution and wrap into [0..res-1].
    # This undoes the earlier shift by -half-resolution.
    ticks = (adjusted_pos + (resolutions // 2)) % resolutions

    return ticks


def convert_to_bytes(value, n_bytes: int):
    import scservo_sdk as scs

    # Note: No need to convert back into unsigned int, since this byte preprocessing
    # already handles it for us.
    if n_bytes == 1:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
        ]
    elif n_bytes == 2:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
        ]
    elif n_bytes == 4:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),
        ]
    else:
        raise NotImplementedError(
            f"Value of the number of bytes to be sent is expected to be in [1, 2, 4], but "
            f"{n_bytes} is provided instead."
        )
    return data


class FeetechMotorsBus(MotorsBus):
    """
    The FeetechMotorsBus class allows to efficiently read and write to the attached motors. It relies on the
    python feetech sdk to communicate with the motors, which is itself based on the dynamixel sdk.
    """

    model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
    model_resolution_table = deepcopy(MODEL_RESOLUTION)
    model_baudrate_table = deepcopy(MODEL_BAUDRATE_TABLE)

    def __init__(
        self,
        port: str,
        motors: dict[str, tuple[int, str]],
    ):
        super().__init__(port, motors)

    def _set_handlers(self):
        import scservo_sdk as scs

        self.port_handler = scs.PortHandler(self.port)
        self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION)

    def _set_timeout(self, timeout: int = TIMEOUT_MS):
        self.port_handler.setPacketTimeoutMillis(timeout)

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from unsigned int32 original range [0, 2**32] to signed float32 range
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                motor_idx, model = self.motors[name]

                # Convert raw motor ticks to homed ticks, then convert the homed ticks to degrees
                values[i] = adjusted_to_homing_ticks(values[i], model, self, motor_idx)
                values[i] = convert_ticks_to_degrees(values[i], model)

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Rescale the present position to a nominal range [0, 100] %,
                # useful for joints with linear motions like Aloha gripper
                values[i] = (values[i] - start_pos) / (end_pos - start_pos) * 100

                if (values[i] < LOWER_BOUND_LINEAR) or (values[i] > UPPER_BOUND_LINEAR):
                    raise JointOutOfRangeError(
                        f"Wrong motor position range detected for {name}. "
                        f"Expected to be in nominal range of [0, 100] % (a full linear translation), "
                        f"with a maximum range of [{LOWER_BOUND_LINEAR}, {UPPER_BOUND_LINEAR}] % to account for some imprecision during calibration, "
                        f"but present value is {values[i]} %. "
                        "This might be due to a cable connection issue creating an artificial jump in motor values. "
                        "You need to recalibrate by running: `python lerobot/scripts/control_robot.py calibrate`"
                    )

        return values

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Inverse of `apply_calibration`."""
        if motor_names is None:
            motor_names = self.motor_names

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                motor_idx, model = self.motors[name]

                # Convert degrees to homed ticks, then convert the homed ticks to raw ticks
                values[i] = convert_degrees_to_ticks(values[i], model)
                values[i] = adjusted_to_motor_ticks(values[i], model, self, motor_idx)

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Convert from nominal lnear range of [0, 100] % to
                # actual motor range of values which can be arbitrary.
                values[i] = values[i] / 100 * (end_pos - start_pos) + start_pos

        values = np.round(values).astype(np.int32)
        return values

    def read_with_motor_ids(self, motor_models, motor_ids, data_name, num_retry=NUM_READ_RETRY):
        import scservo_sdk as scs

        return_list = True
        if not isinstance(motor_ids, list):
            return_list = False
            motor_ids = [motor_ids]

        assert_same_address(self.model_ctrl_table, self.motor_models, data_name)
        addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
        group = scs.GroupSyncRead(self.port_handler, self.packet_handler, addr, bytes)
        for idx in motor_ids:
            group.addParam(idx)

        for _ in range(num_retry):
            comm = group.txRxPacket()
            if comm == scs.COMM_SUCCESS:
                break

        if comm != scs.COMM_SUCCESS:
            raise ConnectionError(
                f"Read failed due to communication error on port {self.port_handler.port_name} for indices {motor_ids}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

        values = []
        for idx in motor_ids:
            value = group.getData(idx, addr, bytes)
            values.append(value)

        if return_list:
            return values
        else:
            return values[0]

    def _read(self, data_name: str, motor_names: list[str]):
        import scservo_sdk as scs

        motor_ids = []
        models = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)

        assert_same_address(self.model_ctrl_table, models, data_name)
        addr, bytes = self.model_ctrl_table[model][data_name]
        group_key = get_group_sync_key(data_name, motor_names)

        if data_name not in self.group_readers:
            # Very Important to flush the buffer!
            self.port_handler.ser.reset_output_buffer()
            self.port_handler.ser.reset_input_buffer()

            # Create new group reader
            self.group_readers[group_key] = scs.GroupSyncRead(
                self.port_handler, self.packet_handler, addr, bytes
            )
            for idx in motor_ids:
                self.group_readers[group_key].addParam(idx)

        for _ in range(NUM_READ_RETRY):
            comm = self.group_readers[group_key].txRxPacket()
            if comm == scs.COMM_SUCCESS:
                break

        if comm != scs.COMM_SUCCESS:
            raise ConnectionError(
                f"Read failed due to communication error on port {self.port} for group_key {group_key}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

        values = []
        for idx in motor_ids:
            value = self.group_readers[group_key].getData(idx, addr, bytes)
            values.append(value)

        values = np.array(values)

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.apply_calibration(values, motor_names)

        return values

    def write_with_motor_ids(self, motor_models, motor_ids, data_name, values, num_retry=NUM_WRITE_RETRY):
        import scservo_sdk as scs

        if not isinstance(motor_ids, list):
            motor_ids = [motor_ids]
        if not isinstance(values, list):
            values = [values]

        assert_same_address(self.model_ctrl_table, motor_models, data_name)
        addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
        group = scs.GroupSyncWrite(self.port_handler, self.packet_handler, addr, bytes)
        for idx, value in zip(motor_ids, values, strict=True):
            data = convert_to_bytes(value, bytes)
            group.addParam(idx, data)

        for _ in range(num_retry):
            comm = group.txPacket()
            if comm == scs.COMM_SUCCESS:
                break

        if comm != scs.COMM_SUCCESS:
            raise ConnectionError(
                f"Write failed due to communication error on port {self.port_handler.port_name} for indices {motor_ids}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

    def _write(self, data_name: str, values: list[int], motor_names: list[str]) -> None:
        import scservo_sdk as scs

        motor_ids = []
        models = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.revert_calibration(values, motor_names)

        assert_same_address(self.model_ctrl_table, models, data_name)
        addr, bytes = self.model_ctrl_table[model][data_name]
        group_key = get_group_sync_key(data_name, motor_names)

        init_group = data_name not in self.group_readers
        if init_group:
            self.group_writers[group_key] = scs.GroupSyncWrite(
                self.port_handler, self.packet_handler, addr, bytes
            )

        for idx, value in zip(motor_ids, values, strict=True):
            data = convert_to_bytes(value, bytes)
            if init_group:
                self.group_writers[group_key].addParam(idx, data)
            else:
                self.group_writers[group_key].changeParam(idx, data)

        comm = self.group_writers[group_key].txPacket()
        if comm != scs.COMM_SUCCESS:
            raise ConnectionError(
                f"Write failed due to communication error on port {self.port} for group_key {group_key}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

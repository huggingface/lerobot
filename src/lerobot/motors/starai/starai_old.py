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

import enum
import logging
import math
import time
import traceback
from copy import deepcopy
import struct

import numpy as np
import tqdm

from lerobot.common.robot_devices.motors.configs import StaraiMotorsBusConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc
import fashionstar_uart_sdk as uservo
import serial


BAUDRATE = 1_000_000
TIMEOUT_MS = 1000

MAX_ID_RANGE = 252

# The following bounds define the lower and upper joints range (after calibration).
# For joints in degree (i.e. revolute joints), their nominal range is [-180, 180] degrees
# which corresponds to a half rotation on the left and half rotation on the right.
# Some joints might require higher range, so we allow up to [-270, 270] degrees until
# an error is raised.
LOWER_BOUND_DEGREE = -360
UPPER_BOUND_DEGREE = 360
# For joints in percentage (i.e. joints that move linearly like the prismatic joint of a gripper),
# their nominal range is [0, 100] %. For instance, for Aloha gripper, 0% is fully
# closed, and 100% is fully open. To account for slight calibration issue, we allow up to
# [-10, 110] until an error is raised.
LOWER_BOUND_LINEAR = -40
UPPER_BOUND_LINEAR = 150
HALF_TURN_DEGREE = 90

# data_name: (address, size_byte)

# U_SERIES_BAUDRATE_TABLE = {
#     0: 115_200,
#     1: 1_000_000,
# }

CALIBRATION_REQUIRED = ["Goal_Position", "Present_Position"]
CONVERT_UINT32_TO_INT32_REQUIRED = [""]
COMM_SUCCESS = 128



NUM_READ_RETRY = 10
NUM_WRITE_RETRY = 10


def get_group_sync_key(data_name, motor_names):
    group_key = f"{data_name}_" + "_".join(motor_names)
    return group_key


def get_result_name(fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    rslt_name = f"{fn_name}_{group_key}"
    return rslt_name


def get_queue_name(fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    queue_name = f"{fn_name}_{group_key}"
    return queue_name


def get_log_name(var_name, fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    log_name = f"{var_name}_{fn_name}_{group_key}"
    return log_name


class TorqueMode(enum.Enum):
    ENABLED = 1
    DISABLED = 0


class DriveMode(enum.Enum):
    NON_INVERTED = 0
    INVERTED = 1


class CalibrationMode(enum.Enum):
    # Joints with rotational motions are expressed in degrees in nominal range of [-180, 180]
    DEGREE = 0
    # Joints with linear motions (like gripper of Aloha) are expressed in nominal range of [0, 100]
    LINEAR = 1


class JointOutOfRangeError(Exception):
    def __init__(self, message="Joint is out of range"):
        self.message = message
        super().__init__(self.message)


class StaraiMotorsBus:

    def __init__(
        self,
        config: StaraiMotorsBusConfig,
    ):
        self.port = config.port
        self.motors = config.motors
        self.mock = config.mock
        self.interval = config.interval


        self.port_handler = None
        self.packet_handler = None
        self.calibration = None
        self.is_connected = False
        self.group_readers = {}
        self.group_writers = {}
        self.logs = {}
        self.gripper_degree_record = 0.0

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"StaraiMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        if self.mock:
            # import tests.motors.mock_dynamixel_sdk as dxl
            raise RobotDeviceAlreadyConnectedError(
                f"mock is not supported."
            )


        self.uart = serial.Serial(port=self.port,baudrate=BAUDRATE,parity=serial.PARITY_NONE,stopbits=1,bytesize=8,timeout=0)
        motor_ids = []
        try:
            self.port_handler = uservo.UartServoManager(self.uart)
            motor_names = self.motor_names

            # for name in motor_names:
            #     motor_idx, model = self.motors[name]
            #     motor_ids.append(motor_idx)
                # self.port_handler.disable_torque(motor_idx)
            time.sleep(0.005)
            self.port_handler.reset_multi_turn_angle(0xff)
            time.sleep(0.01)

        except Exception:
            traceback.print_exc()
            print(
                "\nTry running `python lerobot/scripts/find_motors_bus_port.py` to make sure you are using the correct port.\n"
            )
            raise OSError(f"Failed to open port '{self.port}'.")
        time.sleep(1)
        self.port_handler.send_sync_servo_monitor(motor_ids)

        # Allow to read and write
        self.is_connected = True

    def reconnect(self):

        if self.mock:
            # import tests.motors.mock_dynamixel_sdk as dxl
            raise RobotDeviceAlreadyConnectedError(
                f"mock is not supported."
            )

        self.uart = serial.Serial(port=self.port,baudrate=BAUDRATE,parity=serial.PARITY_NONE,stopbits=1,bytesize=8,timeout=0.001)
        
        try:
            self.port_handler = uservo.UartServoManager(self.uart, srv_num=7)
        except Exception:
            raise OSError(f"Failed to open port '{self.port}'.")


        self.is_connected = True


    def find_motor_indices(self, possible_ids=None, num_retry=2):
        if possible_ids is None:
            possible_ids = range(MAX_ID_RANGE)

        indices = []
        for idx in tqdm.tqdm(possible_ids):
            try:
                present_idx = self.read_with_motor_ids(self.motor_models, [idx], "ID", num_retry=num_retry)[0]
            except ConnectionError:
                continue

            if idx != present_idx:
                # sanity check
                raise OSError(
                    "Motor index used to communicate through the bus is not the same as the one present in the motor memory. The motor memory might be damaged."
                )
            indices.append(idx)

        return indices


    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def set_calibration(self, calibration: dict[str, list]):
        self.calibration = calibration

    def apply_calibration_autocorrect(self, values: np.ndarray | list, motor_names: list[str] | None):
        """This function applies the calibration, automatically detects out of range errors for motors values and attempts to correct.

        For more info, see docstring of `apply_calibration` and `autocorrect_calibration`.
        """
        try:
            values = self.apply_calibration(values, motor_names)
        except JointOutOfRangeError as e:
            print(e)
            # self.autocorrect_calibration(values, motor_names)
            values = self.apply_calibration(values, motor_names)
        return values

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        if motor_names is None:
            motor_names = self.motor_names
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]

                # Update direction of rotation of the motor to match between leader and follower.
                # In fact, the motor of the leader for a given joint can be assembled in an
                # opposite direction in term of rotation than the motor of the follower on the same joint.

                values[i] += homing_offset


                if (values[i] < LOWER_BOUND_DEGREE) or (values[i] > UPPER_BOUND_DEGREE):
                    raise JointOutOfRangeError(
                        f"Wrong motor position range detected for {name}. "
                        f"Expected to be in nominal range of [-{HALF_TURN_DEGREE}, {HALF_TURN_DEGREE}] degrees (a full rotation), "
                        f"with a maximum range of [{LOWER_BOUND_DEGREE}, {UPPER_BOUND_DEGREE}] degrees to account for joints that can rotate a bit more, "
                        f"but present value is {values[i]} degree. "
                        "This might be due to a cable connection issue creating an artificial 360 degrees jump in motor values. "
                        "You need to recalibrate by running: `python lerobot/scripts/control_robot.py calibrate`"
                    )

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Rescale the present position to a nominal range [0, 100] %,
                # useful for joints with linear motions like Aloha gripper
                values[i] = (values[i] - start_pos) / (end_pos - start_pos) * 100

                if (values[i] < LOWER_BOUND_LINEAR) or (values[i] > UPPER_BOUND_LINEAR):
                    raise JointOutOfRangeError(
                        f"Wrong port name:{self.port}"
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
                homing_offset = self.calibration["homing_offset"][calib_idx]
                values[i] -= homing_offset
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]
                if i == 1:
                    if values[i] < start_pos:
                        values[i] = start_pos
                elif i == 2:
                    if values[i] > end_pos:
                        values[i] = end_pos
                    
            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Convert from nominal lnear range of [0, 100] % to
                # actual motor range of values which can be arbitrary.
                values[i] = values[i] / 100 * (end_pos - start_pos) + start_pos

        values = np.round(values).astype(np.int32)

        return values

    # def read_with_motor_ids(self, motor_models, motor_ids, data_name, num_retry=NUM_READ_RETRY):
    #     if self.mock:
    #         import tests.motors.mock_dynamixel_sdk as dxl
    #     else:
    #         import dynamixel_sdk as dxl

    #     return_list = True
    #     if not isinstance(motor_ids, list):
    #         return_list = False
    #         motor_ids = [motor_ids]

    #     assert_same_address(self.model_ctrl_table, self.motor_models, data_name)
    #     addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
    #     group = dxl.GroupSyncRead(self.port_handler, self.packet_handler, addr, bytes)
    #     for idx in motor_ids:
    #         group.addParam(idx)

    #     for _ in range(num_retry):
    #         comm = group.txRxPacket()
    #         if comm == dxl.COMM_SUCCESS:
    #             break

    #     if comm != dxl.COMM_SUCCESS:
    #         raise ConnectionError(
    #             f"Read failed due to communication error on port {self.port_handler.port_name} for indices {motor_ids}: "
    #             f"{self.packet_handler.getTxRxResult(comm)}"
    #         )

    #     values = []
    #     for idx in motor_ids:
    #         value = group.getData(idx, addr, bytes)
    #         values.append(value)

    #     if return_list:
    #         return values
    #     else:
    #         return values[0]

    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()


        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        motor_ids = []
        models = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)

        # for _ in range(NUM_READ_RETRY):
        if data_name == "Present_Position":
            self.port_handler.send_sync_servo_monitor(motor_ids)
            comm = COMM_SUCCESS
            # break

        else:
            raise ConnectionError(
                f"function read not implemented for {data_name}"
            )  

        if comm != COMM_SUCCESS:
            raise ConnectionError(
                f"Read failed due to communication error on port {self.port}"
            )

        values = []
        for idx in motor_ids:
            values.append(self.port_handler.servos[idx].angle_monitor)
        # print(values[0],values[1],values[2],values[3],values[4],values[5],values[6])
        values = np.array(values)
        if data_name in CONVERT_UINT32_TO_INT32_REQUIRED:
            values = values.astype(np.int32)

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.apply_calibration_autocorrect(values, motor_names)

        # log the number of seconds it took to read the data from the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time at which the data was received
        ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    def write_with_motor_ids(self, motor_models, motor_ids, data_name, values, num_retry=NUM_WRITE_RETRY):
        # if self.mock:
        #     import tests.motors.mock_dynamixel_sdk as dxl
        # else:
        #     import dynamixel_sdk as dxl

        if not isinstance(motor_ids, list):
            motor_ids = [motor_ids]
        if not isinstance(values, list):
            values = [values]

        # assert_same_address(self.model_ctrl_table, motor_models, data_name)
        # addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
        # group = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, addr, bytes)
        # for idx, value in zip(motor_ids, values, strict=True):
        #     data = convert_to_bytes(value, bytes, self.mock)
        #     group.addParam(idx, data)

        # for _ in range(num_retry):
        #     comm = group.txPacket()
        #     if comm == dxl.COMM_SUCCESS:
        #         break

        if comm != dxl.COMM_SUCCESS:
            raise ConnectionError(
                f"Write failed due to communication error on port {self.port_handler.port_name} for indices {motor_ids}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        if isinstance(values, (int, float, np.integer)):
            values = [int(values)] * len(motor_names)

        values = np.array(values)
        # print(data_name,values)
        motor_ids = []
        models = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.revert_calibration(values, motor_names)

        values = values.tolist()
        if data_name == "Torque_Enable":
            comm = COMM_SUCCESS
        elif data_name == "Goal_Position":
            
            if  motor_names[6] != None and motor_names[6] == "gripper":
                if self.gripper_degree_record != values[6] :
                    self.gripper_degree_record = values[6]
                    command_data_list = [struct.pack("<BlLHHH", motor_ids[i], int(values[i]*10), self.interval, 20, 20, 0)for i in motor_ids]
                    self.port_handler.send_sync_multiturnanglebyinterval(self.port_handler.CODE_SET_SERVO_ANGLE_MTURN_BY_INTERVAL,len(motor_ids), command_data_list)
                else:
                    command_data_list = [struct.pack("<BlLHHH", motor_ids[i], int(values[i]*10), self.interval, 20, 20, 0)for i in (motor_ids[:-1])]
                    self.port_handler.send_sync_multiturnanglebyinterval(self.port_handler.CODE_SET_SERVO_ANGLE_MTURN_BY_INTERVAL,len(motor_ids[:-1]), command_data_list)
            else:
                command_data_list = [struct.pack("<BlLHHH", motor_ids[i], int(values[i]*10), self.interval, 20, 20, 0)for i in motor_ids]
                self.port_handler.send_sync_multiturnanglebyinterval(self.port_handler.CODE_SET_SERVO_ANGLE_MTURN_BY_INTERVAL,len(motor_ids), command_data_list)
            
            
            comm = COMM_SUCCESS
 



        else :
            raise ValueError(
                f"Write failed for data_name {data_name} because it is not supported. "
            )





        if comm != COMM_SUCCESS:
            raise ConnectionError(
                f"Write failed due to communication error on port {self.port} for data_name {data_name}: "
                # f"{self.packet_handler.getTxRxResult(comm)}"
            )

        # log the number of seconds it took to write the data to the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "write", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # TODO(rcadene): should we log the time before sending the write command?
        # log the utc time when the write has been completed
        ts_utc_name = get_log_name("timestamp_utc", "write", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"({self.port}) is not connected. Try running `motors_bus.connect()` first."
            )

        if self.port_handler is not None:
            self.port_handler = None

        self.packet_handler = None
        self.group_readers = {}
        self.group_writers = {}
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

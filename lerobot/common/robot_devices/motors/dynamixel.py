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

import numpy as np
import tqdm

from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

PROTOCOL_VERSION = 2.0
BAUDRATE = 1_000_000
TIMEOUT_MS = 1000

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


def convert_to_bytes(value, bytes, mock=False):
    if mock:
        return value

    import dynamixel_sdk as dxl

    # Note: No need to convert back into unsigned int, since this byte preprocessing
    # already handles it for us.
    if bytes == 1:
        data = [
            dxl.DXL_LOBYTE(dxl.DXL_LOWORD(value)),
        ]
    elif bytes == 2:
        data = [
            dxl.DXL_LOBYTE(dxl.DXL_LOWORD(value)),
            dxl.DXL_HIBYTE(dxl.DXL_LOWORD(value)),
        ]
    elif bytes == 4:
        data = [
            dxl.DXL_LOBYTE(dxl.DXL_LOWORD(value)),
            dxl.DXL_HIBYTE(dxl.DXL_LOWORD(value)),
            dxl.DXL_LOBYTE(dxl.DXL_HIWORD(value)),
            dxl.DXL_HIBYTE(dxl.DXL_HIWORD(value)),
        ]
    else:
        raise NotImplementedError(
            f"Value of the number of bytes to be sent is expected to be in [1, 2, 4], but "
            f"{bytes} is provided instead."
        )
    return data


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


def assert_same_address(model_ctrl_table, motor_models, data_name):
    all_addr = []
    all_bytes = []
    for model in motor_models:
        addr, bytes = model_ctrl_table[model][data_name]
        all_addr.append(addr)
        all_bytes.append(bytes)

    if len(set(all_addr)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different address for `data_name`='{data_name}' ({list(zip(motor_models, all_addr, strict=False))}). Contact a LeRobot maintainer."
        )

    if len(set(all_bytes)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different bytes representation for `data_name`='{data_name}' ({list(zip(motor_models, all_bytes, strict=False))}). Contact a LeRobot maintainer."
        )


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


class DynamixelMotorsBus:
    """
    The DynamixelMotorsBus class allows to efficiently read and write to the attached motors. It relies on
    the python dynamixel sdk to communicate with the motors. For more info, see the [Dynamixel SDK Documentation](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python_read_write_protocol_2_0/#python-read-write-protocol-20).

    A DynamixelMotorsBus instance requires a port (e.g. `DynamixelMotorsBus(port="/dev/tty.usbmodem575E0031751"`)).
    To find the port, you can run our utility script:
    ```bash
    python lerobot/scripts/find_motors_bus_port.py
    >>> Finding all available ports for the MotorBus.
    >>> ['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
    >>> Remove the usb cable from your DynamixelMotorsBus and press Enter when done.
    >>> The port of this DynamixelMotorsBus is /dev/tty.usbmodem575E0031751.
    >>> Reconnect the usb cable.
    ```

    Example of usage for 1 motor connected to the bus:
    ```python
    motor_name = "gripper"
    motor_index = 6
    motor_model = "xl330-m288"

    config = DynamixelMotorsBusConfig(
        port="/dev/tty.usbmodem575E0031751",
        motors={motor_name: (motor_index, motor_model)},
    )
    motors_bus = DynamixelMotorsBus(config)
    motors_bus.connect()

    position = motors_bus.read("Present_Position")

    # move from a few motor steps as an example
    few_steps = 30
    motors_bus.write("Goal_Position", position + few_steps)

    # when done, consider disconnecting
    motors_bus.disconnect()
    ```
    """

    def __init__(
        self,
        config: DynamixelMotorsBusConfig,
    ):
        self.port = config.port
        self.motors = config.motors
        self.mock = config.mock

        self.model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
        self.model_resolution = deepcopy(MODEL_RESOLUTION)

        self.port_handler = None
        self.packet_handler = None
        self.calibration = None
        self.is_connected = False
        self.group_readers = {}
        self.group_writers = {}
        self.logs = {}

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"DynamixelMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        if self.mock:
            import tests.motors.mock_dynamixel_sdk as dxl
        else:
            import dynamixel_sdk as dxl

        self.port_handler = dxl.PortHandler(self.port)
        self.packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)

        try:
            if not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'.")
        except Exception:
            traceback.print_exc()
            print(
                "\nTry running `python lerobot/scripts/find_motors_bus_port.py` to make sure you are using the correct port.\n"
            )
            raise

        # Allow to read and write
        self.is_connected = True

        self.port_handler.setPacketTimeoutMillis(TIMEOUT_MS)

    def reconnect(self):
        if self.mock:
            import tests.motors.mock_dynamixel_sdk as dxl
        else:
            import dynamixel_sdk as dxl

        self.port_handler = dxl.PortHandler(self.port)
        self.packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            raise OSError(f"Failed to open port '{self.port}'.")

        self.is_connected = True

    def are_motors_configured(self):
        # Only check the motor indices and not baudrate, since if the motor baudrates are incorrect,
        # a ConnectionError will be raised anyway.
        try:
            return (self.motor_indices == self.read("ID")).all()
        except ConnectionError as e:
            print(e)
            return False

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

    def set_bus_baudrate(self, baudrate):
        present_bus_baudrate = self.port_handler.getBaudRate()
        if present_bus_baudrate != baudrate:
            print(f"Setting bus baud rate to {baudrate}. Previously {present_bus_baudrate}.")
            self.port_handler.setBaudRate(baudrate)

            if self.port_handler.getBaudRate() != baudrate:
                raise OSError("Failed to write bus baud rate.")

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
            self.autocorrect_calibration(values, motor_names)
            values = self.apply_calibration(values, motor_names)
        return values

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Convert from unsigned int32 joint position range [0, 2**32[ to the universal float32 nominal degree range ]-180.0, 180.0[ with
        a "zero position" at 0 degree.

        Note: We say "nominal degree range" since the motors can take values outside this range. For instance, 190 degrees, if the motor
        rotate more than a half a turn from the zero position. However, most motors can't rotate more than 180 degrees and will stay in this range.

        Joints values are original in [0, 2**32[ (unsigned int32). Each motor are expected to complete a full rotation
        when given a goal position that is + or - their resolution. For instance, dynamixel xl330-m077 have a resolution of 4096, and
        at any position in their original range, let's say the position 56734, they complete a full rotation clockwise by moving to 60830,
        or anticlockwise by moving to 52638. The position in the original range is arbitrary and might change a lot between each motor.
        To harmonize between motors of the same model, different robots, or even models of different brands, we propose to work
        in the centered nominal degree range ]-180, 180[.
        """
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from unsigned int32 original range [0, 2**32] to signed float32 range
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                # Update direction of rotation of the motor to match between leader and follower.
                # In fact, the motor of the leader for a given joint can be assembled in an
                # opposite direction in term of rotation than the motor of the follower on the same joint.
                if drive_mode:
                    values[i] *= -1

                # Convert from range [-2**31, 2**31] to
                # nominal range [-resolution//2, resolution//2] (e.g. [-2048, 2048])
                values[i] += homing_offset

                # Convert from range [-resolution//2, resolution//2] to
                # universal float32 centered degree range [-180, 180]
                # (e.g. 2048 / (4096 // 2) * 180 = 180)
                values[i] = values[i] / (resolution // 2) * HALF_TURN_DEGREE

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
                        f"Wrong motor position range detected for {name}. "
                        f"Expected to be in nominal range of [0, 100] % (a full linear translation), "
                        f"with a maximum range of [{LOWER_BOUND_LINEAR}, {UPPER_BOUND_LINEAR}] % to account for some imprecision during calibration, "
                        f"but present value is {values[i]} %. "
                        "This might be due to a cable connection issue creating an artificial jump in motor values. "
                        "You need to recalibrate by running: `python lerobot/scripts/control_robot.py calibrate`"
                    )

        return values

    def autocorrect_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """This function automatically detects issues with values of motors after calibration, and correct for these issues.

        Some motors might have values outside of expected maximum bounds after calibration.
        For instance, for a joint in degree, its value can be outside [-270, 270] degrees, which is totally unexpected given
        a nominal range of [-180, 180] degrees, which represents half a turn to the left or right starting from zero position.

        Known issues:
        #1: Motor value randomly shifts of a full turn, caused by hardware/connection errors.
        #2: Motor internal homing offset is shifted by a full turn, caused by using default calibration (e.g Aloha).
        #3: motor internal homing offset is shifted by less or more than a full turn, caused by using default calibration
            or by human error during manual calibration.

        Issues #1 and #2 can be solved by shifting the calibration homing offset by a full turn.
        Issue #3 will be visually detected by user and potentially captured by the safety feature `max_relative_target`,
        that will slow down the motor, raise an error asking to recalibrate. Manual recalibrating will solve the issue.

        Note: A full turn corresponds to 360 degrees but also to 4096 steps for a motor resolution of 4096.
        """
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from unsigned int32 original range [0, 2**32] to signed float32 range
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                # Update direction of rotation of the motor to match between leader and follower.
                # In fact, the motor of the leader for a given joint can be assembled in an
                # opposite direction in term of rotation than the motor of the follower on the same joint.
                if drive_mode:
                    values[i] *= -1

                # Convert from initial range to range [-180, 180] degrees
                calib_val = (values[i] + homing_offset) / (resolution // 2) * HALF_TURN_DEGREE
                in_range = (calib_val > LOWER_BOUND_DEGREE) and (calib_val < UPPER_BOUND_DEGREE)

                # Solve this inequality to find the factor to shift the range into [-180, 180] degrees
                # values[i] = (values[i] + homing_offset + resolution * factor) / (resolution // 2) * HALF_TURN_DEGREE
                # - HALF_TURN_DEGREE <= (values[i] + homing_offset + resolution * factor) / (resolution // 2) * HALF_TURN_DEGREE <= HALF_TURN_DEGREE
                # (- (resolution // 2) - values[i] - homing_offset) / resolution <= factor <= ((resolution // 2) - values[i] - homing_offset) / resolution
                low_factor = (-(resolution // 2) - values[i] - homing_offset) / resolution
                upp_factor = ((resolution // 2) - values[i] - homing_offset) / resolution

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Convert from initial range to range [0, 100] in %
                calib_val = (values[i] - start_pos) / (end_pos - start_pos) * 100
                in_range = (calib_val > LOWER_BOUND_LINEAR) and (calib_val < UPPER_BOUND_LINEAR)

                # Solve this inequality to find the factor to shift the range into [0, 100] %
                # values[i] = (values[i] - start_pos + resolution * factor) / (end_pos + resolution * factor - start_pos - resolution * factor) * 100
                # values[i] = (values[i] - start_pos + resolution * factor) / (end_pos - start_pos) * 100
                # 0 <= (values[i] - start_pos + resolution * factor) / (end_pos - start_pos) * 100 <= 100
                # (start_pos - values[i]) / resolution <= factor <= (end_pos - values[i]) / resolution
                low_factor = (start_pos - values[i]) / resolution
                upp_factor = (end_pos - values[i]) / resolution

            if not in_range:
                # Get first integer between the two bounds
                if low_factor < upp_factor:
                    factor = math.ceil(low_factor)

                    if factor > upp_factor:
                        raise ValueError(f"No integer found between bounds [{low_factor=}, {upp_factor=}]")
                else:
                    factor = math.ceil(upp_factor)

                    if factor > low_factor:
                        raise ValueError(f"No integer found between bounds [{low_factor=}, {upp_factor=}]")

                if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                    out_of_range_str = f"{LOWER_BOUND_DEGREE} < {calib_val} < {UPPER_BOUND_DEGREE} degrees"
                    in_range_str = f"{LOWER_BOUND_DEGREE} < {calib_val} < {UPPER_BOUND_DEGREE} degrees"
                elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                    out_of_range_str = f"{LOWER_BOUND_LINEAR} < {calib_val} < {UPPER_BOUND_LINEAR} %"
                    in_range_str = f"{LOWER_BOUND_LINEAR} < {calib_val} < {UPPER_BOUND_LINEAR} %"

                logging.warning(
                    f"Auto-correct calibration of motor '{name}' by shifting value by {abs(factor)} full turns, "
                    f"from '{out_of_range_str}' to '{in_range_str}'."
                )

                # A full turn corresponds to 360 degrees but also to 4096 steps for a motor resolution of 4096.
                self.calibration["homing_offset"][calib_idx] += resolution * factor

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Inverse of `apply_calibration`."""
        if motor_names is None:
            motor_names = self.motor_names

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                # Convert from nominal 0-centered degree range [-180, 180] to
                # 0-centered resolution range (e.g. [-2048, 2048] for resolution=4096)
                values[i] = values[i] / HALF_TURN_DEGREE * (resolution // 2)

                # Subtract the homing offsets to come back to actual motor range of values
                # which can be arbitrary.
                values[i] -= homing_offset

                # Remove drive mode, which is the rotation direction of the motor, to come back to
                # actual motor rotation direction which can be arbitrary.
                if drive_mode:
                    values[i] *= -1

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Convert from nominal lnear range of [0, 100] % to
                # actual motor range of values which can be arbitrary.
                values[i] = values[i] / 100 * (end_pos - start_pos) + start_pos

        values = np.round(values).astype(np.int32)
        return values

    def read_with_motor_ids(self, motor_models, motor_ids, data_name, num_retry=NUM_READ_RETRY):
        if self.mock:
            import tests.motors.mock_dynamixel_sdk as dxl
        else:
            import dynamixel_sdk as dxl

        return_list = True
        if not isinstance(motor_ids, list):
            return_list = False
            motor_ids = [motor_ids]

        assert_same_address(self.model_ctrl_table, self.motor_models, data_name)
        addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
        group = dxl.GroupSyncRead(self.port_handler, self.packet_handler, addr, bytes)
        for idx in motor_ids:
            group.addParam(idx)

        for _ in range(num_retry):
            comm = group.txRxPacket()
            if comm == dxl.COMM_SUCCESS:
                break

        if comm != dxl.COMM_SUCCESS:
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

    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"DynamixelMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        if self.mock:
            import tests.motors.mock_dynamixel_sdk as dxl
        else:
            import dynamixel_sdk as dxl

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

        assert_same_address(self.model_ctrl_table, models, data_name)
        addr, bytes = self.model_ctrl_table[model][data_name]
        group_key = get_group_sync_key(data_name, motor_names)

        if data_name not in self.group_readers:
            # create new group reader
            self.group_readers[group_key] = dxl.GroupSyncRead(
                self.port_handler, self.packet_handler, addr, bytes
            )
            for idx in motor_ids:
                self.group_readers[group_key].addParam(idx)

        for _ in range(NUM_READ_RETRY):
            comm = self.group_readers[group_key].txRxPacket()
            if comm == dxl.COMM_SUCCESS:
                break

        if comm != dxl.COMM_SUCCESS:
            raise ConnectionError(
                f"Read failed due to communication error on port {self.port} for group_key {group_key}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

        values = []
        for idx in motor_ids:
            value = self.group_readers[group_key].getData(idx, addr, bytes)
            values.append(value)

        values = np.array(values)

        # Convert to signed int to use range [-2048, 2048] for our motor positions.
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
        if self.mock:
            import tests.motors.mock_dynamixel_sdk as dxl
        else:
            import dynamixel_sdk as dxl

        if not isinstance(motor_ids, list):
            motor_ids = [motor_ids]
        if not isinstance(values, list):
            values = [values]

        assert_same_address(self.model_ctrl_table, motor_models, data_name)
        addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
        group = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, addr, bytes)
        for idx, value in zip(motor_ids, values, strict=True):
            data = convert_to_bytes(value, bytes, self.mock)
            group.addParam(idx, data)

        for _ in range(num_retry):
            comm = group.txPacket()
            if comm == dxl.COMM_SUCCESS:
                break

        if comm != dxl.COMM_SUCCESS:
            raise ConnectionError(
                f"Write failed due to communication error on port {self.port_handler.port_name} for indices {motor_ids}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"DynamixelMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        if self.mock:
            import tests.motors.mock_dynamixel_sdk as dxl
        else:
            import dynamixel_sdk as dxl

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        if isinstance(values, (int, float, np.integer)):
            values = [int(values)] * len(motor_names)

        values = np.array(values)

        motor_ids = []
        models = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.revert_calibration(values, motor_names)

        values = values.tolist()

        assert_same_address(self.model_ctrl_table, models, data_name)
        addr, bytes = self.model_ctrl_table[model][data_name]
        group_key = get_group_sync_key(data_name, motor_names)

        init_group = data_name not in self.group_readers
        if init_group:
            self.group_writers[group_key] = dxl.GroupSyncWrite(
                self.port_handler, self.packet_handler, addr, bytes
            )

        for idx, value in zip(motor_ids, values, strict=True):
            data = convert_to_bytes(value, bytes, self.mock)
            if init_group:
                self.group_writers[group_key].addParam(idx, data)
            else:
                self.group_writers[group_key].changeParam(idx, data)

        comm = self.group_writers[group_key].txPacket()
        if comm != dxl.COMM_SUCCESS:
            raise ConnectionError(
                f"Write failed due to communication error on port {self.port} for group_key {group_key}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
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
                f"DynamixelMotorsBus({self.port}) is not connected. Try running `motors_bus.connect()` first."
            )

        if self.port_handler is not None:
            self.port_handler.closePort()
            self.port_handler = None

        self.packet_handler = None
        self.group_readers = {}
        self.group_writers = {}
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

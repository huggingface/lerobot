import enum
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
from dynamixel_sdk import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

PROTOCOL_VERSION = 2.0
BAUD_RATE = 1_000_000
TIMEOUT_MS = 1000

# https://emanual.robotis.com/docs/en/dxl/x/xl330-m077
# https://emanual.robotis.com/docs/en/dxl/x/xl330-m288
# https://emanual.robotis.com/docs/en/dxl/x/xl430-w250
# https://emanual.robotis.com/docs/en/dxl/x/xm430-w350
# https://emanual.robotis.com/docs/en/dxl/x/xm540-w270

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

CALIBRATION_REQUIRED = ["Goal_Position", "Present_Position"]
CONVERT_UINT32_TO_INT32_REQUIRED = ["Goal_Position", "Present_Position"]

MODEL_CONTROL_TABLE = {
    "x_series": X_SERIES_CONTROL_TABLE,
    "xl330-m077": X_SERIES_CONTROL_TABLE,
    "xl330-m288": X_SERIES_CONTROL_TABLE,
    "xl430-w250": X_SERIES_CONTROL_TABLE,
    "xm430-w350": X_SERIES_CONTROL_TABLE,
    "xm540-w270": X_SERIES_CONTROL_TABLE,
}

NUM_READ_RETRY = 10


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


def find_available_ports():
    ports = []
    for path in Path("/dev").glob("tty*"):
        ports.append(str(path))
    return ports


def find_port():
    print("Finding all available ports for the DynamixelMotorsBus.")
    ports_before = find_available_ports()
    print(ports_before)

    print("Remove the usb cable from your DynamixelMotorsBus and press Enter when done.")
    input()

    time.sleep(0.5)
    ports_after = find_available_ports()
    ports_diff = list(set(ports_before) - set(ports_after))

    if len(ports_diff) == 1:
        port = ports_diff[0]
        print(f"The port of this DynamixelMotorsBus is '{port}'")
        print("Reconnect the usb cable.")
    elif len(ports_diff) == 0:
        raise OSError(f"Could not detect the port. No difference was found ({ports_diff}).")
    else:
        raise OSError(f"Could not detect the port. More than one port was found ({ports_diff}).")


class TorqueMode(enum.Enum):
    ENABLED = 1
    DISABLED = 0


class OperatingMode(enum.Enum):
    VELOCITY = 1
    POSITION = 3
    EXTENDED_POSITION = 4
    CURRENT_CONTROLLED_POSITION = 5
    PWM = 16
    UNKNOWN = -1


class DriveMode(enum.Enum):
    NON_INVERTED = 0
    INVERTED = 1


class DynamixelMotorsBus:
    # TODO(rcadene): Add a script to find the motor indices without DynamixelWizzard2
    """
    The DynamixelMotorsBus class allows to efficiently read and write to the attached motors. It relies on
    the python dynamixel sdk to communicate with the motors. For more info, see the [Dynamixel SDK Documentation](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python_read_write_protocol_2_0/#python-read-write-protocol-20).

    A DynamixelMotorsBus instance requires a port (e.g. `DynamixelMotorsBus(port="/dev/tty.usbmodem575E0031751"`)).
    To find the port, you can run our utility script:
    ```bash
    python lerobot/common/robot_devices/motors/dynamixel.py
    >>> Finding all available ports for the DynamixelMotorsBus.
    >>> ['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
    >>> Remove the usb cable from your DynamixelMotorsBus and press Enter when done.
    >>> The port of this DynamixelMotorsBus is /dev/tty.usbmodem575E0031751.
    >>> Reconnect the usb cable.
    ```
    To find the motor indices, use [DynamixelWizzard2](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2).

    Example of usage for 1 motor connected to the bus:
    ```python
    motor_name = "gripper"
    motor_index = 6
    motor_model = "xl330-m077"

    motors_bus = DynamixelMotorsBus(
        port="/dev/tty.usbmodem575E0031751",
        motors={motor_name: (motor_index, motor_model)},
    )
    motors_bus.connect()

    motors_bus.teleop_step()

    # when done, consider disconnecting
    motors_bus.disconnect()
    ```
    """

    def __init__(
        self,
        port: str,
        motors: dict[str, tuple[int, str]],
        extra_model_control_table: dict[str, list[tuple]] | None = None,
    ):
        self.port = port
        self.motors = motors

        self.model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
        if extra_model_control_table:
            self.model_ctrl_table.update(extra_model_control_table)

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

        self.port_handler = PortHandler(self.port)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        try:
            if not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'.")
        except Exception:
            traceback.print_exc()
            print(
                "\nTry running `python lerobot/common/robot_devices/motors/dynamixel.py` to make sure you are using the correct port.\n"
            )
            raise

        self.port_handler.setBaudRate(BAUD_RATE)
        self.port_handler.setPacketTimeoutMillis(TIMEOUT_MS)

        self.is_connected = True

    @property
    def motor_names(self) -> list[int]:
        return list(self.motors.keys())

    def set_calibration(self, calibration: dict[str, tuple[int, bool]]):
        self.calibration = calibration

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        if not self.calibration:
            return values

        if motor_names is None:
            motor_names = self.motor_names

        for i, name in enumerate(motor_names):
            homing_offset, drive_mode = self.calibration[name]

            if values[i] is not None:
                if drive_mode:
                    values[i] *= -1
                values[i] += homing_offset

        return values

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        if not self.calibration:
            return values

        if motor_names is None:
            motor_names = self.motor_names

        for i, name in enumerate(motor_names):
            homing_offset, drive_mode = self.calibration[name]

            if values[i] is not None:
                values[i] -= homing_offset
                if drive_mode:
                    values[i] *= -1

        return values

    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"DynamixelMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
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

        assert_same_address(self.model_ctrl_table, models, data_name)
        addr, bytes = self.model_ctrl_table[model][data_name]
        group_key = get_group_sync_key(data_name, motor_names)

        if data_name not in self.group_readers:
            # create new group reader
            self.group_readers[group_key] = GroupSyncRead(self.port_handler, self.packet_handler, addr, bytes)
            for idx in motor_ids:
                self.group_readers[group_key].addParam(idx)

        for _ in range(NUM_READ_RETRY):
            comm = self.group_readers[group_key].txRxPacket()
            if comm == COMM_SUCCESS:
                break

        if comm != COMM_SUCCESS:
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

        if data_name in CALIBRATION_REQUIRED:
            values = self.apply_calibration(values, motor_names)

        # log the number of seconds it took to read the data from the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time at which the data was received
        ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"DynamixelMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

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

        if data_name in CALIBRATION_REQUIRED:
            values = self.revert_calibration(values, motor_names)

        values = values.tolist()

        assert_same_address(self.model_ctrl_table, models, data_name)
        addr, bytes = self.model_ctrl_table[model][data_name]
        group_key = get_group_sync_key(data_name, motor_names)

        init_group = data_name not in self.group_readers
        if init_group:
            self.group_writers[group_key] = GroupSyncWrite(
                self.port_handler, self.packet_handler, addr, bytes
            )

        for idx, value in zip(motor_ids, values, strict=True):
            # Note: No need to convert back into unsigned int, since this byte preprocessing
            # already handles it for us.
            if bytes == 1:
                data = [
                    DXL_LOBYTE(DXL_LOWORD(value)),
                ]
            elif bytes == 2:
                data = [
                    DXL_LOBYTE(DXL_LOWORD(value)),
                    DXL_HIBYTE(DXL_LOWORD(value)),
                ]
            elif bytes == 4:
                data = [
                    DXL_LOBYTE(DXL_LOWORD(value)),
                    DXL_HIBYTE(DXL_LOWORD(value)),
                    DXL_LOBYTE(DXL_HIWORD(value)),
                    DXL_HIBYTE(DXL_HIWORD(value)),
                ]
            else:
                raise NotImplementedError(
                    f"Value of the number of bytes to be sent is expected to be in [1, 2, 4], but "
                    f"{bytes} is provided instead."
                )

            if init_group:
                self.group_writers[group_key].addParam(idx, data)
            else:
                self.group_writers[group_key].changeParam(idx, data)

        comm = self.group_writers[group_key].txPacket()
        if comm != COMM_SUCCESS:
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


if __name__ == "__main__":
    # Helper to find the usb port associated to all your DynamixelMotorsBus.
    find_port()

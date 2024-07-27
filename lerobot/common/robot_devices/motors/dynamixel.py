import enum
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import tqdm
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
BAUDRATE = 1_000_000
TIMEOUT_MS = 1000

MAX_ID_RANGE = 252

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
}

MODEL_RESOLUTION = {
    "x_series": 4096,
    "xl330-m077": 4096,
    "xl330-m288": 4096,
    "xl430-w250": 4096,
    "xm430-w350": 4096,
    "xm540-w270": 4096,
}

MODEL_BAUDRATE_TABLE = {
    "x_series": X_SERIES_BAUDRATE_TABLE,
    "xl330-m077": X_SERIES_BAUDRATE_TABLE,
    "xl330-m288": X_SERIES_BAUDRATE_TABLE,
    "xl430-w250": X_SERIES_BAUDRATE_TABLE,
    "xm430-w350": X_SERIES_BAUDRATE_TABLE,
    "xm540-w270": X_SERIES_BAUDRATE_TABLE,
}

NUM_READ_RETRY = 10
NUM_WRITE_RETRY = 10


def convert_indices_to_baudrates(values: np.ndarray | list[int], models: list[str]):
    assert len(values) == len(models)
    for i in range(len(values)):
        model = models[i]
        index = values[i]
        values[i] = MODEL_BAUDRATE_TABLE[model][index]
    return values


def convert_baudrates_to_indices(values: np.ndarray | list[int], models: list[str]):
    assert len(values) == len(models)
    for i in range(len(values)):
        model = models[i]
        brate = values[i]
        table_values = list(MODEL_BAUDRATE_TABLE[model].values())
        table_keys = list(MODEL_BAUDRATE_TABLE[model].keys())
        values[i] = table_keys[table_values.index(brate)]
    return values


def convert_to_bytes(value, bytes):
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


def find_available_ports():
    ports = []
    for path in Path("/dev").glob("*"):
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

    degrees = motors_bus.read("Present_Position")

    motors_bus.write("Goal_Position", degrees + 30)

    # when done, consider disconnecting
    motors_bus.disconnect()
    ```
    """

    def __init__(
        self,
        port: str,
        motors: dict[str, tuple[int, str]],
        extra_model_control_table: dict[str, list[tuple]] | None = None,
        extra_model_resolution: dict[str, int] | None = None,
    ):
        self.port = port
        self.motors = motors

        self.model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
        if extra_model_control_table:
            self.model_ctrl_table.update(extra_model_control_table)

        self.model_resolution = deepcopy(MODEL_RESOLUTION)
        if extra_model_resolution:
            self.model_resolution.update(extra_model_resolution)

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

        # Allow to read and write
        self.is_connected = True

        self.port_handler.setPacketTimeoutMillis(TIMEOUT_MS)

        # Set expected baudrate for the bus
        self.set_bus_baudrate(BAUDRATE)

        if not self.are_motors_configured():
            print(
                r"/!\ First, verify that all the cables are connected the proper way. If you detect an issue, before making any modification, unplug the power cord to not damage the motors. Rewire correctly. Then plug the power again and relaunch the script."
            )
            print(
                r"/!\ Secondly, if the cables connection look correct and it is the first time that you use these motors, follow these manual steps to configure them."
            )
            input("Press Enter to configure your motors...")
            print()
            self.configure_motors()

    def reconnect(self):
        self.port_handler = PortHandler(self.port)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)
        if not self.port_handler.openPort():
            raise OSError(f"Failed to open port '{self.port}'.")
        self.is_connected = True

    def are_motors_configured(self):
        try:
            return (self.motor_indices == self.read("ID")).all()
        except ConnectionError as e:
            print(e)
            return False

    def configure_motors(self):
        # TODO(rcadene): This script assumes motors follow the X_SERIES baudrates

        print("Scanning all baudrates and motor indices")
        all_baudrates = set(X_SERIES_BAUDRATE_TABLE.values())
        ids_per_baudrate = {}
        for baudrate in all_baudrates:
            self.set_bus_baudrate(baudrate)
            present_ids = self.find_motor_indices()
            if len(present_ids) > 0:
                ids_per_baudrate[baudrate] = present_ids
        print(f"Motor indices detected: {ids_per_baudrate}")
        print()

        possible_baudrates = list(ids_per_baudrate.keys())
        possible_ids = list({idx for sublist in ids_per_baudrate.values() for idx in sublist})
        untaken_ids = list(set(range(MAX_ID_RANGE)) - set(possible_ids) - set(self.motor_indices))

        # Connect successively one motor to the chain and write a unique random index for each
        for i in range(len(self.motors)):
            self.disconnect()
            print("1. Unplug the power cord")
            print(
                f"2. Plug/unplug minimal number of cables to only have the first {i+1} motor(s) ({self.motor_names[:i+1]}) connected."
            )
            print("3. Re-plug the power cord.")
            input("Press Enter to continue...")
            print()
            self.reconnect()

            if i > 0:
                try:
                    self._read_with_motor_ids(self.motor_models, untaken_ids[:i], "ID")
                except ConnectionError:
                    print(f"Failed to read from {untaken_ids[:i+1]}. Make sure the power cord is plugged in.")
                    input("Press Enter to continue...")
                    print()
                    self.reconnect()

            print("Scanning possible baudrates and motor indices")
            motor_found = False
            for baudrate in possible_baudrates:
                self.set_bus_baudrate(baudrate)
                present_ids = self.find_motor_indices(possible_ids)
                if len(present_ids) == 1:
                    present_idx = present_ids[0]
                    print(f"Detected motor with index {present_idx}")

                    if baudrate != BAUDRATE:
                        print(f"Setting its baudrate to {BAUDRATE}")
                        baudrate_idx = list(X_SERIES_BAUDRATE_TABLE.values()).index(BAUDRATE)

                        # The write can fail, so we allow retries
                        for _ in range(NUM_WRITE_RETRY):
                            self._write_with_motor_ids(
                                self.motor_models, present_idx, "Baud_Rate", baudrate_idx
                            )
                            time.sleep(0.5)
                            self.set_bus_baudrate(BAUDRATE)
                            try:
                                present_baudrate_idx = self._read_with_motor_ids(
                                    self.motor_models, present_idx, "Baud_Rate"
                                )
                            except ConnectionError:
                                print("Failed to write baudrate. Retrying.")
                                self.set_bus_baudrate(baudrate)
                                continue
                            break
                        else:
                            raise

                        if present_baudrate_idx != baudrate_idx:
                            raise OSError("Failed to write baudrate.")

                    print(f"Setting its index to a temporary untaken index ({untaken_ids[i]})")
                    self._write_with_motor_ids(self.motor_models, present_idx, "ID", untaken_ids[i])

                    present_idx = self._read_with_motor_ids(self.motor_models, untaken_ids[i], "ID")
                    if present_idx != untaken_ids[i]:
                        raise OSError("Failed to write index.")

                    motor_found = True
                    break
                elif len(present_ids) > 1:
                    raise OSError(f"More than one motor detected ({present_ids}), but only one was expected.")

            if not motor_found:
                raise OSError(
                    "No motor found, but one new motor expected. Verify power cord is plugged in and retry."
                )
            print()

        print(f"Setting expected motor indices: {self.motor_indices}")
        self.set_bus_baudrate(BAUDRATE)
        self._write_with_motor_ids(
            self.motor_models, untaken_ids[: len(self.motors)], "ID", self.motor_indices
        )
        print()

        if (self.read("ID") != self.motor_indices).any():
            raise OSError("Failed to write motors indices.")

        print("Configuration is done!")

    def find_motor_indices(self, possible_ids=None):
        if possible_ids is None:
            possible_ids = range(MAX_ID_RANGE)

        indices = []
        for idx in tqdm.tqdm(possible_ids):
            try:
                present_idx = self._read_with_motor_ids(self.motor_models, [idx], "ID")[0]
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

    def set_calibration(self, calibration: dict[str, tuple[int, bool]]):
        self.calibration = calibration

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Convert from unsigned int32 joint position range [0, 2**32[ to the universal float32 centered degree range [-180.0, 180.0[

        Joints values are original in [0, 2**32[ (unsigned int32). Each motor are expected to complete a full rotation
        when given a goal position that is + or - their resolution. For instance, dynamixel xl330-m077 have a resolution of 4096, and
        at any position in their original range, let's say the position 56734, they complete a full rotation clockwise by moving to 60830,
        or anticlockwise by moving to 42638. The position in the original range is arbitrary and might change a lot between each motor.
        To harmonize between motors of the same model, different robots, or even models of different brands, we propose to work
        in the centered degree range [-180, 180[. This function first applies the pre-computed calibration to convert
        from [0, 2**32[ to [-2048, 2048[, then divide by 2048.
        """
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from unsigned int32 original range [0, 2**32[ to centered signed int32 range [-2**31, 2**31[
        values = values.astype(np.int32)

        for i, name in enumerate(motor_names):
            homing_offset, drive_mode = self.calibration[name]

            # Update direction of rotation of the motor to match between leader and follower. In fact, the motor of the leader for a given joint
            # can be assembled in an opposite direction in term of rotation than the motor of the follower on the same joint.
            if drive_mode:
                values[i] *= -1

            # Convert from range [-2**31, 2**31[ to centered resolution range [-resolution, resolution[ (e.g. [-2048, 2048[)
            values[i] += homing_offset

        # Convert from range [-resolution, resolution[ to the universal float32 centered degree range [-180, 180[
        values = values.astype(np.float32)
        for i, name in enumerate(motor_names):
            _, model = self.motors[name]
            resolution = self.model_resolution[model]
            values[i] = values[i] / (resolution // 2) * 180

        return values

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from the universal float32 centered degree range [-180, 180[ to centered resolution range [-resolution, resolution[
        for i, name in enumerate(motor_names):
            _, model = self.motors[name]
            resolution = self.model_resolution[model]

            values[i] = values[i] / 180 * (resolution // 2)

        values = np.round(values).astype(np.int32)

        # Convert from range [-resolution, resolution[ to centered signed int32 range [-2**31, 2**31[
        for i, name in enumerate(motor_names):
            homing_offset, drive_mode = self.calibration[name]
            values[i] -= homing_offset

            # Update direction of rotation of the motor that was matching between leader and follower to their original direction.
            # In fact, the motor of the leader for a given joint can be assembled in an opposite direction in term of rotation
            # than the motor of the follower on the same joint.
            if drive_mode:
                values[i] *= -1

        return values

    def _read_with_motor_ids(self, motor_models, motor_ids, data_name):
        return_list = True
        if not isinstance(motor_ids, list):
            return_list = False
            motor_ids = [motor_ids]

        assert_same_address(self.model_ctrl_table, self.motor_models, data_name)
        addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
        group = GroupSyncRead(self.port_handler, self.packet_handler, addr, bytes)
        for idx in motor_ids:
            group.addParam(idx)

        comm = group.txRxPacket()
        if comm != COMM_SUCCESS:
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

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.apply_calibration(values, motor_names)

        # log the number of seconds it took to read the data from the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time at which the data was received
        ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    def _write_with_motor_ids(self, motor_models, motor_ids, data_name, values):
        if not isinstance(motor_ids, list):
            motor_ids = [motor_ids]
        if not isinstance(values, list):
            values = [values]

        assert_same_address(self.model_ctrl_table, motor_models, data_name)
        addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
        group = GroupSyncWrite(self.port_handler, self.packet_handler, addr, bytes)
        for idx, value in zip(motor_ids, values, strict=True):
            data = convert_to_bytes(value, bytes)
            group.addParam(idx, data)

        comm = group.txPacket()
        if comm != COMM_SUCCESS:
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
            self.group_writers[group_key] = GroupSyncWrite(
                self.port_handler, self.packet_handler, addr, bytes
            )

        for idx, value in zip(motor_ids, values, strict=True):
            data = convert_to_bytes(value, bytes)
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

from copy import deepcopy
import enum
import numpy as np

from dynamixel_sdk import PacketHandler, PortHandler, COMM_SUCCESS, GroupSyncRead, GroupSyncWrite
from dynamixel_sdk import DXL_HIBYTE, DXL_HIWORD, DXL_LOBYTE, DXL_LOWORD

PROTOCOL_VERSION = 2.0
BAUD_RATE = 1_000_000
TIMEOUT_MS = 1000


def u32_to_i32(value: int | np.array) -> int | np.array:
    """
    Convert an unsigned 32-bit integer array to a signed 32-bit integer array.
    """

    if isinstance(value, int):
        if value > 2147483647:
            value = value - 4294967296
    else:
        for i in range(len(value)):
            if value[i] is not None and value[i] > 2147483647:
                value[i] = value[i] - 4294967296

    return value


def i32_to_u32(value: int | np.array) -> int | np.array:
    """
    Convert a signed 32-bit integer array to an unsigned 32-bit integer array.
    """

    if isinstance(value, int):
        if value < 0:
            value = value + 4294967296
    else:
        for i in range(len(value)):
            if value[i] is not None and value[i] < 0:
                value[i] = value[i] + 4294967296

    return value


def retrieve_ids_and_command(values: np.array, ids: np.array) -> (list[int], np.array):
    """
    Convert the values to a chain command. Skip the None values and return the ids and values.
    """

    non_none_values = np.array([value for value in values if value is not None])
    non_none_values_ids = [ids[i] for i, value in enumerate(values) if value is not None]

    return non_none_values_ids, non_none_values


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


# https://emanual.robotis.com/docs/en/dxl/x/xl330-m077
# https://emanual.robotis.com/docs/en/dxl/x/xl330-m288
# https://emanual.robotis.com/docs/en/dxl/x/xl430-w250
# https://emanual.robotis.com/docs/en/dxl/x/xm430-w350
# https://emanual.robotis.com/docs/en/dxl/x/xm540-w270

# data_name, address, size (byte)
X_SERIES_CONTROL_TABLE = [
    ("Model_Number", 0, 2),
    ("Model_Information", 2, 4),
    ("Firmware_Version", 6, 1),
    ("ID", 7, 1),
    ("Baud_Rate", 8, 1),
    ("Return_Delay_Time", 9, 1),
    ("Drive_Mode", 10, 1),
    ("Operating_Mode", 11, 1),
    ("Secondary_ID", 12, 1),
    ("Protocol_Type", 13, 1),
    ("Homing_Offset", 20, 4),
    ("Moving_Threshold", 24, 4),
    ("Temperature_Limit", 31, 1),
    ("Max_Voltage_Limit", 32, 2),
    ("Min_Voltage_Limit", 34, 2),
    ("PWM_Limit", 36, 2),
    ("Current_Limit", 38, 2),
    ("Acceleration_Limit", 40, 4),
    ("Velocity_Limit", 44, 4),
    ("Max_Position_Limit", 48, 4),
    ("Min_Position_Limit", 52, 4),
    ("Shutdown", 63, 1),
    ("Torque_Enable", 64, 1),
    ("LED", 65, 1),
    ("Status_Return_Level", 68, 1),
    ("Registered_Instruction", 69, 1),
    ("Hardware_Error_Status", 70, 1),
    ("Velocity_I_Gain", 76, 2),
    ("Velocity_P_Gain", 78, 2),
    ("Position_D_Gain", 80, 2),
    ("Position_I_Gain", 82, 2),
    ("Position_P_Gain", 84, 2),
    ("Feedforward_2nd_Gain", 88, 2),
    ("Feedforward_1st_Gain", 90, 2),
    ("Bus_Watchdog", 98, 1),
    ("Goal_PWM", 100, 2),
    ("Goal_Current", 102, 2),
    ("Goal_Velocity", 104, 4),
    ("Profile_Acceleration", 108, 4),
    ("Profile_Velocity", 112, 4),
    ("Goal_Position", 116, 4),
    ("Realtime_Tick", 120, 2),
    ("Moving", 122, 1),
    ("Moving_Status", 123, 1),
    ("Present_PWM", 124, 2),
    ("Present_Current", 126, 2),
    ("Present_Velocity", 128, 4),
    ("Present_Position", 132, 4),
    ("Velocity_Trajectory", 136, 4),
    ("Position_Trajectory", 140, 4),
    ("Present_Input_Voltage", 144, 2),
    ("Present_Temperature", 146, 1)
]

MODEL_CONTROL_TABLE = {
    "xl330-m077": X_SERIES_CONTROL_TABLE,
    "xl330-m288": X_SERIES_CONTROL_TABLE,
    "xl430-w250": X_SERIES_CONTROL_TABLE,
    "xm430-w350": X_SERIES_CONTROL_TABLE,
    "xm540-w270": X_SERIES_CONTROL_TABLE,
}


class DynamixelBus:

    def __init__(self, port: str, motor_models: dict[int, str],
                 extra_model_control_table: dict[str, list[tuple]] | None = None):
        self.port = port
        self.motor_models = motor_models

        self.model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
        if extra_model_control_table:
            self.model_ctrl_table.update(extra_model_control_table)

        # Find read/write addresses and number of bytes for each motor
        self.motor_ctrl = {}
        for idx, model in self.motor_models.items():
            for data_name, addr, bytes in self.model_ctrl_table[model]:
                if idx not in self.motor_ctrl:
                    self.motor_ctrl[idx] = {}
                self.motor_ctrl[idx][data_name] = {
                    "addr": addr,
                    "bytes": bytes,
                }

        self.port_handler = PortHandler(self.port)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            raise OSError(f"Failed to open port {self.port}")

        self.port_handler.setBaudRate(BAUD_RATE)
        self.port_handler.setPacketTimeoutMillis(TIMEOUT_MS)

        self.group_readers = {}
        self.group_writers = {}

    @property
    def motor_ids(self) -> list[int]:
        return list(self.motor_models.keys())

    def write(self, data_name, value, motor_idx: int):
        addr = self.motor_ctrl[motor_idx][data_name]["addr"]
        bytes = self.motor_ctrl[motor_idx][data_name]["bytes"]
        args = (self.port_handler, motor_idx, addr, value)
        if bytes == 1:
            comm, err = self.packet_handler.write1ByteTxRx(*args)
        elif bytes == 2:
            comm, err = self.packet_handler.write2ByteTxRx(*args)
        elif bytes == 4:
            comm, err = self.packet_handler.write4ByteTxRx(*args)
        else:
            raise NotImplementedError(
                f"Value of the number of bytes to be sent is expected to be in [1, 2, 4], but {bytes} "
                f"is provided instead.")

        if comm != COMM_SUCCESS:
            raise ConnectionError(
                f"Write failed due to communication error on port {self.port} for motor {motor_idx}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )
        elif err != 0:
            raise ConnectionError(
                f"Write failed due to error {err} on port {self.port} for motor {motor_idx}: "
                f"{self.packet_handler.getTxRxResult(err)}"
            )

    def read(self, data_name, motor_idx: int):
        addr = self.motor_ctrl[motor_idx][data_name]["addr"]
        bytes = self.motor_ctrl[motor_idx][data_name]["bytes"]
        args = (self.port_handler, motor_idx, addr)
        if bytes == 1:
            value, comm, err = self.packet_handler.read1ByteTxRx(*args)
        elif bytes == 2:
            value, comm, err = self.packet_handler.read2ByteTxRx(*args)
        elif bytes == 4:
            value, comm, err = self.packet_handler.read4ByteTxRx(*args)
        else:
            raise NotImplementedError(
                f"Value of the number of bytes to be sent is expected to be in [1, 2, 4], but "
                f"{bytes} is provided instead.")

        if comm != COMM_SUCCESS:
            raise ConnectionError(
                f"Read failed due to communication error on port {self.port} for motor {motor_idx}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )
        elif err != 0:
            raise ConnectionError(
                f"Read failed due to error {err} on port {self.port} for motor {motor_idx}: "
                f"{self.packet_handler.getTxRxResult(err)}"
            )

        return value

    def sync_read(self, data_name, motor_ids: list[int] | None = None):
        if motor_ids is None:
            motor_ids = self.motor_ids

        group_key = f"{data_name}_" + "_".join([str(idx) for idx in motor_ids])
        first_motor_idx = list(self.motor_ctrl.keys())[0]
        addr = self.motor_ctrl[first_motor_idx][data_name]["addr"]
        bytes = self.motor_ctrl[first_motor_idx][data_name]["bytes"]

        if data_name not in self.group_readers:
            self.group_readers[group_key] = GroupSyncRead(self.port_handler, self.packet_handler, addr, bytes)
            for idx in motor_ids:
                self.group_readers[group_key].addParam(idx)

        comm = self.group_readers[group_key].txRxPacket()
        if comm != COMM_SUCCESS:
            raise ConnectionError(
                f"Read failed due to communication error on port {self.port} for group_key {group_key}: "
                f"{self.packet_handler.getTxRxResult(comm)}"
            )

        values = []
        for idx in motor_ids:
            value = self.group_readers[group_key].getData(idx, addr, bytes)
            values.append(value)

        return np.array(values)

    def sync_write(self, data_name, values: int | list[int], motor_ids: int | list[int] | None = None):
        if motor_ids is None:
            motor_ids = self.motor_ids

        if isinstance(motor_ids, int):
            motor_ids = [motor_ids]

        if isinstance(values, (int, np.integer)):
            values = [int(values)] * len(motor_ids)

        if isinstance(values, np.ndarray):
            values = values.tolist()

        group_key = f"{data_name}_" + "_".join([str(idx) for idx in motor_ids])

        first_motor_idx = list(self.motor_ctrl.keys())[0]
        addr = self.motor_ctrl[first_motor_idx][data_name]["addr"]
        bytes = self.motor_ctrl[first_motor_idx][data_name]["bytes"]
        init_group = data_name not in self.group_readers

        if init_group:
            self.group_writers[group_key] = GroupSyncWrite(self.port_handler, self.packet_handler, addr, bytes)

        for idx, value in zip(motor_ids, values):
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
                    f"{bytes} is provided instead.")

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

    def read_model_number(self, motor_idx: int):
        return self.read("Model_Number", motor_idx)

    def read_model_information(self, motor_idx: int):
        return self.read("Model_Information", motor_idx)

    def read_firmware_version(self, motor_idx: int):
        return self.read("Firmware_Version", motor_idx)

    def write_id(self, value, motor_idx: int):
        self.write("ID", value, motor_idx)

    def read_id(self, motor_idx: int):
        return self.read("ID", motor_idx)

    def write_baud_rate(self, value, motor_idx: int):
        self.write("Baud_Rate", value, motor_idx)

    def read_baud_rate(self, motor_idx: int):
        return self.read("Baud_Rate", motor_idx)

    def read_return_delay_time(self, motor_idx: int):
        return self.read("Return_Delay_Time", motor_idx)

    def write_drive_mode(self, value, motor_idx: int):
        self.write("Drive_Mode", value, motor_idx)

    def read_drive_mode(self, motor_idx: int):
        return self.read("Drive_Mode", motor_idx)

    def write_operating_mode(self, value, motor_idx: int):
        self.write("Operating_Mode", value, motor_idx)

    def read_operating_mode(self, motor_idx: int):
        return self.read("Operating_Mode", motor_idx)

    def read_protocol_type(self, motor_idx: int):
        return self.read("Protocol_Type", motor_idx)

    def write_homing_offset(self, value, motor_idx: int):
        self.write("Homing_Offset", value, motor_idx)

    def read_homing_offset(self, motor_idx: int):
        return self.read("Homing_Offset", motor_idx)

    def write_moving_threshold(self, value, motor_idx: int):
        self.write("Moving_Threshold", value, motor_idx)

    def read_moving_threshold(self, motor_idx: int):
        return self.read("Moving_Threshold", motor_idx)

    def write_temperature_limit(self, value, motor_idx: int):
        self.write("Temperature_Limit", value, motor_idx)

    def read_temperature_limit(self, motor_idx: int):
        return self.read("Temperature_Limit", motor_idx)

    def write_max_voltage_limit(self, value, motor_idx: int):
        self.write("Max_Voltage_Limit", value, motor_idx)

    def read_max_voltage_limit(self, motor_idx: int):
        return self.read("Max_Voltage_Limit", motor_idx)

    def write_min_voltage_limit(self, value, motor_idx: int):
        self.write("Min_Voltage_Limit", value, motor_idx)

    def read_min_voltage_limit(self, motor_idx: int):
        return self.read("Min_Voltage_Limit", motor_idx)

    def write_pwm_limit(self, value, motor_idx: int):
        self.write("PWM_Limit", value, motor_idx)

    def read_pwm_limit(self, motor_idx: int):
        return self.read("PWM_Limit", motor_idx)

    def write_current_limit(self, value, motor_idx: int):
        self.write("Current_Limit", value, motor_idx)

    def read_current_limit(self, motor_idx: int):
        return self.read("Current_Limit", motor_idx)

    def write_acceleration_limit(self, value, motor_idx: int):
        self.write("Acceleration_Limit", value, motor_idx)

    def read_acceleration_limit(self, motor_idx: int):
        return self.read("Acceleration_Limit", motor_idx)

    def write_velocity_limit(self, value, motor_idx: int):
        self.write("Velocity_Limit", value, motor_idx)

    def read_velocity_limit(self, motor_idx: int):
        return self.read("Velocity_Limit", motor_idx)

    def write_max_position_limit(self, value, motor_idx: int):
        self.write("Max_Position_Limit", value, motor_idx)

    def read_max_position_limit(self, motor_idx: int):
        return self.read("Max_Position_Limit", motor_idx)

    def write_min_position_limit(self, value, motor_idx: int):
        self.write("Min_Position_Limit", value, motor_idx)

    def read_min_position_limit(self, motor_idx: int):
        return self.read("Min_Position_Limit", motor_idx)

    def write_torque_enable(self, value, motor_idx: int):
        self.write("Torque_Enable", value, motor_idx)

    def read_torque_enable(self, motor_idx: int):
        return self.read("Torque_Enable", motor_idx)

    def write_velocity_i_gain(self, value, motor_idx: int):
        self.write("Velocity_I_Gain", value, motor_idx)

    def read_velocity_i_gain(self, motor_idx: int):
        return self.read("Velocity_I_Gain", motor_idx)

    def write_velocity_p_gain(self, value, motor_idx: int):
        self.write("Velocity_P_Gain", value, motor_idx)

    def read_velocity_p_gain(self, motor_idx: int):
        return self.read("Velocity_P_Gain", motor_idx)

    def write_position_d_gain(self, value, motor_idx: int):
        self.write("Position_D_Gain", value, motor_idx)

    def read_position_d_gain(self, motor_idx: int):
        return self.read("Position_D_Gain", motor_idx)

    def write_position_i_gain(self, value, motor_idx: int):
        self.write("Position_I_Gain", value, motor_idx)

    def read_position_i_gain(self, motor_idx: int):
        return self.read("Position_I_Gain", motor_idx)

    def write_position_p_gain(self, value, motor_idx: int):
        self.write("Position_P_Gain", value, motor_idx)

    def read_position_p_gain(self, motor_idx: int):
        return self.read("Position_P_Gain", motor_idx)

    def write_goal_pwm(self, value, motor_idx: int):
        self.write("Goal_PWM", value, motor_idx)

    def read_goal_pwm(self, motor_idx: int):
        return self.read("Goal_PWM", motor_idx)

    def write_goal_current(self, value, motor_idx: int):
        self.write("Goal_Current", value, motor_idx)

    def read_goal_current(self, motor_idx: int):
        return self.read("Goal_Current", motor_idx)

    def write_goal_velocity(self, value, motor_idx: int):
        self.write("Goal_Velocity", value, motor_idx)

    def read_goal_velocity(self, motor_idx: int):
        return self.read("Goal_Velocity", motor_idx)

    def write_goal_position_u32(self, value, motor_idx: int):
        self.write("Goal_Position", value, motor_idx)

    def write_goal_position_i32(self, value, motor_idx: int):
        self.write("Goal_Position", i32_to_u32(value), motor_idx)

    def read_goal_position_u32(self, motor_idx: int):
        return self.read("Goal_Position", motor_idx)

    def read_goal_position_i32(self, motor_idx: int):
        goal_position_u32 = self.read_goal_position_u32(motor_idx)

        return u32_to_i32(goal_position_u32)

    def read_present_pwm(self, motor_idx: int):
        return self.read("Present_PWM", motor_idx)

    def read_present_current(self, motor_idx: int):
        return self.read("Present_Current", motor_idx)

    def read_present_velocity(self, motor_idx: int):
        return self.read("Present_Velocity", motor_idx)

    def read_present_position_u32(self, motor_idx: int):
        return self.read("Present_Position", motor_idx)

    def read_present_position_i32(self, motor_idx: int):
        present_position_u32 = self.read_present_position_u32(motor_idx)

        return u32_to_i32(present_position_u32)

    def read_present_input_voltage(self, motor_idx: int):
        return self.read("Present_Input_Voltage", motor_idx)

    def read_present_temperature(self, motor_idx: int):
        return self.read("Present_Temperature", motor_idx)

    def sync_read_model_number(self, motor_ids: list[int] | None = None):
        return self.sync_read("Model_Number", motor_ids)

    def sync_read_model_information(self, motor_ids: list[int] | None = None):
        return self.sync_read("Model_Information", motor_ids)

    def sync_read_firmware_version(self, motor_ids: list[int] | None = None):
        return self.sync_read("Firmware_Version", motor_ids)

    def sync_read_id(self, motor_ids: list[int] | None = None):
        return self.sync_read("ID", motor_ids)

    def sync_read_baud_rate(self, motor_ids: list[int] | None = None):
        return self.sync_read("Baud_Rate", motor_ids)

    def sync_read_return_delay_time(self, motor_ids: list[int] | None = None):
        return self.sync_read("Return_Delay_Time", motor_ids)

    def sync_read_drive_mode(self, motor_ids: list[int] | None = None):
        return self.sync_read("Drive_Mode", motor_ids)

    def sync_read_operating_mode(self, motor_ids: list[int] | None = None):
        return self.sync_read("Operating_Mode", motor_ids)

    def sync_read_protocol_type(self, motor_ids: list[int] | None = None):
        return self.sync_read("Protocol_Type", motor_ids)

    def sync_read_homing_offset(self, motor_ids: list[int] | None = None):
        return self.sync_read("Homing_Offset", motor_ids)

    def sync_read_moving_threshold(self, motor_ids: list[int] | None = None):
        return self.sync_read("Moving_Threshold", motor_ids)

    def sync_read_temperature_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("Temperature_Limit", motor_ids)

    def sync_read_max_voltage_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("Max_Voltage_Limit", motor_ids)

    def sync_read_min_voltage_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("Min_Voltage_Limit", motor_ids)

    def sync_read_pwm_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("PWM_Limit", motor_ids)

    def sync_read_current_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("Current_Limit", motor_ids)

    def sync_read_acceleration_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("Acceleration_Limit", motor_ids)

    def sync_read_velocity_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("Velocity_Limit", motor_ids)

    def sync_read_max_position_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("Max_Position_Limit", motor_ids)

    def sync_read_min_position_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("Min_Position_Limit", motor_ids)

    def sync_read_torque_enable(self, motor_ids: list[int] | None = None):
        return self.sync_read("Torque_Enable", motor_ids)

    def sync_read_velocity_i_gain(self, motor_ids: list[int] | None = None):
        return self.sync_read("Velocity_I_Gain", motor_ids)

    def sync_read_velocity_p_gain(self, motor_ids: list[int] | None = None):
        return self.sync_read("Velocity_P_Gain", motor_ids)

    def sync_read_position_d_gain(self, motor_ids: list[int] | None = None):
        return self.sync_read("Position_D_Gain", motor_ids)

    def sync_read_position_i_gain(self, motor_ids: list[int] | None = None):
        return self.sync_read("Position_I_Gain", motor_ids)

    def sync_read_position_p_gain(self, motor_ids: list[int] | None = None):
        return self.sync_read("Position_P_Gain", motor_ids)

    def sync_read_goal_pwm(self, motor_ids: list[int] | None = None):
        return self.sync_read("Goal_PWM", motor_ids)

    def sync_read_goal_current(self, motor_ids: list[int] | None = None):
        return self.sync_read("Goal_Current", motor_ids)

    def sync_read_goal_velocity(self, motor_ids: list[int] | None = None):
        return self.sync_read("Goal_Velocity", motor_ids)

    def sync_read_goal_position_u32(self, motor_ids: list[int] | None = None):
        return self.sync_read("Goal_Position", motor_ids)

    def sync_read_goal_position_i32(self, motor_ids: list[int] | None = None):
        goal_position_u32 = self.sync_read_goal_position_u32(motor_ids)

        return u32_to_i32(goal_position_u32)

    def sync_read_present_pwm(self, motor_ids: list[int] | None = None):
        return self.sync_read("Present_PWM", motor_ids)

    def sync_read_present_current(self, motor_ids: list[int] | None = None):
        return self.sync_read("Present_Current", motor_ids)

    def sync_read_present_velocity(self, motor_ids: list[int] | None = None):
        return self.sync_read("Present_Velocity", motor_ids)

    def sync_read_present_position_u32(self, motor_ids: list[int] | None = None):
        return self.sync_read("Present_Position", motor_ids)

    def sync_read_present_position_i32(self, motor_ids: list[int] | None = None):
        present_position_u32 = self.sync_read_present_position_u32(motor_ids)

        return u32_to_i32(present_position_u32)

    def sync_read_present_input_voltage(self, motor_ids: list[int] | None = None):
        return self.sync_read("Present_Input_Voltage", motor_ids)

    def sync_read_present_temperature(self, motor_ids: list[int] | None = None):
        return self.sync_read("Present_Temperature", motor_ids)

    def sync_write_id(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("ID", values, motor_ids)

    def sync_write_baud_rate(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Baud_Rate", values, motor_ids)

    def sync_write_drive_mode(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Drive_Mode", values, motor_ids)

    def sync_write_operating_mode(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Operating_Mode", values, motor_ids)

    def sync_write_homing_offset(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Homing_Offset", values, motor_ids)

    def sync_write_moving_threshold(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Moving_Threshold", values, motor_ids)

    def sync_write_temperature_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Temperature_Limit", values, motor_ids)

    def sync_write_max_voltage_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Max_Voltage_Limit", values, motor_ids)

    def sync_write_min_voltage_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Min_Voltage_Limit", values, motor_ids)

    def sync_write_pwm_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("PWM_Limit", values, motor_ids)

    def sync_write_current_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Current_Limit", values, motor_ids)

    def sync_write_acceleration_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Acceleration_Limit", values, motor_ids)

    def sync_write_velocity_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Velocity_Limit", values, motor_ids)

    def sync_write_max_position_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Max_Position_Limit", values, motor_ids)

    def sync_write_min_position_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Min_Position_Limit", values, motor_ids)

    def sync_write_torque_enable(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Torque_Enable", values, motor_ids)

    def sync_write_velocity_i_gain(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Velocity_I_Gain", values, motor_ids)

    def sync_write_velocity_p_gain(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Velocity_P_Gain", values, motor_ids)

    def sync_write_position_d_gain(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Position_D_Gain", values, motor_ids)

    def sync_write_position_i_gain(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Position_I_Gain", values, motor_ids)

    def sync_write_position_p_gain(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Position_P_Gain", values, motor_ids)

    def sync_write_goal_pwm(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Goal_PWM", values, motor_ids)

    def sync_write_goal_current(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Goal_Current", values, motor_ids)

    def sync_write_goal_velocity(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Goal_Velocity", values, motor_ids)

    def sync_write_goal_position_u32(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Goal_Position", values, motor_ids)

    def sync_write_goal_position_i32(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Goal_Position", i32_to_u32(values), motor_ids)

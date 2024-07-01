from copy import deepcopy
import enum
import numpy as np

from scservo_sdk import PacketHandler, PortHandler, COMM_SUCCESS, GroupSyncRead, GroupSyncWrite
from scservo_sdk import SCS_HIBYTE, SCS_HIBYTE, SCS_LOBYTE, SCS_LOWORD

PROTOCOL_VERSION = 0
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
    pass


class DriveMode(enum.Enum):
    pass


SCS_SERIES_CONTROL_TABLE = [
    ("Model", 3, 2),
    ("ID", 5, 1),
    ("Baud_Rate", 6, 1),
    ("Return_Delay", 7, 1),
    ("Response_Status_Level", 8, 1),
    ("Min_Angle_Limit", 9, 2),
    ("Max_Angle_Limit", 11, 2),
    ("Max_Temperature_Limit", 13, 1),
    ("Max_Voltage_Limit", 14, 1),
    ("Min_Voltage_Limit", 15, 1),
    ("Max_Torque_Limit", 16, 2),
    ("Phase", 18, 1),
    ("Unloading_Condition", 19, 1),
    ("LED_Alarm_Condition", 20, 1),
    ("P_Coefficient", 21, 1),
    ("D_Coefficient", 22, 1),
    ("I_Coefficient", 23, 1),
    ("Minimum_Startup_Force", 24, 2),
    ("CW_Dead_Zone", 26, 1),
    ("CCW_Dead_Zone", 27, 1),
    ("Protection_Current", 28, 2),
    ("Angular_Resolution", 30, 1),
    ("Offset", 31, 2),
    ("Mode", 33, 1),
    ("Protective_Torque", 34, 1),
    ("Protection_Time", 35, 1),
    ("Overload_Torque", 36, 1),
    ("Speed_closed_loop_P_proportional_coefficient", 37, 1),
    ("Over_Current_Protection_Time", 38, 1),
    ("Velocity_closed_loop_I_integral_coefficient", 39, 1),
    ("Torque_Enable", 40, 1),
    ("Acceleration", 41, 1),
    ("Goal_Position", 42, 2),
    ("Goal_Time", 44, 2),
    ("Goal_Speed", 46, 2),
    ("Lock", 55, 1),
    ("Present_Position", 56, 2),
    ("Present_Speed", 58, 2),
    ("Present_Load", 60, 2),
    ("Present_Voltage", 62, 1),
    ("Present_Temperature", 63, 1),
    ("Status", 65, 1),
    ("Moving", 66, 1),
    ("Present_Current", 69, 2)
]

MODEL_CONTROL_TABLE = {
    "scs_series": SCS_SERIES_CONTROL_TABLE,
    "sts3215": SCS_SERIES_CONTROL_TABLE,
}


class FeetechBus:

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

    def close(self):
        self.port_handler.closePort()

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
                    SCS_LOBYTE(SCS_LOWORD(value)),
                ]
            elif bytes == 2:
                data = [
                    SCS_LOBYTE(SCS_LOWORD(value)),
                    SCS_HIBYTE(SCS_LOWORD(value)),
                ]
            elif bytes == 4:
                data = [
                    SCS_LOBYTE(SCS_LOWORD(value)),
                    SCS_HIBYTE(SCS_LOWORD(value)),
                    SCS_LOBYTE(SCS_HIBYTE(value)),
                    SCS_HIBYTE(SCS_HIBYTE(value)),
                ]
            else:
                raise NotImplementedError(
                    f"Value of the number of bytes to be sent is expected to be in [1, 2, 4], but {bytes} "
                    f"is provided instead.")

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

    def read_model(self, motor_idx: int):
        return self.read("Model", motor_idx)

    def sync_read_model(self, motor_ids: list[int] | None = None):
        return self.sync_read("Model", motor_ids)

    def write_id(self, value, motor_idx: int):
        self.write("ID", value, motor_idx)

    def read_id(self, motor_idx: int):
        return self.read("ID", motor_idx)

    def sync_read_id(self, motor_ids: list[int] | None = None):
        return self.sync_read("ID", motor_ids)

    def sync_write_id(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("ID", values, motor_ids)

    def write_baud_rate(self, value, motor_idx: int):
        self.write("Baud_Rate", value, motor_idx)

    def read_baud_rate(self, motor_idx: int):
        return self.read("Baud_Rate", motor_idx)

    def sync_read_baud_rate(self, motor_ids: list[int] | None = None):
        return self.sync_read("Baud_Rate", motor_ids)

    def sync_write_baud_rate(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Baud_Rate", values, motor_ids)

    def read_return_delay(self, motor_idx: int):
        return self.read("Return_Delay", motor_idx)

    def sync_read_return_delay(self, motor_ids: list[int] | None = None):
        return self.sync_read("Return_Delay", motor_ids)

    def read_response_status_level(self, motor_idx: int):
        return self.read("Response_Status_Level", motor_idx)

    def sync_read_response_status_level(self, motor_ids: list[int] | None = None):
        return self.sync_read("Response_Status_Level", motor_ids)

    def write_min_angle_limit(self, value, motor_idx: int):
        self.write("Min_Angle_Limit", value, motor_idx)

    def read_min_angle_limit(self, motor_idx: int):
        return self.read("Min_Angle_Limit", motor_idx)

    def sync_read_min_angle_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("Min_Angle_Limit", motor_ids)

    def sync_write_min_angle_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Min_Angle_Limit", values, motor_ids)

    def write_max_angle_limit(self, value, motor_idx: int):
        self.write("Max_Angle_Limit", value, motor_idx)

    def read_max_angle_limit(self, motor_idx: int):
        return self.read("Max_Angle_Limit", motor_idx)

    def sync_read_max_angle_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("Max_Angle_Limit", motor_ids)

    def sync_write_max_angle_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Max_Angle_Limit", values, motor_ids)

    def write_max_temperature_limit(self, value, motor_idx: int):
        self.write("Max_Temperature_Limit", value, motor_idx)

    def read_max_temperature_limit(self, motor_idx: int):
        return self.read("Max_Temperature_Limit", motor_idx)

    def sync_read_max_temperature_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("Max_Temperature_Limit", motor_ids)

    def sync_write_max_temperature_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Max_Temperature_Limit", values, motor_ids)

    def write_max_voltage_limit(self, value, motor_idx: int):
        self.write("Max_Voltage_Limit", value, motor_idx)

    def read_max_voltage_limit(self, motor_idx: int):
        return self.read("Max_Voltage_Limit", motor_idx)

    def sync_read_max_voltage_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("Max_Voltage_Limit", motor_ids)

    def sync_write_max_voltage_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Max_Voltage_Limit", values, motor_ids)

    def write_min_voltage_limit(self, value, motor_idx: int):
        self.write("Min_Voltage_Limit", value, motor_idx)

    def read_min_voltage_limit(self, motor_idx: int):
        return self.read("Min_Voltage_Limit", motor_idx)

    def sync_read_min_voltage_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("Min_Voltage_Limit", motor_ids)

    def sync_write_min_voltage_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Min_Voltage_Limit", values, motor_ids)

    def write_max_torque_limit(self, value, motor_idx: int):
        self.write("Max_Torque_Limit", value, motor_idx)

    def read_max_torque_limit(self, motor_idx: int):
        return self.read("Max_Torque_Limit", motor_idx)

    def sync_read_max_torque_limit(self, motor_ids: list[int] | None = None):
        return self.sync_read("Max_Torque_Limit", motor_ids)

    def sync_write_max_torque_limit(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Max_Torque_Limit", values, motor_ids)

    def write_p_coefficient(self, value, motor_idx: int):
        self.write("P_Coefficient", value, motor_idx)

    def read_p_coefficient(self, motor_idx: int):
        return self.read("P_Coefficient", motor_idx)

    def sync_read_p_coefficient(self, motor_ids: list[int] | None = None):
        return self.sync_read("P_Coefficient", motor_ids)

    def sync_write_p_coefficient(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("P_Coefficient", values, motor_ids)

    def write_d_coefficient(self, value, motor_idx: int):
        self.write("D_Coefficient", value, motor_idx)

    def read_d_coefficient(self, motor_idx: int):
        return self.read("D_Coefficient", motor_idx)

    def sync_read_d_coefficient(self, motor_ids: list[int] | None = None):
        return self.sync_read("D_Coefficient", motor_ids)

    def sync_write_d_coefficient(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("D_Coefficient", values, motor_ids)

    def write_i_coefficient(self, value, motor_idx: int):
        self.write("I_Coefficient", value, motor_idx)

    def read_i_coefficient(self, motor_idx: int):
        return self.read("I_Coefficient", motor_idx)

    def sync_read_i_coefficient(self, motor_ids: list[int] | None = None):
        return self.sync_read("I_Coefficient", motor_ids)

    def sync_write_i_coefficient(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("I_Coefficient", values, motor_ids)

    def write_minimum_startup_force(self, value, motor_idx: int):
        self.write("Minimum_Startup_Force", value, motor_idx)

    def read_minimum_startup_force(self, motor_idx: int):
        return self.read("Minimum_Startup_Force", motor_idx)

    def sync_read_minimum_startup_force(self, motor_ids: list[int] | None = None):
        return self.sync_read("Minimum_Startup_Force", motor_ids)

    def sync_write_minimum_startup_force(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Minimum_Startup_Force", values, motor_ids)

    def write_cw_dead_zone(self, value, motor_idx: int):
        self.write("CW_Dead_Zone", value, motor_idx)

    def read_cw_dead_zone(self, motor_idx: int):
        return self.read("CW_Dead_Zone", motor_idx)

    def sync_read_cw_dead_zone(self, motor_ids: list[int] | None = None):
        return self.sync_read("CW_Dead_Zone", motor_ids)

    def sync_write_cw_dead_zone(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("CW_Dead_Zone", values, motor_ids)

    def write_ccw_dead_zone(self, value, motor_idx: int):
        self.write("CCW_Dead_Zone", value, motor_idx)

    def read_ccw_dead_zone(self, motor_idx: int):
        return self.read("CCW_Dead_Zone", motor_idx)

    def sync_read_ccw_dead_zone(self, motor_ids: list[int] | None = None):
        return self.sync_read("CCW_Dead_Zone", motor_ids)

    def sync_write_ccw_dead_zone(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("CCW_Dead_Zone", values, motor_ids)

    def write_protection_current(self, value, motor_idx: int):
        self.write("Protection_Current", value, motor_idx)

    def read_protection_current(self, motor_idx: int):
        return self.read("Protection_Current", motor_idx)

    def sync_read_protection_current(self, motor_ids: list[int] | None = None):
        return self.sync_read("Protection_Current", motor_ids)

    def sync_write_protection_current(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Protection_Current", values, motor_ids)

    def read_angular_resolution(self, motor_idx: int):
        return self.read("Angular_Resolution", motor_idx)

    def sync_read_angular_resolution(self, motor_ids: list[int] | None = None):
        return self.sync_read("Angular_Resolution", motor_ids)

    def write_offset(self, value, motor_idx: int):
        self.write("Offset", value, motor_idx)

    def read_offset(self, motor_idx: int):
        return self.read("Offset", motor_idx)

    def sync_read_offset(self, motor_ids: list[int] | None = None):
        return self.sync_read("Offset", motor_ids)

    def sync_write_offset(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Offset", values, motor_ids)

    def write_mode(self, value, motor_idx: int):
        self.write("Mode", value, motor_idx)

    def read_mode(self, motor_idx: int):
        return self.read("Mode", motor_idx)

    def sync_read_mode(self, motor_ids: list[int] | None = None):
        return self.sync_read("Mode", motor_ids)

    def sync_write_mode(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Mode", values, motor_ids)

    def write_protective_torque(self, value, motor_idx: int):
        self.write("Protective_Torque", value, motor_idx)

    def read_protective_torque(self, motor_idx: int):
        return self.read("Protective_Torque", motor_idx)

    def sync_read_protective_torque(self, motor_ids: list[int] | None = None):
        return self.sync_read("Protective_Torque", motor_ids)

    def sync_write_protective_torque(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Protective_Torque", values, motor_ids)

    def read_protection_time(self, motor_idx: int):
        return self.read("Protection_Time", motor_idx)

    def sync_read_protection_time(self, motor_ids: list[int] | None = None):
        return self.sync_read("Protection_Time", motor_ids)

    def write_speed_closed_loop_p_proportional_coefficient(self, value, motor_idx: int):
        self.write("Speed_closed_loop_P_proportional_coefficient", value, motor_idx)

    def read_speed_closed_loop_p_proportional_coefficient(self, motor_idx: int):
        return self.read("Speed_closed_loop_P_proportional_coefficient", motor_idx)

    def sync_read_speed_closed_loop_p_proportional_coefficient(self, motor_ids: list[int] | None = None):
        return self.sync_read("Speed_closed_loop_P_proportional_coefficient", motor_ids)

    def sync_write_speed_closed_loop_p_proportional_coefficient(self, values: int | list[int],
                                                                motor_ids: list[int] | None = None):
        self.sync_write("Speed_closed_loop_P_proportional_coefficient", values, motor_ids)

    def write_over_current_protection_time(self, value, motor_idx: int):
        self.write("Over_Current_Protection_Time", value, motor_idx)

    def read_over_current_protection_time(self, motor_idx: int):
        return self.read("Over_Current_Protection_Time", motor_idx)

    def sync_read_over_current_protection_time(self, motor_ids: list[int] | None = None):
        return self.sync_read("Over_Current_Protection_Time", motor_ids)

    def sync_write_over_current_protection_time(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Over_Current_Protection_Time", values, motor_ids)

    def write_velocity_closed_loop_i_integral_coefficient(self, value, motor_idx: int):
        self.write("Velocity_closed_loop_I_integral_coefficient", value, motor_idx)

    def read_velocity_closed_loop_i_integral_coefficient(self, motor_idx: int):
        return self.read("Velocity_closed_loop_I_integral_coefficient", motor_idx)

    def sync_read_velocity_closed_loop_i_integral_coefficient(self, motor_ids: list[int] | None = None):
        return self.sync_read("Velocity_closed_loop_I_integral_coefficient", motor_ids)

    def sync_write_velocity_closed_loop_i_integral_coefficient(self, values: int | list[int],
                                                               motor_ids: list[int] | None = None):
        self.sync_write("Velocity_closed_loop_I_integral_coefficient", values, motor_ids)

    def write_torque_enable(self, value, motor_idx: int):
        self.write("Torque_Enable", value, motor_idx)

    def read_torque_enable(self, motor_idx: int):
        return self.read("Torque_Enable", motor_idx)

    def sync_read_torque_enable(self, motor_ids: list[int] | None = None):
        return self.sync_read("Torque_Enable", motor_ids)

    def sync_write_torque_enable(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Torque_Enable", values, motor_ids)

    def write_goal_position_u32(self, value, motor_idx: int):
        self.write("Goal_Position", value, motor_idx)

    def write_goal_position_i32(self, value, motor_idx: int):
        self.write("Goal_Position", i32_to_u32(value), motor_idx)

    def read_goal_position_u32(self, motor_idx: int):
        return self.read("Goal_Position", motor_idx)

    def read_goal_position_i32(self, motor_idx: int):
        goal_position_u32 = self.read_goal_position_u32(motor_idx)

        return u32_to_i32(goal_position_u32)

    def sync_read_goal_position_u32(self, motor_ids: list[int] | None = None):
        return self.sync_read("Goal_Position", motor_ids)

    def sync_read_goal_position_i32(self, motor_ids: list[int] | None = None):
        goal_position_u32 = self.sync_read_goal_position_u32(motor_ids)

        return u32_to_i32(goal_position_u32)

    def sync_write_goal_position_u32(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Goal_Position", values, motor_ids)

    def sync_write_goal_position_i32(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Goal_Position", i32_to_u32(values), motor_ids)

    def write_goal_time(self, value, motor_idx: int):
        self.write("Goal_Time", value, motor_idx)

    def read_goal_time(self, motor_idx: int):
        return self.read("Goal_Time", motor_idx)

    def sync_read_goal_time(self, motor_ids: list[int] | None = None):
        return self.sync_read("Goal_Time", motor_ids)

    def sync_write_goal_time(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Goal_Time", values, motor_ids)

    def write_goal_speed(self, value, motor_idx: int):
        self.write("Goal_Speed", value, motor_idx)

    def read_goal_speed(self, motor_idx: int):
        return self.read("Goal_Speed", motor_idx)

    def sync_read_goal_speed(self, motor_ids: list[int] | None = None):
        return self.sync_read("Goal_Speed", motor_ids)

    def sync_write_goal_speed(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Goal_Speed", values, motor_ids)

    def write_lock(self, value, motor_idx: int):
        self.write("Lock", value, motor_idx)

    def read_lock(self, motor_idx: int):
        return self.read("Lock", motor_idx)

    def sync_read_lock(self, motor_ids: list[int] | None = None):
        return self.sync_read("Lock", motor_ids)

    def sync_write_lock(self, values: int | list[int], motor_ids: list[int] | None = None):
        self.sync_write("Lock", values, motor_ids)

    def read_present_position_u32(self, motor_idx: int):
        return self.read("Present_Position", motor_idx)

    def read_present_position_i32(self, motor_idx: int):
        present_position_u32 = self.read_present_position_u32(motor_idx)

        return u32_to_i32(present_position_u32)

    def sync_read_present_position_u32(self, motor_ids: list[int] | None = None):
        return self.sync_read("Present_Position", motor_ids)

    def sync_read_present_position_i32(self, motor_ids: list[int] | None = None):
        present_position_u32 = self.sync_read_present_position_u32(motor_ids)

        return u32_to_i32(present_position_u32)

    def read_present_speed(self, motor_idx: int):
        return self.read("Present_Speed", motor_idx)

    def sync_read_present_speed(self, motor_ids: list[int] | None = None):
        return self.sync_read("Present_Speed", motor_ids)

    def read_present_load(self, motor_idx: int):
        return self.read("Present_Load", motor_idx)

    def sync_read_present_load(self, motor_ids: list[int] | None = None):
        return self.sync_read("Present_Load", motor_ids)

    def read_present_voltage(self, motor_idx: int):
        return self.read("Present_Voltage", motor_idx)

    def sync_read_present_voltage(self, motor_ids: list[int] | None = None):
        return self.sync_read("Present_Voltage", motor_ids)

    def read_present_temperature(self, motor_idx: int):
        return self.read("Present_Temperature", motor_idx)

    def sync_read_present_temperature(self, motor_ids: list[int] | None = None):
        return self.sync_read("Present_Temperature", motor_ids)

    def read_moving(self, motor_idx: int):
        return self.read("Moving", motor_idx)

    def sync_read_moving(self, motor_ids: list[int] | None = None):
        return self.sync_read("Moving", motor_ids)

    def read_present_current(self, motor_idx: int):
        return self.read("Present_Current", motor_idx)

    def sync_read_present_current(self, motor_ids: list[int] | None = None):
        return self.sync_read("Present_Current", motor_ids)

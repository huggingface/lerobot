from copy import deepcopy
import enum
import numpy as np

from dynamixel_sdk import PacketHandler, PortHandler, COMM_SUCCESS, GroupSyncRead, GroupSyncWrite


def write_goal_current(io: PacketHandler, serial: PortHandler, servo_id: int, address: int, goal_current: int | None = None):
    """
    Write the goal current to the puppet robot
    :param io: PacketHandler
    :param serial: PortHandler
    :param servo_id: int
    :param goal_current: current
    """
    comm, error = io.write2ByteTxRx(serial, servo_id, 102, goal_current)

    if goal_current is not None:
        if comm != COMM_SUCCESS:
            print(f"Failed to communicate with motor {servo_id}")
            print("%s" % io.getTxRxResult(comm))
        if error != 0:
            print(f"Error while writing goal current to motor {servo_id}")
            print("%s" % io.getRxPacketError(error))


def read_present_current(io: PacketHandler, serial: PortHandler, servo_id: int, address: int) -> int | None:
    """
    Read the present current from the puppet robot
    :param io: PacketHandler
    :param serial: PortHandler
    :param servo_id: int
    :return: int
    """

    current, comm, error = io.read2ByteTxRx(serial, servo_id, 126)

    if comm != COMM_SUCCESS:
        print(f"Failed to communicate with motor {servo_id}")
        print("%s" % io.getTxRxResult(comm))
        if error != 0:
            print(f"Error while reading present current from motor {servo_id}")
            print("%s" % io.getRxPacketError(error))

    return current if comm == COMM_SUCCESS and error == 0 else None


def write_goal_position(io: PacketHandler, serial: PortHandler, servo_id: int, address: int, goal_position: int | None = None):
    """
    Write the goal position to the puppet robot
    :param io: PacketHandler
    :param serial: PortHandler
    :param servo_id: int
    :param goal_position:
    """
    comm, error = io.write4ByteTxRx(serial, servo_id, 116, goal_position)

    if goal_position is not None:
        if comm != COMM_SUCCESS:
            print(f"Failed to communicate with motor {servo_id}")
            print("%s" % io.getTxRxResult(comm))
        if error != 0:
            print(f"Error while writing goal position to motor {servo_id}")
            print("%s" % io.getRxPacketError(error))


def write_goal_positions(io: PacketHandler, serial: PortHandler, ids: np.array, address: int, goal_positions: np.array):
    """
    Write the goal positions to the puppet robot
    :param io: PacketHandler
    :param serial: PortHandler
    :param ids: np.array
    :param goal_positions: np.array
    """

    for i in range(len(ids)):
        if goal_positions[i] is not None:
            comm, error = io.write4ByteTxRx(serial, ids[i], 116, goal_positions[i])
            if comm != COMM_SUCCESS:
                print(f"Failed to communicate with motor {ids[i]}")
                print("%s" % io.getTxRxResult(comm))
            if error != 0:
                print(f"Error while writing goal position to motor {ids[i]}")
                print("%s" % io.getRxPacketError(error))


def read_present_position(io: PacketHandler, serial: PortHandler, servo_id: int, address: int):
    """
    Read the present position from the puppet robot
    :param io: PacketHandler
    :param serial: PortHandler
    :param servo_id: int
    :return: int
    """

    position, comm, error = io.read4ByteTxRx(serial, servo_id, 132)

    if comm != COMM_SUCCESS:
        print(f"Failed to communicate with motor {servo_id}")
        print("%s" % io.getTxRxResult(comm))
    if error != 0:
        print(f"Error while reading present position from motor {servo_id}")
        print("%s" % io.getRxPacketError(error))

    return position if comm == COMM_SUCCESS and error == 0 else None


def read_present_positions(io: PacketHandler, serial: PortHandler, ids: np.array, address: int):
    """
    Read the present positions from the puppet robot
    :param io: PacketHandler
    :param serial: PortHandler
    :param ids: np.array
    :return: np.array
    """
    present_positions = []

    for id_ in ids:
        position, comm, error = io.read4ByteTxRx(serial, id_, 132)

        if comm != COMM_SUCCESS:
            print(f"Failed to communicate with motor {id_}")
            print("%s" % io.getTxRxResult(comm))
        if error != 0:
            print(f"Error while reading present position from motor {id_}")
            print("%s" % io.getRxPacketError(error))

        present_positions.append(position if comm == COMM_SUCCESS and error == 0 else None)

    return np.array(present_positions)


def read_present_velocity(io: PacketHandler, serial: PortHandler, servo_id: int, address: int):
    """
    Read the present velocity from the puppet robot
    :param io: PacketHandler
    :param serial: PortHandler
    :param id: int
    :return: int
    """

    velocity, comm, error = io.read4ByteTxRx(serial, servo_id, 128)

    if comm != COMM_SUCCESS:
        print(f"Failed to communicate with motor {servo_id}")
        print("%s" % io.getTxRxResult(comm))
    if error != 0:
        print(f"Error while reading present velocity from motor {servo_id}")
        print("%s" % io.getRxPacketError(error))

    return velocity if comm == COMM_SUCCESS and error == 0 else None


def read_present_velocities(io: PacketHandler, serial: PortHandler, ids: np.array, address: int):
    """
    Read the present velocities from the puppet robot
    :param io: PacketHandler
    :param serial: PortHandler
    :param ids: list
    :return: list
    """
    present_velocities = []

    for id_ in ids:
        velocity, comm, error = io.read4ByteTxRx(serial, id_, 128)

        if comm != COMM_SUCCESS:
            print(f"Failed to communicate with motor {id_}")
            print("%s" % io.getTxRxResult(comm))
        if error != 0:
            print(f"Error while reading present velocity from motor {id_}")
            print("%s" % io.getRxPacketError(error))

        present_velocities.append(velocity if comm == COMM_SUCCESS and error == 0 else None)

    return np.array(present_velocities)


def enable_torque(io: PacketHandler, serial: PortHandler, servo_id: int, address: int):
    """
    Enable the torque of the puppet robot
    :param io: PacketHandler
    :param serial: PortHandler
    :param servo_id: int
    """
    comm, error = io.write1ByteTxRx(serial, servo_id, 64, 1)

    if comm != COMM_SUCCESS:
        print(f"Failed to communicate with motor {servo_id}")
        print("%s" % io.getTxRxResult(comm))
    if error != 0:
        print(f"Error while enabling torque for motor {servo_id}")
        print("%s" % io.getRxPacketError(error))


def enable_torques(io: PacketHandler, serial: PortHandler, ids: np.array, address: int):
    """
    Enable the torques of the puppet robot
    :param io: PacketHandler
    :param serial: PortHandler
    :param ids: np.array
    """
    for id_ in ids:
        comm, error = io.write1ByteTxRx(serial, id_, 64, 1)

        if comm != COMM_SUCCESS:
            print(f"Failed to communicate with motor {id_}")
            print("%s" % io.getTxRxResult(comm))
        if error != 0:
            print(f"Error while enabling torque for motor {id_}")
            print("%s" % io.getRxPacketError(error))


def disable_torque(io: PacketHandler, serial: PortHandler, servo_id: int, address: int):
    """
    Disable the torque of the puppet robot
    :param io: PacketHandler
    :param serial: PortHandler
    :param servo_id: int
    """
    comm, error = io.write1ByteTxRx(serial, servo_id, 64, 0)

    if comm != COMM_SUCCESS:
        print(f"Failed to communicate with motor {servo_id}")
        print("%s" % io.getTxRxResult(comm))
    if error != 0:
        print(f"Error while disabling torque for motor {servo_id}")
        print("%s" % io.getRxPacketError(error))


def disable_torques(io: PacketHandler, serial: PortHandler, ids: np.array, address: int):
    """
    Disable the torques of the puppet robot
    :param io: PacketHandler
    :param serial: PortHandler
    :param ids: np.array
    """
    for id_ in ids:
        comm, error = io.write1ByteTxRx(serial, id_, 64, 0)

        if comm != COMM_SUCCESS:
            print(f"Failed to communicate with motor {id_}")
            print("%s" % io.getTxRxResult(comm))
        if error != 0:
            print(f"Error while disabling torque for motor {id_}")
            print("%s" % io.getRxPacketError(error))


def write_operating_mode(io: PacketHandler, serial: PortHandler, servo_id: int, address: int, mode: int):
    """
    Write the appropriate operating mode for the LCR.
    :param io: PacketHandler
    :param serial: PortHandler
    :param servo_id: int
    :param mode: mode to write
    """
    comm, error = io.write1ByteTxRx(serial, servo_id, 11, mode)

    if mode is not None:
        if comm != COMM_SUCCESS:
            print(f"Failed to communicate with motor {servo_id}")
            print("%s" % io.getTxRxResult(comm))
        if error != 0:
            print(f"Error while writing operating mode for motor {servo_id}")
            print("%s" % io.getRxPacketError(error))


def write_operating_modes(io: PacketHandler, serial: PortHandler, ids: np.array, address: int, mode: int):
    """
    Write the appropriate operating mode for the LCR.
    :param io: PacketHandler
    :param serial: PortHandler
    :param ids: numpy array of motor ids
    :param mode: mode to write
    """
    for id_ in ids:
        comm, error = io.write1ByteTxRx(serial, id_, 11, mode)

        if mode is not None:
            if comm != COMM_SUCCESS:
                print(f"Failed to communicate with motor {id_}")
                print("%s" % io.getTxRxResult(comm))
            if error != 0:
                print(f"Error while writing operating mode for motor {id_}")
                print("%s" % io.getRxPacketError(error))


def write_homing_offset(io: PacketHandler, serial: PortHandler, servo_id: int, address: int, offset: int):
    """
    Write the homing offset for the LCR.
    :param io: PacketHandler
    :param serial: PortHandler
    :param servo_id: int
    :param offset: int
    """
    comm, error = io.write4ByteTxRx(serial, servo_id, 20, offset)

    if offset is not None:
        if comm != COMM_SUCCESS:
            print(f"Failed to communicate with motor {servo_id}")
            print("%s" % io.getTxRxResult(comm))
        if error != 0:
            print(f"Error while writing homing offset for motor {servo_id}")
            print("%s" % io.getRxPacketError(error))


def write_homing_offsets(io: PacketHandler, serial: PortHandler, ids: np.array, address: int, offsets: np.array):
    """
    Write the homing offset for the LCR.
    :param io: PacketHandler
    :param serial: PortHandler
    :param ids: numpy array of motor ids
    :param offsets: numpy array of offsets
    """
    for i, id_ in enumerate(ids):
        comm, error = io.write4ByteTxRx(serial, id_, 20, int(offsets[i]))

        if offsets[i] is not None:
            if comm != COMM_SUCCESS:
                print(f"Failed to communicate with motor {id_}")
                print("%s" % io.getTxRxResult(comm))
            if error != 0:
                print(f"Error while writing homing offset for motor {id_}")
                print("%s" % io.getRxPacketError(error))


def write_drive_mode(io: PacketHandler, serial: PortHandler, servo_id: int, address: int, mode: int):
    """
    Write the drive mode for the LCR.
    :param io: PacketHandler
    :param serial: PortHandler
    :param servo_id: int
    :param mode: int
    """
    comm, error = io.write1ByteTxRx(serial, servo_id, 10, mode)

    if mode is not None:
        if comm != COMM_SUCCESS:
            print(f"Failed to communicate with motor {servo_id}")
            print("%s" % io.getTxRxResult(comm))
        if error != 0:
            print(f"Error while writing drive mode for motor {servo_id}")
            print("%s" % io.getRxPacketError(error))


def write_drive_modes(io: PacketHandler, serial: PortHandler, ids: np.array, address: int, modes: np.array):
    """
    Write the drive mode for the LCR.
    :param io: PacketHandler
    :param serial: PortHandler
    :param ids: numpy array of motor ids
    :param modes: numpy array of drive modes
    """
    for i, id_ in enumerate(ids):
        comm, error = io.write1ByteTxRx(serial, id_, 10, modes[i])

        if modes[i] is not None:
            if comm != COMM_SUCCESS:
                print(f"Failed to communicate with motor {id_}")
                print("%s" % io.getTxRxResult(comm))
            if error != 0:
                print(f"Error while writing drive mode for motor {id_}")
                print("%s" % io.getRxPacketError(error))


def write_current_limit(io: PacketHandler, serial: PortHandler, servo_id: int, address: int, limit: int):
    """
    Write the current limit for the LCR.
    :param io: PacketHandler
    :param serial: PortHandler
    :param servo_id: int
    :param limit: int
    """
    comm, error = io.write2ByteTxRx(serial, servo_id, 38, limit)

    if limit is not None:
        if comm != COMM_SUCCESS:
            print(f"Failed to communicate with motor {servo_id}")
            print("%s" % io.getTxRxResult(comm))
        if error != 0:
            print(f"Error while writing current limit for motor {servo_id}")
            print("%s" % io.getRxPacketError(error))


def write_current_limits(io: PacketHandler, serial: PortHandler, ids: np.array, address: int, limits: np.array):
    """
    Write the current limit for the LCR.
    :param io: PacketHandler
    :param serial: PortHandler
    :param ids: numpy array of motor ids
    :param limits: numpy array of current limits
    """
    for i, id_ in enumerate(ids):
        comm, error = io.write2ByteTxRx(serial, id_, 38, limits[i])

        if limits[i] is not None:
            if comm != COMM_SUCCESS:
                print(f"Failed to communicate with motor {id_}")
                print("%s" % io.getTxRxResult(comm))
            if error != 0:
                print(f"Error while writing current limit for motor {id_}")
                print("%s" % io.getRxPacketError(error))


PROTOCOL_VERSION = 2.0
BAUDRATE = 1_000_000
TIMOUT_MS = 1000

class TorqueMode(enum.Enum):
    ENABLED = 1
    DISABLED = 0

class OperatingMode(enum.Enum):
    VELOCITY = 1
    POSITION = 3
    CURRENT_CONTROLLED_POSITION = 5
    PWM = 16
    UNKNOWN = -1


# https://emanual.robotis.com/docs/en/dxl/x/xl330-m077
# https://emanual.robotis.com/docs/en/dxl/x/xl330-m288
# https://emanual.robotis.com/docs/en/dxl/x/xl430-w250
# https://emanual.robotis.com/docs/en/dxl/x/xm430-w350
# https://emanual.robotis.com/docs/en/dxl/x/xm540-w270
# data_name, address, size (byte)
X_SERIE_CONTROL_TABLE = [
    ("goal_position", 116, 4),
    ("goal_current", 102, 2),
    ("goal_pwm", 100, 2),
    ("goal_velocity", 104, 4),
    ("position", 132, 4),
    ("current", 126, 2),
    ("pwm", 124, 2),
    ("velocity", 128, 4),
    ("torque", 64, 1),
    ("temperature", 146, 1),
    ("temperature_limit", 31, 1),
    ("pwm_limit", 36, 2),
    ("current_limit", 38, 2),
]


MODEL_CONTROL_TABLE = {
    "xl330-m077": X_SERIE_CONTROL_TABLE,
    "xl330-m288": X_SERIE_CONTROL_TABLE,
    "xl430-w250": X_SERIE_CONTROL_TABLE,
    "xm430-w350": X_SERIE_CONTROL_TABLE,
    "xm540-w270": X_SERIE_CONTROL_TABLE,
}



def process_response(packet_handler, dxl_comm_result: int, dxl_error: int, motor_idx: int):
    if dxl_comm_result != COMM_SUCCESS:
        raise ConnectionError(
            f"dxl_comm_result for motor {motor_idx}: {packet_handler.getTxRxResult(dxl_comm_result)}"
        )
    elif dxl_error != 0:
        print(f"dxl error {dxl_error}")
        raise ConnectionError(
            f"dynamixel error for motor {motor_idx}: {packet_handler.getTxRxResult(dxl_error)}"
        )



class DynamixelMotorChain:

    def __init__(self, port: str, motor_models: dict[int, str], extra_model_control_table: dict[str, list[tuple]] | None = None):
        self.port = port
        self.motor_models = motor_models

        self.model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
        if extra_model_control_table:
            self.model_ctrl_table.update(extra_model_control_table)

        # Find read/write addresses and number of bytes for each motor
        self.motor_ctrl = {}
        for idx, model in self.motor_models.items():
            self.motor_data[idx] = {}
            for data_name, addr, bytes in self.model_ctrl_table[model]:
                self.motor_ctrl[idx][data_name] = {
                    "addr": addr,
                    "bytes": bytes,
                }

        self.port_handler = PortHandler(self.port)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        if not self.portHandler.openPort():
            raise OSError(f"Failed to open port {self.port}")

        if not self.portHandler.setBaudRate(BAUDRATE):
            raise OSError(f"Failed to set baudrate to {BAUDRATE}")
        
        if not self.portHandler.setPacketTimeoutMillis(TIMOUT_MS):
            raise OSError(f"Failed to set packet timeout to {TIMOUT_MS} (ms)")

        self.position_reader = GroupSyncRead(self.port_handler, self.packet_handler, 

        )

        self.torque_enabled = {idx: False for idx in self.motor_models}
        self.torque_enabled = {idx: False for idx in self.motor_models}

    @property
    def motor_ids(self) -> list[int]:
        return list(self.motor_models.keys())

    def write(self, data_name, value, motor_idx: int | None):
        # By default, enable torque on all motors
        motor_ids = [motor_idx] if motor_idx else self.motor_ids
        
        for idx in motor_ids:
            addr = self.motor_ctrl[idx][data_name]["addr"]
            bytes = self.motor_ctrl[idx][data_name]["bytes"]
            args = (self.port_handler, idx, addr, value)
            if bytes == 1:
                rslt, err = self.packet_handler.write1ByteTxRx(*args)
            elif bytes == 2:
                rslt, err = self.packet_handler.write2ByteTxRx(*args)
            elif bytes == 4:
                rslt, err = self.packet_handler.write4ByteTxRx(*args)
            else:
                raise NotImplementedError(f"Value of the number of bytes to be sent is expected to be in [1, 2, 4], but {num_byte} is provided instead.")
            
            process_response(self.packet_handler, rslt, err, idx)

    def read(self, data_name, motor_idx: int | None):


    def enable_torque(self, motor_idx: int | None):
        self.send_paquet("torque", TorqueMode.ENABLED, motor_idx)

    def disable_torque(self, motor_idx: int | None):
        self.send_paquet("torque", TorqueMode.DISABLED, motor_idx)

    def set_operating_mode(self, motor_idx: int | None, mode: OperatingMode):
        self.send_paquet("torque", mode, motor_idx, num_byte=2)
        


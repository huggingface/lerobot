from copy import deepcopy
import enum
import numpy as np

from dynamixel_sdk import PacketHandler, PortHandler, COMM_SUCCESS, GroupSyncRead, GroupSyncWrite


def pos2pwm(pos: np.ndarray) -> np.ndarray:
    """
    :param pos: numpy array of joint positions in range [-pi, pi]
    :return: numpy array of pwm values in range [0, 4096]
    """
    return ((pos / 3.14 + 1.0) * 2048).astype(np.int64)


def pwm2pos(pwm: np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm values in range [0, 4096]
    :return: numpy array of joint positions in range [-pi, pi]
    """
    return (pwm / 2048 - 1) * 3.14


def pwm2vel(pwm: np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm/s joint velocities
    :return: numpy array of rad/s joint velocities
    """
    return pwm * 3.14 / 2048


def vel2pwm(vel: np.ndarray) -> np.ndarray:
    """
    :param vel: numpy array of rad/s joint velocities
    :return: numpy array of pwm/s joint velocities
    """
    return (vel * 2048 / 3.14).astype(np.int64)

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



class DynamixelMotorsChain:

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

        # self.torque_enabled = {idx: False for idx in self.motor_models}
        self.group_readers = {}
        self.group_writters = {}

    @property
    def motor_ids(self) -> list[int]:
        return list(self.motor_models.keys())

    def write(self, data_name, value, motor_idx: int | None):
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
                raise NotImplementedError(f"Value of the number of bytes to be sent is expected to be in [1, 2, 4], but {bytes} is provided instead.")
            
            if rslt != COMM_SUCCESS:
                raise ConnectionError(
                    f"Write failed due to communication error on port {self.port} for motor {motor_idx}: {self.packet_handler.getTxRxResult(rslt)}"
                )
            elif err != 0:
                raise ConnectionError(
                    f"Write failed due to error {err} on port {self.port} for motor {motor_idx}: {self.packet_handler.getTxRxResult(err)}"
                )

    def read_(self, data_name, motor_idx: int | None):
        motor_ids = self.motor_ids if motor_idx is None else [motor_idx]

        if data_name not in self.group_readers:
            addr = self.motor_ctrl[0]["addr"]
            bytes = self.motor_ctrl[0]["bytes"]
            group_reader = GroupSyncRead(self.port_handler, self.packet_handler, addr, bytes)

            for idx in motor_idx:
                group_reader.addParam(idx)

            self.group_readers[data_name] = group_reader
        
        rslt = self.group_readers[data_name].txRxPacket()
        

    def write_(self, data_name, motor_ids: int | None):
        if data_name not in self.group_readers:
            addr = self.motor_ctrl[0]["addr"]
            bytes = self.motor_ctrl[0]["bytes"]
            self.group_readers[data_name] = GroupSyncWrite(self.port_handler, self.packet_handler, addr, bytes)


    def read(self, data_name, motor_idx: int | None):
        # TODO(rcadene): implement retry
        motor_ids = self.motor_ids if motor_idx is None else [motor_idx]
        
        values = []
        for idx in motor_ids:
            addr = self.motor_ctrl[idx][data_name]["addr"]
            bytes = self.motor_ctrl[idx][data_name]["bytes"]
            args = (self.port_handler, idx, addr)
            if bytes == 1:
                value, rslt, err = self.packet_handler.write1ByteTxRx(*args)
            elif bytes == 2:
                value, rslt, err = self.packet_handler.write2ByteTxRx(*args)
            elif bytes == 4:
                value, rslt, err = self.packet_handler.write4ByteTxRx(*args)
            else:
                raise NotImplementedError(f"Value of the number of bytes to be sent is expected to be in [1, 2, 4], but {bytes} is provided instead.")
                
            if rslt != COMM_SUCCESS:
                raise ConnectionError(
                    f"Read failed due to communication error on port {self.port} for motor {motor_idx}: {self.packet_handler.getTxRxResult(rslt)}"
                )
            elif err != 0:
                raise ConnectionError(
                    f"Read failed due to error {err} on port {self.port} for motor {motor_idx}: {self.packet_handler.getTxRxResult(err)}"
                )
            
            values.append(value)
            
        if motor_idx is None:
            return values
        else:
            return values[0]
        
    def enable_torque(self, motor_idx: int | None):
        # By default, enable torque on all motors
        self.write("torque", TorqueMode.ENABLED, motor_idx)

    def disable_torque(self, motor_idx: int | None):
        self.write("torque", TorqueMode.DISABLED, motor_idx)

    def set_operating_mode(self, motor_idx: int | None, mode: OperatingMode):
        self.write("torque", mode, motor_idx, num_byte=2)
        
    def read_position(self, motor_idx: int | None):
        return self.read("position", motor_idx)
    
    def read_position(self, motor_idx: int | None):
        return self.read("position", motor_idx)
    
    def write_goal_position(self, motor_idx: int | None):
        return self.write("goal_position", motor_idx)
    



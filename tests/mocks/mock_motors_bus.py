# ruff: noqa: N802

from lerobot.motors.motors_bus import (
    Motor,
    MotorsBus,
)

DUMMY_CTRL_TABLE_1 = {
    "Firmware_Version": (0, 1),
    "Model_Number": (1, 2),
    "Present_Position": (3, 4),
    "Goal_Position": (11, 2),
}

DUMMY_CTRL_TABLE_2 = {
    "Model_Number": (0, 2),
    "Firmware_Version": (2, 1),
    "Present_Position": (3, 4),
    "Present_Velocity": (7, 4),
    "Goal_Position": (11, 4),
    "Goal_Velocity": (15, 4),
    "Lock": (19, 1),
}

DUMMY_MODEL_CTRL_TABLE = {
    "model_1": DUMMY_CTRL_TABLE_1,
    "model_2": DUMMY_CTRL_TABLE_2,
    "model_3": DUMMY_CTRL_TABLE_2,
}

DUMMY_BAUDRATE_TABLE = {
    0: 1_000_000,
    1: 500_000,
    2: 250_000,
}

DUMMY_MODEL_BAUDRATE_TABLE = {
    "model_1": DUMMY_BAUDRATE_TABLE,
    "model_2": DUMMY_BAUDRATE_TABLE,
    "model_3": DUMMY_BAUDRATE_TABLE,
}

DUMMY_ENCODING_TABLE = {
    "Present_Position": 8,
    "Goal_Position": 10,
}

DUMMY_MODEL_ENCODING_TABLE = {
    "model_1": DUMMY_ENCODING_TABLE,
    "model_2": DUMMY_ENCODING_TABLE,
    "model_3": DUMMY_ENCODING_TABLE,
}

DUMMY_MODEL_NUMBER_TABLE = {
    "model_1": 1234,
    "model_2": 5678,
    "model_3": 5799,
}

DUMMY_MODEL_RESOLUTION_TABLE = {
    "model_1": 4096,
    "model_2": 1024,
    "model_3": 4096,
}


class MockPortHandler:
    def __init__(self, port_name):
        self.is_open: bool = False
        self.baudrate: int
        self.packet_start_time: float
        self.packet_timeout: float
        self.tx_time_per_byte: float
        self.is_using: bool = False
        self.port_name: str = port_name
        self.ser = None

    def openPort(self):
        self.is_open = True
        return self.is_open

    def closePort(self):
        self.is_open = False

    def clearPort(self): ...
    def setPortName(self, port_name):
        self.port_name = port_name

    def getPortName(self):
        return self.port_name

    def setBaudRate(self, baudrate):
        self.baudrate: baudrate

    def getBaudRate(self):
        return self.baudrate

    def getBytesAvailable(self): ...
    def readPort(self, length): ...
    def writePort(self, packet): ...
    def setPacketTimeout(self, packet_length): ...
    def setPacketTimeoutMillis(self, msec): ...
    def isPacketTimeout(self): ...
    def getCurrentTime(self): ...
    def getTimeSinceStart(self): ...
    def setupPort(self, cflag_baud): ...
    def getCFlagBaud(self, baudrate): ...


class MockMotorsBus(MotorsBus):
    available_baudrates = [500_000, 1_000_000]
    default_timeout = 1000
    model_baudrate_table = DUMMY_MODEL_BAUDRATE_TABLE
    model_ctrl_table = DUMMY_MODEL_CTRL_TABLE
    model_encoding_table = DUMMY_MODEL_ENCODING_TABLE
    model_number_table = DUMMY_MODEL_NUMBER_TABLE
    model_resolution_table = DUMMY_MODEL_RESOLUTION_TABLE
    normalized_data = ["Present_Position", "Goal_Position"]

    def __init__(self, port: str, motors: dict[str, Motor]):
        super().__init__(port, motors)
        self.port_handler = MockPortHandler(port)

    def _assert_protocol_is_compatible(self, instruction_name): ...
    def _handshake(self): ...
    def _find_single_motor(self, motor, initial_baudrate): ...
    def configure_motors(self): ...
    def is_calibrated(self): ...
    def read_calibration(self): ...
    def write_calibration(self, calibration_dict): ...
    def disable_torque(self, motors, num_retry): ...
    def _disable_torque(self, motor, model, num_retry): ...
    def enable_torque(self, motors, num_retry): ...
    def _get_half_turn_homings(self, positions): ...
    def _encode_sign(self, data_name, ids_values): ...
    def _decode_sign(self, data_name, ids_values): ...
    def _split_into_byte_chunks(self, value, length): ...
    def broadcast_ping(self, num_retry, raise_on_error): ...

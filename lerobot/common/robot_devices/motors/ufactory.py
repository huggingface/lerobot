from copy import deepcopy
import enum
from lerobot.common.robot_devices.utils import \
    RobotDeviceAlreadyConnectedError, \
    RobotDeviceNotConnectedError
from xarm.wrapper import XArmAPI
import numpy as np


class TorqueMode(enum.Enum):
    ENABLED = 1
    DISABLED = 0

class xArmWrapper:
    """Wrapper for the xArm Python SDK"""

    def __init__(
        self,
        port: str,
        motors: dict[str, tuple[int, str]],
        mock=False,
    ):
        self.port = port
        self.motors = motors
        self.mock = mock

        self.calibration = None
        self.is_connected = False
        self.logs = {}

        self.api = None

        self.MAX_SPEED_LIMIT = None
        self.MAX_ACC_LIMIT = None


    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"DynamixelMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        if self.mock:
            return
        else:
            self.api = XArmAPI(self.port)

        try:
            if not self.api.connected:
                raise OSError(f"Failed to connect to xArm API @ '{self.port}'.")
        except Exception:
            print(
                "\Exception while connecting in xArmWrapper.\n"
            )
            raise

        # Allow to read and write
        self.is_connected = True

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        pass  # TODO (@vmayoral): implement if of interest

    def read(self, data_name, motor_names: str | list[str] | None = None):
        pass  # TODO (@vmayoral): implement if of interest

    def enable(self, follower: bool = False):
        self.api.motion_enable(enable=True)
        self.api.clean_error()
        if follower:
            self.api.set_mode(1)
        else:
            self.api.set_mode(0)
        self.api.set_state(state=0)
        #
        self.api.set_gripper_mode(0)
        self.api.set_gripper_enable(True)
        self.api.set_gripper_speed(5000)  # default speed, as there's no way to fetch gripper speed from API

        # Initialize the global speed and acceleration limits
        self.initialize_limits()

        # assume leader by default
        if not follower:
            self.api.set_mode(2)
            self.api.set_state(0)
            # Light up the digital output 2 (button), to signal manual mode
            self.api.set_tgpio_digital(ionum=2, value=1)

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FeetechMotorsBus({self.port}) is not connected. Try running `motors_bus.connect()` first."
            )

        # Turn off manual mode after recording
        self.api.set_mode(0)
        self.api.set_state(0)
        # Light down the digital output 2 (button), to signal manual mode
        self.api.set_tgpio_digital(ionum=2, value=0)
        # Disconnect both arms
        self.api.disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

    def initialize_limits(self):
        # heuristic: 1/3 of the max speed and acceleration limits
        #  for testing purposes
        self.MAX_SPEED_LIMIT = max(self.api.joint_speed_limit)/3
        self.MAX_ACC_LIMIT = max(self.api.joint_acc_limit)/3

    def get_position(self):
        code, angles = self.api.get_servo_angle()
        code_gripper, pos_gripper = self.api.get_gripper_position()
        # pos = angles[:-1] + [pos_gripper]  # discard 7th dof, which is not present in U850
        pos = angles + [pos_gripper]
        return pos

    def set_position(self, position: np.ndarray):
        angles = position[:-1].tolist()
        gripper_pos = int(position[-1])
        #
        self.api.set_servo_angle_j(angles=angles, is_radian=False, wait=False)
        self.api.set_gripper_position(pos=gripper_pos, wait=False)

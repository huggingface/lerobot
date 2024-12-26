import enum
import numpy as np
from dynamixel_sdk import *
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError

from gello.agents.gello_agent import GelloAgent

class TorqueMode(enum.Enum):
    ENABLED = 1
    DISABLED = 0

class GelloDynamixelWrapper:
    """
    Wrapper for Dynamixel motors used in Gello robot
    
    NOTE: Gello builds upon inter-process communication using ZeroMQ. In particular, the communication happens through the RobotEnv class instanciated within the run_env.py script which connects to the GelloAgent, which abstracts away the hardware and the actuation.

    This wrapper is used to interface with the Gello robot's hardware part only. For that purpose, only the GelloAgent is used without the ZeroMQ part and/or its communication overhead.
    """
    GRIPPER_OPEN = 800
    GRIPPER_CLOSE = 0

    def __init__(
        self,
        port: str,
        motors: dict[str, tuple[int, str]],
        mock: bool = False,
    ):
        """Initialize the Gello Dynamixel wrapper
        
        Args:
            port: Serial port for Dynamixel communication
            motors: Dictionary mapping motor names to (ID, model) tuples
            joint_ids: IDs of joint motors in order
            joint_offsets: Joint offset angles in radians
            joint_signs: Signs (+1/-1) for joint directions
            gripper_config: Tuple of (gripper_id, open_pos, closed_pos)
            mock: If True, run in simulation mode
            baudrate: Communication baudrate
        """
        self.port = port
        self.motors = motors
        self.mock = mock

        self.is_connected = False
        self.gello = None

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property 
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        pass  # TODO (@vmayoral): implement if of interest

    def read(self, data_name, motor_names: str | list[str] | None = None):
        pass  # TODO (@vmayoral): implement if of interest

    def enable(self, follower: bool = False):
        pass  # TODO (@vmayoral): implement if of interest

    def connect(self):
        """Connect to the Dynamixel motors"""
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"GelloDynamixelWrapper({self.port}) is already connected."
            )

        if self.mock:
            return

        self.gello = agent = GelloAgent(port=self.port, start_joints=None)

        self.is_connected = True

    def disconnect(self):
        """Disconnect from the Dynamixel motors"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"GelloDynamixelWrapper({self.port}) is not connected."
            )

        self.is_connected = False

    def get_position(self) -> list[float]:
        """Get current positions of all joints and gripper
        
        Returns:
            List of positions [joint1, joint2, ..., gripper]
        """
        if self.mock:
            return [0.0] * (len(self.joint_ids) + 1)

        rads_pos = self.gello.act(None).tolist()
        degrees_pos = np.rad2deg(rads_pos[:-1])  # Convert all but last to degrees

        # NOTE: heuristic obtained from the code of the XArmRobot in the `gello` package
        #
        gripper_value = self.GRIPPER_OPEN + rads_pos[-1] * (self.GRIPPER_CLOSE - self.GRIPPER_OPEN)
        degrees_pos = degrees_pos.tolist()  # Convert to list for appending
        degrees_pos.append(gripper_value)
        
        return degrees_pos

    def set_position(self, position: np.ndarray):
        """Set positions for all joints and gripper
        
        Args:
            position: Array of positions [joint1, joint2, ..., gripper]
        """
        if self.mock:
            return
        pass

    def robot_reset(self):
        """Reset the robot to a safe state"""
        if not self.mock:
            # Open gripper to safe position
            pass

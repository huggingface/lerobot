import time
import traceback
import trossen_arm as trossen
import numpy as np

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.robot_devices.motors.configs import TrossenArmDriverConfig

PITCH_CIRCLE_RADIUS = 0.00875 # meters
VEL_LIMITS = [3.375, 3.375, 3.375, 7.0, 7.0, 7.0, 12.5 * PITCH_CIRCLE_RADIUS]

TROSSEN_ARM_MODELS = {
    "V0_LEADER": [trossen.Model.wxai_v0, trossen.StandardEndEffector.wxai_v0_leader],
    "V0_FOLLOWER": [trossen.Model.wxai_v0, trossen.StandardEndEffector.wxai_v0_follower],
}

class TrossenArmDriver:
    """
        The `TrossenArmDriver` class provides an interface for controlling 
        Trossen Robotics' robotic arms. It leverages the trossen_arm for communication with arms.

        This class allows for configuration, torque management, and motion control of robotic arms. It includes features for handling connection states, moving the 
        arm to specified poses, and logging timestamps for debugging and performance analysis.

        ### Key Features:
        - **Multi-motor Control:** Supports multiple motors connected to a bus.
        - **Mode Switching:** Enables switching between position and gravity 
        compensation modes.
        - **Home and Sleep Pose Management:** Automatically transitions the arm to home and sleep poses for safe operation.
        - **Error Handling:** Raises specific exceptions for connection and operational errors.
        - **Logging:** Captures timestamps for operations to aid in debugging.

        ### Example Usage:
        ```python
        motors = {
            "joint_0": (1, "4340"),
            "joint_1": (2, "4340"),
            "joint_2": (4, "4340"),
            "joint_3": (6, "4310"),
            "joint_4": (7, "4310"),
            "joint_5": (8, "4310"),
            "joint_6": (9, "4310"),
        }
        arm_driver = TrossenArmDriver(
            motors=motors,
            ip="192.168.1.2",
            model="V0_LEADER",
        )
        arm_driver.connect()

        # Read motor positions
        positions = arm_driver.read("Present_Position")

        # Move to a new position (Home Pose)
        # Last joint is the gripper, which is in range [0, 450]
        arm_driver.write("Goal_Position", [0, 15, 15, 0, 0, 0, 200])

        # Disconnect when done
        arm_driver.disconnect()
        ```
    """


    def __init__(
        self,
        config: TrossenArmDriverConfig,
    ):
        self.ip = config.ip
        self.model = config.model
        self.mock = config.mock
        self.driver = None
        self.calibration = None
        self.is_connected = False
        self.group_readers = {}
        self.group_writers = {}
        self.logs = {}
        self.fps = 30
        self.home_pose = [0, np.pi/12, np.pi/12, 0, 0, 0, 0]
        self.sleep_pose = [0, 0, 0, 0, 0, 0, 0]

        self.motors={
                    # name: (index, model)
                    "joint_0": [1, "4340"],
                    "joint_1": [2, "4340"],
                    "joint_2": [3, "4340"],
                    "joint_3": [4, "4310"],
                    "joint_4": [5, "4310"],
                    "joint_5": [6, "4310"],
                    "joint_6": [7, "4310"],
                }

        self.prev_write_time = 0
        self.current_write_time = None
        
        # To prevent DiscontinuityError due to large jumps in position in short time.
        # We scale the time to move based on the distance between the start and goal values and the maximum speed of the motors.
        # The below factor is used to scale the time to move.
        self.TIME_SCALING_FACTOR = 3.0

        # Minimum time to move for the arm (This is a tuning parameter)
        self.MIN_TIME_TO_MOVE = 3.0 / self.fps

    def connect(self):
        print(f"Connecting to {self.model} arm at {self.ip}...")
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"TrossenArmDriver({self.ip}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        print("Initializing the drivers...")

        # Initialize the driver
        self.driver = trossen.TrossenArmDriver()

        # Get the model configuration
        try:
            model_name, model_end_effector = TROSSEN_ARM_MODELS[self.model]
        except KeyError:
            raise ValueError(f"Unsupported model: {self.model}")

        print("Configuring the drivers...")

        # Configure the driver
        try:
            self.driver.configure(model_name, model_end_effector, self.ip, True)
        except Exception:
            traceback.print_exc()
            print(
                f"Failed to configure the driver for the {self.model} arm at {self.ip}."
            )
            raise

        # Move the arms to the home pose
        self.driver.set_all_modes(trossen.Mode.position)
        self.driver.set_all_positions(self.home_pose, 2.0, True)

        # Allow to read and write
        self.is_connected = True


    def reconnect(self):
        try:
            model_name, model_end_effector = TROSSEN_ARM_MODELS[self.model]
        except KeyError:
            raise ValueError(f"Unsupported model: {self.model}")
        try:
            self.driver.configure(model_name, model_end_effector, self.ip, True)
        except Exception:
            traceback.print_exc()
            print(
                f"Failed to configure the driver for the {self.model} arm at {self.ip}."
            )
            raise

        self.is_connected = True


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
        pass

    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def autocorrect_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        pass

    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"TrossenArmMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        # Read the present position of the motors
        if data_name == "Present_Position":
            # Get the positions of the motors
            values = self.driver.get_positions()
            values[:-1] = np.degrees(values[:-1])  # Convert all joints except gripper
            values[-1] = values[-1] * 10000  # Convert gripper to range (0-450)
        else:
            values = None
            print(f"Data name: {data_name} is not supported for reading.")

        # TODO: Add support for reading other data names as required

        self.logs["delta_timestamp_s_read"] = time.perf_counter() - start_time

        values = np.array(values, dtype=np.float32)
        return values

    def compute_time_to_move(self, goal_values: np.ndarray):
        # Compute the time to move based on the distance between the start and goal values
        # and the maximum speed of the motors
        current_pose = self.driver.get_positions()
        displacement = abs(goal_values - current_pose)
        time_to_move_all_joints = self.TIME_SCALING_FACTOR*displacement / VEL_LIMITS
        time_to_move = max(time_to_move_all_joints)
        time_to_move = max(time_to_move, self.MIN_TIME_TO_MOVE)
        return time_to_move

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"TrossenAIArm({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        # Write the goal position of the motors
        if data_name == "Goal_Position":
            values = np.array(values, dtype=np.float32)
            # Convert back to radians for joints
            values[:-1] = np.radians(values[:-1])  # Convert all joints except gripper
            values[-1] = values[-1] / 10000  # Convert gripper back to range (0-0.045)
            self.driver.set_all_positions(values.tolist(), self.compute_time_to_move(values), False)
            self.prev_write_time = self.current_write_time

        # Enable or disable the torque of the motors
        elif data_name == "Torque_Enable":
            # Set the arms to POSITION mode
            if values == 1:
                self.driver.set_all_modes(trossen.Mode.position)
            else:
                self.driver.set_all_modes(trossen.Mode.external_effort)
                self.driver.set_all_external_efforts([0.0] * 7, 0.0, True)
        elif data_name == "Reset":
            self.driver.set_all_modes(trossen.Mode.position)
            self.driver.set_all_positions(self.home_pose, 2.0, True)
        else:
            print(f"Data name: {data_name} value: {values} is not supported for writing.")

        self.logs["delta_timestamp_s_write"] = time.perf_counter() - start_time

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"Trossen Arm Driver ({self.port}) is not connected. Try running `motors_bus.connect()` first."
            )
        self.driver.set_all_modes(trossen.Mode.position)
        self.driver.set_all_positions(self.home_pose, 2.0, True)
        self.driver.set_all_positions(self.sleep_pose, 2.0, True)

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

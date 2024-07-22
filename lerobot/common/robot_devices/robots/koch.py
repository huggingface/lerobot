import pickle
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
    TorqueMode,
)

from lerobot.common.robot_devices.motors.position_control.configure import (
    build_logical_to_physical_tables,
    build_physical_to_logical_tables,
    build_logical_to_physical_function,
    build_physical_to_logical_function,
)

from lerobot.common.robot_devices.motors.utils import MotorsBus
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError

URL_HORIZONTAL_POSITION = {
    "follower": "https://raw.githubusercontent.com/huggingface/lerobot/main/media/koch/follower_horizontal.png",
    "leader": "https://raw.githubusercontent.com/huggingface/lerobot/main/media/koch/leader_horizontal.png",
}
URL_90_DEGREE_POSITION = {
    "follower": "https://raw.githubusercontent.com/huggingface/lerobot/main/media/koch/follower_90_degree.png",
    "leader": "https://raw.githubusercontent.com/huggingface/lerobot/main/media/koch/leader_90_degree.png",
}

########################################################################
# Calibration logic
########################################################################

TARGET_HORIZONTAL_POSITION = np.array([0, -90, 90, 0, -90, 0])
TARGET_90_DEGREE_POSITION = np.array([90, 0, 0, 90, 0, -90])

GRIPPER_OPEN = np.array([-40])


def reset_arm(arm: MotorsBus):
    # To be configured, all servos must be in "torque disable" mode
    arm.write("Torque_Enable", TorqueMode.DISABLED.value)

    # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
    # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
    # you could end up with a servo with a position 0 or 4095 at a crucial point See [
    # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
    all_motors_except_gripper = [name for name in arm.motor_names if name != "gripper"]
    arm.write("Operating_Mode", OperatingMode.EXTENDED_POSITION.value, all_motors_except_gripper)

    # TODO(rcadene): why?
    # Use 'position control current based' for gripper
    arm.write("Operating_Mode", OperatingMode.CURRENT_CONTROLLED_POSITION.value, "gripper")


def run_arm_calibration(arm: MotorsBus, name: str, arm_type: str):
    """Example of usage:
    ```python
    run_arm_calibration(arm, "left", "follower")
    ```
    """
    reset_arm(arm)

    # TODO(rcadene): document what position 1 mean
    print(
        f"Please move the '{name} {arm_type}' arm to the horizontal position (gripper fully closed, see {URL_HORIZONTAL_POSITION[arm_type]})"
    )
    input("Press Enter to continue...")

    physical_position_1 = arm.read("Present_Position")

    # TODO(rcadene): document what position 2 mean
    print(
        f"Please move the '{name} {arm_type}' arm to the 90 degree position (gripper fully open, see {URL_90_DEGREE_POSITION[arm_type]})"
    )
    input("Press Enter to continue...")

    physical_position_2 = arm.read("Present_Position")

    wanted = np.array(
        [
            (TARGET_HORIZONTAL_POSITION[i], TARGET_90_DEGREE_POSITION[i])
            for i in range(len(TARGET_HORIZONTAL_POSITION))
        ]
    )

    position = np.array(
        [(physical_position_1[i], physical_position_2[i]) for i in range(len(physical_position_1))]
    )

    physical_to_logical_tables = build_physical_to_logical_tables(position, wanted)

    logical_to_physical_tables = build_logical_to_physical_tables(position, wanted)

    return physical_to_logical_tables, logical_to_physical_tables


########################################################################
# Alexander Koch robot arm
########################################################################


@dataclass
class KochRobotConfig:
    """
    Example of usage:
    ```python
    KochRobotConfig()
    ```
    """

    # Define all components of the robot
    leader_arms: dict[str, MotorsBus] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBus] = field(default_factory=lambda: {})
    cameras: dict[str, Camera] = field(default_factory=lambda: {})


class KochRobot:
    # TODO(rcadene): Implement force feedback
    """Tau Robotics: https://tau-robotics.com

    Example of highest frequency teleoperation without camera:
    ```python
    # Defines how to communicate with the motors of the leader and follower arms
    leader_arms = {
        "main": DynamixelMotorsBus(
            port="/dev/tty.usbmodem575E0031751",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl330-m077"),
                "shoulder_lift": (2, "xl330-m077"),
                "elbow_flex": (3, "xl330-m077"),
                "wrist_flex": (4, "xl330-m077"),
                "wrist_roll": (5, "xl330-m077"),
                "gripper": (6, "xl330-m077"),
            },
        ),
    }
    follower_arms = {
        "main": DynamixelMotorsBus(
            port="/dev/tty.usbmodem575E0032081",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl430-w250"),
                "shoulder_lift": (2, "xl430-w250"),
                "elbow_flex": (3, "xl330-m288"),
                "wrist_flex": (4, "xl330-m288"),
                "wrist_roll": (5, "xl330-m288"),
                "gripper": (6, "xl330-m288"),
            },
        ),
    }
    robot = KochRobot(leader_arms, follower_arms)

    # Connect motors buses and cameras if any (Required)
    robot.connect()

    while True:
        robot.teleop_step()
    ```

    Example of highest frequency data collection without camera:
    ```python
    # Assumes leader and follower arms have been instantiated already (see first example)
    robot = KochRobot(leader_arms, follower_arms)
    robot.connect()
    while True:
        observation, action = robot.teleop_step(record_data=True)
    ```

    Example of highest frequency data collection with cameras:
    ```python
    # Defines how to communicate with 2 cameras connected to the computer.
    # Here, the webcam of the mackbookpro and the iphone (connected in USB to the macbookpro)
    # can be reached respectively using the camera indices 0 and 1. These indices can be
    # arbitrary. See the documentation of `OpenCVCamera` to find your own camera indices.
    cameras = {
        "macbookpro": OpenCVCamera(camera_index=0, fps=30, width=640, height=480),
        "iphone": OpenCVCamera(camera_index=1, fps=30, width=640, height=480),
    }

    # Assumes leader and follower arms have been instantiated already (see first example)
    robot = KochRobot(leader_arms, follower_arms, cameras)
    robot.connect()
    while True:
        observation, action = robot.teleop_step(record_data=True)
    ```

    Example of controlling the robot with a policy (without running multiple policies in parallel to ensure highest frequency):
    ```python
    # Assumes leader and follower arms + cameras have been instantiated already (see previous example)
    robot = KochRobot(leader_arms, follower_arms, cameras)
    robot.connect()
    while True:
        # Uses the follower arms and cameras to capture an observation
        observation = robot.capture_observation()

        # Assumes a policy has been instantiated
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Orders the robot to move
        robot.send_action(action)
    ```

    Example of disconnecting which is not mandatory since we disconnect when the object is deleted:
    ```python
    robot.disconnect()
    ```
    """

    def __init__(
        self,
        config: KochRobotConfig | None = None,
        calibration_path: Path = ".cache/calibration/koch.pkl",
        **kwargs,
    ):
        if config is None:
            config = KochRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)
        self.calibration_path = Path(calibration_path)

        self.leader_arms = self.config.leader_arms
        self.follower_arms = self.config.follower_arms
        self.cameras = self.config.cameras
        self.is_connected = False
        self.logs = {}

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "KochRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "KochRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # Connect the arms
        for name in self.follower_arms:
            self.follower_arms[name].connect()
            self.leader_arms[name].connect()

        # Reset the arms and load or run calibration
        if self.calibration_path.exists():
            # Reset all arms before setting calibration
            for name in self.follower_arms:
                reset_arm(self.follower_arms[name])
            for name in self.leader_arms:
                reset_arm(self.leader_arms[name])

            with open(self.calibration_path, "rb") as f:
                calibration = pickle.load(f)
        else:
            # Run calibration process which begins by reseting all arms
            calibration = self.run_calibration()

            self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.calibration_path, "wb") as f:
                pickle.dump(calibration, f)

        print(calibration)

        # Set calibration
        for name in self.follower_arms:
            for motor in self.follower_arms[name].motor_names:
                calibration[f"follower_{name}"][motor]["physical_to_logical"] = (
                    build_physical_to_logical_function(
                        calibration[f"follower_{name}"][motor]["physical_to_logical"]
                    )
                )

                calibration[f"follower_{name}"][motor]["logical_to_physical"] = (
                    build_logical_to_physical_function(
                        calibration[f"follower_{name}"][motor]["logical_to_physical"]
                    )
                )

            self.follower_arms[name].set_calibration(calibration[f"follower_{name}"])
        for name in self.leader_arms:
            for motor in self.leader_arms[name].motor_names:
                calibration[f"leader_{name}"][motor]["physical_to_logical"] = (
                    build_physical_to_logical_function(
                        calibration[f"leader_{name}"][motor]["physical_to_logical"]
                    )
                )

                calibration[f"leader_{name}"][motor]["logical_to_physical"] = (
                    build_logical_to_physical_function(
                        calibration[f"leader_{name}"][motor]["logical_to_physical"]
                    )
                )

            self.leader_arms[name].set_calibration(calibration[f"leader_{name}"])

        # Set better PID values to close the gap between recored states and actions
        # TODO(rcadene): Implement an automatic procedure to set optimial PID values for each motor
        for name in self.follower_arms:
            self.follower_arms[name].write("Position_P_Gain", 1500, "elbow_flex")
            self.follower_arms[name].write("Position_I_Gain", 0, "elbow_flex")
            self.follower_arms[name].write("Position_D_Gain", 600, "elbow_flex")

        # Enable torque on all motors of the follower arms
        for name in self.follower_arms:
            self.follower_arms[name].write("Torque_Enable", 1)

        # Enable torque on the gripper of the leader arms, and move it to 45 degrees,
        # so that we can use it as a trigger to close the gripper of the follower arms.
        for name in self.leader_arms:
            self.leader_arms[name].write("Torque_Enable", 1, "gripper")
            self.leader_arms[name].write("Goal_Position", GRIPPER_OPEN, "gripper")
            self.leader_arms[name].write("Goal_Current", 40, "gripper")

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

    def run_calibration(self):
        calibration = {}

        for name in self.follower_arms:
            physical_to_logical_tables, logical_to_physical_tables = run_arm_calibration(
                self.follower_arms[name], name, "follower"
            )

            calibration[f"follower_{name}"] = {}
            for idx, motor_name in enumerate(self.follower_arms[name].motor_names):
                calibration[f"follower_{name}"][motor_name] = {}
                calibration[f"follower_{name}"][motor_name]["physical_to_logical"] = (
                    physical_to_logical_tables[idx]
                )
                calibration[f"follower_{name}"][motor_name]["logical_to_physical"] = (
                    logical_to_physical_tables[idx]
                )

        for name in self.leader_arms:
            physical_to_logical_tables, logical_to_physical_tables = run_arm_calibration(
                self.leader_arms[name], name, "leader"
            )

            calibration[f"leader_{name}"] = {}
            for idx, motor_name in enumerate(self.leader_arms[name].motor_names):
                calibration[f"leader_{name}"][motor_name] = {}
                calibration[f"leader_{name}"][motor_name]["physical_to_logical"] = physical_to_logical_tables[
                    idx
                ]
                calibration[f"leader_{name}"][motor_name]["logical_to_physical"] = logical_to_physical_tables[
                    idx
                ]

        return calibration

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "KochRobot is not connected. You need to run `robot.connect()`."
            )

        # Prepare to assign the positions of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            now = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - now

        follower_goal_pos = {}
        for name in self.leader_arms:
            follower_goal_pos[name] = leader_pos[name]

        # Send action
        for name in self.follower_arms:
            now = time.perf_counter()
            follower_position = self.follower_arms[name].read("Present_Position", raw=True)
            self.follower_arms[name].write(
                "Goal_Position", follower_goal_pos[name], supplementary_values=follower_position
            )
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - now

        # Early exit when recording data is not requested
        if not record_data:
            return

        # TODO(rcadene): Add velocity and other info
        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            now = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - now

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = np.concatenate(state)

        # Create action by concatenating follower goal position
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = np.concatenate(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            now = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - now

        # Populate output dictionnaries and format to pytorch
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = torch.from_numpy(state)
        action_dict["action"] = torch.from_numpy(action)
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = torch.from_numpy(images[name])

        return obs_dict, action_dict

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "KochRobot is not connected. You need to run `robot.connect()`."
            )

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            now = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - now

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = np.concatenate(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            now = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - now

        # Populate output dictionnaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = torch.from_numpy(state)
        for name in self.cameras:
            # Convert to pytorch format: channel first and float32 in [0,1]
            img = torch.from_numpy(images[name])
            img = img.type(torch.float32) / 255
            img = img.permute(2, 0, 1).contiguous()
            obs_dict[f"observation.images.{name}"] = img
        return obs_dict

    def send_action(self, action: torch.Tensor):
        """The provided action is expected to be a vector."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "KochRobot is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        to_idx = 0
        follower_goal_pos = {}
        for name in self.follower_arms:
            if name in self.follower_arms:
                to_idx += len(self.follower_arms[name].motor_names)
                follower_goal_pos[name] = action[from_idx:to_idx].numpy()
                from_idx = to_idx

        for name in self.follower_arms:
            self.follower_arms[name].write("Goal_Position", follower_goal_pos[name].astype(np.int32))

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "KochRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

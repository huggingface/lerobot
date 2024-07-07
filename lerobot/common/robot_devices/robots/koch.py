import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path
import time

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
    TorqueMode,
)
from lerobot.common.robot_devices.motors.utils import MotorsBus

########################################################################
# Calibration logic
########################################################################

# TARGET_HORIZONTAL_POSITION = motor_position_to_angle(np.array([0, -1024, 1024, 0, -1024, 0]))
# TARGET_90_DEGREE_POSITION = motor_position_to_angle(np.array([1024, 0, 0, 1024, 0, -1024]))
# GRIPPER_OPEN = motor_position_to_angle(np.array([-400]))

TARGET_HORIZONTAL_POSITION = np.array([0, -1024, 1024, 0, -1024, 0])
TARGET_90_DEGREE_POSITION = np.array([1024, 0, 0, 1024, 0, -1024])
GRIPPER_OPEN = np.array([-400])


def apply_homing_offset(values: np.array, homing_offset: np.array) -> np.array:
    for i in range(len(values)):
        if values[i] is not None:
            values[i] += homing_offset[i]
    return values


def apply_drive_mode(values: np.array, drive_mode: np.array) -> np.array:
    for i in range(len(values)):
        if values[i] is not None and drive_mode[i]:
            values[i] = -values[i]
    return values


def apply_calibration(values: np.array, homing_offset: np.array, drive_mode: np.array) -> np.array:
    values = apply_drive_mode(values, drive_mode)
    values = apply_homing_offset(values, homing_offset)
    return values


def revert_calibration(values: np.array, homing_offset: np.array, drive_mode: np.array) -> np.array:
    """
    Transform working position into real position for the robot.
    """
    values = apply_homing_offset(
        values,
        np.array([-homing_offset if homing_offset is not None else None for homing_offset in homing_offset]),
    )
    values = apply_drive_mode(values, drive_mode)
    return values


def revert_appropriate_positions(positions: np.array, drive_mode: list[bool]) -> np.array:
    for i, revert in enumerate(drive_mode):
        if not revert and positions[i] is not None:
            positions[i] = -positions[i]
    return positions


def compute_corrections(positions: np.array, drive_mode: list[bool], target_position: np.array) -> np.array:
    correction = revert_appropriate_positions(positions, drive_mode)

    for i in range(len(positions)):
        if correction[i] is not None:
            if drive_mode[i]:
                correction[i] -= target_position[i]
            else:
                correction[i] += target_position[i]

    return correction


def compute_nearest_rounded_positions(positions: np.array) -> np.array:
    return np.array(
        [
            round(positions[i] / 1024) * 1024 if positions[i] is not None else None
            for i in range(len(positions))
        ]
    )


def compute_homing_offset(
    arm: DynamixelMotorsBus, drive_mode: list[bool], target_position: np.array
) -> np.array:
    # Get the present positions of the servos
    present_positions = apply_calibration(
        arm.read("Present_Position"), np.array([0, 0, 0, 0, 0, 0]), drive_mode
    )

    nearest_positions = compute_nearest_rounded_positions(present_positions)
    correction = compute_corrections(nearest_positions, drive_mode, target_position)
    return correction


def compute_drive_mode(arm: DynamixelMotorsBus, offset: np.array):
    # Get current positions
    present_positions = apply_calibration(
        arm.read("Present_Position"), offset, np.array([False, False, False, False, False, False])
    )

    nearest_positions = compute_nearest_rounded_positions(present_positions)

    # construct 'drive_mode' list comparing nearest_positions and TARGET_90_DEGREE_POSITION
    drive_mode = []
    for i in range(len(nearest_positions)):
        drive_mode.append(nearest_positions[i] != TARGET_90_DEGREE_POSITION[i])
    return drive_mode


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

    # Make sure the native calibration (homing offset abd drive mode) is disabled, since we use our own calibration layer to be more generic
    arm.write("Homing_Offset", 0)
    arm.write("Drive_Mode", DriveMode.NON_INVERTED.value)


def run_arm_calibration(arm: MotorsBus, name: str):
    reset_arm(arm)

    # TODO(rcadene): document what position 1 mean
    print(f"Please move the '{name}' arm to the horizontal position (gripper fully closed)")
    input("Press Enter to continue...")

    horizontal_homing_offset = compute_homing_offset(
        arm, [False, False, False, False, False, False], TARGET_HORIZONTAL_POSITION
    )

    # TODO(rcadene): document what position 2 mean
    print(f"Please move the '{name}' arm to the 90 degree position (gripper fully open)")
    input("Press Enter to continue...")

    drive_mode = compute_drive_mode(arm, horizontal_homing_offset)
    homing_offset = compute_homing_offset(arm, drive_mode, TARGET_90_DEGREE_POSITION)

    # Invert offset for all drive_mode servos
    for i in range(len(drive_mode)):
        if drive_mode[i]:
            homing_offset[i] = -homing_offset[i]

    print("Calibration is done!")

    print("=====================================")
    print("      HOMING_OFFSET: ", " ".join([str(i) for i in homing_offset]))
    print("      DRIVE_MODE: ", " ".join([str(i) for i in drive_mode]))
    print("=====================================")

    return homing_offset, drive_mode


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

    # Define all the components of the robot
    leader_arms: dict[str, MotorsBus] = field(
        default_factory=lambda: {
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
    )
    follower_arms: dict[str, MotorsBus] = field(
        default_factory=lambda: {
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
    )
    cameras: dict[str, Camera] = field(default_factory=lambda: {})


class KochRobot:
    """Tau Robotics: https://tau-robotics.com

    Example of usage:
    ```python
    robot = KochRobot()
    robot.connect()
    while True:
        robot.teleop_step()
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
            raise ValueError(f"KochRobot is already connected.")

        for name in self.follower_arms:
            self.follower_arms[name].connect()
            self.leader_arms[name].connect()

        if self.calibration_path.exists():
            # Reset all arms before setting calibration
            for name in self.follower_arms:
                reset_arm(self.follower_arms[name])

            for name in self.leader_arms:
                reset_arm(self.leader_arms[name])

            with open(self.calibration_path, "rb") as f:
                calibration = pickle.load(f)
        else:
            calibration = self.run_calibration()

            self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.calibration_path, "wb") as f:
                pickle.dump(calibration, f)

        for name in self.follower_arms:
            self.follower_arms[name].set_calibration(calibration[f"follower_{name}"])
            self.follower_arms[name].write("Torque_Enable", 1)
            self.follower_arms[name].write("Torque_Enable", 1)

        for name in self.leader_arms:
            self.leader_arms[name].set_calibration(calibration[f"leader_{name}"])
            # TODO(rcadene): add comments
            self.leader_arms[name].write("Goal_Position", GRIPPER_OPEN, "gripper")
            self.leader_arms[name].write("Torque_Enable", 1, "gripper")

        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

    def run_calibration(self):
        if not self.is_connected:
            raise ValueError(f"KochRobot is not connected. You need to run `robot.connect()`.")

        calibration = {}

        for name in self.follower_arms:
            homing_offset, drive_mode = run_arm_calibration(self.follower_arms[name], f"{name} follower")

            calibration[f"follower_{name}"] = {}
            for idx, motor_name in enumerate(self.follower_arms[name].motor_names):
                calibration[f"follower_{name}"][motor_name] = (homing_offset[idx], drive_mode[idx])

        for name in self.leader_arms:
            homing_offset, drive_mode = run_arm_calibration(self.leader_arms[name], f"{name} leader")

            calibration[f"leader_{name}"] = {}
            for idx, motor_name in enumerate(self.leader_arms[name].motor_names):
                calibration[f"leader_{name}"][motor_name] = (homing_offset[idx], drive_mode[idx])

        return calibration

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise ValueError(f"KochRobot is not connected. You need to run `robot.connect()`.")

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
            self.follower_arms[name].write("Goal_Position", follower_goal_pos[name])
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - now

        # Early exit when recording data is not requested
        if not record_data:
            return

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
        if not self.is_connected:
            raise ValueError(f"KochRobot is not connected. You need to run `robot.connect()`.")

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            follower_pos[name] = self.follower_arms[name].read("Present_Position")

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = np.concatenate(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            images[name] = self.cameras[name].async_read()

        # Populate output dictionnaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = torch.from_numpy(state)
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = torch.from_numpy(images[name])
        return obs_dict

    def send_action(self, action):
        if not self.is_connected:
            raise ValueError(f"KochRobot is not connected. You need to run `robot.connect()`.")

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

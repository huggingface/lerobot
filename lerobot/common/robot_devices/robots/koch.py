import pickle
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.motors.dynamixel import (
    OperatingMode,
    TorqueMode,
    convert_degrees_to_steps,
)
from lerobot.common.robot_devices.motors.utils import MotorsBus
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError

########################################################################
# Calibration logic
########################################################################

URL_TEMPLATE = (
    "https://raw.githubusercontent.com/huggingface/lerobot/main/media/{robot}/{arm}_{position}.webp"
)

# In nominal degree range ]-180, +180[
ZERO_POSITION_DEGREE = 0
ROTATED_POSITION_DEGREE = 90
GRIPPER_OPEN_DEGREE = 35.156


def assert_drive_mode(drive_mode):
    # `drive_mode` is in [0,1] with 0 means original rotation direction for the motor, and 1 means inverted.
    if not np.all(np.isin(drive_mode, [0, 1])):
        raise ValueError(f"`drive_mode` contains values other than 0 or 1: ({drive_mode})")


def apply_drive_mode(position, drive_mode):
    assert_drive_mode(drive_mode)
    # Convert `drive_mode` from [0, 1] with 0 indicates original rotation direction and 1 inverted,
    # to [-1, 1] with 1 indicates original rotation direction and -1 inverted.
    signed_drive_mode = -(drive_mode * 2 - 1)
    position *= signed_drive_mode
    return position


def reset_torque_mode(arm: MotorsBus):
    # To be configured, all servos must be in "torque disable" mode
    arm.write("Torque_Enable", TorqueMode.DISABLED.value)

    # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
    # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
    # you could end up with a servo with a position 0 or 4095 at a crucial point See [
    # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
    all_motors_except_gripper = [name for name in arm.motor_names if name != "gripper"]
    if len(all_motors_except_gripper) > 0:
        arm.write("Operating_Mode", OperatingMode.EXTENDED_POSITION.value, all_motors_except_gripper)

    # Use 'position control current based' for gripper to be limited by the limit of the current.
    # For the follower gripper, it means it can grasp an object without forcing too much even tho,
    # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
    # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
    # to make it move, and it will move back to its original target position when we release the force.
    arm.write("Operating_Mode", OperatingMode.CURRENT_CONTROLLED_POSITION.value, "gripper")


def run_arm_calibration(arm: MotorsBus, name: str, arm_type: str):
    """This function ensures that a neural network trained on data collected on a given robot
    can work on another robot. For instance before calibration, setting a same goal position
    for each motor of two different robots will get two very different positions. But after calibration,
    the two robots will move to the same position.To this end, this function computes the homing offset
    and the drive mode for each motor of a given robot.

    Homing offset is used to shift the motor position to a ]-2048, +2048[ nominal range (when the motor uses 2048 steps
    to complete a half a turn). This range is set around an arbitrary "zero position" corresponding to all motor positions
    being 0. During the calibration process, you will need to manually move the robot to this "zero position".

    Drive mode is used to invert the rotation direction of the motor. This is useful when some motors have been assembled
    in the opposite orientation for some robots. During the calibration process, you will need to manually move the robot
    to the "rotated position".

    After calibration, the homing offsets and drive modes are stored in a cache.

    Example of usage:
    ```python
    run_arm_calibration(arm, "left", "follower")
    ```
    """
    reset_torque_mode(arm)

    print(f"\nRunning calibration of {name} {arm_type}...")

    print("\nMove arm to zero position")
    print("See: " + URL_TEMPLATE.format(robot="koch", arm=arm_type, position="zero"))
    input("Press Enter to continue...")

    # We arbitrarely choosed our zero target position to be a straight horizontal position with gripper upwards and closed.
    # It is easy to identify and all motors are in a "quarter turn" position. Once calibration is done, this position will
    # corresponds to every motor angle being 0. If you set all 0 as Goal Position, the arm will move in this position.
    zero_position = convert_degrees_to_steps(ZERO_POSITION_DEGREE, arm.motor_models)

    def _compute_nearest_rounded_position(position, models):
        # TODO(rcadene): Rework this function since some motors cant physically rotate a quarter turn
        # (e.g. the gripper of Aloha arms can only rotate ~50 degree)
        quarter_turn_degree = 90
        quarter_turn = convert_degrees_to_steps(quarter_turn_degree, models)
        nearest_pos = np.round(position.astype(float) / quarter_turn) * quarter_turn
        return nearest_pos.astype(position.dtype)

    # Compute homing offset so that `present_position + homing_offset ~= target_position`.
    position = arm.read("Present_Position")
    position = _compute_nearest_rounded_position(position, arm.motor_models)
    homing_offset = zero_position - position

    print("\nMove arm to rotated target position")
    print("See: " + URL_TEMPLATE.format(robot="koch", arm=arm_type, position="rotated"))
    input("Press Enter to continue...")

    # The rotated target position corresponds to a rotation of a quarter turn from the zero position.
    # This allows to identify the rotation direction of each motor.
    # For instance, if the motor rotates 90 degree, and its value is -90 after applying the homing offset, then we know its rotation direction
    # is inverted. However, for the calibration being successful, we need everyone to follow the same target position.
    # Sometimes, there is only one possible rotation direction. For instance, if the gripper is closed, there is only one direction which
    # corresponds to opening the gripper. When the rotation direction is ambiguous, we arbitrarely rotate clockwise from the point of view
    # of the previous motor in the kinetic chain.
    rotated_position = convert_degrees_to_steps(ROTATED_POSITION_DEGREE, arm.motor_models)

    # Find drive mode by rotating each motor by a quarter of a turn.
    # Drive mode indicates if the motor rotation direction should be inverted (=1) or not (=0).
    position = arm.read("Present_Position")
    position += homing_offset
    position = _compute_nearest_rounded_position(position, arm.motor_models)
    drive_mode = (position != rotated_position).astype(np.int32)

    # Re-compute homing offset to take into account drive mode
    position = arm.read("Present_Position")
    position = apply_drive_mode(position, drive_mode)
    position = _compute_nearest_rounded_position(position, arm.motor_models)
    homing_offset = rotated_position - position

    print("\nMove arm to rest position")
    print("See: " + URL_TEMPLATE.format(robot="koch", arm=arm_type, position="rest"))
    input("Press Enter to continue...")
    print()

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

    # Define all components of the robot
    leader_arms: dict[str, MotorsBus] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBus] = field(default_factory=lambda: {})
    cameras: dict[str, Camera] = field(default_factory=lambda: {})


class KochRobot:
    # TODO(rcadene): Implement force feedback
    """This class allows to control any Koch robot of various number of motors.

    A few versions are available:
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow expansion, which was developed
    by Alexander Koch from [Tau Robotics](https://tau-robotics.com): [Github for sourcing and assembly](
    - [Koch v1.1])https://github.com/jess-moss/koch-v1-1), which was developed by Jess Moss.

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
    # Here, the webcam of the laptop and the phone (connected in USB to the laptop)
    # can be reached respectively using the camera indices 0 and 1. These indices can be
    # arbitrary. See the documentation of `OpenCVCamera` to find your own camera indices.
    cameras = {
        "laptop": OpenCVCamera(camera_index=0, fps=30, width=640, height=480),
        "phone": OpenCVCamera(camera_index=1, fps=30, width=640, height=480),
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
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].connect()
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()

        # Reset the arms and load or run calibration
        if self.calibration_path.exists():
            # Reset all arms before setting calibration
            for name in self.follower_arms:
                reset_torque_mode(self.follower_arms[name])
            for name in self.leader_arms:
                reset_torque_mode(self.leader_arms[name])

            with open(self.calibration_path, "rb") as f:
                calibration = pickle.load(f)
        else:
            print(f"Missing calibration file '{self.calibration_path}'. Starting calibration precedure.")
            # Run calibration process which begins by reseting all arms
            calibration = self.run_calibration()

            print(f"Calibration is done! Saving calibration file '{self.calibration_path}'")
            self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.calibration_path, "wb") as f:
                pickle.dump(calibration, f)

        # Set calibration
        for name in self.follower_arms:
            self.follower_arms[name].set_calibration(calibration[f"follower_{name}"])
        for name in self.leader_arms:
            self.leader_arms[name].set_calibration(calibration[f"leader_{name}"])

        # Set better PID values to close the gap between recored states and actions
        # TODO(rcadene): Implement an automatic procedure to set optimial PID values for each motor
        for name in self.follower_arms:
            self.follower_arms[name].write("Position_P_Gain", 1500, "elbow_flex")
            self.follower_arms[name].write("Position_I_Gain", 0, "elbow_flex")
            self.follower_arms[name].write("Position_D_Gain", 600, "elbow_flex")

        # Enable torque on all motors of the follower arms
        for name in self.follower_arms:
            print(f"Activating torque on {name} follower arm.")
            self.follower_arms[name].write("Torque_Enable", 1)

        # Enable torque on the gripper of the leader arms, and move it to 45 degrees,
        # so that we can use it as a trigger to close the gripper of the follower arms.
        for name in self.leader_arms:
            self.leader_arms[name].write("Torque_Enable", 1, "gripper")
            self.leader_arms[name].write("Goal_Position", GRIPPER_OPEN_DEGREE, "gripper")

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

    def run_calibration(self):
        calibration = {}

        for name in self.follower_arms:
            homing_offset, drive_mode = run_arm_calibration(self.follower_arms[name], name, "follower")

            calibration[f"follower_{name}"] = {}
            for idx, motor_name in enumerate(self.follower_arms[name].motor_names):
                calibration[f"follower_{name}"][motor_name] = (homing_offset[idx], drive_mode[idx])

        for name in self.leader_arms:
            homing_offset, drive_mode = run_arm_calibration(self.leader_arms[name], name, "leader")

            calibration[f"leader_{name}"] = {}
            for idx, motor_name in enumerate(self.leader_arms[name].motor_names):
                calibration[f"leader_{name}"][motor_name] = (homing_offset[idx], drive_mode[idx])

        return calibration

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "KochRobot is not connected. You need to run `robot.connect()`."
            )

        # Prepare to assign the position of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        follower_goal_pos = {}
        for name in self.leader_arms:
            follower_goal_pos[name] = leader_pos[name]

        # Send action
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            self.follower_arms[name].write("Goal_Position", follower_goal_pos[name])
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

        # Early exit when recording data is not requested
        if not record_data:
            return

        # TODO(rcadene): Add velocity and other info
        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

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
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

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
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = np.concatenate(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = torch.from_numpy(state)
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = torch.from_numpy(images[name])
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

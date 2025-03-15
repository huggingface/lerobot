# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains logic to instantiate a robot, read information from its motors and cameras,
and send orders to its motors.
"""
# TODO(rcadene, aliberts): reorganize the codebase into one file per robot, with the associated
# calibration procedure, to make it easy for people to add their own robot.

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from lerobot.common.cameras.configs import CameraConfig
from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors.configs import MotorsBusConfig
from lerobot.common.motors.motors_bus import MotorsBus
from lerobot.common.motors.utils import make_motors_buses_from_configs
from lerobot.common.robots.config import RobotConfig
from lerobot.common.robots.utils import ensure_safe_goal_position, get_arm_id


@dataclass
class ManipulatorRobotConfig(RobotConfig):
    leader_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    # Optionally limit the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length
    # as the number of motors in your follower arms (assumes all follower arms have the same number of
    # motors).
    max_relative_target: list[float] | float | None = None

    # Optionally set the leader arm in torque mode with the gripper motor set to this angle. This makes it
    # possible to squeeze the gripper and have it spring back to an open position on its own. If None, the
    # gripper is not put in torque mode.
    gripper_open_degree: float | None = None

    mock: bool = False

    def __post_init__(self):
        if self.mock:
            for arm in self.leader_arms.values():
                if not arm.mock:
                    arm.mock = True
            for arm in self.follower_arms.values():
                if not arm.mock:
                    arm.mock = True
            for cam in self.cameras.values():
                if not cam.mock:
                    cam.mock = True

        if self.max_relative_target is not None and isinstance(self.max_relative_target, Sequence):
            for name in self.follower_arms:
                if len(self.follower_arms[name].motors) != len(self.max_relative_target):
                    raise ValueError(
                        f"len(max_relative_target)={len(self.max_relative_target)} but the follower arm with name {name} has "
                        f"{len(self.follower_arms[name].motors)} motors. Please make sure that the "
                        f"`max_relative_target` list has as many parameters as there are motors per arm. "
                        "Note: This feature does not yet work with robots where different follower arms have "
                        "different numbers of motors."
                    )


def apply_feetech_offsets_from_calibration(motorsbus, calibration_dict: dict):
    """
    Reads 'calibration_dict' containing 'homing_offset' and 'motor_names',
    then writes each motor's offset to the servo's internal Offset (0x1F) in EPROM.

    This version is modified so each homed position (originally 0) will now read
    2047, i.e. 180° away from 0 in the 4096-count circle. Offsets are permanently
    stored in EEPROM, so the servo's Present_Position is hardware-shifted even
    after power cycling.

    Steps:
      1) Subtract 2047 from the old offset (so 0 -> 2047).
      2) Clamp to [-2047..+2047].
      3) Encode sign bit and magnitude into a 12-bit number.
    """

    homing_offsets = calibration_dict["homing_offset"]
    motor_names = calibration_dict["motor_names"]
    start_pos = calibration_dict["start_pos"]

    # Open the write lock, changes to EEPROM do NOT persist yet
    motorsbus.write("Lock", 1)

    # For each motor, set the 'Offset' parameter
    for m_name, old_offset in zip(motor_names, homing_offsets, strict=False):
        # If bus doesn’t have a motor named m_name, skip
        if m_name not in motorsbus.motors:
            print(f"Warning: '{m_name}' not found in motorsbus.motors; skipping offset.")
            continue

        if m_name == "gripper":
            old_offset = start_pos  # If gripper set the offset to the start position of the gripper
            continue

        # Shift the offset so the homed position reads 2047
        new_offset = old_offset - 2047

        # Clamp to [-2047..+2047]
        if new_offset > 2047:
            new_offset = 2047
            print(
                f"Warning: '{new_offset}' is getting clamped because its larger then 2047; This should not happen!"
            )
        elif new_offset < -2047:
            new_offset = -2047
            print(
                f"Warning: '{new_offset}' is getting clamped because its smaller then -2047; This should not happen!"
            )

        # Determine the direction (sign) bit and magnitude
        direction_bit = 1 if new_offset < 0 else 0
        magnitude = abs(new_offset)

        # Combine sign bit (bit 11) with the magnitude (bits 0..10)
        servo_offset = (direction_bit << 11) | magnitude

        # Write offset to servo
        motorsbus.write("Offset", servo_offset, motor_names=m_name)
        print(
            f"Set offset for {m_name}: "
            f"old_offset={old_offset}, new_offset={new_offset}, servo_encoded={magnitude} + direction={direction_bit}"
        )

    motorsbus.write("Lock", 0)
    print("Offsets have been saved to EEPROM successfully.")


class ManipulatorRobot:
    # TODO(rcadene): Implement force feedback
    """This class allows to control any manipulator robot of various number of motors.

    Non exhaustive list of robots:
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow expansion, developed
    by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    - [Aloha](https://www.trossenrobotics.com/aloha-kits) developed by Trossen Robotics

    Example of instantiation, a pre-defined robot config is required:
    ```python
    robot = ManipulatorRobot(KochRobotConfig())
    ```

    Example of overwriting motors during instantiation:
    ```python
    # Defines how to communicate with the motors of the leader and follower arms
    leader_arms = {
        "main": DynamixelMotorsBusConfig(
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
        "main": DynamixelMotorsBusConfig(
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
    robot_config = KochRobotConfig(leader_arms=leader_arms, follower_arms=follower_arms)
    robot = ManipulatorRobot(robot_config)
    ```

    Example of overwriting cameras during instantiation:
    ```python
    # Defines how to communicate with 2 cameras connected to the computer.
    # Here, the webcam of the laptop and the phone (connected in USB to the laptop)
    # can be reached respectively using the camera indices 0 and 1. These indices can be
    # arbitrary. See the documentation of `OpenCVCamera` to find your own camera indices.
    cameras = {
        "laptop": OpenCVCamera(camera_index=0, fps=30, width=640, height=480),
        "phone": OpenCVCamera(camera_index=1, fps=30, width=640, height=480),
    }
    robot = ManipulatorRobot(KochRobotConfig(cameras=cameras))
    ```

    Once the robot is instantiated, connect motors buses and cameras if any (Required):
    ```python
    robot.connect()
    ```

    Example of highest frequency teleoperation, which doesn't require cameras:
    ```python
    while True:
        robot.teleop_step()
    ```

    Example of highest frequency data collection from motors and cameras (if any):
    ```python
    while True:
        observation, action = robot.teleop_step(record_data=True)
    ```

    Example of controlling the robot with a policy:
    ```python
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
        config: ManipulatorRobotConfig,
    ):
        self.config = config
        self.robot_type = self.config.type
        self.calibration_dir = Path(self.config.calibration_dir)
        self.leader_arms = make_motors_buses_from_configs(self.config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(self.config.follower_arms)
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}

    def get_motor_names(self, arm: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arm.items() for motor in bus.motors]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = self.get_motor_names(self.leader_arms)
        state_names = self.get_motor_names(self.leader_arms)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # Connect the arms
        for name in self.follower_arms:
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()

        if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
            from lerobot.common.motors.dynamixel.dynamixel import TorqueMode
        elif self.robot_type in ["so100", "moss", "lekiwi"]:
            from lerobot.common.motors.feetech.feetech import TorqueMode

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        for name in self.follower_arms:
            self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        for name in self.leader_arms:
            self.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

        self.activate_calibration()

        # Set robot preset (e.g. torque in leader gripper for Koch v1.1)
        if self.robot_type in ["koch", "koch_bimanual"]:
            self.set_koch_robot_preset()
        elif self.robot_type == "aloha":
            self.set_aloha_robot_preset()
        elif self.robot_type in ["so100", "moss", "lekiwi"]:
            self.set_so100_robot_preset()

        # Enable torque on all motors of the follower arms
        for name in self.follower_arms:
            print(f"Activating torque on {name} follower arm.")
            self.follower_arms[name].write("Torque_Enable", 1)

        if self.config.gripper_open_degree is not None:
            if self.robot_type not in ["koch", "koch_bimanual"]:
                raise NotImplementedError(
                    f"{self.robot_type} does not support position AND current control in the handle, which is require to set the gripper open."
                )
            # Set the leader arm in torque mode with the gripper motor set to an angle. This makes it possible
            # to squeeze the gripper and have it spring back to an open position on its own.
            for name in self.leader_arms:
                self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                self.leader_arms[name].write("Goal_Position", self.config.gripper_open_degree, "gripper")

        # Check both arms can be read
        for name in self.follower_arms:
            self.follower_arms[name].read("Present_Position")
        for name in self.leader_arms:
            self.leader_arms[name].read("Present_Position")

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

    def activate_calibration(self):
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """

        def load_or_run_calibration_(name, arm, arm_type):
            arm_id = get_arm_id(name, arm_type)
            arm_calib_path = self.calibration_dir / f"{arm_id}.json"

            if arm_calib_path.exists():
                with open(arm_calib_path) as f:
                    calibration = json.load(f)
            else:
                # TODO(rcadene): display a warning in __init__ if calibration file not available
                print(f"Missing calibration file '{arm_calib_path}'")

                if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
                    from lerobot.common.motors.dynamixel.dynamixel_calibration import run_arm_calibration

                    calibration = run_arm_calibration(arm, self.robot_type, name, arm_type)

                elif self.robot_type in ["so100", "moss", "lekiwi"]:
                    from lerobot.common.motors.feetech.feetech_calibration import (
                        run_full_arm_calibration,
                    )

                    calibration = run_full_arm_calibration(arm, self.robot_type, name, arm_type)

                print(f"Calibration is done! Saving calibration file '{arm_calib_path}'")
                arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
                with open(arm_calib_path, "w") as f:
                    json.dump(calibration, f)

            return calibration

            # For each follower arm

        for name, arm_bus in self.follower_arms.items():
            calibration = load_or_run_calibration_(name, arm_bus, "follower")
            arm_bus.set_calibration(calibration)

            # If this is a Feetech robot, also set the servo offset into EEPROM
            if self.robot_type in ["so100", "lekiwi"]:
                apply_feetech_offsets_from_calibration(arm_bus, calibration)

        # For each leader arm
        for name, arm_bus in self.leader_arms.items():
            calibration = load_or_run_calibration_(name, arm_bus, "leader")
            arm_bus.set_calibration(calibration)

            # Optionally also set offset for leader if you want the servo offsets as well
            if self.robot_type in ["so100", "lekiwi"]:
                apply_feetech_offsets_from_calibration(arm_bus, calibration)

    def set_koch_robot_preset(self):
        def set_operating_mode_(arm):
            from lerobot.common.motors.dynamixel.dynamixel import TorqueMode

            if (arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
                raise ValueError("To run set robot preset, the torque must be disabled on all motors.")

            # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
            # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
            # you could end up with a servo with a position 0 or 4095 at a crucial point See [
            # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
            all_motors_except_gripper = [name for name in arm.motor_names if name != "gripper"]
            if len(all_motors_except_gripper) > 0:
                # 4 corresponds to Extended Position on Koch motors
                arm.write("Operating_Mode", 4, all_motors_except_gripper)

            # Use 'position control current based' for gripper to be limited by the limit of the current.
            # For the follower gripper, it means it can grasp an object without forcing too much even tho,
            # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
            # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
            # to make it move, and it will move back to its original target position when we release the force.
            # 5 corresponds to Current Controlled Position on Koch gripper motors "xl330-m077, xl330-m288"
            arm.write("Operating_Mode", 5, "gripper")

        for name in self.follower_arms:
            set_operating_mode_(self.follower_arms[name])

            # Set better PID values to close the gap between recorded states and actions
            # TODO(rcadene): Implement an automatic procedure to set optimal PID values for each motor
            self.follower_arms[name].write("Position_P_Gain", 1500, "elbow_flex")
            self.follower_arms[name].write("Position_I_Gain", 0, "elbow_flex")
            self.follower_arms[name].write("Position_D_Gain", 600, "elbow_flex")

        if self.config.gripper_open_degree is not None:
            for name in self.leader_arms:
                set_operating_mode_(self.leader_arms[name])

                # Enable torque on the gripper of the leader arms, and move it to 45 degrees,
                # so that we can use it as a trigger to close the gripper of the follower arms.
                self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                self.leader_arms[name].write("Goal_Position", self.config.gripper_open_degree, "gripper")

    def set_aloha_robot_preset(self):
        def set_shadow_(arm):
            # Set secondary/shadow ID for shoulder and elbow. These joints have two motors.
            # As a result, if only one of them is required to move to a certain position,
            # the other will follow. This is to avoid breaking the motors.
            if "shoulder_shadow" in arm.motor_names:
                shoulder_idx = arm.read("ID", "shoulder")
                arm.write("Secondary_ID", shoulder_idx, "shoulder_shadow")

            if "elbow_shadow" in arm.motor_names:
                elbow_idx = arm.read("ID", "elbow")
                arm.write("Secondary_ID", elbow_idx, "elbow_shadow")

        for name in self.follower_arms:
            set_shadow_(self.follower_arms[name])

        for name in self.leader_arms:
            set_shadow_(self.leader_arms[name])

        for name in self.follower_arms:
            # Set a velocity limit of 131 as advised by Trossen Robotics
            self.follower_arms[name].write("Velocity_Limit", 131)

            # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
            # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
            # you could end up with a servo with a position 0 or 4095 at a crucial point See [
            # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
            all_motors_except_gripper = [
                name for name in self.follower_arms[name].motor_names if name != "gripper"
            ]
            if len(all_motors_except_gripper) > 0:
                # 4 corresponds to Extended Position on Aloha motors
                self.follower_arms[name].write("Operating_Mode", 4, all_motors_except_gripper)

            # Use 'position control current based' for follower gripper to be limited by the limit of the current.
            # It can grasp an object without forcing too much even tho,
            # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
            # 5 corresponds to Current Controlled Position on Aloha gripper follower "xm430-w350"
            self.follower_arms[name].write("Operating_Mode", 5, "gripper")

            # Note: We can't enable torque on the leader gripper since "xc430-w150" doesn't have
            # a Current Controlled Position mode.

        if self.config.gripper_open_degree is not None:
            warnings.warn(
                f"`gripper_open_degree` is set to {self.config.gripper_open_degree}, but None is expected for Aloha instead",
                stacklevel=1,
            )

    def set_so100_robot_preset(self):
        for name in self.follower_arms:
            # Mode=0 for Position Control
            self.follower_arms[name].write("Mode", 0)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.follower_arms[name].write("P_Coefficient", 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.follower_arms[name].write("I_Coefficient", 0)
            self.follower_arms[name].write("D_Coefficient", 32)
            # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
            # which is mandatory for Maximum_Acceleration to take effect after rebooting.
            self.follower_arms[name].write("Lock", 0)
            # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
            # the motors. Note: this configuration is not in the official STS3215 Memory Table
            self.follower_arms[name].write("Maximum_Acceleration", 254)
            self.follower_arms[name].write("Acceleration", 254)

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Prepare to assign the position of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Send goal position to the follower
        follower_goal_pos = {}
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name]

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Used when record_data=True
            follower_goal_pos[name] = goal_pos

            goal_pos = goal_pos.numpy().astype(np.int32)
            self.follower_arms[name].write("Goal_Position", goal_pos)
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
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Create action by concatenating follower goal position
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Command the follower arms to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action: tensor containing the concatenated goal positions for the follower arms.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:
            # Get goal position of each follower arm by splitting the action vector
            to_idx += len(self.follower_arms[name].motor_names)
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Save tensor to concat and return
            action_sent.append(goal_pos)

            # Send goal position to each follower
            goal_pos = goal_pos.numpy().astype(np.int32)
            self.follower_arms[name].write("Goal_Position", goal_pos)

        return torch.cat(action_sent)

    def print_logs(self):
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
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

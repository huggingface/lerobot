#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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


import copy
import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import transforms3d as t3d

# Adapt imports to your environment.
# Assuming woanarm_api_py is available in python path
import woanarm_api_py as woanarm
import woangripper_api_py as woangripper

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_woan_arm import WoanRobotConfig

logger = logging.getLogger("woan_arm")


class WoanAdapter(Robot):
    """
    Adapter for Woan Robot Arm (7-DOF) to be used with LeRobot.
    """

    config_class = WoanRobotConfig
    name = "woan_arm"

    def __init__(self, config: WoanRobotConfig):
        """
        Initialize the WoanAdapter with the given configuration.

        Args:
            config (WoanRobotConfig): Configuration object containing device path and other settings.
        """
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._arm = None
        self._prev_observation = None
        self._gripper = None

        # Internal state for observation calculation
        self.dof = 7  # As per documentation

        # Initialize cameras from config
        if not config.is_teleop_leader:
            self.cameras = make_cameras_from_configs(config.cameras)

    def __str__(self) -> str:
        return f"woan {self.config.id}"

    def connect(self) -> None:
        """
        Connect to the Woan Arm robot.

        Establish communication with the robot arm hardware using the configuration provided.
        Enables motors upon successful connection.

        Raises:
            DeviceAlreadyConnectedError: If the device is already connected.
            ConnectionError: If enabling motors fails.
            Exception: Propagates any other exceptions during connection.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 1. Setup API Config
        api_config = woanarm.WoanConfig()
        api_config.device = self.config.port
        api_config.baud_rate = self.config.baud_rate

        api_config.robot_model = self.config.robot_model
        api_config.version = self.config.version
        api_config.model_description_path = self.config.woan_description_path
        api_config.slcan_type = self.config.slcan_type
        # Set other defaults or from config if needed
        api_config.dof = self.dof
        api_config.is_teleop_leader = self.config.is_teleop_leader

        try:
            # 2. Initialize and Connect Arm
            self._arm = woanarm.WoanArm(api_config)

            # 3. Enable Motors
            if not self._arm.enable_motors():
                raise ConnectionError("Failed to enable motors")

            if self.config.enable_gripper:
                # 4. Optionally Initialize Gripper
                self._gripper = woangripper.GripperControl()
                if not self._gripper.initialize(self.config.port, self.config.slcan_type):
                    logger.warning("Failed to initialize gripper, continuing without it.")

            # Connect cameras
            if not self.config.is_teleop_leader:
                for cam_name, cam in self.cameras.items():
                    cam.connect()
                    logger.info(f"{self} camera '{cam_name}' connected.")

            self.is_connected = True
            logger.info(f"{self} connected.")

        except Exception as e:
            logger.error(f"Failed to connect to {self}: {e}")
            self.disconnect()
            raise e

    def disconnect(self) -> None:
        """
        Disconnect from the Woan Arm robot.

        Disables motors and tears down the connection to the hardware.
        """
        if not self.is_connected:
            return

        # Disconnect cameras
        if not self.config.is_teleop_leader:
            for cam_name, cam in self.cameras.items():
                if cam.is_connected:
                    cam.disconnect()
                    logger.info(f"{self} camera '{cam_name}' disconnected.")
        if self._arm:
            self._arm.disable_motors()
            self._arm = None

        self.is_connected = False
        logger.info(f"{self} disconnected.")

    @property
    def is_connected(self) -> bool:
        """
        Check if the robot is currently connected (arm + all cameras).

        Returns:
            bool: True if connected, False otherwise.
        """
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = value

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action = copy.copy(action)

        # extract common parameters
        params = {
            "speed_scale": action.get("speed_scale", 1.0),
            "trajectory_connect": int(action.get("trajectory_connect", 0)),
            "block": action.get("block", True),
        }

        # define action handlers
        handlers: list[tuple[Callable, Callable]] = [
            (self._is_joint_control, self._execute_joint_control),
            (self._is_pose_control, self._execute_pose_control),
            (self._is_reset_control, self._execute_reset_position),
            (self._is_gripper_control, self._execute_gripper_control),
        ]

        # dispatch action
        for check_func, exec_func in handlers:
            if check_func(action):
                exec_func(action, params)
                return action

        return action

    # --- Predicate Functions ---

    def _is_joint_control(self, action: dict) -> bool:
        # Check if all joint data is present
        return all(f"joint{i}.pos" in action for i in range(1, self.dof + 1))

    def _is_pose_control(self, action: dict) -> bool:
        return "pose" in action

    def _is_reset_control(self, action: dict) -> bool:
        return action.get("reset", False)

    def _is_gripper_control(self, action: dict) -> bool:
        if not self.config.enable_gripper:
            return False
        return "gripper.position" in action

    # -- Execution Functions ---

    def _execute_joint_control(self, action: dict, params: dict):
        target_joints = [float(action[f"joint{i}.pos"]) for i in range(1, self.dof + 1)]
        self._arm.movej(target_joints, **params)

    def _execute_pose_control(self, action: dict, params: dict):
        target_pose = self._parse_pose(action["pose"])
        if target_pose:
            self._arm.movep(target_pose, **params)

    def _execute_reset_position(self):
        home_joints = list(self.config.home_joints_positions)

        # Reset may require forcing specific parameters, overriding common parameters
        self._arm.movej(home_joints, speed_scale=0.5, trajectory_connect=0)

    def _execute_gripper_control(self, action: dict, params: dict):
        if not self._gripper:
            logger.warning("Gripper control requested but gripper is not initialized.")
            return

        position = int(action["gripper.position"])
        self._gripper.set_position(position)

    # --- Utility Functions: Complex Data Conversion ---

    def _parse_pose(self, raw_pose) -> Any:
        """Convert various input formats to a woanarm.Pose object"""
        if isinstance(raw_pose, woanarm.Pose):
            return raw_pose

        target_pose = woanarm.Pose()
        pose_arr = np.array(raw_pose)

        if pose_arr.size == 7:
            # [x, y, z, qw, qx, qy, qz]
            flat = pose_arr.flatten()
            target_pose.x, target_pose.y, target_pose.z = map(float, flat[:3])
            target_pose.qw, target_pose.qx, target_pose.qy, target_pose.qz = map(float, flat[3:])
        elif pose_arr.shape == (4, 4):
            # 4x4 Matrix
            translation = pose_arr[:3, 3]
            quat = t3d.quaternions.mat2quat(pose_arr[:3, :3])
            target_pose.x, target_pose.y, target_pose.z = map(float, translation)
            target_pose.qw, target_pose.qx, target_pose.qy, target_pose.qz = map(float, quat)
        else:
            # logger.warning(...)
            return None
        return target_pose

    def get_observation(self) -> dict[str, Any]:
        """
        Get the current observation of the robot state.

        Retrieves joint positions from the motors and computes joint velocities.

        Returns:
            Dict[str, Any]: Dictionary containing joint positions and velocities.
            Keys are 'joint1.pos', 'joint1.vel', etc.

        Raises:
            DeviceNotConnectedError: If the robot is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        current_joints = self._arm.get_joint_positions_from_motors()
        current_joints_vel = self._arm.get_joint_velocities()

        for i in range(self.dof):
            # API uses 0-based index likely, joint names usually 1-based
            obs_dict[f"joint{i + 1}.pos"] = current_joints[i]
            if self.config.use_velocity:
                obs_dict[f"joint{i + 1}.vel"] = current_joints_vel[i]

        if self.config.enable_gripper:
            gripper_status = self._gripper.get_gripper_status()
            obs_dict["gripper.position"] = gripper_status.position
            obs_dict["gripper.force"] = gripper_status.force

        # Capture images from cameras
        if not self.config.is_teleop_leader:
            for cam_key, cam in self.cameras.items():
                start = time.perf_counter()
                obs_dict[cam_key] = cam.async_read()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        self._prev_observation = obs_dict
        return obs_dict

    @property
    def _motors_ft(self) -> dict[str, type]:
        """
        Internal property defining the feature types for motors.
        """
        motors = {f"joint{i}.pos": float for i in range(1, self.dof + 1)}
        return motors

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """
        Internal property defining the feature types for cameras.
        Returns a dict mapping camera name to (height, width, channels) tuple.
        """
        if self.config.is_teleop_leader:
            return {}

        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def action_features(self) -> dict[str, type]:
        """
        Get the action space features.

        Returns:
            Dict[str, type]: Dictionary mapping action keys to their types.
        """
        return self._motors_ft

    @property
    def observation_features(self) -> dict[str, Any]:
        # depack observation features, avoid original dict mutation
        features = {**self._motors_ft}

        if self.config.use_velocity:
            for i in range(1, self.dof + 1):
                features[f"joint{i}.vel"] = float

        # Add camera features
        features.update(self._cameras_ft)

        return features

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True


class WoanTeleopFollower(WoanAdapter):
    """
    Specialized Adapter for Teleoperation scenarios.

    This adapter overrides send_action to prioritize direct MIT control.
    """

    def unpack_action(self, action_tensor: torch.Tensor, dof: int = 7) -> dict:
        """
        unpack data from tensor to action dict for teleoperation.

        Args:
            action_tensor (torch.Tensor): A 1D tensor of shape (2 * dof,),
                                        where the first dof elements are positions and the last dof elements are velocities.
            dof (int): Degrees of freedom, default is 7.

        Returns:
            dict: A dictionary containing 'jointX.pos' and 'jointX.vel' keys.
        """
        action_list = action_tensor.tolist()

        # Sanity check
        if len(action_list) != dof * 2:
            logger.warning(f"Action tensor length {len(action_list)} does not match expected {dof * 2}")

        pos = action_list[:dof]
        vel = action_list[dof:]

        action_dict = {}
        for i in range(dof):
            joint_name = f"joint{i + 1}"
            action_dict[f"{joint_name}.pos"] = pos[i]
            action_dict[f"{joint_name}.vel"] = vel[i]

        return action_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action to the robot with teleoperation-specific optimizations.

        Optimizations:
        1. Enforces non-blocking execution (block=False).
        2. Streamlines the control path (skips complex checks if simple joint control is detected).
        3. Robust error handling for high-frequency loops.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if action.get("reset", False):
            # Follower directly handles reset with fixed params
            self._execute_reset_position()
            return action

        # Teleop primarily relies on Joint Position Control and Velocity Control
        # Check if we have a standard joint position command and velocity command
        # (This check is slightly faster than calling the parent's generic handler loop)
        try:
            if action.get("mirror", False):
                action = self.mirror_action(action)
            r = range(1, self.dof + 1)
            target_pos = [float(action[f"joint{i}.pos"]) for i in r]
            target_vel = [float(action[f"joint{i}.vel"]) for i in r]

            self._arm.send_trajectory_point(target_pos, target_vel)

            gripper_pos = action.get("gripper.position")
            if self.config.enable_gripper and gripper_pos is not None:
                self._gripper.set_position(gripper_pos)

        except KeyError:
            # Fall back to standard handler if not all joints are specified
            logger.error("Incomplete teleop action command, falling back to standard handler.")
            return super().send_action(action)

        except (ValueError, TypeError) as e:
            # Catch invalid value types
            logger.warning(f"Invalid joint values in action: {e}. Command ignored.")
            return action

        return action

    def mirror_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Mirror the action for teleoperation follower.
        """
        mirrored_action = copy.deepcopy(action)

        # Apply negation to joint positions and velocities
        for i in range(1, self.dof + 1):
            joint_pos = f"joint{i}.pos"
            if joint_pos in mirrored_action:
                mirrored_action[joint_pos] = -mirrored_action[joint_pos]

            joint_vel = f"joint{i}.vel"
            if joint_vel in mirrored_action:
                mirrored_action[joint_vel] = -mirrored_action[joint_vel]

        return mirrored_action

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

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_moveit2 import ActionType, MoveIt2Config
from .moveit2_interface import MoveIt2Interface

logger = logging.getLogger(__name__)


class MoveIt2(Robot):
    config_class = MoveIt2Config
    name = "moveit2"

    def __init__(self, config: MoveIt2Config):
        super().__init__(config)
        self.config = config
        self.moveit2_interface = MoveIt2Interface(config.moveit2_interface)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        all_joint_names = self.config.moveit2_interface.arm_joint_names.copy()
        if self.config.moveit2_interface.gripper_joint_name:
            all_joint_names.append(self.config.moveit2_interface.gripper_joint_name)
        motor_state_ft = {f"{motor}.pos": float for motor in all_joint_names}
        return {**motor_state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.action_type == ActionType.CARTESIAN_VELOCITY:
            return {
                "linear_x.vel": float,
                "linear_y.vel": float,
                "linear_z.vel": float,
                "angular_x.vel": float,
                "angular_y.vel": float,
                "angular_z.vel": float,
                "gripper.pos": float,
            }
        elif self.config.action_type == ActionType.JOINT_POSITION:
            return {f"{joint}.pos": float for joint in self.config.moveit2_interface.arm_joint_names} | {
                "gripper.pos": float
            }
        else:
            raise ValueError(f"Unsupported action type: {self.config.action_type}")

    @property
    def is_connected(self) -> bool:
        return self.moveit2_interface.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        for cam in self.cameras.values():
            cam.connect()
        self.moveit2_interface.connect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass  # robot must be calibrated before running LeRobot

    def configure(self) -> None:
        pass  # robot must be configured before running LeRobot

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}
        joint_state = self.moveit2_interface.joint_state
        if joint_state is None:
            raise ValueError("Joint state is not available yet.")
        obs_dict.update({f"{joint}.pos": pos for joint, pos in joint_state["position"].items()})

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (dict[str, float]): The goal positions for the motors or pressed_keys dict.

        Raises:
            DeviceNotConnectedError: if robot is not connected.

        Returns:
            dict[str, float]: The action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.action_type == ActionType.CARTESIAN_VELOCITY:
            if self.config.action_from_keyboard:
                action = self.keyboard_to_velocity_action(action)

            if self.config.max_relative_target is not None:
                # We don't have the current velocity of the arm, so set it to 0.0
                # Effectively the goal velocity gets clipped by max_relative_target
                goal_present_vel = {key: (act, 0.0) for key, act in action.items()}
                action = ensure_safe_goal_position(goal_present_vel, self.config.max_relative_target)

            linear_vel = (
                action["linear_x.vel"],
                action["linear_y.vel"],
                action["linear_z.vel"],
            )
            angular_vel = (
                action["angular_x.vel"],
                action["angular_y.vel"],
                action["angular_z.vel"],
            )
            self.moveit2_interface.servo(linear=linear_vel, angular=angular_vel)
        elif self.config.action_type == ActionType.JOINT_POSITION:
            if self.config.action_from_keyboard:
                action = self.keyboard_to_joint_position_action(action)

            if self.config.max_relative_target is not None:
                goal_present_pos = {}
                joint_state = self.moveit2_interface.joint_state
                if joint_state is None:
                    raise ValueError("Joint state is not available yet.")

                for key, goal in action.items():
                    present_pos = joint_state["position"].get(key.replace(".pos", ""), 0.0)
                    goal_present_pos[key] = (goal, present_pos)
                action = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

            joint_positions = [
                action[joint + ".pos"] for joint in self.config.moveit2_interface.arm_joint_names
            ]
            self.moveit2_interface.send_joint_position_command(joint_positions)

        gripper_pos = action["gripper.pos"]
        self.moveit2_interface.send_gripper_command(gripper_pos)
        return action

    def keyboard_to_joint_position_action(self, pressed_keys: dict[str, Any]) -> dict[str, float]:
        """Convert pressed keys to joint position action commands for teleop.
        hardcoded for a 6-DOF arm with a gripper.
        """
        action = {f"{joint}.pos": 0.0 for joint in self.config.moveit2_interface.arm_joint_names}
        if "q" in pressed_keys:
            action["joint_1.pos"] += 0.1
        if "a" in pressed_keys:
            action["joint_1.pos"] -= 0.1
        if "w" in pressed_keys:
            action["joint_2.pos"] += 0.1
        if "s" in pressed_keys:
            action["joint_2.pos"] -= 0.1
        if "e" in pressed_keys:
            action["joint_3.pos"] += 0.1
        if "d" in pressed_keys:
            action["joint_3.pos"] -= 0.1
        if "r" in pressed_keys:
            action["joint_4.pos"] += 0.1
        if "f" in pressed_keys:
            action["joint_4.pos"] -= 0.1
        if "t" in pressed_keys:
            action["joint_5.pos"] += 0.1
        if "g" in pressed_keys:
            action["joint_5.pos"] -= 0.1
        if "y" in pressed_keys:
            action["joint_6.pos"] += 0.1
        if "h" in pressed_keys:
            action["joint_6.pos"] -= 0.1

        gripper_pos = 0.0
        if "space" in pressed_keys:
            gripper_pos = 1.0

        action["gripper.pos"] = gripper_pos
        return action

    def keyboard_to_velocity_action(self, pressed_keys: dict[str, Any]) -> dict[str, float]:
        """Convert pressed keys to velocity action commands for teleop."""
        lin_vel_x = 0.0
        if "a" in pressed_keys:
            lin_vel_x += 1.0
        if "d" in pressed_keys:
            lin_vel_x -= 1.0
        lin_vel_y = 0.0
        if "w" in pressed_keys:
            lin_vel_y -= 1.0
        if "s" in pressed_keys:
            lin_vel_y += 1.0
        lin_vel_z = 0.0
        if "n" in pressed_keys:
            lin_vel_z -= 1.0
        if "m" in pressed_keys:
            lin_vel_z += 1.0

        ang_vel_y = 0.0
        if "j" in pressed_keys:
            ang_vel_y -= 1.0
        if "l" in pressed_keys:
            ang_vel_y += 1.0

        ang_vel_x = 0.0
        if "i" in pressed_keys:
            ang_vel_x -= 1.0
        if "k" in pressed_keys:
            ang_vel_x += 1.0
        ang_vel_z = 0.0
        if "u" in pressed_keys:
            ang_vel_z += 1.0
        if "o" in pressed_keys:
            ang_vel_z -= 1.0

        gripper_pos = 0.0
        if "space" in pressed_keys:
            gripper_pos = 1.0

        return {
            "linear_x.vel": lin_vel_x,
            "linear_y.vel": lin_vel_y,
            "linear_z.vel": lin_vel_z,
            "angular_x.vel": ang_vel_x,
            "angular_y.vel": ang_vel_y,
            "angular_z.vel": ang_vel_z,
            "gripper.pos": gripper_pos,
        }

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()
        self.moveit2_interface.disconnect()

        logger.info(f"{self} disconnected.")

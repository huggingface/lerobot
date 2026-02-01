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

"""
Piper Dual Arm Robot with Software-level Teleoperation.

This robot plugin connects to 4 Piper arms (2 leaders, 2 followers) and implements
software-level teleoperation:
- Reads joint positions from leader arms
- Writes joint positions to follower arms
- During inference, only followers are controlled (no leaders needed)
"""

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors.piper.piper import PiperMotorsBus, PiperMotorsBusConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_piper_dual_teleop import PIPERDualTeleopConfig

logger = logging.getLogger(__name__)


def get_motor_names(arm: dict[str, Any]) -> list[str]:
    return [motor for arm_key, bus in arm.items() for motor in bus.motors]


class PIPERDualTeleop(Robot):
    """Piper Dual Arm Robot with Software-level Teleoperation.

    This robot connects to 4 Piper arms:
    - 2 Leader arms: used for reading teleop commands (operator moves these)
    - 2 Follower arms: used for executing actions (these follow the leaders)

    In teleop mode (use_teleop=True):
    - get_observation() reads follower states (what actually happened)
    - send_action() writes to followers
    - teleop_step() is called internally to read leaders and command followers

    In inference mode (use_teleop=False):
    - Only followers are connected/controlled
    - Leaders are not needed
    """

    config_class = PIPERDualTeleopConfig
    name = "piper_dual_teleop"

    def __init__(self, config: PIPERDualTeleopConfig):
        super().__init__(config)
        self.config = config

        # Define motor configuration (same for all arms)
        motor_config = {
            "joint_1": (1, "agilex_piper"),
            "joint_2": (2, "agilex_piper"),
            "joint_3": (3, "agilex_piper"),
            "joint_4": (4, "agilex_piper"),
            "joint_5": (5, "agilex_piper"),
            "joint_6": (6, "agilex_piper"),
            "gripper": (7, "agilex_piper"),
        }

        # Create follower buses (always needed)
        self.left_follower_bus = PiperMotorsBus(
            PiperMotorsBusConfig(
                can_name=config.left_follower_port,
                motors=motor_config.copy(),
            )
        )
        self.right_follower_bus = PiperMotorsBus(
            PiperMotorsBusConfig(
                can_name=config.right_follower_port,
                motors=motor_config.copy(),
            )
        )

        # Create leader buses (only for teleop mode)
        self.left_leader_bus = None
        self.right_leader_bus = None
        if config.use_teleop:
            self.left_leader_bus = PiperMotorsBus(
                PiperMotorsBusConfig(
                    can_name=config.left_leader_port,
                    motors=motor_config.copy(),
                )
            )
            self.right_leader_bus = PiperMotorsBus(
                PiperMotorsBusConfig(
                    can_name=config.right_leader_port,
                    motors=motor_config.copy(),
                )
            )

        self.logs = {}
        self._is_connected = False
        self._is_calibrated = False
        self.cameras = make_cameras_from_configs(config.cameras)

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
        # Left Arm
        left_arm_dict = {"follower": self.left_follower_bus}
        left_action_names = [f"left_{name}" for name in get_motor_names(left_arm_dict)]
        left_state_names = [f"left_{name}" for name in get_motor_names(left_arm_dict)]

        # Right Arm
        right_arm_dict = {"follower": self.right_follower_bus}
        right_action_names = [f"right_{name}" for name in get_motor_names(right_arm_dict)]
        right_state_names = [f"right_{name}" for name in get_motor_names(right_arm_dict)]

        action_names = left_action_names + right_action_names
        state_names = left_state_names + right_state_names

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
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for record/replay"""
        left_arm_dict = {"follower": self.left_follower_bus}
        left_motor_names = get_motor_names(left_arm_dict)

        right_arm_dict = {"follower": self.right_follower_bus}
        right_motor_names = get_motor_names(right_arm_dict)

        features = {}
        for name in left_motor_names:
            features[f"left_{name}.pos"] = float
        for name in right_motor_names:
            features[f"right_{name}.pos"] = float

        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for record/replay"""
        return {cam_key: (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def configure(self, **kwargs):
        pass

    @property
    def is_connected(self) -> bool:
        """Check if robot and all cameras are connected"""
        followers_connected = self.left_follower_bus.is_connected and self.right_follower_bus.is_connected

        if self.config.use_teleop:
            leaders_connected = (
                self.left_leader_bus is not None
                and self.right_leader_bus is not None
                and self.left_leader_bus.is_connected
                and self.right_leader_bus.is_connected
            )
            return (
                followers_connected
                and leaders_connected
                and all(cam.is_connected for cam in self.cameras.values())
            )

        return followers_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated"""
        return self._is_calibrated

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self) -> None:
        """Connect all arms and cameras"""
        if self._is_connected:
            raise DeviceAlreadyConnectedError("Piper is already connected. Do not run robot.connect() twice.")

        # Connect followers (always enabled for control)
        print("Connecting left follower arm...")
        self.left_follower_bus.connect(enable=True)
        print("Left follower connected (ACTIVE).")

        print("Connecting right follower arm...")
        self.right_follower_bus.connect(enable=True)
        print("Right follower connected (ACTIVE).")

        # Connect leaders if in teleop mode (passive reading, no enable needed)
        if self.config.use_teleop:
            print("Connecting left leader arm (read-only)...")
            # Leaders don't need enable - we just read from them
            # But SDK requires connect call to open CAN port
            # Note: We don't enable leaders so they remain passive
            try:
                self.left_leader_bus.connect(enable=False)
            except Exception as e:
                logger.warning(f"Left leader connect warning (expected): {e}")
            print("Left leader connected (PASSIVE).")

            print("Connecting right leader arm (read-only)...")
            try:
                self.right_leader_bus.connect(enable=False)
            except Exception as e:
                logger.warning(f"Right leader connect warning (expected): {e}")
            print("Right leader connected (PASSIVE).")

        print(f"piper_dual_teleop connected (use_teleop={self.config.use_teleop})")

        # Connect cameras
        for name in self.cameras:
            self.cameras[name].connect()
            print(f"camera {name} connected")

        print("All connected")
        self._is_connected = True

        # Calibrate followers to home position
        self.calibrate()

    def disconnect(self) -> None:
        """Disconnect all arms and cameras"""
        print("Disconnecting left follower arm...")
        self.left_follower_bus.safe_disconnect()
        print("Disconnecting right follower arm...")
        self.right_follower_bus.safe_disconnect()

        print("piper disable after 5 seconds")
        time.sleep(5)

        self.left_follower_bus.connect(enable=False)
        self.right_follower_bus.connect(enable=False)

        # Leaders don't need explicit disconnect since they're passive
        if self.config.use_teleop:
            print("Leaders were in passive mode, no action needed.")

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self._is_connected = False

    def calibrate(self):
        """Move followers to home positions"""
        if not self._is_connected:
            raise ConnectionError()

        self.left_follower_bus.apply_calibration()
        self.right_follower_bus.apply_calibration()
        self._is_calibrated = True

    def get_observation(self) -> dict:
        """Capture current follower joint positions and camera images.

        Note: We read from FOLLOWERS because we want to record the actual robot state,
        not the commanded state from leaders.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected. Run `robot.connect()` first.")

        # Read left follower arm
        left_state = self.left_follower_bus.read()
        obs_dict = {f"left_{joint}.pos": float(val) for joint, val in left_state.items()}

        # Read right follower arm
        right_state = self.right_follower_bus.read()
        obs_dict.update({f"right_{joint}.pos": float(val) for joint, val in right_state.items()})

        # Read cameras
        for name, cam in self.cameras.items():
            obs_dict[name] = cam.async_read()

        return obs_dict

    def get_leader_action(self) -> dict[str, float]:
        """Read current joint positions from leader arms.

        This is used during teleop recording to get the commanded action.
        """
        if not self.config.use_teleop:
            raise RuntimeError("Leader arms not available in non-teleop mode")

        if self.left_leader_bus is None or self.right_leader_bus is None:
            raise RuntimeError("Leader buses not initialized")

        # Read left leader arm
        left_state = self.left_leader_bus.read()
        action = {f"left_{joint}.pos": float(val) for joint, val in left_state.items()}

        # Read right leader arm
        right_state = self.right_leader_bus.read()
        action.update({f"right_{joint}.pos": float(val) for joint, val in right_state.items()})

        return action

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Send action to follower arms.

        If in teleop mode and action is empty/None, read from leaders and execute.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected.")

        motor_order = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]

        # Check for legacy dataset format (single "action" key with array)
        if "action" in action and "left_joint_1.pos" not in action:
            raw_action = action["action"]
            if len(raw_action) == 14:
                left_target_joints = raw_action[:7]
                right_target_joints = raw_action[7:]
            else:
                print(f"WARNING: Unexpected action array length {len(raw_action)}, expected 14 for dual arm.")
                return action
        else:
            # Standard named format
            left_target_joints = [action[f"left_{motor}.pos"] for motor in motor_order]
            right_target_joints = [action[f"right_{motor}.pos"] for motor in motor_order]

        # Write to followers
        self.left_follower_bus.write(left_target_joints)
        self.right_follower_bus.write(right_target_joints)

        return action

    def teleop_step(self) -> dict[str, float]:
        """Perform one step of software teleoperation.

        Reads from leaders, writes to followers, returns the action taken.
        This should be called in the control loop when doing teleop recording.
        """
        if not self.config.use_teleop:
            raise RuntimeError("teleop_step() not available in non-teleop mode")

        # Read leader positions
        action = self.get_leader_action()

        # Send to followers
        self.send_action(action)

        return action

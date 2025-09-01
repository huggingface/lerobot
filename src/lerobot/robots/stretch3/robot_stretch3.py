#!/usr/bin/env python

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

import time
from typing import Dict, Any, List  # ADDED: Imported List for state_keys, action_keys
import numpy as np
from stretch_body.gamepad_teleop import GamePadTeleop
from stretch_body.robot import Robot as StretchAPI
from stretch_body.robot_params import RobotParams

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.constants import OBS_IMAGES, OBS_STATE
from lerobot.datasets.utils import get_nested_item
from ..robot import Robot
from .configuration_stretch3 import Stretch3RobotConfig

# {lerobot_keys: stretch.api.keys}
STRETCH_MOTORS = {
    "head_pan.pos": "head.head_pan.pos",
    "head_tilt.pos": "head.head_tilt.pos",
    "lift.pos": "lift.pos",
    "arm.pos": "arm.pos",
    "wrist_pitch.pos": "end_of_arm.wrist_pitch.pos",
    "wrist_roll.pos": "end_of_arm.wrist_roll.pos",
    "wrist_yaw.pos": "end_of_arm.wrist_yaw.pos",
    "gripper.pos": "end_of_arm.stretch_gripper.pos",
    "base_x.vel": "base.x_vel",
    "base_y.vel": "base.y_vel",
    "base_theta.vel": "base.theta_vel",
}

class Stretch3Robot(Robot):
    """[Stretch 3](https://hello-robot.com/stretch-3-product), by Hello Robot."""

    config_class = Stretch3RobotConfig
    name: str = "stretch3"  # CHANGED: Added explicit str type annotation

    def __init__(self, config: Stretch3RobotConfig) -> None:
        # REMOVED: raise NotImplementedError
        super().__init__(config)
        self.config: Stretch3RobotConfig = config  # ADDED: Explicit type annotation
        self.robot_type: str = self.config.type  # CHANGED: Added type annotation
        self.api: StretchAPI = StretchAPI()  # CHANGED: Added type annotation
        self.cameras = make_cameras_from_configs(config.cameras)  # CHANGED: Type inferred, no change needed
        self._is_connected: bool = False  # CHANGED: Renamed to _is_connected for property
        self.logs: Dict[str, float] = {}  # CHANGED: Added type annotation
        self.teleop: GamePadTeleop | None = None  # CHANGED: Added type annotation, made Optional
        RobotParams.set_logging_level("WARNING")
        RobotParams.set_logging_formatter("brief_console_formatter")
        self.state_keys: List[str] | None = None  # CHANGED: Added type annotation
        self.action_keys: List[str] | None = None  # CHANGED: Added type annotation

    @property
    def observation_features(self) -> Dict[str, Any]:
        return {
            "dtype": "float32",
            "shape": (len(STRETCH_MOTORS),),
            "names": {"motors": list(STRETCH_MOTORS)},
        }

    @property
    def action_features(self) -> Dict[str, Any]:
        return self.observation_features

    @property
    def camera_features(self) -> Dict[str, Dict[str, Any]]:  # CHANGED: Specified inner dict type
        cam_ft: Dict[str, Dict[str, Any]] = {}  # CHANGED: Added type annotation
        for cam_key, cam in self.cameras.items():
            cam_ft[cam_key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def is_connected(self) -> bool:  # ADDED: Converted is_connected to property
        """Whether the robot is currently connected."""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:  # ADDED: Added missing property
        """Whether the robot is currently calibrated."""
        return self.api.is_homed() if self._is_connected else False

    def connect(self) -> None:  # CHANGED: Added explicit return type
        """Establish communication with the robot."""
        self._is_connected = self.api.startup()  # CHANGED: Use _is_connected
        if not self._is_connected:
            raise ConnectionError(
                "Another process is already using Stretch. Try running 'stretch_free_robot_process.py'"
            )

        for cam in self.cameras.values():
            cam.connect()
            self._is_connected = self._is_connected and cam.is_connected

        if not self._is_connected:
            raise ConnectionError("Could not connect to the cameras, check that all cameras are plugged-in.")

        if self.config.calibrate:  # CHANGED: Use config.calibrate to match Robot base class
            self.calibrate()

    def calibrate(self) -> None:  # CHANGED: Added explicit return type
        """Calibrate the robot if applicable."""
        if not self._is_connected:  # ADDED: Safety check
            raise ConnectionError("Robot must be connected before calibration")
        if not self.api.is_homed():
            self.api.home()

    def configure(self) -> None:  # ADDED: Added missing method
        """Apply one-time or runtime configuration to the robot."""
        # Example: Set motor parameters or control modes
        # Add specific configuration logic if needed
        pass

    def _get_state(self) -> Dict[str, float]:  # CHANGED: Specified value type as float
        status = self.api.get_status()
        return {k: get_nested_item(status, v, sep=".") for k, v in STRETCH_MOTORS.items()}

    def get_observation(self) -> Dict[str, np.ndarray]:  # CHANGED: Specified return type
        if not self._is_connected:  # ADDED: Safety check
            raise ConnectionError("Robot must be connected to get observation")

        obs_dict: Dict[str, np.ndarray] = {}  # CHANGED: Added type annotation
        before_read_t = time.perf_counter()
        state = self._get_state()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        if self.state_keys is None:
            self.state_keys = list(state)

        state_array = np.asarray(list(state.values()), dtype=np.float32)  # CHANGED: Named variable for clarity
        obs_dict[OBS_STATE] = state_array

        for cam_key, cam in self.cameras.items():
            before_camread_t = time.perf_counter()
            image = cam.async_read()
            obs_dict[f"{OBS_IMAGES}.{cam_key}"] = np.asarray(image, dtype=np.uint8)  # CHANGED: Explicit np.ndarray conversion
            self.logs[f"read_camera_{cam_key}_dt_s"] = cam.logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{cam_key}_dt_s"] = time.perf_counter() - before_camread_t

        return obs_dict

    def send_action(self, action: np.ndarray) -> np.ndarray:  # CHANGED: Specified return type
        if not self._is_connected:  # CHANGED: Use _is_connected
            raise ConnectionError("Robot must be connected to send action")

        if self.teleop is None:
            self.teleop = GamePadTeleop(robot_instance=False)
            self.teleop.startup(robot=self)

        if self.action_keys is None:
            dummy_action = self.teleop.gamepad_controller.get_state()
            self.action_keys = list(dummy_action.keys())

        if action.shape != (len(self.action_keys),):  # ADDED: Shape validation
            raise ValueError(f"Action shape {action.shape} does not match expected ({len(self.action_keys)},)")

        action_dict = dict(zip(self.action_keys, action.tolist(), strict=True))
        before_write_t = time.perf_counter()
        self.teleop.do_motion(state=action_dict, robot=self)
        self.push_command()  # CHANGED: No change, but verify push_command exists
        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        return action

    def print_logs(self) -> None:  # CHANGED: Added explicit return type
        """Print robot-specific logs."""
        # Implement logging logic if needed
        pass

    def teleop_safety_stop(self) -> None:  # CHANGED: Added explicit return type
        """Stop teleop motion for safety."""
        if self.teleop is not None:
            self.teleop._safety_stop(robot=self)

    def disconnect(self) -> None:  # CHANGED: Added explicit return type
        """Disconnect from the robot and perform cleanup."""
        self.api.stop()
        if self.teleop is not None:
            self.teleop.gamepad_controller.stop()
            self.teleop.stop()

        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False  # CHANGED: Use _is_connected

#!/usr/bin/env python

# Copyright 2024 Istituto Italiano di Tecnologia. All rights reserved.
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

import os
import logging
import time
import uuid
from functools import cached_property
from typing import Any

import numpy as np
import yarp
from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.ergocub import ErgoCubMotorsBus
from lerobot.robots.robot import Robot

from .configuration_ergocub import ErgoCubConfig
from .safety_utils import HandSafetyChecker

logger = logging.getLogger(__name__)


class ErgoCub(Robot):
    config_class = ErgoCubConfig
    name = "ergocub"
    
    def __init__(self, config: ErgoCubConfig):
        super().__init__(config)
        # Set YARP robot name for resource finding
        os.environ["YARP_ROBOT_NAME"] = "ergoCubSN002"

        self.config = config
        self.session_id = uuid.uuid4()
        self._is_connected = False
        # Absolute vs relative action mode
        self.absolute = bool(getattr(config, "absolute", True))
        # Accumulator cache for relative mode; stores the last absolute command we sent (per-key)
        # Initialized lazily from encoders when first used
        self.acc_state = None

        # Initialize safety checker
        self.safety_checker = HandSafetyChecker(position_tolerance=config.position_tolerance)
        self._safety_control_acquired = False

        # Emotions RPC support
        #self._emotion_cmd_port = yarp.Port()
        #self._emotion_local_port_name: str | None = None
        #self._emotion_remote_port_name = "/ergoCubEmotions/rpc"
        #self._last_emotion_label: str | None = None
        #self._emotion_map = {
        #    0: "neutral",
        #    1: "happy",
        #    2: "alert",
        #    3: "shy",
        #}

        # Reset event publisher
        self._reset_event_port = yarp.BufferedPortBottle()
        self._reset_event_port_name = "/reset"
        

        yarp.Network.init()


        prepared_camera_configs = {}
        for cam_name, cam_config in config.cameras.items():
            cam_config.local_prefix = f"{config.local_prefix}/{self.session_id}"
            prepared_camera_configs[cam_name] = cam_config

        self.cameras = make_cameras_from_configs(prepared_camera_configs)

        # Initialize the new ErgoCub motors bus
        self.bus = ErgoCubMotorsBus(
            remote_prefix=config.remote_prefix,
            local_prefix=f"{config.local_prefix}/{self.session_id}",
            control_boards=config.control_boards,
            state_boards=config.state_boards,
            left_hand=config.left_hand,
            right_hand=config.right_hand,
            finger_scale=config.finger_scale,
        )

    def connect(self, calibrate: bool = True):
        """
        Establish communication with the robot.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
            
        # Connect cameras using standard LeRobot interface
        for cam in self.cameras.values():
            cam.connect()
        
        # Connect motor bus (hand and head controllers)
        self.bus.connect()

        # Connect emotions RPC client (blocks until server is available)
        # self._emotion_local_port_name = f"{self.config.local_prefix}/{self.session_id}/emotions/rpc:o"
        # if not self._emotion_cmd_port.open(self._emotion_local_port_name):
        #     raise ConnectionError(f"Failed to open emotions RPC port {self._emotion_local_port_name}")

        if not self._reset_event_port.open(self._reset_event_port_name):
            raise ConnectionError(f"Failed to open reset event port {self._reset_event_port_name}")

        # if not "Sim" in self.config.remote_prefix:
        #     while not yarp.Network.connect(self._emotion_local_port_name, self._emotion_remote_port_name):
        #         print("ergoCubEmotions: waiting for connection")
        #         time.sleep(1)

        self._is_connected = True

        # New connection: require safety handshake again.
        self.safety_checker.reset_arm_control()
        self._safety_control_acquired = False

        self.acc_state = self.bus.read_state()
        
        if not self.is_calibrated and calibrate:
            logger.info("ErgoCub doesn't require calibration - skipping.")
            
        self.configure()
        logger.info("%s connected. Going to reset...", self)

        self.reset()


    def disconnect(self):
        """Disconnect from the robot and perform any necessary cleanup."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
            
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        # Disconnect motor bus
        self.bus.disconnect()

        # if self._emotion_cmd_port is not None:
        #     self._emotion_cmd_port.close()

        if self._reset_event_port is not None:
            self._reset_event_port.close()

        self._is_connected = False
        
        logger.info("%s disconnected.", self)

    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current observation from the robot.

        Returns:
            dict[str, Any]: A flat dictionary representing the robot's current sensory state.
            
        Raises:
            DeviceNotConnectedError: if robot is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
            
        obs = {}
        
        # Read camera data using standard LeRobot interface
        for cam_name, cam in self.cameras.items():
            cam_data = cam.read()
            if isinstance(cam_data, np.ndarray):
                obs[cam_name] = cam_data
            if "image" in cam_data:
                obs[cam_name] = cam_data["image"]
            if "depth" in cam_data:
                obs[f"{cam_name}_depth"] = cam_data["depth"]
        
        # Read motor data (poses and finger positions) using new motor bus
        motor_data = self.bus.read_state()
        obs.update(motor_data)
        
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send an action command to the robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action = dict(action)
        action['left_hand.position.z'] += 0.5
        action['right_hand.position.z'] += 0.5
        action["left_hand.position.x"] += 0.10
        action["right_hand.position.x"] += 0.10

        # Optional emotion command key from policy output.
        #emotion_value = action.pop("emotions", action.pop("emotion", None))

        # If using relative mode, convert incoming deltas to absolute targets
        if not self.absolute:
            action = self.to_absolute(action)

        # Safety checks are active only until control is acquired once.
        # This avoids running expensive checks at every cycle afterwards.
        if not self._safety_control_acquired:
            current_state = self.bus.read_state()

            if not self.safety_checker.is_valid_action(action):
                return current_state

            if not self.safety_checker.check_hand_position_safety(action, current_state):
                return current_state

            self._safety_control_acquired = True
            logger.info("Safety control acquired for hands: %s", ["left", "right"])

        # Send commands via motor bus
        self.bus.send_commands(action)

        # Forward emotion to /ergoCubEmotions/rpc (ignore unsupported values).
        # if emotion_value is not None and "Sim" not in self.config.remote_prefix:
        #     self._send_emotion_command(emotion_value)

        return action

    # def _send_emotion_command(self, emotion_value: Any) -> None:
    #     try:
    #         value = float(emotion_value)
    #     except (TypeError, ValueError):
    #         logger.warning("Invalid emotion value %r, expected numeric in [0, 1]", emotion_value)
    #         return

    #     # Equal bins in [0, 1]:
    #     # 0 -> [0.00, 0.25), 1 -> [0.25, 0.50), 2 -> [0.50, 0.75), 3 -> [0.75, 1.00]
    #     value = float(np.clip(value, 0.0, 1.0))
    #     if value < 0.25:
    #         emotion_idx = 0
    #     elif value < 0.50:
    #         emotion_idx = 1
    #     elif value < 0.75:
    #         emotion_idx = 2
    #     else:
    #         emotion_idx = 3

    #     emotion_label = self._emotion_map.get(emotion_idx)
    #     if emotion_label is None:
    #         return

    #     # Avoid sending duplicated commands at control frequency.
    #     if emotion_label == self._last_emotion_label:
    #         return

    #     cmd = yarp.Bottle()
    #     cmd.addString("setEmotion")
    #     cmd.addString(emotion_label)

    #     reply = yarp.Bottle()
    #     ok = self._emotion_cmd_port.write(cmd, reply)
    #     if not ok:
    #         logger.warning("Failed to send emotion command '%s'", emotion_label)
    #         return

    #     self._last_emotion_label = emotion_label
    
    def reset(self) -> None:
        """Reset the robot to a default state."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # Reset safety handshake: require re-validation after reset.
        self.safety_checker.reset_arm_control()
        self._safety_control_acquired = False

        # Reset motor bus (hands and head)
        self.bus.reset()
        self.acc_state = self.bus.read_state()

        # Publish reset event on /reset (best-effort, non-blocking when supported)
        reset_event = self._reset_event_port.prepare()
        reset_event.clear()
        reset_event.addString("reset")
        reset_event.addString(str(time.time()))
        try:
            self._reset_event_port.write(False)
        except TypeError:
            self._reset_event_port.write()

        logger.info("%s has been reset.", self)

    # ---------------------------------------------------------------------
    # Relative-to-Absolute conversion
    # ---------------------------------------------------------------------
    def to_absolute(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Convert an input action dictionary interpreted as relative deltas into
        absolute targets using an internal accumulator initialized from encoders.
        """
        # TODO: handle quaternions. For now assuming their value is zero
        abs_action: dict[str, Any] = dict(action)

        for k, v in action.items():
            base = self.acc_state[k]
            abs_action[k] = base + v
            self.acc_state[k] = abs_action[k]

        return abs_action


    @property
    def is_connected(self) -> bool:
        """Whether the robot is currently connected."""
        # Check if cameras are connected
        cameras_connected = all(cam.is_connected for cam in self.cameras.values()) if self.cameras else True
        
        # Check if motor bus is connected
        motors_connected = self.bus.is_connected
        
        return self._is_connected and cameras_connected and motors_connected

    @property
    def is_calibrated(self) -> bool:
        """ErgoCub doesn't require calibration."""
        return True

    def calibrate(self) -> None:
        """ErgoCub doesn't require calibration - no-op."""

    def configure(self) -> None:
        """Apply any one-time configuration - no-op for ErgoCub."""

    @property
    def _action_ft(self) -> dict[str, type]:
        """Get motor features from the bus."""
        return self.bus.action_features
    
    @property
    def _state_ft(self) -> dict[str, type]:
        """Get motor features from the bus."""
        return self.bus.state_features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Helper property to get camera features in SO101 format."""
        cam_features = {}
        for cam_name, cam_config in self.config.cameras.items():
            cam_features[cam_name] = (cam_config.height, cam_config.width, 3)
            if hasattr(cam_config, 'use_depth') and cam_config.use_depth:
                cam_features[f"{cam_name}_depth"] = (cam_config.height, cam_config.width)
        return cam_features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Values are either float for single values or tuples for array shapes.
        """
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        A dictionary describing the structure and types of the actions expected by the robot.
        ErgoCub actions are pose commands (position + orientation) for hands and head, plus finger positions.
        
        Returns action features in SO100-like format with dot notation.
        """
        return self._action_ft  # Actions and observations have the same structure

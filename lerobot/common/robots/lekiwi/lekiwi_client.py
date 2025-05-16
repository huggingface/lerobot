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

import base64
import json
import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import zmq

from lerobot.common.constants import OBS_IMAGES, OBS_STATE
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_lekiwi import LeKiwiClientConfig


class LeKiwiClient(Robot):
    config_class = LeKiwiClientConfig
    name = "lekiwi_client"

    def __init__(self, config: LeKiwiClientConfig):
        super().__init__(config)
        self.config = config
        self.id = config.id
        self.robot_type = config.type

        self.remote_ip = config.remote_ip
        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations

        self.teleop_keys = config.teleop_keys

        self.polling_timeout_ms = config.polling_timeout_ms
        self.connect_timeout_s = config.connect_timeout_s

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None

        self.last_frames = {}

        self.last_remote_arm_state = {}
        self.last_remote_base_state = {"base_left_wheel": 0, "base_back_wheel": 0, "base_right_wheel": 0}

        # Define three speed levels and a current index
        self.speed_levels = [
            {"xy": 0.1, "theta": 30},  # slow
            {"xy": 0.2, "theta": 60},  # medium
            {"xy": 0.3, "theta": 90},  # fast
        ]
        self.speed_index = 0  # Start at slow

        self._is_connected = False
        self.logs = {}

    @property
    def state_feature(self) -> dict:
        state_ft = {
            "arm_shoulder_pan": {"shape": (1,), "info": None, "dtype": "float32"},
            "arm_shoulder_lift": {"shape": (1,), "info": None, "dtype": "float32"},
            "arm_elbow_flex": {"shape": (1,), "info": None, "dtype": "float32"},
            "arm_wrist_flex": {"shape": (1,), "info": None, "dtype": "float32"},
            "arm_wrist_roll": {"shape": (1,), "info": None, "dtype": "float32"},
            "arm_gripper": {"shape": (1,), "info": None, "dtype": "float32"},
            "x_cmd": {"shape": (1,), "info": None, "dtype": "float32"},
            "y_cmd": {"shape": (1,), "info": None, "dtype": "float32"},
            "theta_cmd": {"shape": (1,), "info": None, "dtype": "float32"},
        }
        return state_ft

    @property
    def action_feature(self) -> dict:
        action_ft = {
            "arm_shoulder_pan": {"shape": (1,), "info": None, "dtype": "float32"},
            "arm_shoulder_lift": {"shape": (1,), "info": None, "dtype": "float32"},
            "arm_elbow_flex": {"shape": (1,), "info": None, "dtype": "float32"},
            "arm_wrist_flex": {"shape": (1,), "info": None, "dtype": "float32"},
            "arm_wrist_roll": {"shape": (1,), "info": None, "dtype": "float32"},
            "arm_gripper": {"shape": (1,), "info": None, "dtype": "float32"},
            "base_left_wheel": {"shape": (1,), "info": None, "dtype": "float32"},
            "base_right_wheel": {"shape": (1,), "info": None, "dtype": "float32"},
            "base_back_wheel": {"shape": (1,), "info": None, "dtype": "float32"},
        }
        return action_ft

    @property
    def camera_features(self) -> dict[str, dict]:
        cam_ft = {
            f"{OBS_IMAGES}.front": {
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
                "info": None,
                "dtype": "image",
            },
            f"{OBS_IMAGES}.wrist": {
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"],
                "dtype": "image",
                "info": None,
            },
        }
        return cam_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        pass

    def connect(self) -> None:
        """Establishes ZMQ sockets with the remote mobile robot"""

        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                "LeKiwi Daemon is already connected. Do not run `robot.connect()` twice."
            )

        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        zmq_cmd_locator = f"tcp://{self.remote_ip}:{self.port_zmq_cmd}"
        self.zmq_cmd_socket.connect(zmq_cmd_locator)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        zmq_observations_locator = f"tcp://{self.remote_ip}:{self.port_zmq_observations}"
        self.zmq_observation_socket.connect(zmq_observations_locator)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)

        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.connect_timeout_s * 1000))
        if self.zmq_observation_socket not in socks or socks[self.zmq_observation_socket] != zmq.POLLIN:
            raise DeviceNotConnectedError("Timeout waiting for LeKiwi Host to connect expired.")

        self._is_connected = True

    def calibrate(self) -> None:
        pass

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap the value to fit within signed 16-bit range (-32768 to 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> maximum positive value
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> minimum negative value
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    def _body_to_wheel_raw(
        self,
        x_cmd: float,
        y_cmd: float,
        theta_cmd: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert desired body-frame velocities into wheel raw commands.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          y_cmd      : Linear velocity in y (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the center of rotation to each wheel (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"base_left_wheel": value, "base_back_wheel": value, "base_right_wheel": value}.

        Notes:
          - Internally, the method converts theta_cmd to rad/s for the kinematics.
          - The raw command is computed from the wheels angular speed in deg/s
            using _degps_to_raw(). If any command exceeds max_raw, all commands
            are scaled down proportionally.
        """
        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta_cmd * (np.pi / 180.0)
        # Create the body velocity vector [x, y, theta_rad].
        velocity_vector = np.array([x_cmd, y_cmd, theta_rad])

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 120, 0]) - 90)
        # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
        # The third column (base_radius) accounts for the effect of rotation.
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # Scaling
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert each wheel’s angular speed (deg/s) to a raw integer.
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]

        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def _wheel_raw_to_body(
        self, wheel_raw: dict[str, Any], wheel_radius: float = 0.05, base_radius: float = 0.125
    ) -> dict[str, Any]:
        """
        Convert wheel raw command feedback back into body-frame velocities.

        Parameters:
          wheel_raw   : Vector with raw wheel commands ("base_left_wheel", "base_back_wheel", "base_right_wheel").
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the robot center to each wheel (meters).

        Returns:
          A dict (x_cmd, y_cmd, theta_cmd) where:
             OBS_STATE.x_cmd      : Linear velocity in x (m/s).
             OBS_STATE.y_cmd      : Linear velocity in y (m/s).
             OBS_STATE.theta_cmd  : Rotational velocity in deg/s.
        """

        # Convert each raw command back to an angular speed in deg/s.
        wheel_degps = np.array([LeKiwiClient._raw_to_degps(int(v)) for _, v in wheel_raw.items()])
        # Convert from deg/s to rad/s.
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # Compute each wheel’s linear speed (m/s) from its angular speed.
        wheel_linear_speeds = wheel_radps * wheel_radius

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 120, 0]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Solve the inverse kinematics: body_velocity = M⁻¹ · wheel_linear_speeds.
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x_cmd, y_cmd, theta_rad = velocity_vector
        theta_cmd = theta_rad * (180.0 / np.pi)
        return {
            f"{OBS_STATE}.x_cmd": x_cmd * 1000,
            f"{OBS_STATE}.y_cmd": y_cmd * 1000,
            f"{OBS_STATE}.theta_cmd": theta_cmd,
        }  # Convert to mm/s

    def _poll_and_get_latest_message(self) -> Optional[str]:
        """Polls the ZMQ socket for a limited time and returns the latest message string."""
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)

        try:
            socks = dict(poller.poll(self.polling_timeout_ms))
        except zmq.ZMQError as e:
            logging.error(f"ZMQ polling error: {e}")
            return None

        if self.zmq_observation_socket not in socks:
            logging.info("No new data available within timeout.")
            return None

        last_msg = None
        while True:
            try:
                msg = self.zmq_observation_socket.recv_string(zmq.NOBLOCK)
                last_msg = msg
            except zmq.Again:
                break

        if last_msg is None:
            logging.warning("Poller indicated data, but failed to retrieve message.")

        return last_msg

    def _parse_observation_json(self, obs_string: str) -> Optional[Dict[str, Any]]:
        """Parses the JSON observation string."""
        try:
            return json.loads(obs_string)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON observation: {e}")
            return None

    def _decode_image_from_b64(self, image_b64: str) -> Optional[np.ndarray]:
        """Decodes a base64 encoded image string to an OpenCV image."""
        if not image_b64:
            return None
        try:
            jpg_data = base64.b64decode(image_b64)
            np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                logging.warning("cv2.imdecode returned None for an image.")
            return frame
        except (TypeError, ValueError) as e:
            logging.error(f"Error decoding base64 image data: {e}")
            return None

    def _remote_state_from_obs(
        self, observation: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
        """Extracts frames, speed, and arm state from the parsed observation."""

        # Separate image and state data
        image_observation = {k: v for k, v in observation.items() if k.startswith(OBS_IMAGES)}
        state_observation = {k: v for k, v in observation.items() if k.startswith(OBS_STATE)}

        # Decode images
        current_frames: Dict[str, np.ndarray] = {}
        for cam_name, image_b64 in image_observation.items():
            frame = self._decode_image_from_b64(image_b64)
            if frame is not None:
                current_frames[cam_name] = frame

        # Extract state components
        current_arm_state = {k: v for k, v in state_observation.items() if k.startswith(f"{OBS_STATE}.arm")}
        current_base_state = {k: v for k, v in state_observation.items() if k.startswith(f"{OBS_STATE}.base")}

        return current_frames, current_arm_state, current_base_state

    def _get_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
        """
        Polls the video socket for the latest observation data.

        Attempts to retrieve and decode the latest message within a short timeout.
        If successful, updates and returns the new frames, speed, and arm state.
        If no new data arrives or decoding fails, returns the last known values.
        """

        # 1. Get the latest message string from the socket
        latest_message_str = self._poll_and_get_latest_message()

        # 2. If no message, return cached data
        if latest_message_str is None:
            return self.last_frames, self.last_remote_arm_state, self.last_remote_base_state

        # 3. Parse the JSON message
        observation = self._parse_observation_json(latest_message_str)

        # 4. If JSON parsing failed, return cached data
        if observation is None:
            return self.last_frames, self.last_remote_arm_state, self.last_remote_base_state

        # 5. Process the valid observation data
        try:
            new_frames, new_arm_state, new_base_state = self._remote_state_from_obs(observation)
        except Exception as e:
            logging.error(f"Error processing observation data, serving last observation: {e}")
            return self.last_frames, self.last_remote_arm_state, self.last_remote_base_state

        self.last_frames = new_frames
        self.last_remote_arm_state = new_arm_state
        self.last_remote_base_state = new_base_state

        return new_frames, new_arm_state, new_base_state

    def get_observation(self) -> dict[str, Any]:
        """
        Capture observations from the remote robot: current follower arm positions,
        present wheel speeds (converted to body-frame velocities: x, y, theta),
        and a camera frame. Receives over ZMQ, translate to body-frame vel
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("LeKiwiClient is not connected. You need to run `robot.connect()`.")

        frames, remote_arm_state, remote_base_state = self._get_data()
        remote_body_state = self._wheel_raw_to_body(remote_base_state)

        obs_dict = {**remote_arm_state, **remote_body_state}

        # TODO(Steven): Remove this when it is possible to record a non-numpy array value
        obs_dict = {k: np.array([v], dtype=np.float32) for k, v in obs_dict.items()}

        # Loop over each configured camera
        for cam_name, frame in frames.items():
            if frame is None:
                logging.warning("Frame is None")
                frame = np.zeros((640, 480, 3), dtype=np.uint8)
            obs_dict[cam_name] = torch.from_numpy(frame)

        return obs_dict

    def _from_keyboard_to_wheel_action(self, pressed_keys: np.ndarray):
        # Speed control
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # e.g. 0.1, 0.25, or 0.4
        theta_speed = speed_setting["theta"]  # e.g. 30, 60, or 90

        x_cmd = 0.0  # m/s forward/backward
        y_cmd = 0.0  # m/s lateral
        theta_cmd = 0.0  # deg/s rotation

        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed
        return self._body_to_wheel_raw(x_cmd, y_cmd, theta_cmd)

    def configure(self):
        pass

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command lekiwi to move to a target joint configuration. Translates to motor space + sends over ZMQ

        Args:
            action (np.ndarray): array containing the goal positions for the motors.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            np.ndarray: the action sent to the motors, potentially clipped.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        goal_pos = {}

        common_keys = [
            key
            for key in action
            if key in (motor.replace("arm_", "") for motor, _ in self.action_feature.items())
        ]

        arm_actions = {"arm_" + arm_motor: action[arm_motor] for arm_motor in common_keys}
        goal_pos = arm_actions

        keyboard_keys = np.array(list(set(action.keys()) - set(common_keys)))
        wheel_actions = self._from_keyboard_to_wheel_action(keyboard_keys)
        goal_pos = {**arm_actions, **wheel_actions}

        self.zmq_cmd_socket.send_string(json.dumps(goal_pos))  # action is in motor space

        # TODO(Steven): Remove the np conversion when it is possible to record a non-numpy array value
        goal_pos = {"action." + k: np.array([v], dtype=np.float32) for k, v in goal_pos.items()}
        return goal_pos

    def disconnect(self):
        """Cleans ZMQ comms"""

        if not self._is_connected:
            raise DeviceNotConnectedError(
                "LeKiwi is not connected. You need to run `robot.connect()` before disconnecting."
            )
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
        self._is_connected = False

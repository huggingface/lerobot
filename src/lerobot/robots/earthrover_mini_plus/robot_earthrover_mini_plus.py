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
"""EarthRover Mini Plus robot using Frodobots SDK."""

import base64
import logging
from functools import cached_property
from typing import Any

import cv2
import numpy as np
import requests

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_earthrover_mini_plus import EarthRoverMiniPlusConfig

logger = logging.getLogger(__name__)


class EarthRoverMiniPlus(Robot):
    """
    EarthRover Mini Plus robot controlled via Frodobots SDK HTTP API.

    This robot uses cloud-based control through the Frodobots SDK instead of direct
    hardware connection. Cameras stream via WebRTC through Agora cloud, and control
    commands are sent via HTTP POST requests.

    The robot supports:
    - Dual cameras (front and rear) accessed via SDK HTTP endpoints
    - Linear and angular velocity control
    - Battery and orientation telemetry

    Attributes:
        config: Robot configuration
        sdk_base_url: URL of the Frodobots SDK server (default: http://localhost:8000)
    """

    config_class = EarthRoverMiniPlusConfig
    name = "earthrover_mini_plus"

    def __init__(self, config: EarthRoverMiniPlusConfig):
        """Initialize EarthRover Mini Plus robot.

        Args:
            config: Robot configuration including SDK URL
        """
        super().__init__(config)
        self.config = config
        self.sdk_base_url = "http://localhost:8000"

        self._is_connected = False

        logger.info(f"Initialized {self.name} with SDK at {self.sdk_base_url}")

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected to SDK."""
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to robot via Frodobots SDK.

        Args:
            calibrate: Not used for SDK-based robot (kept for API compatibility)

        Raises:
            DeviceAlreadyConnectedError: If robot is already connected
            DeviceNotConnectedError: If cannot connect to SDK server
        """
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected")

        # Verify SDK is running and accessible
        try:
            response = requests.get(f"{self.sdk_base_url}/data", timeout=10.0)
            if response.status_code != 200:
                raise DeviceNotConnectedError(
                    f"Cannot connect to SDK at {self.sdk_base_url}. "
                    "Make sure it's running: hypercorn main:app --reload"
                )
        except requests.RequestException as e:
            raise DeviceNotConnectedError(f"Cannot connect to SDK at {self.sdk_base_url}: {e}") from e

        self._is_connected = True
        logger.info(f"{self.name} connected to SDK")

        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        """Calibration not needed for SDK-based robot."""
        logger.info("Calibration not required for SDK-based robot")

    @property
    def is_calibrated(self) -> bool:
        """SDK robot doesn't require calibration.

        Returns:
            bool: Always True for SDK-based robots
        """
        return True

    def configure(self) -> None:
        """Configure robot (no-op for SDK-based robot)."""
        pass

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define the observation space for dataset recording.

        Returns:
            dict: Observation features with types/shapes:
                - front: (480, 640, 3) - Front camera RGB image
                - rear: (480, 640, 3) - Rear camera RGB image
                - linear.vel: float - Current speed (0-1, SDK reports only positive speeds)
                - angular.vel: float - Angular velocity (always 0, SDK doesn't report this)
                - battery.level: float - Battery level (0-1, normalized from 0-100)
                - orientation.deg: float - Robot orientation (0-1, normalized from raw value)
                - gps.latitude: float - GPS latitude coordinate
                - gps.longitude: float - GPS longitude coordinate
                - gps.signal: float - GPS signal strength (0-1, normalized from percentage)
                - signal.level: float - Network signal level (0-1, normalized from 0-5)
                - vibration: float - Vibration sensor reading
                - lamp.state: float - Lamp state (0=off, 1=on)
        """
        return {
            # Cameras (height, width, channels)
            "front": (480, 640, 3),
            "rear": (480, 640, 3),
            # Motion state
            "linear.vel": float,
            "angular.vel": float,
            # Robot state
            "battery.level": float,
            "orientation.deg": float,
            # GPS
            "gps.latitude": float,
            "gps.longitude": float,
            "gps.signal": float,
            # Sensors
            "signal.level": float,
            "vibration": float,
            "lamp.state": float,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define the action space.

        Returns:
            dict: Action features with types:
                - linear.vel: float - Target linear velocity
                - angular.vel: float - Target angular velocity
        """
        return {
            "linear.vel": float,
            "angular.vel": float,
        }

    def get_observation(self) -> dict[str, Any]:
        """Get current robot observation from SDK.

        Returns:
            dict: Observation containing:
                - front: Front camera image (480, 640, 3) in RGB format
                - rear: Rear camera image (480, 640, 3) in RGB format
                - linear.vel: Current speed (0-1, SDK reports only positive speeds)
                - angular.vel: Angular velocity (0, SDK doesn't report this separately)
                - battery.level: Battery level (0-1, normalized from 0-100)
                - orientation.deg: Robot orientation (0-1, normalized from raw value)
                - gps.latitude: GPS latitude coordinate
                - gps.longitude: GPS longitude coordinate
                - gps.signal: GPS signal strength (0-1, normalized from percentage)
                - signal.level: Network signal level (0-1, normalized from 0-5)
                - vibration: Vibration sensor reading
                - lamp.state: Lamp state (0=off, 1=on)

        Raises:
            DeviceNotConnectedError: If robot is not connected

        Note:
            Camera frames are retrieved from SDK endpoints /v2/front and /v2/rear.
            Frames are decoded from base64 and converted from BGR to RGB format.
            Robot telemetry is retrieved from /data endpoint.
            All SDK values are normalized to appropriate ranges for dataset recording.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected")

        observation = {}

        # Get camera images from SDK (converted to RGB)
        try:
            frames = self._get_camera_frames()
            observation["front"] = frames.get("front", np.zeros((480, 640, 3), dtype=np.uint8))
            observation["rear"] = frames.get("rear", np.zeros((480, 640, 3), dtype=np.uint8))
        except Exception as e:
            logger.warning(f"Error getting camera frames: {e}")
            observation["front"] = np.zeros((480, 640, 3), dtype=np.uint8)
            observation["rear"] = np.zeros((480, 640, 3), dtype=np.uint8)

        # Get robot state from SDK
        try:
            robot_data = self._get_robot_data()

            # Motion state
            observation["linear.vel"] = robot_data.get("speed", 0) / 100.0  # Normalize 0-100 to 0-1
            observation["angular.vel"] = 0.0  # SDK doesn't report angular velocity separately

            # Robot state
            observation["battery.level"] = robot_data.get("battery", 0) / 100.0  # Normalize 0-100 to 0-1
            observation["orientation.deg"] = robot_data.get("orientation", 0) / 360.0  # Normalize to 0-1

            # GPS data
            observation["gps.latitude"] = robot_data.get("latitude", 0.0)
            observation["gps.longitude"] = robot_data.get("longitude", 0.0)
            observation["gps.signal"] = robot_data.get("gps_signal", 0) / 100.0  # Normalize percentage to 0-1

            # Sensors
            observation["signal.level"] = robot_data.get("signal_level", 0) / 5.0  # Normalize 0-5 to 0-1
            observation["vibration"] = robot_data.get("vibration", 0.0)
            observation["lamp.state"] = float(robot_data.get("lamp", 0))  # 0 or 1

        except Exception as e:
            logger.warning(f"Error getting robot state: {e}")
            # Set all observations to default values
            observation["linear.vel"] = 0.0
            observation["angular.vel"] = 0.0
            observation["battery.level"] = 0.0
            observation["orientation.deg"] = 0.0
            observation["gps.latitude"] = 0.0
            observation["gps.longitude"] = 0.0
            observation["gps.signal"] = 0.0
            observation["signal.level"] = 0.0
            observation["vibration"] = 0.0
            observation["lamp.state"] = 0.0

        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action to robot via SDK.

        Args:
            action: Action dict with keys:
                - linear.vel: Target linear velocity (-1 to 1)
                - angular.vel: Target angular velocity (-1 to 1)

        Returns:
            dict: The action that was sent (matches action_features keys)

        Raises:
            DeviceNotConnectedError: If robot is not connected

        Note:
            Actions are sent to SDK via POST /control endpoint.
            SDK expects commands in range [-1, 1].
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected")

        # Extract action values
        linear = action.get("linear.vel", 0.0)
        angular = action.get("angular.vel", 0.0)

        # Send command to SDK
        try:
            self._send_command_to_sdk(linear, angular)
        except Exception as e:
            logger.error(f"Error sending action: {e}")

        # Return action in format matching action_features
        return {
            "linear.vel": linear,
            "angular.vel": angular,
        }

    def disconnect(self) -> None:
        """Disconnect from robot.

        Stops the robot and closes connection to SDK.

        Raises:
            DeviceNotConnectedError: If robot is not connected
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected")

        # Stop the robot before disconnecting
        try:
            self._send_command_to_sdk(0.0, 0.0)
        except Exception as e:
            logger.warning(f"Failed to stop robot during disconnect: {e}")

        self._is_connected = False
        logger.info(f"{self.name} disconnected")

    # Private helper methods for SDK communication

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """Get camera frames from SDK using v2 endpoints.

        Returns:
            dict: Dictionary with 'front' and 'rear' keys containing decoded images in RGB format

        Note:
            Uses /v2/front and /v2/rear endpoints which are 15x faster than /screenshot.
            Images are base64 encoded, resized to 640x480, and converted from BGR to RGB.
        """
        frames = {}

        # Get front camera
        try:
            response = requests.get(f"{self.sdk_base_url}/v2/front", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                if "front_frame" in data and data["front_frame"]:
                    front_img = self._decode_base64_image(data["front_frame"])
                    if front_img is not None:
                        # Resize and convert BGR to RGB
                        front_img = cv2.resize(front_img, (640, 480))
                        frames["front"] = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Error fetching front camera: {e}")

        # Get rear camera
        try:
            response = requests.get(f"{self.sdk_base_url}/v2/rear", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                if "rear_frame" in data and data["rear_frame"]:
                    rear_img = self._decode_base64_image(data["rear_frame"])
                    if rear_img is not None:
                        # Resize and convert BGR to RGB
                        rear_img = cv2.resize(rear_img, (640, 480))
                        frames["rear"] = cv2.cvtColor(rear_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Error fetching rear camera: {e}")

        return frames

    def _decode_base64_image(self, base64_string: str) -> np.ndarray | None:
        """Decode base64 string to image.

        Args:
            base64_string: Base64 encoded image string

        Returns:
            np.ndarray: Decoded image in BGR format (OpenCV default), or None if decoding fails
        """
        try:
            img_bytes = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img  # Return in BGR format (OpenCV default)
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None

    def _get_robot_data(self) -> dict:
        """Get robot telemetry data from SDK.

        Returns:
            dict: Robot telemetry data including battery, speed, orientation, GPS, etc.
                 Empty dict if request fails.

        Note:
            Uses /data endpoint which provides comprehensive robot state.
        """
        try:
            response = requests.get(f"{self.sdk_base_url}/data", timeout=2.0)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Error fetching robot data: {e}")

        return {}

    def _send_command_to_sdk(self, linear: float, angular: float, lamp: int = 0) -> bool:
        """Send control command to SDK.

        Args:
            linear: Linear velocity command (-1 to 1)
            angular: Angular velocity command (-1 to 1)
            lamp: Lamp control (0=off, 1=on)

        Returns:
            bool: True if command sent successfully, False otherwise

        Note:
            Uses POST /control endpoint. Commands are sent as JSON payload.
        """
        try:
            payload = {
                "command": {
                    "linear": linear,
                    "angular": angular,
                    "lamp": lamp,
                }
            }

            response = requests.post(
                f"{self.sdk_base_url}/control",
                json=payload,
                timeout=1.0,
            )

            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return False

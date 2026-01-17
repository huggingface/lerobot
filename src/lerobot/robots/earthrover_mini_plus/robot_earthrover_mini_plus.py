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

import cv2
import numpy as np
import requests

from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_earthrover_mini_plus import EarthRoverMiniPlusConfig

logger = logging.getLogger(__name__)

# Action feature keys
ACTION_LINEAR_VEL = "linear.vel"
ACTION_ANGULAR_VEL = "angular.vel"

# Observation feature keys
OBS_FRONT = "front"
OBS_REAR = "rear"
OBS_LINEAR_VEL = "linear.vel"
OBS_BATTERY_LEVEL = "battery.level"
OBS_ORIENTATION_DEG = "orientation.deg"
OBS_GPS_LATITUDE = "gps.latitude"
OBS_GPS_LONGITUDE = "gps.longitude"
OBS_GPS_SIGNAL = "gps.signal"
OBS_SIGNAL_LEVEL = "signal.level"
OBS_VIBRATION = "vibration"
OBS_LAMP_STATE = "lamp.state"


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

        # Empty cameras dict for compatibility with recording script
        # Cameras are accessed directly via SDK, not through Camera objects
        self.cameras = {}
        self._is_connected = False

        # Cache for camera frames (fallback when requests fail)
        self._last_front_frame = None
        self._last_rear_frame = None

        # Cache for robot telemetry data (fallback when requests fail)
        self._last_robot_data = None

        logger.info(f"Initialized {self.name} with SDK at {self.sdk_base_url}")

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected to SDK."""
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect to robot via Frodobots SDK.

        Args:
            calibrate: Not used for SDK-based robot (kept for API compatibility)

        Raises:
            DeviceAlreadyConnectedError: If robot is already connected
            DeviceNotConnectedError: If cannot connect to SDK server
        """

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
            OBS_FRONT: (480, 640, 3),
            OBS_REAR: (480, 640, 3),
            # Motion state
            OBS_LINEAR_VEL: float,
            # Robot state
            OBS_BATTERY_LEVEL: float,
            OBS_ORIENTATION_DEG: float,
            # GPS
            OBS_GPS_LATITUDE: float,
            OBS_GPS_LONGITUDE: float,
            OBS_GPS_SIGNAL: float,
            # Sensors
            OBS_SIGNAL_LEVEL: float,
            OBS_VIBRATION: float,
            OBS_LAMP_STATE: float,
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
            ACTION_LINEAR_VEL: float,
            ACTION_ANGULAR_VEL: float,
        }

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Get current robot observation from SDK.

        Returns:
            RobotObservation: Observation containing:
                - front: Front camera image (480, 640, 3) in RGB format
                - rear: Rear camera image (480, 640, 3) in RGB format
                - linear.vel: Current speed (0-1, SDK reports only positive speeds)
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

        observation = {}

        # Get camera images from SDK
        frames = self._get_camera_frames()
        observation[OBS_FRONT] = frames["front"]
        observation[OBS_REAR] = frames["rear"]

        # Get robot state from SDK
        robot_data = self._get_robot_data()

        # Motion state
        observation[OBS_LINEAR_VEL] = robot_data["speed"] / 100.0  # Normalize 0-100 to 0-1

        # Robot state
        observation[OBS_BATTERY_LEVEL] = robot_data["battery"] / 100.0  # Normalize 0-100 to 0-1
        observation[OBS_ORIENTATION_DEG] = robot_data["orientation"] / 360.0  # Normalize to 0-1

        # GPS data
        observation[OBS_GPS_LATITUDE] = robot_data["latitude"]
        observation[OBS_GPS_LONGITUDE] = robot_data["longitude"]
        observation[OBS_GPS_SIGNAL] = robot_data["gps_signal"] / 100.0  # Normalize percentage to 0-1

        # Sensors
        observation[OBS_SIGNAL_LEVEL] = robot_data["signal_level"] / 5.0  # Normalize 0-5 to 0-1
        observation[OBS_VIBRATION] = robot_data["vibration"]
        observation[OBS_LAMP_STATE] = float(robot_data["lamp"])  # 0 or 1

        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Send action to robot via SDK.

        Args:
            action: Action dict with keys:
                - linear.vel: Target linear velocity (-1 to 1)
                - angular.vel: Target angular velocity (-1 to 1)

        Returns:
            RobotAction: The action that was sent (matches action_features keys)
        Raises:
            DeviceNotConnectedError: If robot is not connected

        Note:
            Actions are sent to SDK via POST /control endpoint.
            SDK expects commands in range [-1, 1].
        """

        # Extract action values and convert to float
        linear = float(action.get(ACTION_LINEAR_VEL, 0.0))
        angular = float(action.get(ACTION_ANGULAR_VEL, 0.0))

        # Send command to SDK
        try:
            self._send_command_to_sdk(linear, angular)
        except Exception as e:
            logger.error(f"Error sending action: {e}")

        # Return action in format matching action_features
        return {
            ACTION_LINEAR_VEL: linear,
            ACTION_ANGULAR_VEL: angular,
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        """Disconnect from robot.

        Stops the robot and closes connection to SDK.

        Raises:
            DeviceNotConnectedError: If robot is not connected
        """

        # Stop the robot before disconnecting
        try:
            self._send_command_to_sdk(0.0, 0.0)
        except Exception as e:
            logger.warning(f"Failed to stop robot during disconnect: {e}")

        self._is_connected = False
        logger.info(f"{self.name} disconnected")

    # Private helper methods for SDK communication

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """Get camera frames from SDK using v2 endpoints with caching fallback.

        Returns:
            dict: Dictionary with 'front' and 'rear' keys containing:
                - Current frame (if request succeeds)
                - Cached frame (if request fails but cache exists)
                - Zero array (if request fails and no cache exists yet)

        Note:
            Uses /v2/front and /v2/rear endpoints which are 15x faster than /screenshot.
            Images are base64 encoded, resized to 640x480, and converted from BGR to RGB.
            If request fails, returns the last successfully retrieved frame (cached).
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
                        front_rgb = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
                        frames["front"] = front_rgb
                        # Cache the successful frame
                        self._last_front_frame = front_rgb
        except Exception as e:
            logger.warning(f"Error fetching front camera: {e}")

        # Fallback: use cache or zero array
        if "front" not in frames:
            if self._last_front_frame is not None:
                frames["front"] = self._last_front_frame
            else:
                frames["front"] = np.zeros((480, 640, 3), dtype=np.uint8)

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
                        rear_rgb = cv2.cvtColor(rear_img, cv2.COLOR_BGR2RGB)
                        frames["rear"] = rear_rgb
                        # Cache the successful frame
                        self._last_rear_frame = rear_rgb
        except Exception as e:
            logger.warning(f"Error fetching rear camera: {e}")

        # Fallback: use cache or zero array
        if "rear" not in frames:
            if self._last_rear_frame is not None:
                frames["rear"] = self._last_rear_frame
            else:
                frames["rear"] = np.zeros((480, 640, 3), dtype=np.uint8)

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
            dict: Robot telemetry data including battery, speed, orientation, GPS, etc:
                - Current data (if request succeeds)
                - Cached data (if request fails but cache exists)
                - Default values (if request fails and no cache exists yet)

        Note:
            Uses /data endpoint which provides comprehensive robot state.
            If request fails, returns the last successfully retrieved data (cached).
        """
        try:
            response = requests.get(f"{self.sdk_base_url}/data", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                # Cache the successful data
                self._last_robot_data = data
                return data
        except Exception as e:
            logger.warning(f"Error fetching robot data: {e}")

        # Fallback: use cache or default values
        if self._last_robot_data is not None:
            return self._last_robot_data
        else:
            # Return dict with default values (used only on first failure before any cache exists)
            return {
                "speed": 0,
                "battery": 0,
                "orientation": 0,
                "latitude": 0.0,
                "longitude": 0.0,
                "gps_signal": 0,
                "signal_level": 0,
                "vibration": 0.0,
                "lamp": 0,
            }

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

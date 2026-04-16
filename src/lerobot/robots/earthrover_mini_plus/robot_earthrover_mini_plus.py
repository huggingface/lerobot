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

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_earthrover_mini_plus import EarthRoverMiniPlusConfig

logger = logging.getLogger(__name__)

# Action feature keys
ACTION_LINEAR_VEL = "linear_velocity"
ACTION_ANGULAR_VEL = "angular_velocity"

# Observation feature keys — cameras
OBS_FRONT = "front"
OBS_REAR = "rear"

# Observation feature keys — telemetry
OBS_SPEED = "speed"
OBS_BATTERY_LEVEL = "battery_level"
OBS_ORIENTATION = "orientation"
OBS_GPS_LATITUDE = "gps_latitude"
OBS_GPS_LONGITUDE = "gps_longitude"
OBS_GPS_SIGNAL = "gps_signal"
OBS_SIGNAL_LEVEL = "signal_level"
OBS_VIBRATION = "vibration"
OBS_LAMP = "lamp"

# Observation feature keys — IMU sensors
OBS_ACCELEROMETER_X = "accelerometer_x"
OBS_ACCELEROMETER_Y = "accelerometer_y"
OBS_ACCELEROMETER_Z = "accelerometer_z"
OBS_GYROSCOPE_X = "gyroscope_x"
OBS_GYROSCOPE_Y = "gyroscope_y"
OBS_GYROSCOPE_Z = "gyroscope_z"
OBS_MAGNETOMETER_X = "magnetometer_filtered_x"
OBS_MAGNETOMETER_Y = "magnetometer_filtered_y"
OBS_MAGNETOMETER_Z = "magnetometer_filtered_z"

# Observation feature keys — wheel RPMs
OBS_WHEEL_RPM_0 = "wheel_rpm_0"
OBS_WHEEL_RPM_1 = "wheel_rpm_1"
OBS_WHEEL_RPM_2 = "wheel_rpm_2"
OBS_WHEEL_RPM_3 = "wheel_rpm_3"


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
                - speed: float - Current speed (raw SDK value)
                - battery_level: float - Battery level (0-100)
                - orientation: float - Robot orientation in degrees
                - gps_latitude: float - GPS latitude coordinate
                - gps_longitude: float - GPS longitude coordinate
                - gps_signal: float - GPS signal strength (percentage)
                - signal_level: float - Network signal level (0-5)
                - vibration: float - Vibration sensor reading
                - lamp: float - Lamp state (0=off, 1=on)
                - accelerometer_x: float - Accelerometer X axis (raw SDK value)
                - accelerometer_y: float - Accelerometer Y axis (raw SDK value)
                - accelerometer_z: float - Accelerometer Z axis (raw SDK value)
                - gyroscope_x: float - Gyroscope X axis (raw SDK value)
                - gyroscope_y: float - Gyroscope Y axis (raw SDK value)
                - gyroscope_z: float - Gyroscope Z axis (raw SDK value)
                - magnetometer_filtered_x: float - Magnetometer X axis (raw SDK value)
                - magnetometer_filtered_y: float - Magnetometer Y axis (raw SDK value)
                - magnetometer_filtered_z: float - Magnetometer Z axis (raw SDK value)
                - wheel_rpm_0: float - Wheel 0 RPM
                - wheel_rpm_1: float - Wheel 1 RPM
                - wheel_rpm_2: float - Wheel 2 RPM
                - wheel_rpm_3: float - Wheel 3 RPM
        """
        return {
            # Cameras (height, width, channels)
            OBS_FRONT: (480, 640, 3),
            OBS_REAR: (480, 640, 3),
            # Telemetry
            OBS_SPEED: float,
            OBS_BATTERY_LEVEL: float,
            OBS_ORIENTATION: float,
            OBS_GPS_LATITUDE: float,
            OBS_GPS_LONGITUDE: float,
            OBS_GPS_SIGNAL: float,
            OBS_SIGNAL_LEVEL: float,
            OBS_VIBRATION: float,
            OBS_LAMP: float,
            # IMU — accelerometer
            OBS_ACCELEROMETER_X: float,
            OBS_ACCELEROMETER_Y: float,
            OBS_ACCELEROMETER_Z: float,
            # IMU — gyroscope
            OBS_GYROSCOPE_X: float,
            OBS_GYROSCOPE_Y: float,
            OBS_GYROSCOPE_Z: float,
            # IMU — magnetometer
            OBS_MAGNETOMETER_X: float,
            OBS_MAGNETOMETER_Y: float,
            OBS_MAGNETOMETER_Z: float,
            # Wheel RPMs
            OBS_WHEEL_RPM_0: float,
            OBS_WHEEL_RPM_1: float,
            OBS_WHEEL_RPM_2: float,
            OBS_WHEEL_RPM_3: float,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define the action space.

        Returns:
            dict: Action features with types:
                - linear_velocity: float - Target linear velocity (-1 to 1)
                - angular_velocity: float - Target angular velocity (-1 to 1)
        """
        return {
            ACTION_LINEAR_VEL: float,
            ACTION_ANGULAR_VEL: float,
        }

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Get current robot observation from SDK.

        Camera frames are retrieved from SDK endpoints /v2/front and /v2/rear.
        Frames are decoded from base64 and converted from BGR to RGB format.
        Robot telemetry is retrieved from /data endpoint.
        Sensor arrays (accels, gyros, mags, rpms) each contain entries of
        [values..., timestamp]; the latest reading from each array is used.

        Returns:
            RobotObservation: Observation containing:
                - front: Front camera image (480, 640, 3) in RGB format
                - rear: Rear camera image (480, 640, 3) in RGB format
                - speed: float - Current speed (raw SDK value)
                - battery_level: float - Battery level (0-100)
                - orientation: float - Robot orientation in degrees
                - gps_latitude: float - GPS latitude coordinate
                - gps_longitude: float - GPS longitude coordinate
                - gps_signal: float - GPS signal strength (percentage)
                - signal_level: float - Network signal level (0-5)
                - vibration: float - Vibration sensor reading
                - lamp: float - Lamp state (0=off, 1=on)
                - accelerometer_x/y/z: float - Accelerometer axes (raw SDK value)
                - gyroscope_x/y/z: float - Gyroscope axes (raw SDK value)
                - magnetometer_filtered_x/y/z: float - Magnetometer axes (raw SDK value)
                - wheel_rpm_0/1/2/3: float - Wheel RPMs

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

        # Telemetry
        observation[OBS_SPEED] = float(robot_data["speed"])
        observation[OBS_BATTERY_LEVEL] = float(robot_data["battery"])
        observation[OBS_ORIENTATION] = float(robot_data["orientation"])
        observation[OBS_GPS_LATITUDE] = float(robot_data["latitude"])
        observation[OBS_GPS_LONGITUDE] = float(robot_data["longitude"])
        observation[OBS_GPS_SIGNAL] = float(robot_data["gps_signal"])
        observation[OBS_SIGNAL_LEVEL] = float(robot_data["signal_level"])
        observation[OBS_VIBRATION] = float(robot_data["vibration"])
        observation[OBS_LAMP] = float(robot_data["lamp"])

        # Accelerometer — latest reading from accels array [x, y, z, ts]
        accel = self._latest_sensor_reading(robot_data, "accels", n_values=3)
        observation[OBS_ACCELEROMETER_X] = accel[0]
        observation[OBS_ACCELEROMETER_Y] = accel[1]
        observation[OBS_ACCELEROMETER_Z] = accel[2]

        # Gyroscope — latest reading from gyros array [x, y, z, ts]
        gyro = self._latest_sensor_reading(robot_data, "gyros", n_values=3)
        observation[OBS_GYROSCOPE_X] = gyro[0]
        observation[OBS_GYROSCOPE_Y] = gyro[1]
        observation[OBS_GYROSCOPE_Z] = gyro[2]

        # Magnetometer — latest reading from mags array [x, y, z, ts]
        mag = self._latest_sensor_reading(robot_data, "mags", n_values=3)
        observation[OBS_MAGNETOMETER_X] = mag[0]
        observation[OBS_MAGNETOMETER_Y] = mag[1]
        observation[OBS_MAGNETOMETER_Z] = mag[2]

        # Wheel RPMs — latest reading from rpms array [w0, w1, w2, w3, ts]
        rpm = self._latest_sensor_reading(robot_data, "rpms", n_values=4)
        observation[OBS_WHEEL_RPM_0] = rpm[0]
        observation[OBS_WHEEL_RPM_1] = rpm[1]
        observation[OBS_WHEEL_RPM_2] = rpm[2]
        observation[OBS_WHEEL_RPM_3] = rpm[3]

        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Send action to robot via SDK.

        Args:
            action: Action dict with keys:
                - linear_velocity: Target linear velocity (-1 to 1)
                - angular_velocity: Target angular velocity (-1 to 1)

        Returns:
            RobotAction: The action that was sent (matches action_features keys)

        Raises:
            DeviceNotConnectedError: If robot is not connected

        Note:
            Actions are sent to SDK via POST /control endpoint.
            SDK expects commands in range [-1, 1].
        """
        linear = float(action.get(ACTION_LINEAR_VEL, 0.0))
        angular = float(action.get(ACTION_ANGULAR_VEL, 0.0))

        try:
            self._send_command_to_sdk(linear, angular)
        except Exception as e:
            logger.error(f"Error sending action: {e}")

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

    @staticmethod
    def _latest_sensor_reading(robot_data: dict, key: str, n_values: int) -> list[float]:
        """Extract the latest sensor reading from an SDK sensor array.

        The SDK returns sensor arrays like ``accels``, ``gyros``, ``mags``,
        ``rpms`` where each entry is ``[value_0, ..., value_n, timestamp]``.
        This helper returns the *n_values* leading floats from the last entry,
        falling back to zeros when the key is missing or the array is empty.
        """
        readings = robot_data.get(key)
        if readings and len(readings) > 0:
            latest = readings[-1]
            return [float(v) for v in latest[:n_values]]
        return [0.0] * n_values

    def _get_robot_data(self) -> dict:
        """Get robot telemetry data from SDK.

        Returns:
            dict: Robot telemetry data including battery, speed, orientation, GPS,
                and sensor arrays (accels, gyros, mags, rpms):
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
            "accels": [],
            "gyros": [],
            "mags": [],
            "rpms": [],
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

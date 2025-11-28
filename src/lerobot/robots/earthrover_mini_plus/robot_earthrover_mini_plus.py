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

import base64
import logging
from functools import cached_property
from typing import Any

import cv2
import numpy as np
import requests

from lerobot.cameras.earthrover_mini_camera import VirtualCamera
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_earthrover_mini_plus import EarthRoverMiniPlusConfig

logger = logging.getLogger(__name__)


class EarthRoverMiniPlus(Robot):
    """
    EarthRover Mini robot using Frodobots SDK HTTP API.
    
    This implementation uses the cloud-based Frodobots SDK instead of direct TCP connection.
    Perfect for dataset recording and teleoperation through LeRobot.
    """

    config_class = EarthRoverMiniPlusConfig
    name = "earthrover_mini_plus"

    def __init__(self, config: EarthRoverMiniPlusConfig):
        super().__init__(config)
        self.config = config
        self.sdk_base_url = "http://localhost:8000"
        
        # SDK cameras (base64 from HTTP API)
        self.cameras = {}
        self._is_connected = False
        self._last_observation = {}
        
        logger.info(f"Initialized {self.name} with SDK at {self.sdk_base_url}")

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to robot via Frodobots SDK."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected")
        
        # Check SDK connection
        try:
            response = requests.get(f"{self.sdk_base_url}/data", timeout=10.0)
            if response.status_code != 200:
                raise DeviceNotConnectedError(
                    f"Cannot connect to SDK at {self.sdk_base_url}. "
                    "Make sure it's running: hypercorn main:app --reload"
                )
        except requests.RequestException as e:
            raise DeviceNotConnectedError(
                f"Cannot connect to SDK at {self.sdk_base_url}: {e}"
            ) from e
        
        # Initialize virtual cameras for dataset recording
        # We'll populate them with frames from SDK
        self.cameras = {
            "front": VirtualCamera("front", self, fps=30, width=640, height=480),
            "rear": VirtualCamera("rear", self, fps=30, width=640, height=480),
        }
        
        self._is_connected = True
        logger.info(f"✓ {self.name} connected to SDK")
        
        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        """Calibration not needed for SDK-based robot."""
        logger.info("Calibration not required for SDK-based robot")

    @property
    def is_calibrated(self) -> bool:
        """SDK robot doesn't require calibration."""
        return True

    def configure(self) -> None:
        """No configuration needed."""
        pass

    @cached_property
    def observation_features(self) -> dict:
        """Define the observation space for dataset recording."""
        return {
            # Cameras (height, width, channels)
            "front": (480, 640, 3),
            "rear": (480, 640, 3),
            # Robot state (individual features) - using .vel suffix like LeKiwi
            "linear.vel": float,
            "angular.vel": float,
            "battery.level": float,
            "orientation.deg": float,
        }

    @cached_property
    def action_features(self) -> dict:
        """Define the action space - matches observation keys for velocities."""
        return {
            "linear.vel": float,
            "angular.vel": float,
        }

    def get_observation(self) -> dict[str, Any]:
        """
        Get current robot observation from SDK.
        
        Returns observation dict with:
        - front: front camera image
        - rear: rear camera image  
        - linear.vel, angular.vel: current velocities
        - battery.level, orientation.deg: robot state
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected")
        
        observation = {}
        
        # Get camera images
        try:
            frames = self._get_camera_frames()
            
            # Front camera
            if "front" in frames:
                observation["front"] = frames["front"]
            else:
                observation["front"] = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Rear camera
            if "rear" in frames:
                observation["rear"] = frames["rear"]
            else:
                observation["rear"] = np.zeros((480, 640, 3), dtype=np.uint8)
                
        except Exception as e:
            logger.warning(f"Error getting camera frames: {e}")
            observation["front"] = np.zeros((480, 640, 3), dtype=np.uint8)
            observation["rear"] = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Get robot state
        try:
            robot_data = self._get_robot_data()
            
            # Current velocities (normalized from SDK speed value)
            # SDK gives speed 0-100, we normalize to match our action space
            observation["linear.vel"] = robot_data.get("speed", 0) / 100.0 * 50.0  # Assume max 50 units
            observation["angular.vel"] = 0.0  # SDK doesn't report angular velocity separately
            
            # Robot state
            observation["battery.level"] = robot_data.get("battery", 0) / 100.0  # Normalize to 0-1
            observation["orientation.deg"] = robot_data.get("orientation", 0) / 360.0  # Normalize to 0-1
            
        except Exception as e:
            logger.warning(f"Error getting robot state: {e}")
            observation["linear.vel"] = 0.0
            observation["angular.vel"] = 0.0
            observation["battery.level"] = 0.0
            observation["orientation.deg"] = 0.0
        
        self._last_observation = observation
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action to robot via SDK.
        
        Args:
            action: Dict with 'linear.vel' and 'angular.vel' keys (from teleop)
            
        Returns:
            The action that was sent (matches action_features keys)
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected")
        
        # Extract action values - handle both old and new key formats
        linear = action.get("linear.vel", action.get("linear_velocity", 0.0))
        angular = action.get("angular.vel", action.get("angular_velocity", 0.0))
        
        # Send to SDK
        try:
            self._send_command_to_sdk(linear, angular)
        except Exception as e:
            logger.error(f"Error sending action: {e}")
        
        # Return in the format that matches action_features
        return {
            "linear.vel": linear,
            "angular.vel": angular,
        }

    def disconnect(self) -> None:
        """Disconnect from robot."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected")
        
        # Stop the robot
        try:
            self._send_command_to_sdk(0.0, 0.0)
        except:
            pass
        
        self._is_connected = False
        logger.info(f"{self.name} disconnected")

    # Helper methods for SDK communication
    
    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """Get camera frames from SDK using separate endpoints."""
        frames = {}
        
        # Get front camera
        try:
            response = requests.get(f"{self.sdk_base_url}/v2/front", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                if "front_frame" in data and data["front_frame"]:
                    front_img = self._decode_base64_image(data["front_frame"])
                    if front_img is not None:
                        frames["front"] = cv2.resize(front_img, (640, 480))
        except Exception as e:
            logger.warning(f"Error fetching front camera: {e}")
        
        # Get rear camera
        try:
            response = requests.get(f"{self.sdk_base_url}/v2/rear", timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                if "rear_frame" in data and data["rear_frame"]:
                    rear_img = self._decode_base64_image(data["rear_frame"])
                    if rear_img is not None:
                        frames["rear"] = cv2.resize(rear_img, (640, 480))
        except Exception as e:
            logger.warning(f"Error fetching rear camera: {e}")
        
        return frames
    
    def _decode_base64_image(self, base64_string: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        import base64
        
        try:
            img_bytes = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None
    
    def _get_robot_data(self) -> dict:
        """Get robot telemetry data from SDK."""
        try:
            response = requests.get(f"{self.sdk_base_url}/data", timeout=10.0)
            
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Error fetching robot data: {e}")
        
        return {}
    
    def _send_command_to_sdk(self, linear: float, angular: float, lamp: int = 0) -> bool:
        """Send control command to SDK."""
        try:
            payload = {
                "command": {
                    "linear": linear,
                    "angular": angular,
                    "lamp": lamp
                }
            }
            
            response = requests.post(
                f"{self.sdk_base_url}/control",
                json=payload,
                timeout=1.0
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return False

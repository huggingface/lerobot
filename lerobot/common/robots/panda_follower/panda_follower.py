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

import numpy as np

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.constants import OBS_STATE
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from ...motors.franka_api.API import API
from .config_panda_follower import PandaConfig

logger = logging.getLogger(__name__)


class PandaRobot(Robot):
    """
    Franka Emika Panda robot integration for LeRobot.
    
    This class provides an interface to control the Franka Emika Panda robot
    through a gRPC API connection.
    """

    config_class = PandaConfig
    name = "panda_follower"

    def __init__(self, config: PandaConfig):
        super().__init__(config)
        self.config = config
        self.api = None
        self._connected = False
        self._calibrated = True  # Panda is factory calibrated
        # Initialize cameras if configured
        self.cameras = make_cameras_from_configs(config.cameras) if config.cameras else {}
        
        # Joint names in order
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7"
        ]

    @property
    def _joint_features(self) -> dict[str, type]:
        """Define joint position features."""
        return {f"{joint}.pos": float for joint in self.joint_names}

    @property 
    def _force_features(self) -> dict[str, type]:
        """Define force/torque features."""
        if not self.config.use_force_feedback:
            return {}
        return {
            "force.x": float, "force.y": float, "force.z": float,
            "torque.x": float, "torque.y": float, "torque.z": float
        }

    @property
    def _cartesian_features(self) -> dict[str, type]:
        """Define Cartesian pose features."""
        if not self.config.use_cartesian_feedback:
            return {}
        return {
            "pose.x": float, "pose.y": float, "pose.z": float,
            "pose.qx": float, "pose.qy": float, "pose.qz": float, "pose.qw": float
        }

    @property
    def _camera_features(self) -> dict[str, tuple]:
        """Define camera features."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) 
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Return all observation features."""
        features = {**self._joint_features}
        
        if self.config.use_force_feedback:
            features.update(self._force_features)
            
        if self.config.use_cartesian_feedback:
            features.update(self._cartesian_features)
            
        if self.cameras:
            features.update(self._camera_features)
            
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Return action features (joint positions)."""
        return self._joint_features

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._connected #and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the Panda robot."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected")

        try:
            # Initialize gRPC connection
            self.api = API(server_address=self.config.ip)
            
            # Test connection by getting joint state
            _ = self.api.get_joint_state()
            self._connected = True
            
            # Connect cameras if any
            for cam in self.cameras.values():
                cam.connect()
                
            logger.info(f"{self} connected successfully")
            
        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect to {self}: {e}")
            raise

    @property
    def is_calibrated(self) -> bool:
        """Panda robots are factory calibrated."""
        return self._calibrated

    def calibrate(self) -> None:
        """Panda robots don't require manual calibration."""
        logger.info(f"{self} is factory calibrated")
        self._calibrated = True

    def configure(self) -> None:
        """Configure robot parameters."""
        # Panda configuration is handled by the gRPC server
        logger.info(f"{self} configuration handled by gRPC server")

    def _validate_joint_positions(self, positions: dict[str, float]) -> dict[str, float]:
        """Validate and clamp joint positions to limits."""
        validated = {}
        
        for joint, position in positions.items():
            if joint in self.config.joint_limits:
                min_pos, max_pos = self.config.joint_limits[joint]
                clamped_pos = np.clip(position, min_pos, max_pos)
                
                if abs(clamped_pos - position) > 1e-6:
                    logger.warning(
                        f"Joint {joint} position {position:.4f} clamped to {clamped_pos:.4f}"
                    )
                    
                validated[joint] = clamped_pos
            else:
                validated[joint] = position
                
        return validated

    def get_observation(self) -> dict[str, Any]:
        """Get current observation from the robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        obs_dict = {}

        try:
            # Get joint positions
            start_time = time.perf_counter()
            joint_angles = self.api.get_joint_position()
            
            # Convert to observation format
            for i, joint_name in enumerate(self.joint_names):
                obs_dict[f"{joint_name}.pos"] = float(joint_angles[i])
                
            dt_ms = (time.perf_counter() - start_time) * 1e3
            logger.debug(f"{self} read joint positions: {dt_ms:.1f}ms")

            # Get force/torque if enabled
            if self.config.use_force_feedback:
                start_time = time.perf_counter()
                wrench = self.api.get_wrench()
                
                obs_dict.update({
                    "force.x": float(wrench.fx),
                    "force.y": float(wrench.fy), 
                    "force.z": float(wrench.fz),
                    "torque.x": float(wrench.tx),
                    "torque.y": float(wrench.ty),
                    "torque.z": float(wrench.tz),
                })
                
                dt_ms = (time.perf_counter() - start_time) * 1e3
                logger.debug(f"{self} read wrench: {dt_ms:.1f}ms")

            # Get Cartesian pose if enabled
            if self.config.use_cartesian_feedback:
                start_time = time.perf_counter()
                cart_pose = self.api.get_cart_pose()
                
                obs_dict.update({
                    "pose.x": float(cart_pose.x),
                    "pose.y": float(cart_pose.y),
                    "pose.z": float(cart_pose.z),
                    "pose.qx": float(cart_pose.qx),
                    "pose.qy": float(cart_pose.qy),
                    "pose.qz": float(cart_pose.qz),
                    "pose.qw": float(cart_pose.qw),
                })
                
                dt_ms = (time.perf_counter() - start_time) * 1e3
                logger.debug(f"{self} read Cartesian pose: {dt_ms:.1f}ms")

            # Capture camera images
            for cam_key, cam in self.cameras.items():
                start_time = time.perf_counter()
                obs_dict[cam_key] = cam.async_read()
                dt_ms = (time.perf_counter() - start_time) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        except Exception as e:
            logger.error(f"Error getting observation from {self}: {e}")
            raise

        return obs_dict

    def send_action(self, action):
        """Send action to the robot.
        
        Args:
            action: Dictionary with joint position targets (keys ending with '.pos')
            
        Returns:
            The actual action sent (potentially clipped for safety)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        action = self._validate_joint_positions(action)
        action_array = np.array([action[joint + ".pos"] for joint in self.joint_names], dtype=np.float32)
        response_msg = self.api.set_joint_position(list(action_array))
        joint_angles = self.api.get_joint_position()
        response = {f"{joint}.pos": float(joint_angles[i]) for i, joint in enumerate(self.joint_names)}
        return response

        
    def disconnect(self) -> None:
        """Disconnect from the robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        try:
            # Disconnect cameras
            for cam in self.cameras.values():
                cam.disconnect()

            # Close gRPC connection
            if self.api and hasattr(self.api, 'channel'):
                self.api.channel.close()

            self._connected = False
            logger.info(f"{self} disconnected")

        except Exception as e:
            logger.error(f"Error disconnecting {self}: {e}")
            raise

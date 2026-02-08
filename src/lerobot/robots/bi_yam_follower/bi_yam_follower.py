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

import logging
import time
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.import_utils import _portal_available

from ..utils import ensure_safe_goal_position

if TYPE_CHECKING or _portal_available:
    import portal
else:
    portal = None

from ..robot import Robot
from .config_bi_yam_follower import BiYamFollowerConfig

logger = logging.getLogger(__name__)


class YamArmClient:
    """Client interface for a single Yam arm using the portal RPC framework."""

    def __init__(self, port: int, host: str = "localhost"):
        """
        Initialize the Yam arm client.

        Args:
            port: Server port for the arm
            host: Server host address
        """
        self.port = port
        self.host = host
        self._client = None

    def connect(self):
        """Connect to the arm server."""
        logger.info(f"Connecting to Yam arm server at {self.host}:{self.port}")
        self._client = portal.Client(f"{self.host}:{self.port}")
        logger.info(f"Successfully connected to Yam arm server at {self.host}:{self.port}")

    def disconnect(self):
        """Disconnect from the arm server."""
        if self._client is not None:
            logger.info(f"Disconnecting from Yam arm server at {self.host}:{self.port}")
            self._client = None

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self._client is not None

    def num_dofs(self) -> int:
        """Get the number of degrees of freedom."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.num_dofs().result()

    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command joint positions."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        self._client.command_joint_pos(joint_pos)

    def get_observations(self) -> dict[str, np.ndarray]:
        """Get current observations including joint positions, velocities, etc."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.get_observations().result()


class BiYamFollower(Robot):
    """
    Bimanual Yam Arms follower robot using the i2rt library.

    This robot controls two Yam arms simultaneously. Each arm communicates via
    the portal RPC framework with servers running on different ports.

    Expected setup:
    - Two Yam follower arms connected via CAN interfaces
    - Server processes running for each arm (see bimanual_lead_follower.py)
    - Left arm server on port 1235 (default)
    - Right arm server on port 1234 (default)
    """

    config_class = BiYamFollowerConfig
    name = "bi_yam_follower"

    def __init__(self, config: BiYamFollowerConfig):
        super().__init__(config)
        self.config = config

        # Create clients for left and right arms
        self.left_arm = YamArmClient(port=config.left_arm_port, host=config.server_host)
        self.right_arm = YamArmClient(port=config.right_arm_port, host=config.server_host)

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

        # Store number of DOFs (will be set after connection)
        self._left_dofs = None
        self._right_dofs = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Define motor feature types for both arms."""
        if self._left_dofs is None or self._right_dofs is None:
            # Default to 7 DOFs (6 joints + 1 gripper) per arm if not yet connected
            left_dofs = 7
            right_dofs = 7
        else:
            left_dofs = self._left_dofs
            right_dofs = self._right_dofs

        features = {}
        # Left arm joints and gripper
        # Assume last DOF is gripper if we have 7 DOFs
        for i in range(left_dofs):
            if left_dofs == 7 and i == left_dofs - 1:  # Last DOF is gripper
                features["left_gripper.pos"] = float
            else:
                features[f"left_joint_{i}.pos"] = float

        # Right arm joints and gripper
        # Assume last DOF is gripper if we have 7 DOFs
        for i in range(right_dofs):
            if right_dofs == 7 and i == right_dofs - 1:  # Last DOF is gripper
                features["right_gripper.pos"] = float
            else:
                features[f"right_joint_{i}.pos"] = float

        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Define camera feature types."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Return observation features including motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Return action features (motor positions)."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if both arms and all cameras are connected."""
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to both arm servers and cameras.

        Args:
            calibrate: Not used for Yam arms (kept for API compatibility)
        """
        if not _portal_available:
            raise ImportError(
                "The 'portal' library is not installed. "
                "Please install it with `pip install 'lerobot[yam]'` to use BiYamFollower. "
                "Note: Yam arms require Linux for hardware operation."
            )

        logger.info("Connecting to bimanual Yam follower robot")

        # Connect to arm servers
        self.left_arm.connect()
        self.right_arm.connect()

        # Get number of DOFs from each arm
        self._left_dofs = self.left_arm.num_dofs()
        self._right_dofs = self.right_arm.num_dofs()

        # Clear cached properties so they get recomputed with actual DOF counts
        for prop in ["observation_features", "action_features"]:
            if prop in self.__dict__:
                del self.__dict__[prop]

        logger.info(f"Left arm DOFs: {self._left_dofs}, Right arm DOFs: {self._right_dofs}")

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        logger.info("Successfully connected to bimanual Yam follower robot")

    @property
    def is_calibrated(self) -> bool:
        """Yam arms don't require calibration in the lerobot sense."""
        return self.is_connected

    def calibrate(self) -> None:
        """Yam arms don't require calibration in the lerobot sense."""
        pass

    def configure(self) -> None:
        """Configure the robot (not needed for Yam arms)."""
        pass

    def setup_motors(self) -> None:
        """Setup motors (not needed for Yam arms)."""
        pass

    @check_if_not_connected
    def get_observation(self, include_cameras: bool = True) -> dict[str, Any]:
        """
        Get current observation from both arms and optionally cameras.

        Args:
            include_cameras: If True, include camera images in observation.
                            Set to False for faster observation when cameras are not needed.

        Returns:
            Dictionary with joint positions for both arms and optionally camera images
        """
        obs_dict = {}

        # Get left arm observations
        left_obs = self.left_arm.get_observations()
        left_joint_pos = left_obs["joint_pos"]
        left_has_gripper = "gripper_pos" in left_obs
        if left_has_gripper:
            left_joint_pos = np.concatenate([left_joint_pos, left_obs["gripper_pos"]])

        # Add with "left_" prefix
        for i, pos in enumerate(left_joint_pos):
            # Gripper is the last DOF if present
            if left_has_gripper and i == len(left_joint_pos) - 1:
                obs_dict["left_gripper.pos"] = pos
            else:
                obs_dict[f"left_joint_{i}.pos"] = pos

        # Get right arm observations
        right_obs = self.right_arm.get_observations()
        right_joint_pos = right_obs["joint_pos"]
        right_has_gripper = "gripper_pos" in right_obs
        if right_has_gripper:
            right_joint_pos = np.concatenate([right_joint_pos, right_obs["gripper_pos"]])

        # Add with "right_" prefix
        for i, pos in enumerate(right_joint_pos):
            # Gripper is the last DOF if present
            if right_has_gripper and i == len(right_joint_pos) - 1:
                obs_dict["right_gripper.pos"] = pos
            else:
                obs_dict[f"right_joint_{i}.pos"] = pos

        # Get camera observations only if requested
        if include_cameras:
            for cam_key, cam in self.cameras.items():
                start = time.perf_counter()
                obs_dict[cam_key] = cam.async_read()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def get_camera_observation(self) -> dict[str, Any]:
        """
        Get current camera observations only.

        Returns:
            Dictionary with camera images
        """
        obs_dict = {}
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        return obs_dict

    @check_if_not_connected
    def get_camera_observation_with_timestamps(self) -> tuple[dict[str, Any], dict[str, float]]:
        """
        Get current camera observations with their capture timestamps.

        This method is useful for data collection where frame-action synchronization
        is needed. Each camera frame is returned along with its capture timestamp.

        Returns:
            tuple containing:
                - Dictionary with camera images (cam_key -> frame)
                - Dictionary with timestamps (cam_key -> timestamp)
        """
        obs_dict = {}
        timestamps = {}
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            # Access the timestamp that was captured with the frame
            if hasattr(cam, "latest_timestamp") and cam.latest_timestamp is not None:
                timestamps[cam_key] = cam.latest_timestamp
            else:
                # Fallback to current time if timestamp not available
                timestamps[cam_key] = time.perf_counter()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        return obs_dict, timestamps

    @check_if_not_connected
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action commands to both arms.

        Args:
            action: Dictionary with joint positions for both arms

        Returns:
            The action that was sent
        """
        # Extract left arm actions
        left_action = []
        for i in range(self._left_dofs):
            # Last DOF is gripper if we have 7 DOFs
            key = (
                "left_gripper.pos"
                if (self._left_dofs == 7 and i == self._left_dofs - 1)
                else f"left_joint_{i}.pos"
            )
            if key in action:
                left_action.append(action[key])

        # Extract right arm actions
        right_action = []
        for i in range(self._right_dofs):
            # Last DOF is gripper if we have 7 DOFs
            key = (
                "right_gripper.pos"
                if (self._right_dofs == 7 and i == self._right_dofs - 1)
                else f"right_joint_{i}.pos"
            )
            if key in action:
                right_action.append(action[key])

        # Apply max_relative_target if configured using shared utility
        if self.config.left_arm_max_relative_target is not None and len(left_action) > 0:
            left_current = self.left_arm.get_joint_pos()
            # Build dict of {key: (goal, present)} for ensure_safe_goal_position
            left_goal_present = {}
            for i, goal_val in enumerate(left_action):
                key = (
                    "left_gripper.pos"
                    if (self._left_dofs == 7 and i == self._left_dofs - 1)
                    else f"left_joint_{i}.pos"
                )
                left_goal_present[key] = (goal_val, float(left_current[i]))
            safe_left = ensure_safe_goal_position(left_goal_present, self.config.left_arm_max_relative_target)
            # Extract back to array in order
            left_action = [safe_left[k] for k in sorted(safe_left.keys())]

        if self.config.right_arm_max_relative_target is not None and len(right_action) > 0:
            right_current = self.right_arm.get_joint_pos()
            # Build dict of {key: (goal, present)} for ensure_safe_goal_position
            right_goal_present = {}
            for i, goal_val in enumerate(right_action):
                key = (
                    "right_gripper.pos"
                    if (self._right_dofs == 7 and i == self._right_dofs - 1)
                    else f"right_joint_{i}.pos"
                )
                right_goal_present[key] = (goal_val, float(right_current[i]))
            safe_right = ensure_safe_goal_position(
                right_goal_present, self.config.right_arm_max_relative_target
            )
            # Extract back to array in order
            right_action = [safe_right[k] for k in sorted(safe_right.keys())]

        # Send commands to arms
        if len(left_action) > 0:
            self.left_arm.command_joint_pos(np.array(left_action))

        if len(right_action) > 0:
            self.right_arm.command_joint_pos(np.array(right_action))

        return action

    def disconnect(self):
        """Disconnect from both arms and cameras."""
        logger.info("Disconnecting from bimanual Yam follower robot")

        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info("Disconnected from bimanual Yam follower robot")

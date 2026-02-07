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
from functools import cached_property

import numpy as np
import portal

from ..teleoperator import Teleoperator
from .config_bi_yam_leader import BiYamLeaderConfig

logger = logging.getLogger(__name__)


class YamLeaderClient:
    """Client interface for a single Yam leader arm using the portal RPC framework."""

    def __init__(self, port: int, host: str = "localhost"):
        """
        Initialize the Yam leader arm client.

        Args:
            port: Server port for the leader arm
            host: Server host address
        """
        self.port = port
        self.host = host
        self._client = None

    def connect(self):
        """Connect to the leader arm server."""
        logger.info(f"Connecting to Yam leader arm server at {self.host}:{self.port}")
        self._client = portal.Client(f"{self.host}:{self.port}")
        logger.info(f"Successfully connected to Yam leader arm server at {self.host}:{self.port}")

    def disconnect(self):
        """Disconnect from the leader arm server."""
        if self._client is not None:
            logger.info(f"Disconnecting from Yam leader arm server at {self.host}:{self.port}")
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
        """Get current joint positions from the leader arm."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.get_joint_pos().result()

    def get_observations(self) -> dict[str, np.ndarray]:
        """Get current observations including joint positions, velocities, etc."""
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.get_observations().result()

    def get_gripper_from_encoder(self) -> float:
        """
        Try to get gripper state from teaching handle encoder button.
        Returns a value between 0 (closed) and 1 (open).
        Falls back to 1.0 (open) if not available.
        """
        if self._client is None:
            raise RuntimeError("Client not connected")
        try:
            # Try to get encoder state if the server exposes it
            # This requires custom method in the i2rt server
            obs = self._client.get_observations().result()
            # Check if encoder button data is available in observations
            # The encoder button state might be in io_inputs or similar field
            if "io_inputs" in obs:
                # Button pressed = closed gripper (0), not pressed = open (1)
                return 0.0 if obs["io_inputs"][0] > 0.5 else 1.0
            return 1.0  # Default to open if no encoder data
        except Exception:
            return 1.0  # Default to open on any error


class BiYamLeader(Teleoperator):
    """
    Bimanual Yam Arms leader (teleoperator) using the i2rt library.

    This teleoperator reads joint positions from two Yam leader arms (with teaching handles)
    and provides them as actions for the follower robot.

    Expected setup:
    - Two Yam leader arms connected via CAN interfaces with teaching handles
    - Server processes running for each leader arm in read-only mode
    - Left leader arm server on port 5002 (default)
    - Right leader arm server on port 5001 (default)

    Note: You'll need to run separate server processes for the leader arms.
    You can modify the i2rt minimum_gello.py script to create read-only
    servers that just expose the leader arm state without trying to control
    a follower.
    """

    config_class = BiYamLeaderConfig
    name = "bi_yam_leader"

    def __init__(self, config: BiYamLeaderConfig):
        super().__init__(config)
        self.config = config

        # Create clients for left and right leader arms
        self.left_arm = YamLeaderClient(port=config.left_arm_port, host=config.server_host)
        self.right_arm = YamLeaderClient(port=config.right_arm_port, host=config.server_host)

        # Store number of DOFs (will be set after connection)
        self._left_dofs = None
        self._right_dofs = None

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action features for both arms."""
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

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """Yam leader arms don't support feedback."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if both leader arms are connected."""
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to both leader arm servers.

        Args:
            calibrate: Not used for Yam arms (kept for API compatibility)
        """
        logger.info("Connecting to bimanual Yam leader arms")

        # Connect to leader arm servers
        self.left_arm.connect()
        self.right_arm.connect()

        # Get number of DOFs from each arm
        self._left_dofs = self.left_arm.num_dofs()
        self._right_dofs = self.right_arm.num_dofs()

        logger.info(f"Left leader arm DOFs: {self._left_dofs}, Right leader arm DOFs: {self._right_dofs}")
        logger.info("Successfully connected to bimanual Yam leader arms")

    @property
    def is_calibrated(self) -> bool:
        """Yam leader arms don't require calibration in the lerobot sense."""
        return self.is_connected

    def calibrate(self) -> None:
        """Yam leader arms don't require calibration in the lerobot sense."""
        pass

    def configure(self) -> None:
        """Configure the teleoperator (not needed for Yam leader arms)."""
        pass

    def setup_motors(self) -> None:
        """Setup motors (not needed for Yam leader arms)."""
        pass

    def get_action(self) -> dict[str, float]:
        """
        Get action from both leader arms by reading their current joint positions.

        For teaching handles (no physical gripper), we try to read encoder button state
        to control the gripper, falling back to fully open if not available.

        Returns:
            Dictionary with joint positions for both arms (including gripper)
        """
        action_dict = {}

        # Get left arm observations
        left_obs = self.left_arm.get_observations()
        left_joint_pos = left_obs["joint_pos"]

        # Handle gripper: either from physical gripper or teaching handle encoder
        left_has_gripper = "gripper_pos" in left_obs
        if left_has_gripper:
            left_joint_pos = np.concatenate([left_joint_pos, left_obs["gripper_pos"]])
        else:
            # Teaching handle: try to get gripper from encoder button
            left_gripper = self.left_arm.get_gripper_from_encoder()
            left_joint_pos = np.concatenate([left_joint_pos, [left_gripper]])
            left_has_gripper = True

        # Add with "left_" prefix
        for i, pos in enumerate(left_joint_pos):
            # Gripper is the last DOF if present
            if left_has_gripper and i == len(left_joint_pos) - 1:
                action_dict["left_gripper.pos"] = float(pos)
            else:
                action_dict[f"left_joint_{i}.pos"] = float(pos)

        # Get right arm observations
        right_obs = self.right_arm.get_observations()
        right_joint_pos = right_obs["joint_pos"]

        # Handle gripper: either from physical gripper or teaching handle encoder
        right_has_gripper = "gripper_pos" in right_obs
        if right_has_gripper:
            right_joint_pos = np.concatenate([right_joint_pos, right_obs["gripper_pos"]])
        else:
            # Teaching handle: try to get gripper from encoder button
            right_gripper = self.right_arm.get_gripper_from_encoder()
            right_joint_pos = np.concatenate([right_joint_pos, [right_gripper]])
            right_has_gripper = True

        # Add with "right_" prefix
        for i, pos in enumerate(right_joint_pos):
            # Gripper is the last DOF if present
            if right_has_gripper and i == len(right_joint_pos) - 1:
                action_dict["right_gripper.pos"] = float(pos)
            else:
                action_dict[f"right_joint_{i}.pos"] = float(pos)

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """
        Send feedback to leader arms (not supported for Yam teaching handles).

        Args:
            feedback: Dictionary with feedback values (ignored)
        """
        # Yam teaching handles are passive devices and don't support feedback
        pass

    def disconnect(self) -> None:
        """Disconnect from both leader arms."""
        logger.info("Disconnecting from bimanual Yam leader arms")

        self.left_arm.disconnect()
        self.right_arm.disconnect()

        logger.info("Disconnected from bimanual Yam leader arms")

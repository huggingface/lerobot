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
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.import_utils import _portal_available

if TYPE_CHECKING or _portal_available:
    import portal
else:
    portal = None

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

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command joint positions to the leader arm.

        Args:
            joint_pos: Joint positions (6 joints, gripper is ignored)
        """
        if self._client is None:
            raise RuntimeError("Client not connected")
        # Strip gripper if included (leader arm only has 6 joints)
        if len(joint_pos) > 6:
            joint_pos = joint_pos[:6]
        self._client.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        """Update PD gains for the leader arm.

        Args:
            kp: Position gains (6 values)
            kd: Derivative gains (6 values)
        """
        if self._client is None:
            raise RuntimeError("Client not connected")
        self._client.update_kp_kd(kp, kd)

    def get_natural_kp(self) -> np.ndarray:
        """Return the leader arm's per-joint kp as configured by i2rt at server startup.

        The unified server snapshots ``robot._kp`` in YAMLeaderRobot.__init__ before
        any update_kp_kd call can overwrite it, so this is the arm-variant-correct
        baseline for bilateral feedback (YAM / YAM_PRO / YAM_ULTRA / BIG_YAM each
        ship distinct configs).
        """
        if self._client is None:
            raise RuntimeError("Client not connected")
        return self._client.get_natural_kp().result()


class BiYamLeader(Teleoperator):
    """
    Bimanual Yam Arms leader (teleoperator) using the i2rt library.

    This teleoperator reads joint positions from two Yam leader arms (with teaching handles)
    and provides them as actions. It supports bilateral control where the leader arms can
    receive position commands to mirror another robot's movements.

    Bilateral control works by adjusting PD gains:
    - Zero gains (kp=0, kd=0): Leader arms move freely with only gravity compensation
    - Non-zero gains (kp>0): Leader arms track commanded positions with force proportional to kp

    Expected setup:
    - Two Yam leader arms connected via CAN interfaces with teaching handles
    - Server processes running for each leader arm exposing update_kp_kd method
    - Left leader arm server on port 5002 (default)
    - Right leader arm server on port 5001 (default)

    Note: Launch the leader RPC servers via
    ``src/lerobot/robots/bi_yam_follower/run_bimanual_yam_server.py``, which
    starts both leader arms (and both followers) in one process with the
    update_kp_kd endpoint bound.
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

        # Store original kp values and bilateral control multiplier
        self._original_kp = None
        self.bilateral_kp = config.bilateral_kp

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
        """Define feedback features - same as action features for position mirroring."""
        # Feedback features match action features to allow leader to mirror robot position
        return self.action_features

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
        if not _portal_available:
            raise ImportError(
                "The 'portal' library is not installed. "
                "Please install it with `pip install 'lerobot[yam]'` to use BiYamLeader. "
                "Note: Yam arms require Linux for hardware operation."
            )

        logger.info("Connecting to bimanual Yam leader arms")

        # Connect to leader arm servers
        self.left_arm.connect()
        self.right_arm.connect()

        # Get number of DOFs from each arm
        self._left_dofs = self.left_arm.num_dofs()
        self._right_dofs = self.right_arm.num_dofs()

        # Clear the cached action_features property so it gets recomputed with actual DOF counts
        if "action_features" in self.__dict__:
            del self.__dict__["action_features"]

        logger.info(f"Left leader arm DOFs: {self._left_dofs}, Right leader arm DOFs: {self._right_dofs}")

        # Apply the configured bilateral_kp so send_feedback is honored from
        # the start. With bilateral_kp == 0.0 we skip the natural-kp fetch
        # entirely so the client stays compatible with older server builds
        # that don't expose the get_natural_kp RPC.
        if self.bilateral_kp > 0.0:
            # Fetch the leader's natural kp so bilateral feedback runs at the
            # i2rt-configured stiffness for whichever arm variant is connected
            # (YAM / YAM_PRO / YAM_ULTRA / BIG_YAM ship different per-joint
            # defaults). Assumes left/right are the same variant; if they
            # diverge in the future, fetch and store per-arm.
            self._original_kp = self.left_arm.get_natural_kp()
            logger.info(f"Leader natural kp from i2rt config: {self._original_kp}")
            self.enable_torque()

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

    @check_if_not_connected
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
            # Teaching handle: use io_inputs from the observations we already fetched
            # above, instead of issuing a second get_observations RPC round-trip.
            if "io_inputs" in left_obs:
                left_gripper = 0.0 if left_obs["io_inputs"][0] > 0.5 else 1.0
            else:
                left_gripper = 1.0  # Default to open
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
            # Teaching handle: use io_inputs from already-fetched observations
            if "io_inputs" in right_obs:
                right_gripper = 0.0 if right_obs["io_inputs"][0] > 0.5 else 1.0
            else:
                right_gripper = 1.0  # Default to open
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
        Command the leader arms to move to specified joint positions.

        The tracking behavior depends on current PD gains:
        - If gains are zero (kp=0): Commands have no effect, arms move freely
        - If gains are non-zero (kp>0): Arms actively track commanded positions

        Args:
            feedback: Dictionary with joint positions (matching feedback_features)
                     Keys are "left_joint_X.pos", "right_joint_X.pos", etc.
        """
        if not feedback:
            return

        # Extract left arm positions
        left_positions = []
        for i in range(6):  # 6 joints per arm (excluding gripper)
            key = f"left_joint_{i}.pos"
            if key in feedback:
                left_positions.append(feedback[key])

        # Extract right arm positions
        right_positions = []
        for i in range(6):  # 6 joints per arm
            key = f"right_joint_{i}.pos"
            if key in feedback:
                right_positions.append(feedback[key])

        # Debug: Log first time to verify data format
        if len(left_positions) == 6 and len(right_positions) == 6:
            logger.debug(
                f"Feedback received - Left: {left_positions[:2]}..., Right: {right_positions[:2]}..."
            )
        elif len(left_positions) == 0 and len(right_positions) == 0:
            logger.warning(
                f"No matching keys found in feedback. Available keys: {list(feedback.keys())[:5]}..."
            )
            return

        # Send positions to leader arms to make them mirror the robot
        if len(left_positions) == 6:
            self.left_arm.command_joint_pos(np.array(left_positions))
        if len(right_positions) == 6:
            self.right_arm.command_joint_pos(np.array(right_positions))

    def enable_torque(self) -> None:
        """Put the leader into position-tracking mode (follows ``send_feedback`` goals).

        Sets non-zero PD gains scaled by ``self.bilateral_kp`` so the leader arms
        actively track commanded positions. The kp baseline is the per-joint
        natural kp fetched from the server in ``connect()``.
        """
        if self._original_kp is None:
            raise RuntimeError("Natural kp not initialized; call connect() before enable_torque().")

        kp = self._original_kp * self.bilateral_kp
        zero_gains = np.zeros(6)
        try:
            self.left_arm.update_kp_kd(kp, zero_gains)
            self.right_arm.update_kp_kd(kp, zero_gains)

            logger.info(f"Bilateral control enabled: kp={self.bilateral_kp}*original={kp[0]:.1f}")
        except Exception as e:
            logger.warning(f"Failed to enable bilateral control: {e}")

    def disable_torque(self) -> None:
        """Put the leader into free-movement mode (gravity compensation only).

        Sets PD gains to zero and commands the current position as the reference so
        the arms remain freely movable under gravity compensation.
        """
        zero_gains = np.zeros(6)
        try:
            # Update gains first (command_joint_pos copies gains into commands)
            self.left_arm.update_kp_kd(zero_gains, zero_gains)
            self.right_arm.update_kp_kd(zero_gains, zero_gains)

            # Command current position as reference for gravity compensation
            left_obs = self.left_arm.get_observations()
            right_obs = self.right_arm.get_observations()
            self.left_arm.command_joint_pos(left_obs["joint_pos"])
            self.right_arm.command_joint_pos(right_obs["joint_pos"])

            logger.info("Bilateral control disabled: arms free to move (kp=0, kd=0)")
        except Exception as e:
            logger.warning(f"Failed to disable bilateral control: {e}")

    def disconnect(self) -> None:
        """Disconnect from both leader arms."""
        logger.info("Disconnecting from bimanual Yam leader arms")

        # Release bilateral PD so the arms don't continue holding their last
        # commanded pose against the operator after the client goes away.
        if self.bilateral_kp > 0.0:
            self.disable_torque()

        self.left_arm.disconnect()
        self.right_arm.disconnect()

        logger.info("Disconnected from bimanual Yam leader arms")

#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_nero_follower import NEOFollowerRobotConfig

logger = logging.getLogger(__name__)

# NERO joint names (7 joints + 1 gripper)
NERO_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
NERO_MOTORS = NERO_JOINTS + ["gripper"]


class NEOFollower(Robot):
    """
    NERO 7-DOF robot arm with AGX Gripper, integrated via pyAgxArm SDK.

    Uses CAN bus communication (socketcan on Linux).
    All joint angles are in radians. Gripper is normalized 0-100.
    """

    config_class = NEOFollowerRobotConfig
    name = "nero_follower"

    def __init__(self, config: NEOFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        self._last_joint_targets: list[float] | None = None

        # Build pyAgxArm config
        self._arm_config = create_agx_arm_config(
            robot=ArmModel.NERO,
            comm="can",
            firmeware_version=config.firmeware_version,
            interface=config.interface,
            channel=config.channel,
            auto_set_motion_mode=config.auto_set_motion_mode,
            enable_joint_limits=config.enable_joint_limits,
        )

        # Create arm instance (do NOT connect yet)
        self._arm = AgxArmFactory.create_arm(self._arm_config)
        self._effector = None

        # Cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in NERO_MOTORS}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        try:
            return self._arm.is_connected() and all(
                cam.is_connected for cam in self.cameras.values()
            )
        except Exception:
            return False

    @property
    def is_calibrated(self) -> bool:
        # NERO has absolute encoders, always calibrated
        return True

    def calibrate(self) -> None:
        # No manual calibration needed for NERO (absolute encoders)
        logger.info(f"{self} uses absolute encoders, no calibration needed.")

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to NERO arm via CAN bus, enable motors, and initialize gripper.
        """
        logger.info(f"Connecting to {self} on {self.config.interface}:{self.config.channel}...")

        # Connect arm
        self._arm.connect()

        # Enable motors
        logger.info("Enabling NERO arm...")
        while not self._arm.enable():
            time.sleep(0.01)
        logger.info("NERO arm enabled.")

        # Initialize gripper
        logger.info("Initializing AGX Gripper...")
        self._effector = self._arm.init_effector(self._arm.OPTIONS.EFFECTOR.AGX_GRIPPER)

        # Set speed
        self._arm.set_speed_percent(self.config.speed_percent)

        # Prime cached targets from live state so partial actions don't zero unspecified joints.
        joint_msg = self._arm.get_joint_angles()
        if joint_msg is not None and joint_msg.msg is not None:
            self._last_joint_targets = [float(v) for v in list(joint_msg.msg)[: len(NERO_JOINTS)]]
            if len(self._last_joint_targets) < len(NERO_JOINTS):
                self._last_joint_targets += [0.0] * (len(NERO_JOINTS) - len(self._last_joint_targets))
        else:
            self._last_joint_targets = [0.0] * len(NERO_JOINTS)

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    def configure(self) -> None:
        """Apply runtime configuration."""
        self._arm.set_speed_percent(self.config.speed_percent)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Read joint angles (7 rad) + gripper state + camera images."""
        start = time.perf_counter()

        # Read joint angles (7 joints, radians)
        joint_msg = self._arm.get_joint_angles()
        if joint_msg is not None and joint_msg.msg is not None:
            joint_vals = list(joint_msg.msg)
        else:
            joint_vals = [0.0] * 7

        obs_dict = {}
        for i, name in enumerate(NERO_JOINTS):
            obs_dict[f"{name}.pos"] = float(joint_vals[i]) if i < len(joint_vals) else 0.0

        # Read gripper state
        if self._effector is not None:
            try:
                gripper_msg = self._effector.get_gripper_ctrl_states()
                if gripper_msg is not None and gripper_msg.msg is not None:
                    obs_dict["gripper.pos"] = float(gripper_msg.msg.value)
                else:
                    obs_dict["gripper.pos"] = 0.0
            except Exception as e:
                logger.debug(f"Failed to read gripper state: {e}")
                obs_dict["gripper.pos"] = 0.0
        else:
            obs_dict["gripper.pos"] = 0.0

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command NERO arm to move to target joint configuration.

        Joint angles in radians, gripper in 0-100 normalized.
        Returns the action actually sent (potentially clipped for safety).
        """
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        joint_goals = {k: float(v) for k, v in goal_pos.items() if k in NERO_JOINTS}

        # Ignore non-joint actions (e.g. raw keyboard chars) to avoid unintended zeroing.
        if not joint_goals and "gripper" not in goal_pos:
            return {}

        # Build a full 7-joint command from current state/cache + provided partial goals.
        if self._last_joint_targets is None:
            self._last_joint_targets = [0.0] * len(NERO_JOINTS)
        base_targets = list(self._last_joint_targets)

        present_obs = self._arm.get_joint_angles()
        if present_obs is not None and present_obs.msg is not None:
            present_vals = [float(v) for v in list(present_obs.msg)[: len(NERO_JOINTS)]]
            if len(present_vals) < len(NERO_JOINTS):
                present_vals += [0.0] * (len(NERO_JOINTS) - len(present_vals))
            base_targets = present_vals

        # Safety clipping if max_relative_target is configured
        if self.config.max_relative_target is not None and present_obs is not None and present_obs.msg is not None:
            present_dict = {name: float(val) for name, val in zip(NERO_JOINTS, list(present_obs.msg))}
            goal_present = {k: (v, present_dict[k]) for k, v in joint_goals.items() if k in present_dict}
            if goal_present:
                joint_goals = ensure_safe_goal_position(goal_present, self.config.max_relative_target)

        for i, name in enumerate(NERO_JOINTS):
            if name in joint_goals:
                base_targets[i] = joint_goals[name]

        if joint_goals:
            self._arm.move_j(base_targets)
            self._last_joint_targets = list(base_targets)

        # Send gripper command
        if "gripper" in goal_pos and self._effector is not None:
            gripper_val = goal_pos["gripper"]
            # Map 0-100 normalized to degrees (0 = closed, 100 = fully open)
            try:
                self._effector.move_gripper_deg(gripper_val)
            except Exception as e:
                logger.warning(f"Failed to send gripper command: {e}")

        sent_action = {f"{motor}.pos": val for motor, val in joint_goals.items()}
        if "gripper" in goal_pos:
            sent_action["gripper.pos"] = float(goal_pos["gripper"])
        return sent_action

    @check_if_not_connected
    def disconnect(self):
        """Disconnect from NERO arm and release resources."""
        try:
            if self.config.disable_torque_on_disconnect:
                self._arm.disable()
                logger.info("NERO arm disabled (torque off).")
        except Exception as e:
            logger.warning(f"Error disabling arm: {e}")

        try:
            self._arm.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting arm: {e}")

        for cam in self.cameras.values():
            cam.disconnect()

        self._last_joint_targets = None

        logger.info(f"{self} disconnected.")

    def setup_motors(self) -> None:
        """No motor setup needed for NERO (CAN bus auto-detects)."""
        logger.info("NERO uses CAN bus, no manual motor setup needed.")

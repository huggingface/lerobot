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

"""LeRobot driver for the LimX Dynamics TRON2 dual-arm robot.

The connection rides on TRON2's WebSocket JSON transport, provided by the
``tron2-env`` package.  Keeping the transport on a wire protocol rather than on
the vendor low-level SDK is deliberate: that SDK ships native extensions bound to
the CPython 3.8 ABI, so it cannot be imported from the Python 3.12+ environments
LeRobot runs in.

``tron2_env`` is imported lazily inside [`LimxTron2.connect`], so a missing
optional dependency surfaces as a one-line ``pip install 'lerobot[limx_tron2]'``
hint at connect time rather than an ImportError at import time.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.import_utils import require_package

from ..robot import Robot
from .config_limx_tron2 import LimxTron2Config
from .joints import gripper_openings, pos_key, servoj_vector, split_state

logger = logging.getLogger(__name__)


class LimxTron2(Robot):
    """Drive a TRON2 dual-arm robot through its WebSocket controller.

    Joint targets are servo positions in radians.  Grippers are only exposed when ``include_grippers`` is
    set on the config; see [`LimxTron2Config.include_grippers`].

    **Attributes**:
        - **config_class** (`type[LimxTron2Config]`) -- The configuration class this robot reads.
        - **name** (`str`) -- The robot type used on the command line, `"limx_tron2"`.
    """

    config_class = LimxTron2Config
    name = "limx_tron2"

    def __init__(self, config: LimxTron2Config):
        super().__init__(config)
        self.config = config
        self._controller: Any | None = None

    # ------------------------------------------------------------------ setup

    @property
    def is_connected(self) -> bool:
        """Whether the controller currently reports a live connection."""
        return self._controller is not None and bool(self._controller.is_connected())

    @property
    def is_calibrated(self) -> bool:
        """TRON2 bring-up and zeroing happen in the robot controller, not here."""
        return True

    def calibrate(self) -> None:
        """No-op: calibration is owned by the robot controller."""

    def configure(self) -> None:
        """Raise any asynchronous fault retained by the controller.

        The motion controller latches faults (a rejected head or end-effector
        command, for instance) and only surfaces them at the next command, so
        callers get them here instead of deep inside a rollout.
        """
        controller = self._require_controller()
        controller.raise_if_faulted()

    def connect(self, calibrate: bool = True) -> None:
        """Open the connection to the robot-side TRON2 controller.

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Accepted for interface compatibility.  TRON2 zeroes itself in the robot controller, so this
                only calls [`LimxTron2.calibrate`], which does nothing.

        Raises:
            ConnectionError: If the controller does not report a live connection.
        """
        if self.is_connected:
            return

        controller = self._make_controller()
        controller.start()

        if not controller.is_connected():
            controller.disconnect()
            raise ConnectionError(
                f"TRON2 controller at ws://{self.config.robot_ip}:{self.config.port} "
                "did not report a live connection"
            )

        self._controller = controller
        if calibrate:
            # Nothing to do, but keep the hook so subclasses can override.
            self.calibrate()

    def _make_controller(self):
        """Build the TRON2 motion controller for this configuration.

        Kept as a separate method so that tests -- and hardware-free dry runs --
        can substitute an in-memory transport instead of opening a WebSocket.
        """
        require_package("tron2-env", extra="limx_tron2", import_name="tron2_env")

        from tron2_env import Tron2Config, create_motion_controller

        transport_config = Tron2Config(
            robot_ip=self.config.robot_ip,
            port=self.config.port,
            init_joints=self.config.init_joints,
            init_head=self.config.init_head,
            init_ee_z_min=self.config.init_ee_z_min,
        )

        logger.info(
            "Connecting to TRON2 at ws://%s:%s (publish_rate=%s Hz)",
            self.config.robot_ip,
            self.config.port,
            self.config.publish_rate,
        )
        return create_motion_controller(
            transport_config,
            backend="websocket",
            publish_rate=self.config.publish_rate,
            eta_default=self.config.eta_default,
        )

    def disconnect(self) -> None:
        """Close the controller connection, swallowing teardown errors."""
        controller, self._controller = self._controller, None
        if controller is None:
            return
        try:
            controller.disconnect()
        except Exception:  # noqa: BLE001 - cleanup must not mask the original error
            logger.exception("Error while disconnecting from TRON2")

    # --------------------------------------------------------------- features

    def _scalar_features(self) -> dict[str, type]:
        """Per-joint scalar features, shared by the observation and action space."""
        features: dict[str, type] = {pos_key(name): float for name in self.config.joint_names}
        if self.config.include_grippers:
            features.update({pos_key(name): float for name in self.config.gripper_names})
        return features

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        """Per-joint scalar features the robot reports."""
        return self._scalar_features()

    @property
    def action_features(self) -> dict[str, type]:
        """Per-joint scalar features the robot accepts."""
        return self._scalar_features()

    # ------------------------------------------------------------------- i/o

    def get_observation(self) -> RobotObservation:
        """Read one state sample and map it onto LeRobot's feature keys.

        Returns:
            `RobotObservation`: Joint positions in radians, keyed by ``"<joint>.pos"``, plus the gripper
            openings when `include_grippers` is set.

        Raises:
            RuntimeError: If the robot is not connected, or the payload carries no `states` field.
        """
        controller = self._require_controller()
        payload = controller.get_joint_states(
            timeout=self.config.observation_timeout,
            max_age=self.config.state_max_age,
        )

        states = payload.get("states")
        if states is None:
            raise RuntimeError(f"TRON2 state payload is missing the 'states' field (keys: {sorted(payload)})")

        left_arm, left_gripper, right_arm, right_gripper, head = split_state(states)
        values = np.concatenate((left_arm, right_arm, head))

        observation: RobotObservation = {
            pos_key(name): float(value) for name, value in zip(self.config.joint_names, values, strict=True)
        }
        if self.config.include_grippers:
            observation[pos_key(self.config.gripper_names[0])] = left_gripper
            observation[pos_key(self.config.gripper_names[1])] = right_gripper
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        """Send a joint command to the robot.

        Args:
            action (`RobotAction`):
                Target joint positions in radians, keyed by ``"<joint>.pos"``.  Gripper keys are read only
                when `include_grippers` is set.

        Returns:
            `RobotAction`: The action that was sent.  The controller interpolates towards the setpoint
            rather than echoing it, so the setpoint is the closest honest answer to "what was sent".

        Raises:
            KeyError: If a joint key is missing, or only one side of the gripper pair is present.
        """
        controller = self._require_controller()

        servoj = servoj_vector(action, self.config.joint_names)
        controller.command_joints(servoj)

        if self.config.include_grippers:
            openings = gripper_openings(action, self.config.gripper_names)
            if openings is not None:
                left, right = openings
                controller.command_end_effector(np.asarray([left]), np.asarray([right]))

        # The controller interpolates towards the setpoint rather than echoing
        # it, so the setpoint is the closest honest answer to "what was sent".
        return action

    # ---------------------------------------------------------------- helpers

    def _require_controller(self):
        """Return the live controller, or raise if the robot is not connected."""
        if self._controller is None:
            raise RuntimeError(
                "TRON2 robot is not connected - call connect() first "
                "(LeRobot does this for you via the Robot context manager)"
            )
        return self._controller


__all__ = ["LimxTron2"]

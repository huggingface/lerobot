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

"""Thread-safe robot wrapper for concurrent observation/action access."""

from __future__ import annotations

from threading import Lock
from typing import Any

from lerobot.robots import Robot


class ThreadSafeRobot:
    """Lock-protected wrapper around a :class:`Robot` for use with background threads.

    When RTC inference runs in a background thread while the main loop
    executes actions, both threads may access the robot concurrently.
    This wrapper serialises ``get_observation`` and ``send_action`` calls.

    Read-only properties are proxied without the lock since they don't
    mutate hardware state.
    """

    def __init__(self, robot: Robot) -> None:
        self._robot = robot
        self._lock = Lock()

    # -- Lock-protected I/O --------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            return self._robot.get_observation()


    def _validate_joint_limits_locked(self, action):
        """Validate SO101 follower joint targets under the existing lock."""
        from dataclasses import asdict, is_dataclass

        from lerobot.rollout.joint_limit_guard import JointLimitGuard

        robot = self._robot

        if getattr(robot, "robot_type", None) not in {"so100_follower", "so101_follower"}:
            return

        config = getattr(robot, "config", None)

        if config is None or getattr(config, "use_degrees", None) is not True:
            raise RuntimeError(
                "SO101 joint guard requires degree mode"
            )

        calibration = getattr(robot, "calibration", None)

        if not isinstance(calibration, dict):
            raise RuntimeError(
                "SO101 calibration unavailable"
            )

        calibration_dict = {}

        for joint in (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ):
            if joint not in calibration:
                raise RuntimeError(
                    f"Missing SO101 calibration: {joint}"
                )

            item = calibration[joint]

            if is_dataclass(item) and not isinstance(item, type):
                calibration_dict[joint] = asdict(item)
            elif isinstance(item, dict):
                calibration_dict[joint] = item.copy()
            else:
                raise RuntimeError(
                    f"Invalid calibration object: {joint}"
                )

        guard = JointLimitGuard(calibration_dict)

        guard.validate(action)

    def send_action(self, action: dict[str, Any] | Any) -> Any:
        with self._lock:
            guard = getattr(self, "_gripper_safety_guard", None)

            if guard is not None and guard.fault_latched:
                from lerobot.rollout.gripper_safety import GripperSafetyError

                raise GripperSafetyError(
                    "GRIPPER LOCKED: robot action dispatch blocked"
                )

            self._validate_joint_limits_locked(action)
            return self._robot.send_action(action)

    def send_action_checked(self, action):
        """Optional SO-ARM101 gripper feedback guard.

        This method is not enabled by ordinary send_action().
        """
        from lerobot.rollout.gripper_safety import (
            GripperStatus,
            GripperSafetyGuard,
            GripperSafetyError,
        )

        with self._lock:
            robot = self._robot

            if getattr(robot, "robot_type", None) not in {"so100_follower", "so101_follower"}:
                raise RuntimeError(
                    "Checked dispatch supports SO101 follower only"
                )

            if not hasattr(robot, "bus"):
                raise RuntimeError("Robot motor bus unavailable")

            if not hasattr(self, "_gripper_safety_guard"):
                self._gripper_safety_guard = GripperSafetyGuard()

            guard = self._gripper_safety_guard

            if guard.fault_latched:
                raise GripperSafetyError("GRIPPER LOCKED")

            try:
                status = GripperStatus(
                    load=robot.bus.read(
                        "Present_Load", "gripper", normalize=False
                    ),
                    current=robot.bus.read(
                        "Present_Current", "gripper", normalize=False
                    ),
                    temperature=robot.bus.read(
                        "Present_Temperature", "gripper", normalize=False
                    ),
                    voltage=robot.bus.read(
                        "Present_Voltage", "gripper", normalize=False
                    ),
                )
            except Exception as exc:
                guard._fault(
                    f"Gripper feedback read failure: {exc!r}"
                )

            guard.check(status)

            # Reject invalid joint targets before attempting dispatch.
            # JointLimitError does not represent a gripper hardware fault.
            self._validate_joint_limits_locked(action)

            try:
                return robot.send_action(action)
            except Exception as exc:
                guard._fault(
                    f"Robot action dispatch failure: {exc!r}"
                )


    # -- Read-only proxies (no lock needed) -----------------------------------

    @property
    def observation_features(self) -> dict:
        return self._robot.observation_features

    @property
    def action_features(self) -> dict:
        return self._robot.action_features

    @property
    def name(self) -> str:
        return self._robot.name

    @property
    def robot_type(self) -> str:
        return self._robot.robot_type

    @property
    def cameras(self):
        return getattr(self._robot, "cameras", {})

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected

    @property
    def inner(self) -> Robot:
        """Access the underlying robot (e.g. for connect/disconnect)."""
        return self._robot

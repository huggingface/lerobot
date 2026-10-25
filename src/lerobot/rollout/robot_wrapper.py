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

import math
import time
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
        self._position_hold_enabled = False
        self._observed_positions: dict[str, float] = {}
        self._last_applied_action: dict[str, float] | None = None
        self._observation_time: float | None = None
        self._hardware_failure: str | None = None

    # -- Lock-protected I/O --------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            sampled_at = time.monotonic()
            observation = self._robot.get_observation()
            self._observed_positions = {
                key: float(observation[key])
                for key in self.action_features
                if key.endswith(".pos") and key in observation
            }
            self._observation_time = sampled_at
            return observation

    @property
    def observation_time(self) -> float | None:
        """Client monotonic sample bound for the last read, before hardware/camera waits."""
        with self._lock:
            return self._observation_time


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
        # SO-100/SO-101 actions must pass the checked safety path.
        # Dispatch before acquiring this method's lock to avoid nested-lock deadlock.
        if getattr(self._robot, "robot_type", None) in {
            "so100_follower",
            "so101_follower",
        }:
            return self.send_action_checked(action)

        with self._lock:
            try:
                applied_action = self._robot.send_action(action)
                if self._position_hold_enabled:
                    self._last_applied_action = self._validated_positions(applied_action)
                return applied_action
            except Exception as exc:
                self._record_hardware_failure("send_action", exc)
                raise

    def _record_hardware_failure(self, operation: str, error: Exception) -> None:
        # Caller owns the I/O lock. Failed commands prohibit further shutdown movement.
        if self._hardware_failure is None:
            self._hardware_failure = f"{operation}: {type(error).__name__}: {error}"

    @property
    def hardware_failure(self) -> str | None:
        """First command/hold failure; never cleared by inference reset."""
        with self._lock:
            return self._hardware_failure

    def _validated_positions(self, action: Any) -> dict[str, float]:
        if not isinstance(action, dict) or set(action) != set(self.action_features):
            raise RuntimeError("Position hold requires the applied position target for every actuator")
        positions = {key: float(value) for key, value in action.items()}
        if not all(math.isfinite(value) for value in positions.values()):
            raise RuntimeError("Position hold requires finite position targets for every actuator")
        return positions

    def configure_position_hold(self) -> None:
        """Enable the explicit, robot-supported position hold; reject mixed control modes."""
        if (
            not self._robot.supports_position_hold
            or not self.action_features
            or any(not key.endswith(".pos") for key in self.action_features)
        ):
            raise ValueError(f"{self.robot_type} has no supported local position-hold contract")
        self._position_hold_enabled = True

    @property
    def supports_hold(self) -> bool:
        return self._position_hold_enabled

    @property
    def supports_position_hold(self) -> bool:
        """Whether this robot declares position-target retention support."""
        return self._robot.supports_position_hold

    def hold(self) -> None:
        """Refresh the last applied position targets without camera reads or inference.

        Driver-returned targets include interpolation and safety clipping. Before
        any command, use the last finite measured pose. The robot may still settle
        toward these targets; this is not an instantaneous physical stop.
        """
        from os import environ
        with self._lock:
            if getattr(self._robot, "robot_type", None) in {"so100_follower", "so101_follower"} and environ.get("SO101_HARDWARE_APPROVED") != "1":
                raise RuntimeError("SO101 safety gate: hardware dispatch disabled")
            if self._hardware_failure is not None:
                raise RuntimeError(f"Hardware failure latched: {self._hardware_failure}")
            if not self._position_hold_enabled:
                raise RuntimeError("Local hold was not configured for this robot")
            try:
                target = self._last_applied_action
                if target is None:
                    target = self._validated_positions(self._observed_positions)
                self._validate_joint_limits_locked(target)
                if getattr(self._robot, "robot_type", None) in {
                    "so100_follower",
                    "so101_follower",
                }:
                    self._check_gripper_feedback_locked(self._robot)
                applied_action = self._robot.send_action(target.copy())
                self._last_applied_action = self._validated_positions(applied_action)
            except Exception as exc:
                self._record_hardware_failure("hold", exc)
                raise

    def _check_gripper_feedback_locked(self, robot):
        """Read and validate gripper feedback while the robot I/O lock is held."""
        from lerobot.rollout.gripper_safety import (
            GripperSafetyError,
            GripperSafetyGuard,
            GripperStatus,
        )

        if not hasattr(self, "_gripper_safety_guard"):
            self._gripper_safety_guard = GripperSafetyGuard()

        guard = self._gripper_safety_guard

        if guard.fault_latched:
            raise GripperSafetyError(
                f"GRIPPER LOCKED: {guard.fault_reason}"
            )

        if not hasattr(robot, "bus"):
            guard._fault("Robot motor bus unavailable")

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
        return guard

    def send_action_checked(self, action):
        """Dispatch SO-100/SO-101 actions only after safety checks."""
        from os import environ

        with self._lock:
            robot = self._robot

            if getattr(robot, "robot_type", None) not in {
                "so100_follower",
                "so101_follower",
            }:
                raise RuntimeError(
                    "Checked dispatch supports SO-100/SO-101 followers only"
                )

            if environ.get("SO101_HARDWARE_APPROVED") != "1":
                raise RuntimeError(
                    "SO101 safety gate: hardware dispatch disabled"
                )

            if self._hardware_failure is not None:
                raise RuntimeError(
                    f"Hardware failure latched: {self._hardware_failure}"
                )

            guard = self._check_gripper_feedback_locked(robot)

            # Reject invalid joint targets before attempting dispatch.
            # JointLimitError does not represent a gripper hardware fault.
            self._validate_joint_limits_locked(action)

            try:
                applied_action = robot.send_action(action)
                if self._position_hold_enabled:
                    self._last_applied_action = self._validated_positions(applied_action)
                return applied_action
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

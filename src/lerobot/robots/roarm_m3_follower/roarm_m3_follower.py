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

"""Waveshare RoArm-M3 follower arm (6-DOF incl. gripper, JSON-over-serial via ESP32).

Motor communication goes through the ``roarm_sdk`` package over a USB serial link to the
arm's on-board ESP32 (JSON commands), not a Feetech/Dynamixel binary bus. This is a thin
``Robot`` subclass that wraps the SDK directly (no MotorsBus). Because each SDK call blocks
~35-50 ms, all serial I/O runs on a background ``AsyncArmWorker`` so the synchronous
record/eval loop stays fast: ``get_observation`` reads a cache and ``send_action`` enqueues
the write, both ~0 ms.

Joint order is fixed and defines the dataset schema (see ``roarm_m3_common.JOINT_NAMES``):
``[base, shoulder, elbow, wrist_tilt, wrist_roll, gripper]``.

Gripper B (optional, ``force_control_gripper``): the Waveshare Gripper B uses a CF-3512
constant-current servo. The bundled "move all joints" command zeroes its torque registers,
so when force control is enabled the worker issues a second, gripper-only write to restore
the clamping force. With it off (the default, for the stock gripper / Gripper A) a single
position-only write is sent.
"""

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _roarm_sdk_available, require_package

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .async_worker import AsyncArmWorker
from .config_roarm_m3_follower import RoarmM3FollowerConfig
from .roarm_m3_common import JOINT_NAMES, clamp_limits, disable_info_flow, quantize_action

if TYPE_CHECKING or _roarm_sdk_available:
    from roarm_sdk.roarm import roarm
else:
    roarm = None

logger = logging.getLogger(__name__)


class RoarmM3Follower(Robot):
    """Single Waveshare RoArm-M3 follower arm + its USB cameras, as a LeRobot Robot."""

    config_class = RoarmM3FollowerConfig
    name = "roarm_m3_follower"

    def __init__(self, config: RoarmM3FollowerConfig):
        require_package("roarm_sdk", extra="roarm")
        super().__init__(config)
        self.config = config
        self._arm = None  # roarm SDK handle, created in connect()
        self._worker: AsyncArmWorker | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in JOINT_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._arm is not None and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting {self} on {self.config.port}...")
        self._arm = roarm(roarm_type="roarm_m3", port=self.config.port, baudrate=self.config.baudrate)
        # Probe the serial link (raises if the port is wrong / contended).
        self._arm.joints_angle_get()

        # Kill the firmware's continuous feedback stream so reads stay fresh.
        if self.config.disable_info_flow:
            disable_info_flow(self._arm)

        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()  # enable follower torque

        # One background thread owns ALL serial I/O so a read and a write never race. The
        # gripper-B force double-write is opt-in via force_control_gripper.
        self._worker = AsyncArmWorker(
            self._arm,
            f"follower_{self.id}",
            read_interval_s=0.15,
            force_control_gripper=self.config.force_control_gripper,
        )
        self._worker.start()
        time.sleep(0.20)  # one read cycle to populate the position cache
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # The RoArm-M3 has no per-robot motor-offset calibration file here.
        return True

    def calibrate(self) -> None:
        # No-op: no per-robot motor-offset calibration.
        pass

    def configure(self) -> None:
        """Enable follower torque (active, holds position)."""
        if self._arm is not None:
            self._arm.torque_set(cmd=1)

    @check_if_not_connected
    def disable_torque(self) -> None:
        """Release follower torque so the arm can be moved by hand (drag-teach /
        debugging). Call configure() to re-enable it. The background reader keeps the
        position cache updating, so get_observation tracks the manual motion."""
        self._arm.torque_set(cmd=0)
        logger.info(f"{self} torque disabled.")

    def _read_joint_positions(self) -> np.ndarray:
        """6 joint angles (deg) from the worker cache (~0 ms), with a blocking fallback."""
        cached = self._worker.get_pos() if self._worker is not None else None
        if cached is None:
            cached = np.array(self._arm.joints_angle_get(), dtype=np.float32)
        return np.asarray(cached, dtype=np.float32)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        pos = self._read_joint_positions()
        obs_dict: RobotObservation = {f"{n}.pos": float(pos[i]) for i, n in enumerate(JOINT_NAMES)}
        logger.debug(f"{self} read state: {(time.perf_counter() - start) * 1e3:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            logger.debug(f"{self} read {cam_key}: {(time.perf_counter() - start) * 1e3:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Enqueue a joint goal (degrees); return the action actually sent.

        Reads goals in the fixed joint order (missing keys hold the present position),
        optionally caps the relative step via ``max_relative_target``, quantizes per
        ``action_quantization`` and clamps to the joint limits, then enqueues on the worker
        (the worker performs the bundled write plus, if ``force_control_gripper`` is set,
        the gripper-only force write).
        """
        present = self._read_joint_positions()
        goal_pos = {n: float(action.get(f"{n}.pos", present[i])) for i, n in enumerate(JOINT_NAMES)}

        # Cap relative target when too far from the present position.
        if self.config.max_relative_target is not None:
            present_pos = {n: float(present[i]) for i, n in enumerate(JOINT_NAMES)}
            goal_present_pos = {n: (g, present_pos[n]) for n, g in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        goal = [goal_pos[n] for n in JOINT_NAMES]
        goal = quantize_action(goal, self.config.action_quantization).tolist()
        goal = clamp_limits(goal)

        self._worker.write(goal, self.config.motion_speed, self.config.motion_acc)
        return {f"{n}.pos": float(goal[i]) for i, n in enumerate(JOINT_NAMES)}

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

        if self.config.disable_torque_on_disconnect:
            try:
                self._arm.torque_set(cmd=0)
            except Exception as exc:  # pragma: no cover
                logger.debug(f"torque_set(0) on disconnect failed: {exc}")
        try:
            self._arm.disconnect()
        except Exception as exc:  # pragma: no cover
            logger.debug(f"arm.disconnect() failed: {exc}")
        self._arm = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

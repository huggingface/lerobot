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

"""Waveshare RoArm-M3 used as a passive leader (Teleoperator).

A second RoArm-M3 is held torque-off and moved by hand; its joint angles drive a
``roarm_m3_follower``. Like the follower it talks JSON-over-serial through its ESP32
(``roarm_sdk``), not a Feetech/Dynamixel binary bus, so reads go through a background
poller (``AsyncArmReader``) to keep the synchronous teleop loop fast.

``get_action`` returns positions in the follower's joint space (same fixed joint order, in
degrees): joints 0-4 pass through (the two RoArm-M3 arms share the same kinematics), the
raw read is EMA-smoothed to damp gearbox jitter, the gripper is optionally remapped to the
follower's range (``gripper_remap``, for heterogeneous grippers such as a Gripper B), and
the result is floor-quantized + clamped so the recorded action label equals what the
follower executes (ACT-space alignment).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _roarm_sdk_available, require_package

from ..teleoperator import Teleoperator
from .async_reader import AsyncArmReader
from .config_roarm_m3_leader import RoarmM3LeaderConfig

if TYPE_CHECKING or _roarm_sdk_available:
    from roarm_sdk.roarm import roarm
else:
    roarm = None

logger = logging.getLogger(__name__)

# --- Constants/helpers DUPLICATED from
# lerobot.robots.roarm_m3_follower.roarm_m3_common. No teleoperator imports a robots
# subpackage (same choice as reachy2). They MUST stay in sync with the follower; the test
# suite asserts they agree. ---

# Fixed joint order = the dataset schema.
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_tilt", "wrist_roll", "gripper"]
WRIST_R_LIMIT = 90
ANGLES_MIN = [-190, -110, -70, -110, -WRIST_R_LIMIT, -10]
ANGLES_MAX = [190, 110, 190, 110, WRIST_R_LIMIT, 300]


def quantize_action(goal_pos, mode: str = "floor") -> np.ndarray:
    """Floor/round/float quantization. See roarm_m3_follower.roarm_m3_common."""
    arr = goal_pos.numpy() if hasattr(goal_pos, "numpy") else np.asarray(goal_pos)
    if mode == "float":
        return arr.astype(np.float64)
    if mode == "round":
        return np.rint(arr).astype(np.int32)
    return arr.astype(np.int32)


def clamp_limits(goal_pos: list) -> list:
    """Clamp each joint to [ANGLES_MIN, ANGLES_MAX]."""
    return [max(ANGLES_MIN[i], min(ANGLES_MAX[i], v)) for i, v in enumerate(goal_pos)]


def disable_info_flow(arm) -> bool:
    """Stop the firmware's continuous feedback stream so reads stay fresh."""
    ser = getattr(arm, "_serial_port", None)
    if ser is None or not (hasattr(ser, "write") and hasattr(ser, "reset_input_buffer")):
        logger.warning(
            "RoArm serial handle (_serial_port) not found; cannot disable the firmware "
            "info stream. State reads may be stale."
        )
        return False
    try:
        ser.write(b'{"T":605,"cmd":0}\n')
        time.sleep(0.1)
        ser.reset_input_buffer()
        return True
    except Exception as exc:
        logger.warning(f"Failed to disable the RoArm firmware info stream: {exc}")
        return False


class RoarmM3Leader(Teleoperator):
    """A passive Waveshare RoArm-M3 leader arm, as a LeRobot Teleoperator."""

    config_class = RoarmM3LeaderConfig
    name = "roarm_m3_leader"

    def __init__(self, config: RoarmM3LeaderConfig):
        require_package("roarm_sdk", extra="roarm")
        super().__init__(config)
        self.config = config
        self._arm = None  # roarm SDK handle, created in connect()
        self._reader: AsyncArmReader | None = None
        self._ema: np.ndarray | None = None  # EMA state over the 6 raw leader angles

    @property
    def action_features(self) -> dict[str, type]:
        # Same schema as RoarmM3Follower.action_features.
        return {f"{motor}.pos": float for motor in JOINT_NAMES}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._arm is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting {self} on {self.config.port}...")
        self._arm = roarm(roarm_type="roarm_m3", port=self.config.port, baudrate=self.config.baudrate)
        # Probe the serial link (raises if the port is wrong / contended).
        self._arm.joints_angle_get()

        if self.config.disable_info_flow:
            disable_info_flow(self._arm)

        # Passive: torque off so the operator moves the arm by hand.
        self.configure()

        if not self.is_calibrated and calibrate:
            self.calibrate()

        self._reader = AsyncArmReader(self._arm, f"leader_{self.id}")
        self._reader.start()
        time.sleep(0.20)  # one read cycle to populate the cache
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # No per-leader motor-offset calibration file here.
        return True

    def calibrate(self) -> None:
        # No-op: no per-leader motor-offset calibration.
        pass

    def configure(self) -> None:
        """Put the leader in passive (torque-off) mode so it can be moved by hand."""
        if self._arm is not None:
            self._arm.torque_set(cmd=0)

    def _read_raw(self) -> np.ndarray:
        """6 leader joint angles (deg) from the reader cache, with a blocking fallback."""
        cached = self._reader.get_pos() if self._reader is not None else None
        if cached is None:
            cached = np.array(self._arm.joints_angle_get(), dtype=np.float32)
        return np.asarray(cached, dtype=np.float32)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()
        raw = np.asarray(self._read_raw()[:6], dtype=np.float32)

        # 1. EMA smoothing over the raw 6-vector to damp gearbox jitter.
        alpha = self.config.ema_alpha
        smoothed = raw if self._ema is None else alpha * raw + (1.0 - alpha) * self._ema
        self._ema = smoothed
        goal = smoothed.tolist()

        # 2. Gripper: pass through by default (stock/identical grippers - the common case).
        #    Only when gripper_remap is set do we remap the leader range to the follower
        #    range with a quadratic curve (heterogeneous grippers, e.g. a Gripper B).
        c = self.config
        if c.gripper_remap:
            span = c.gripper_leader_close - c.gripper_leader_open
            ratio = (goal[5] - c.gripper_leader_open) / span if abs(span) >= 1 else 0.0
            ratio = max(0.0, min(1.0, ratio))
            goal[5] = c.gripper_follower_open + ratio * ratio * (
                c.gripper_follower_close - c.gripper_follower_open
            )

        # 3. Floor quantize (ACT space), then 4. clamp limits - same as the follower.
        goal = quantize_action(goal, c.action_quantization).tolist()
        goal = clamp_limits(goal)

        action = {f"{name}.pos": float(goal[i]) for i, name in enumerate(JOINT_NAMES)}
        logger.debug(f"{self} read action: {(time.perf_counter() - start) * 1e3:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("Feedback is not implemented for the RoArm-M3 leader.")

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        try:
            self._arm.disconnect()
        except Exception as exc:  # pragma: no cover
            logger.debug(f"arm.disconnect() failed: {exc}")
        self._arm = None
        logger.info(f"{self} disconnected.")

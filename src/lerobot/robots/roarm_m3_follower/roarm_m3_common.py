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

"""Shared constants and helpers for the RoArm-M3 follower and leader.

The dataset schema (``JOINT_NAMES``), the per-joint limits, the floor quantization and
the stale-state handling must be identical on the follower (Robot) and the leader
(Teleoperator), or recorded action labels would not match what the robot executes.
This module is the single source for the follower side; the leader keeps its own copy of
the small constant set (no teleoperator imports a robots subpackage) and the test suite
asserts they agree.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

# Fixed joint order = the dataset schema. Do not reorder.
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_tilt", "wrist_roll", "gripper"]

# Per-joint angle limits in degrees, in JOINT_NAMES order. Wrist-roll (index 4) is
# restricted to +-90 deg to protect the gripper cables.
WRIST_R_LIMIT = 90
ANGLES_MIN = [-190, -110, -70, -110, -WRIST_R_LIMIT, -10]
ANGLES_MAX = [190, 110, 190, 110, WRIST_R_LIMIT, 300]


def quantize_action(goal_pos, mode: str = "floor") -> np.ndarray:
    """Quantize a joint goal vector before sending it to the firmware.

    The firmware takes integer degrees, which makes a 1-degree staircase on slow
    trajectories. ``mode`` is set from the config (no hidden global state):

      "floor" (default): astype(int32) truncates toward zero -> 1-degree step with a
              systematic downward bias (73.8 -> 73). This is the proven path.
      "round": np.rint then int32 -> removes the downward bias, keeps the 1-degree step.
      "float": send float degrees -> removes the staircase IF the firmware accepts floats
               (validate with a roundtrip on the robot before relying on it).

    Applied identically on the follower (send_action) and the leader (get_action) so the
    recorded action label matches what the robot executes (ACT-space alignment).
    """
    arr = goal_pos.numpy() if hasattr(goal_pos, "numpy") else np.asarray(goal_pos)
    if mode == "float":
        return arr.astype(np.float64)
    if mode == "round":
        return np.rint(arr).astype(np.int32)
    return arr.astype(np.int32)  # "floor" = proven default


def clamp_limits(goal_pos: list) -> list:
    """Clamp each joint to [ANGLES_MIN, ANGLES_MAX]."""
    return [max(ANGLES_MIN[i], min(ANGLES_MAX[i], v)) for i, v in enumerate(goal_pos)]


def disable_info_flow(arm) -> bool:
    """Stop the firmware's continuous feedback stream so reads stay fresh.

    Sends ``{"T":605,"cmd":0}`` once on the arm's serial handle. Firmware shipped with the
    feedback stream on streams frames non-stop, the OS RX buffer fills, and every
    ``joints_angle_get()`` then returns a stale (FIFO) reading -- adding seconds of
    control-loop lag. Returns True on success; logs a warning and returns False if the
    serial handle is missing or the write fails (callers should treat False as
    "stale-read mitigation not applied").
    """
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

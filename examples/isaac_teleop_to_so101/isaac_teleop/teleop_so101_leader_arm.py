#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""SO-101 leader-arm device for NVIDIA Isaac Teleop, exposed to LeRobot.

The leader is a back-drivable SO-101 whose six joint angles are streamed (in radians) by
the native ``so101_leader`` plugin; this device reads them via a ``JointStateSource`` and
converts them into follower-ready ``{joint}.pos``. Same kinematics as the follower, so it
needs no retargeting — a 1:1 joint mirror, direct joint drive.

Units (converted in the device so the output is always follower-valid):

* arm joints: ``rad2deg`` — correct only if the leader's calibrated zero and the follower's
  homing map to the same physical zero (the standard same-hardware assumption).
* gripper: normalized from ``[gripper_open_rad, gripper_close_rad]`` to RANGE_0_100.

``isaacteleop`` imports are guarded behind the availability flag so this module — and the
pure :func:`leader_joints_to_robot_action` converter — import without it (construction
fails fast via the base class).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lerobot.types import RobotAction

from .base import _GRIPPER_MOTOR_SCALE, IsaacTeleopTeleoperator, _isaacteleop_available
from .config_isaac_teleop import SO101LeaderArmConfig

if TYPE_CHECKING or _isaacteleop_available:
    from isaacteleop.retargeting_engine.deviceio_source_nodes import JointStateSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner
else:
    JointStateSource = None
    OutputCombiner = None

# Canonical SO-101 DOF names and order — matches the plugin stream and the follower's motor
# order. Passed to the ``JointStateSource`` as its output layout; the source maps by name and
# :func:`_joints_group_to_rad` reads back by name, so a layout mismatch can't mislabel a DOF.
SO101_LEADER_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def leader_joints_to_robot_action(
    joints_rad: dict[str, float],
    *,
    gripper_joint: str,
    gripper_open_rad: float,
    gripper_close_rad: float,
) -> RobotAction:
    """Convert streamed leader joint angles [rad] to follower-ready ``{joint}.pos``.

    Pure (no ``isaacteleop``, no I/O). Iteration follows ``joints_rad`` insertion order, so
    pass it in :data:`SO101_LEADER_JOINTS` order for a stable layout. Arm joints are
    converted ``rad2deg``; ``gripper_joint`` is normalized from
    ``[gripper_open_rad, gripper_close_rad]`` to RANGE_0_100 (clipped).
    """
    action: RobotAction = {}
    span = gripper_close_rad - gripper_open_rad
    for name, rad in joints_rad.items():
        if name == gripper_joint:
            # Closedness c=0 at open, c=1 at closed; invert to the follower's 100=open jaw.
            closedness = 0.0 if span == 0.0 else (rad - gripper_open_rad) / span
            closedness = min(1.0, max(0.0, closedness))
            action[f"{name}.pos"] = (1.0 - closedness) * _GRIPPER_MOTOR_SCALE
        else:
            action[f"{name}.pos"] = float(np.rad2deg(rad))
    return action


def _joints_group_to_rad(joints) -> dict[str, float]:
    """Read a ``JointStateSource`` output group into ``{joint_name: angle [rad]}``.

    Pure (duck-typed on the group). The group is positional but each slot carries its joint
    name in ``group.group_type.types``; we key off those names (not a positional index) so a
    layout mismatch surfaces as a wrong/missing key here rather than a mislabeled DOF.
    """
    names = [t.name for t in joints.group_type.types]
    return {name: float(joints[i]) for i, name in enumerate(names)}


class SO101LeaderArm(IsaacTeleopTeleoperator):
    """SO-101 leader-arm teleoperator (joint-space), direct joint mirror to the follower.

    Reads the six joint angles off a single ``JointStateSource`` each frame; no retargeter,
    no clutch. When the leader is not streaming, :meth:`get_action` returns the held-last
    joints and :attr:`is_tracking` is ``False`` so the owning loop can hold the follower.
    """

    config_class = SO101LeaderArmConfig
    name = "isaac_teleop_so101_leader"

    def __init__(self, config: SO101LeaderArmConfig):
        super().__init__(config)
        self.config: SO101LeaderArmConfig = config
        # Held-last joint angles [rad], seeded at zero (URDF/home pose) so the first frames
        # before the plugin starts pushing read as the home pose, not garbage.
        self._last_joints_rad: dict[str, float] = dict.fromkeys(SO101_LEADER_JOINTS, 0.0)
        # Whether the most recent get_action() read live leader data (vs held-last).
        self._is_tracking = False

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> OutputCombiner:
        """Build the joint-mirror pipeline: a single ``JointStateSource`` leaf that converts
        the raw stream into a name-keyed joint group. No retargeter (shared kinematics)."""
        source = JointStateSource(
            name="so101_leader",
            collection_id=self.config.collection_id,
            joint_names=SO101_LEADER_JOINTS,
        )
        return OutputCombiner({"joints": source.output(JointStateSource.JOINTS)})

    # ------------------------------------------------------------------
    # Action features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict[str, type]:
        # Matches the serial SOLeader's action features so this is a drop-in joint-space
        # leader: one float `{joint}.pos` per DOF, sendable straight to an SO-101 follower.
        return {f"{name}.pos": float for name in SO101_LEADER_JOINTS}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_tracking(self) -> bool:
        """Whether the last :meth:`get_action` read live leader data (vs held-last)."""
        return self._is_tracking

    # ------------------------------------------------------------------
    # Action extraction
    # ------------------------------------------------------------------

    def get_action(self) -> RobotAction:
        """Step the session and return the leader joints as follower-ready ``{joint}.pos``.

        When the leader is streaming, the live angles are cached and converted; otherwise the
        held-last angles are reused and :attr:`is_tracking` is set ``False``.
        """
        result = self._step(execution_events=self._running_events())

        joints = result["joints"]
        # The JointStateSource output is Optional: absent (is_none) when the device is
        # inactive. Treat that as "not tracking" and reuse the held-last angles.
        self._is_tracking = not getattr(joints, "is_none", False)
        if self._is_tracking:
            try:
                self._last_joints_rad = _joints_group_to_rad(joints)
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                # A partially-populated / malformed group on an odd frame: keep held-last, but
                # report it as not-tracking so the loop holds the follower rather than trusting it.
                self._is_tracking = False

        return leader_joints_to_robot_action(
            self._last_joints_rad,
            gripper_joint="gripper",
            gripper_open_rad=self.config.gripper_open_rad,
            gripper_close_rad=self.config.gripper_close_rad,
        )

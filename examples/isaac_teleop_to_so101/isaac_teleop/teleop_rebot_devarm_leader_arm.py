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

"""reBot DevArm leader-arm device for NVIDIA Isaac Teleop, exposed to LeRobot.

The leader is a back-drivable Seeed Studio reBot DevArm (6-DOF + gripper, seven CAN
motors — Damiao on the B601-DM build, RobStride on the B601-RS build) whose joint angles
are streamed in radians by Isaac Teleop's native ``rebot_devarm_leader`` plugin; this
device reads them via a ``JointStateSource`` and converts them into follower-ready
``{joint}.pos``. The motor backend is the plugin's concern (selected by the device path it
is launched with: serial path -> Damiao, SocketCAN name -> RobStride); this device is
backend-agnostic.

The natural follower is LeRobot's ``rebot_b601_follower`` — the SAME arm hardware — so the
mirror is 1:1 in joint space: every streamed angle (gripper included) is converted
``rad2deg`` and renamed from the plugin's URDF joint names (``joint1..joint6, gripper``) to
the follower's motor names via :attr:`RebotDevArmLeaderArmConfig.joint_name_map`. The
follower's own soft ``joint_limits`` clip out-of-range targets, so no extra normalization
is applied here.

``isaacteleop`` imports are guarded behind the availability flag so this module — and the
pure :func:`rebot_leader_joints_to_robot_action` converter — import without it
(construction fails fast via the base class).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lerobot.types import RobotAction

from .base import IsaacTeleopTeleoperator, _isaacteleop_available
from .config_isaac_teleop import RebotDevArmLeaderArmConfig
from .teleop_so101_leader_arm import _joints_group_to_rad

if TYPE_CHECKING or _isaacteleop_available:
    from isaacteleop.retargeting_engine.deviceio_source_nodes import JointStateSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner
else:
    JointStateSource = None
    OutputCombiner = None

# Canonical reBot DevArm DOF names and order — matches the ``rebot_devarm_leader`` plugin
# stream (the ``reBot-DevArm_fixend`` URDF order: six arm joints then the gripper motor).
# Passed to the ``JointStateSource`` as its output layout; the source maps by name and
# ``_joints_group_to_rad`` reads back by name, so a layout mismatch can't mislabel a DOF.
REBOT_DEVARM_LEADER_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper",
]


def rebot_leader_joints_to_robot_action(
    joints_rad: dict[str, float],
    *,
    joint_name_map: dict[str, str],
) -> RobotAction:
    """Convert streamed leader joint angles [rad] to follower-ready ``{joint}.pos`` [deg].

    Pure (no ``isaacteleop``, no I/O). Every mapped joint — gripper included — is converted
    ``rad2deg`` and renamed through ``joint_name_map`` (stream name -> follower motor name);
    leader and follower are the same B601-DM hardware, so a 1:1 angle mirror is exact and
    the follower's soft joint limits handle clipping. Iteration follows ``joints_rad``
    insertion order, so pass it in :data:`REBOT_DEVARM_LEADER_JOINTS` order for a stable
    layout. Stream joints absent from the map are dropped (not guessed at).
    """
    action: RobotAction = {}
    for name, rad in joints_rad.items():
        target = joint_name_map.get(name)
        if target is None:
            continue
        action[f"{target}.pos"] = float(np.rad2deg(rad))
    return action


class RebotDevArmLeaderArm(IsaacTeleopTeleoperator):
    """reBot DevArm leader-arm teleoperator (joint-space), direct joint mirror.

    Reads the seven joint angles off a single ``JointStateSource`` each frame; no
    retargeter, no clutch (shared kinematics with the ``rebot_b601_follower``). When the
    leader is not streaming, :meth:`get_action` returns the held-last joints and
    :attr:`is_tracking` is ``False`` so the owning loop can hold the follower.
    """

    config_class = RebotDevArmLeaderArmConfig
    name = "isaac_teleop_rebot_devarm_leader"

    def __init__(self, config: RebotDevArmLeaderArmConfig):
        super().__init__(config)
        self.config: RebotDevArmLeaderArmConfig = config
        # Held-last joint angles [rad], seeded at zero (URDF/vendor zero pose) so the first
        # frames before the plugin starts pushing read as the zero pose, not garbage.
        self._last_joints_rad: dict[str, float] = dict.fromkeys(REBOT_DEVARM_LEADER_JOINTS, 0.0)
        # Whether the most recent get_action() read live leader data (vs held-last).
        self._is_tracking = False

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> OutputCombiner:
        """Build the joint-mirror pipeline: a single ``JointStateSource`` leaf that converts
        the raw stream into a name-keyed joint group. No retargeter (shared kinematics)."""
        source = JointStateSource(
            name="rebot_devarm_leader",
            collection_id=self.config.collection_id,
            joint_names=REBOT_DEVARM_LEADER_JOINTS,
        )
        return OutputCombiner({"joints": source.output(JointStateSource.JOINTS)})

    # ------------------------------------------------------------------
    # Action features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict[str, type]:
        # Matches the rebot_b601_follower's action features so this is a drop-in
        # joint-space leader: one float `{joint}.pos` per DOF, in the follower's units.
        return {f"{target}.pos": float for target in self.config.joint_name_map.values()}

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

        When the leader is streaming, the live angles are cached and converted; otherwise
        the held-last angles are reused and :attr:`is_tracking` is set ``False``.
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
                # A partially-populated / malformed group on an odd frame: keep held-last,
                # but report not-tracking so the loop holds the follower rather than trust it.
                self._is_tracking = False

        return rebot_leader_joints_to_robot_action(
            self._last_joints_rad,
            joint_name_map=self.config.joint_name_map,
        )

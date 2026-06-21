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

``SO101LeaderArm`` is the second concrete :class:`IsaacTeleopTeleoperator` device
(after :class:`~lerobot.teleoperators.isaac_teleop.teleop_xr_controller.XRController`)
and the first on Isaac Teleop's *generic joint-space device* path. The leader is a
back-drivable SO-101 arm whose six joint angles are streamed as a ``JointStateOutput``
FlatBuffer by the native ``so101_leader`` plugin over the OpenXR tensor transport; this
device reads them each frame via a ``JointStateSource`` and converts them into
follower-ready ``{joint}.pos`` values.

Unlike the XR controller (raw grip pose -> clutch -> IK in the owning loop), a same-
kinematics leader needs **no** retargeting: the leader and follower share the SO-101
geometry, so the action is a 1:1 joint mirror -- exactly the ``--mode joint`` path of
``Teleop/examples/teleop/python/joint_space_device_example.py``, but emitted in the
follower's native units instead of an Isaac Lab radian action. This makes the device a
drop-in joint-space leader: :meth:`get_action` returns ``{joint}.pos`` ready for
``robot.send_action`` (direct joint drive, like ``lerobot-teleoperate`` with the serial
``so101_leader``), so no LeRobot-side ``JointStateRetargeter`` / ``TensorReorderer`` /
processor pipeline is needed. See ``examples/isaac_teleop_to_so101/teleoperate_leader.py``.

Units (the conversion lives in the device so its output is always follower-valid -- there
is no raw-radians intermediate a caller could accidentally send to the follower):

* arm joints: ``deg = rad2deg(rad)``. Correct only if the leader's calibrated zero
  (``home_ticks``) and the follower's homing map to the same physical zero -- the standard
  same-hardware leader/follower assumption (the plugin README notes it "reproduces LeRobot's
  joint angles in radians rather than degrees"). Cannot be verified from here.
* gripper: normalized from ``[gripper_open_rad, gripper_close_rad]`` to the follower's
  RANGE_0_100 jaw target (100 = open, 0 = closed), matching the SO-101 follower calibration
  used by ``MapXRControllerActionToRobotAction``.

The ``isaacteleop`` package is an optional, separately distributed NVIDIA dependency (the
``isaac-teleop`` extra); all imports of it are deferred so this module -- and the pure
:func:`leader_joints_to_robot_action` converter (unit-tested without the XR runtime) -- can
be imported without it.
"""

from __future__ import annotations

import numpy as np

from lerobot.types import RobotAction

from .base import _GRIPPER_MOTOR_SCALE, IsaacTeleopTeleoperator
from .config_isaac_teleop import SO101LeaderArmConfig

# Canonical SO-101 DOF names and order. Matches the ``so101_leader`` plugin's stream
# (``JointStateOutput.joints[*].name``) and the SO-101 follower's motor order; defines the
# ``JointStateSource`` output layout (read positionally) and the emitted action order.
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

    Pure (no ``isaacteleop``, no I/O), so the unit math is unit-testable without the XR
    runtime. Iteration follows ``joints_rad`` insertion order, so pass it in
    :data:`SO101_LEADER_JOINTS` order for a stable action layout.

    Args:
        joints_rad: ``{joint_name: angle [rad]}`` for every SO-101 DOF (gripper included).
        gripper_joint: Which key in ``joints_rad`` is the gripper (RANGE_0_100 target);
            every other joint is converted ``rad2deg``.
        gripper_open_rad: Leader gripper angle [rad] mapped to jaw 100 (fully open).
        gripper_close_rad: Leader gripper angle [rad] mapped to jaw 0 (fully closed).

    Returns:
        ``{f"{joint}.pos": value}`` -- arm joints in degrees, gripper in RANGE_0_100
        (clipped to ``[0, 100]``) -- directly sendable to an SO-101 follower.
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


class SO101LeaderArm(IsaacTeleopTeleoperator):
    """SO-101 leader-arm teleoperator (joint-space), direct joint mirror to the follower.

    Wraps a single Isaac Teleop ``JointStateSource`` on ``config.collection_id`` (fed by the
    native ``so101_leader`` plugin) and reads the six joint angles off it each frame. There
    is no retargeter and no clutch: :meth:`get_action` returns the leader joints converted
    to the follower's units (arm: degrees, gripper: RANGE_0_100), ready for
    ``robot.send_action``.

    When the leader is not streaming this frame (``JointStateSource`` reports the optional
    output absent), :meth:`get_action` returns the last-known joints (held-last) and
    :attr:`is_tracking` is ``False`` -- the owning loop should hold the follower at its
    measured pose rather than command a stale target.
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

    def _build_pipeline(self):
        """Build the joint-mirror pipeline: a single ``JointStateSource`` leaf.

        ::

            JointStateSource(collection_id) ── joints (name-keyed FloatType per DOF)

        The source converts the raw ``JointStateOutput`` FlatBuffer into a name-keyed group
        of joint positions; ``TeleopSession`` auto-discovers its ``JointStateTracker`` from
        this pipeline leaf. No retargeter is wired: the leader<->follower share the SO-101
        kinematics, so :meth:`get_action` does the unit conversion directly.
        """
        from isaacteleop.retargeting_engine.deviceio_source_nodes import JointStateSource
        from isaacteleop.retargeting_engine.interface import OutputCombiner

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

        Steps the session ``RUNNING`` (no clutch lifecycle to gate) and reads the name-keyed
        joint group off the ``JointStateSource``. When the leader is streaming, the live
        angles are cached and converted; when it is not (optional output absent, or an
        unexpected read failure), the last-known angles are reused and :attr:`is_tracking`
        is set ``False`` so the owning loop can hold the follower.

        Returns:
            ``{f"{joint}.pos": value}`` -- arm joints in degrees, gripper in RANGE_0_100 --
            for all six SO-101 DOFs (see :func:`leader_joints_to_robot_action`).
        """
        result = self._step(execution_events=self._running_events())

        joints = result["joints"]
        # The JointStateSource output is Optional: absent (is_none) when the device is
        # inactive. Treat that as "not tracking" and reuse the held-last angles.
        self._is_tracking = not getattr(joints, "is_none", False)
        if self._is_tracking:
            try:
                self._last_joints_rad = {name: float(joints[i]) for i, name in enumerate(SO101_LEADER_JOINTS)}
            except (IndexError, KeyError, TypeError, ValueError):
                # A partially-populated group on an odd frame: keep held-last, but report
                # it as not-tracking so the loop holds the follower rather than trusting it.
                self._is_tracking = False

        return leader_joints_to_robot_action(
            self._last_joints_rad,
            gripper_joint="gripper",
            gripper_open_rad=self.config.gripper_open_rad,
            gripper_close_rad=self.config.gripper_close_rad,
        )

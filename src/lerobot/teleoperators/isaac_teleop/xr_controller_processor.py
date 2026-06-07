#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Processor step that maps XR controller actions to robot EE targets.

Analogous to ``MapPhoneActionToRobotAction`` in
``lerobot/teleoperators/phone/phone_processor.py``, this step bridges
:meth:`XRController.get_action` output to the input contract of the
downstream ``EEReferenceAndDelta`` closed-loop IK processor.

This module is pure ``numpy`` / ``scipy`` and does **not** import
``isaacteleop``, so it can be unit-tested without the XR runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotActionProcessorStep
from lerobot.types import RobotAction

# Frame change: OpenXR (X=Right, Y=Up, Z=Backward) -> robot (X=Forward, Y=Left, Z=Up).
# This is a proper rotation (det = +1); the inverse is its transpose.
_OPENXR_TO_ROBOT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ],
    dtype=np.float64,
)

# Downstream ``GripperVelocityToJoint(discrete_gripper=True)`` interprets the
# command as a discrete class index {0 = close, 1 = stay, 2 = open} and applies
# the SO100 sign internally. We therefore emit the class index directly and do
# NOT pre-negate here.
_GRIPPER_OPEN = 2.0
_GRIPPER_CLOSE = 0.0


def _remap_openxr_to_robot(pos: np.ndarray, quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remap a position and quaternion from the OpenXR frame to the robot frame."""
    pos_new = (_OPENXR_TO_ROBOT @ pos).astype(np.float32)

    r_old = Rotation.from_quat(quat_xyzw)
    # Change of basis on the rotation matrix: R' = M · R · Mᵀ.
    r_new = Rotation.from_matrix(_OPENXR_TO_ROBOT @ r_old.as_matrix() @ _OPENXR_TO_ROBOT.T)
    quat_new = r_new.as_quat().astype(np.float32)

    return pos_new, quat_new


@ProcessorStepRegistry.register("map_xr_controller_action_to_robot_action")
@dataclass
class MapXRControllerActionToRobotAction(RobotActionProcessorStep):
    """Maps :class:`XRController` output to the robot action format.

    The XR controller reports *absolute* world-space poses in the OpenXR
    coordinate frame plus a clutch (``enabled``) signal. Per frame this step:

    1. Remaps the pose from OpenXR frame (X=Right, Y=Up, Z=Backward) to the
       robot frame (X=Forward, Y=Left, Z=Up).
    2. On the **rising edge** of ``enabled`` (re)captures the controller pose
       as the origin. This co-arms with the downstream
       ``EEReferenceAndDelta``, which latches its FK reference on the same
       rising edge, so the arm does not teleport on engage and the two
       references stay in lock-step across episode resets.
    3. While engaged, emits base-frame-relative deltas: ``target_{x,y,z}`` is
       the position delta from the origin in **real metres** (base frame), and
       ``target_{wx,wy,wz}`` is the rotvec of ``origin_rot_inv · rot``. While
       disengaged, all ``target_*`` are zeroed (phone parity) so the robot
       holds position.

    SO-101 is a 5-DOF, position-dominant IK target, so the orientation channel
    plays a limited role downstream. The world->base seat rotation that
    IsaacLab applies for its sim env is deliberately **not** done here.

    Input keys (from :meth:`XRController.get_action`):
        - ``ee_pos``: ``np.ndarray`` shape ``(3,)`` — EE position (OpenXR frame).
        - ``ee_quat``: ``np.ndarray`` shape ``(4,)`` — EE quaternion ``(x,y,z,w)`` (OpenXR frame).
        - ``gripper``: ``float`` — ``-1.0`` (closed) or ``+1.0`` (open).
        - ``enabled``: ``bool`` — clutch state.

    Output keys (for ``EEReferenceAndDelta``):
        - ``enabled``: ``bool`` — clutch state (passed through).
        - ``target_x``, ``target_y``, ``target_z``: ``float`` — position delta from origin
          in real metres (robot base frame); zeroed when not enabled.
        - ``target_wx``, ``target_wy``, ``target_wz``: ``float`` — orientation delta as a
          rotvec (robot base frame); zeroed when not enabled.
        - ``gripper_vel``: ``float`` — discrete gripper class index for
          ``GripperVelocityToJoint(discrete_gripper=True)``: ``2`` = open, ``0`` = close.
    """

    _origin_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _origin_rot_inv: Rotation | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        ee_pos = action.pop("ee_pos")
        ee_quat = action.pop("ee_quat")
        gripper_cmd = action.pop("gripper")
        enabled = bool(action.pop("enabled"))

        # Remap from OpenXR frame to robot frame.
        ee_pos, ee_quat = _remap_openxr_to_robot(ee_pos, ee_quat)
        rot = Rotation.from_quat(ee_quat)

        # Re-capture the origin on the rising edge of ``enabled`` (or whenever
        # the origin is unset, e.g. the first engaged frame after reset()).
        if enabled and (not self._prev_enabled or self._origin_pos is None):
            self._origin_pos = np.asarray(ee_pos, dtype=float)
            self._origin_rot_inv = rot.inv()

        if enabled and self._origin_pos is not None and self._origin_rot_inv is not None:
            delta_pos = np.asarray(ee_pos, dtype=float) - self._origin_pos
            delta_rotvec = (self._origin_rot_inv * rot).as_rotvec()
            target = (
                float(delta_pos[0]),
                float(delta_pos[1]),
                float(delta_pos[2]),
                float(delta_rotvec[0]),
                float(delta_rotvec[1]),
                float(delta_rotvec[2]),
            )
        else:
            target = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Map the binary gripper state to a discrete class index. The downstream
        # discrete_gripper mode encodes the SO100 sign, so do NOT pre-negate.
        gripper_vel = _GRIPPER_OPEN if float(gripper_cmd) >= 0.0 else _GRIPPER_CLOSE

        action["enabled"] = enabled
        action["target_x"] = target[0]
        action["target_y"] = target[1]
        action["target_z"] = target[2]
        action["target_wx"] = target[3]
        action["target_wy"] = target[4]
        action["target_wz"] = target[5]
        # KEY HONESTY: the "gripper_vel" key name is MANDATED by the downstream
        # contract (EEReferenceAndDelta pops "gripper_vel", GripperVelocityToJoint
        # pops "ee.gripper_vel") and cannot be renamed. The *value* is NOT a
        # velocity: it is the discrete class index {0=close, 1=stay, 2=open}
        # consumed with discrete_gripper=True. A binary trigger only ever emits
        # close(0)/open(2) -- never "stay"(1), which is reserved for continuous
        # controls -- which is intended for this binary clutch grip.
        action["gripper_vel"] = gripper_vel

        self._prev_enabled = enabled
        return action

    def reset(self) -> None:
        """Clear the latched origin and clutch edge so the next engaged frame re-arms.

        Runs per episode in lock-step with ``EEReferenceAndDelta.reset()`` so a
        stale XR origin never desyncs from the downstream FK reference.
        """
        self._origin_pos = None
        self._origin_rot_inv = None
        self._prev_enabled = False

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["ee_pos", "ee_quat", "gripper", "enabled"]:
            features[PipelineFeatureType.ACTION].pop(feat, None)

        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION][feat] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))

        return features

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

This module is pure ``numpy`` and does **not** import ``isaacteleop``, so it
can be unit-tested without the XR runtime.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStepRegistry, RobotActionProcessorStep
from lerobot.types import RobotAction

logger = logging.getLogger(__name__)


def _env_debug_default() -> bool:
    """Default for the ``debug`` field: enabled via ``LEROBOT_XR_DEBUG=1``."""
    return os.environ.get("LEROBOT_XR_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


# Frame change applied to the engage-relative *position delta* only.
# OpenXR (X=Right, Y=Up, Z=Backward) -> robot (X=Forward, Y=Left, Z=Up).
# This is a proper rotation (det = +1); the inverse is its transpose. It maps
# 0 -> 0, so it is inert on the engage frame (zero delta stays zero).
# TODO(verify-on-hardware): confirm this axis remap is correct. A verifier
# watches for mirrored/rotated EE motion (the arm moves left when the controller
# goes right, up<->forward swapped, etc.) -- a sign of a wrong axis permutation.
# TODO(verify-anchor-frame): fold into robot_base_pos once the anchor frame the
# clutch retargeter rebases in is confirmed; if the grip pose is already
# anchor-transformed into the base frame this collapses to identity.
_OPENXR_TO_ROBOT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ],
    dtype=np.float64,
)

# Gripper closedness [0, 1] -> motor units [0, 100] (RANGE_0_100). The affine
# direction (and any polarity flip) lives here; there is no separate invert knob.
# TODO(verify-on-hardware): confirm open/close endpoints + polarity in motor
# units. A verifier watches for the jaw opening when it should close (inverted
# polarity) or not reaching full open/close (wrong travel range).
_GRIPPER_MOTOR_SCALE = 100.0


@ProcessorStepRegistry.register("map_xr_controller_action_to_robot_action")
@dataclass
class MapXRControllerActionToRobotAction(RobotActionProcessorStep):
    """Maps :class:`XRController` output to the closed-loop IK input contract.

    The XR controller (via the three SO-101 retargeters) reports a clutch-rebased
    *absolute* 7D ``ee_pose`` in the robot base frame, a wrist-roll angle [rad],
    a jaw closedness in ``[0, 1]``, and a clutch (``enabled``) signal. The clutch
    retargeter latches its own origin on the *connect* frame (not the squeeze
    edge), so at the moment the operator squeezes, ``ee_pose[:3]`` is generally
    nonzero relative to the EE home. Per frame this step:

    1. On the **rising edge** of ``enabled`` (or whenever the engage-home is
       unset, e.g. the first engaged frame after :meth:`reset`), latches
       ``engage_home = ee_pose[:3]``. This co-arms with the downstream
       ``EEReferenceAndDelta``, which latches its measured-FK reference on the
       same rising edge, so the arm does not teleport on engage and the two
       references stay in lock-step across episode resets.
    2. While engaged, emits a base-frame position *delta*:
       ``target_{x,y,z} = _OPENXR_TO_ROBOT @ (ee_pose[:3] - engage_home)``. On
       the latching frame this is exactly zero (no teleport). While disengaged,
       all ``target_*`` are zeroed (phone parity) so the robot holds position.

    Clutch model: the clutch (``enabled``) gates the whole ARM POSE — both the
    position delta AND the wrist roll. While disengaged the arm is fully frozen:
    ``target_*`` zero (XYZ holds via ``EEReferenceAndDelta``) AND ``wrist_roll``
    is re-emitted as the last commanded roll so the post-IK
    :class:`OverwriteWristRollFromAngle` step keeps the wrist where it was rather
    than continuing to roll. The GRIPPER jaw is controlled independently and is
    NOT clutch-gated: ``ee.gripper_pos`` is emitted every frame regardless of
    ``enabled`` so the operator can open/close the jaw while the arm is held.

    SO-101 is a 5-DOF, position-only IK target, so the orientation delta is
    intentionally unused: ``target_{wx,wy,wz} = 0``. The terminal roll axis is
    recovered separately via the ``wrist_roll`` channel (radians, carried
    through for the post-IK :class:`OverwriteWristRollFromAngle` step). The
    world->base seat rotation is currently folded into ``_OPENXR_TO_ROBOT``.

    TODO(verify-on-hardware): ``wrist_roll`` is an ABSOLUTE angle (no
    engage-relative rebasing), so on the first engage the wrist may snap to the
    controller's current twist (this matches IsaacLab's direct
    ``JointPositionActionCfg`` roll). A verifier watches for the wrist suddenly
    snapping/rotating to a new angle the instant the clutch engages; revisit with
    engage-relative roll rebasing if it is jarring on hardware.

    Input keys (from :meth:`XRController.get_action`):
        - ``ee_pose``: ``np.ndarray`` shape ``(7,)`` — ``[x,y,z,qx,qy,qz,qw]`` (base frame).
        - ``wrist_roll``: ``float`` — absolute wrist-roll angle [rad].
        - ``closedness``: ``float`` — jaw closedness in ``[0, 1]`` (0=open, 1=closed).
        - ``enabled``: ``bool`` — clutch state.

    Output keys:
        - ``enabled``: ``bool`` — clutch state (passed through).
        - ``target_x``, ``target_y``, ``target_z``: ``float`` — engage-relative position
          delta in real metres (robot base frame); zeroed when not enabled.
        - ``target_wx``, ``target_wy``, ``target_wz``: ``float`` — always ``0.0`` (the
          orientation delta is intentionally unused on the 5-DOF arm).
        - ``gripper_vel``: ``float`` — always ``0.0``. KEY HONESTY: this is NOT a
          velocity; the name is mandated only because ``EEReferenceAndDelta`` pops
          ``gripper_vel``. The gripper is driven instead by the absolute
          ``ee.gripper_pos`` target below (the velocity integrator is bypassed).
        - ``wrist_roll``: ``float`` — wrist-roll angle [rad], carried for the post-IK
          :class:`OverwriteWristRollFromAngle` step. Clutch-gated with the arm pose:
          while engaged it is the live command (and is tracked); while disengaged it
          is the last commanded roll (HOLD), so the wrist freezes with the rest of the
          arm. Before the first engage it is OMITTED entirely (no key) so the
          IK-produced roll stands and the wrist is not snapped pre-engage.
        - ``ee.gripper_pos``: ``float`` — absolute jaw target in motor units ``[0, 100]``
          (RANGE_0_100), ``closedness * _GRIPPER_MOTOR_SCALE``. NOT clutch-gated: emitted
          every frame regardless of ``enabled`` (the jaw is independent of the arm pose).
          KEY HONESTY: this is an *absolute joint target*, not a delta/velocity; it passes
          through ``EEReferenceAndDelta`` and ``EEBoundsAndSafety`` untouched and is consumed
          by ``InverseKinematicsEEToJoints`` (which passes it straight to ``gripper.pos``),
          bypassing the velocity integrator.
    """

    # Per-frame debug logging of the clutch + gripper + roll + deltas. Defaults
    # from the LEROBOT_XR_DEBUG env var so it can be toggled without editing code.
    # Logs on the enabled edge and every ``debug_every`` frames.
    debug: bool = field(default_factory=_env_debug_default)
    debug_every: int = 30

    _engage_home: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_wrist_roll_rad: float | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _frame: int = field(default=0, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        ee_pose = np.asarray(action.pop("ee_pose"), dtype=float)
        wrist_roll = float(action.pop("wrist_roll"))
        closedness = float(action.pop("closedness"))
        enabled = bool(action.pop("enabled"))

        ee_pos = ee_pose[:3]

        # Latch the engage-home on the rising edge of ``enabled`` (or whenever
        # it is unset, e.g. the first engaged frame after reset()). The clutch
        # retargeter latches its own origin on connect, so we subtract this
        # engage-home to get a delta that is exactly zero at engage.
        if enabled and (not self._prev_enabled or self._engage_home is None):
            self._engage_home = ee_pos.copy()

        if enabled and self._engage_home is not None:
            # _OPENXR_TO_ROBOT @ v maps the (already base-frame) engage-relative
            # delta; the subtraction-then-matrix order is what keeps the engage
            # frame inert (0 -> 0). Orientation delta is intentionally dropped.
            delta_pos = _OPENXR_TO_ROBOT @ (ee_pos - self._engage_home)
            target = (float(delta_pos[0]), float(delta_pos[1]), float(delta_pos[2]))
        else:
            target = (0.0, 0.0, 0.0)

        action["enabled"] = enabled
        action["target_x"] = target[0]
        action["target_y"] = target[1]
        action["target_z"] = target[2]
        # Orientation delta intentionally unused on the 5-DOF position-only arm.
        action["target_wx"] = 0.0
        action["target_wy"] = 0.0
        action["target_wz"] = 0.0
        # KEY HONESTY: not a velocity -- only emitted because EEReferenceAndDelta
        # pops "gripper_vel". The real gripper command is the absolute
        # "ee.gripper_pos" below.
        action["gripper_vel"] = 0.0
        # Clutch-gate the wrist roll with the rest of the arm pose (radians, for
        # the post-IK OverwriteWristRollFromAngle step). While engaged, the live
        # roll is the command and is tracked. While disengaged, re-emit the last
        # commanded roll (HOLD) so the wrist freezes with the arm; if there has
        # been no engage yet, omit the key entirely so the IK-produced roll stands
        # and the wrist is not snapped pre-engage.
        if enabled:
            self._last_wrist_roll_rad = wrist_roll
            action["wrist_roll"] = wrist_roll
        elif self._last_wrist_roll_rad is not None:
            action["wrist_roll"] = self._last_wrist_roll_rad
        # KEY HONESTY: an ABSOLUTE jaw joint target in motor units [0, 100], not a
        # delta/velocity. NOT clutch-gated -- emitted every frame regardless of
        # ``enabled`` so the jaw is independent of the (clutch-gated) arm pose.
        # Passes through EEReferenceAndDelta + EEBoundsAndSafety untouched;
        # InverseKinematicsEEToJoints forwards it to gripper.pos.
        action["ee.gripper_pos"] = closedness * _GRIPPER_MOTOR_SCALE

        self._frame += 1
        if self.debug:
            self._log_debug(
                enabled=enabled,
                closedness=closedness,
                wrist_roll=wrist_roll,
                delta=np.asarray(target, dtype=float),
            )

        self._prev_enabled = enabled
        return action

    def _log_debug(self, *, enabled: bool, closedness: float, wrist_roll: float, delta: np.ndarray) -> None:
        """Throttled per-frame trace of the clutch/gripper/roll/delta signals.

        Emits on a clutch edge (so transitions are never missed) and otherwise a
        heartbeat every ``debug_every`` frames. The key things this surfaces:
        whether ``enabled`` (the squeeze clutch) ever goes True -- if it never
        does, the arm stays frozen and no controller motion moves it -- plus the
        emitted closedness and wrist roll, to spot polarity / sign issues.
        """
        edge = enabled != self._prev_enabled
        if edge or (self._frame % max(1, self.debug_every) == 0):
            logger.info(
                "[XR map] f=%d enabled=%s home=%s | closedness=%.2f roll=%+.3frad | "
                "|Δpos|=%.4fm Δpos=[%+.3f %+.3f %+.3f]",
                self._frame,
                enabled,
                "set" if self._engage_home is not None else "unset",
                closedness,
                wrist_roll,
                float(np.linalg.norm(delta[:3])),
                delta[0],
                delta[1],
                delta[2],
            )

    def reset(self) -> None:
        """Clear the latched engage-home and clutch edge so the next engaged frame re-arms.

        Runs per episode in lock-step with ``EEReferenceAndDelta.reset()`` so a
        stale XR engage-home never desyncs from the downstream FK reference.
        """
        self._engage_home = None
        self._last_wrist_roll_rad = None
        self._prev_enabled = False

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["ee_pose", "wrist_roll", "closedness"]:
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
            "wrist_roll",
            "ee.gripper_pos",
        ]:
            features[PipelineFeatureType.ACTION][feat] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))

        return features

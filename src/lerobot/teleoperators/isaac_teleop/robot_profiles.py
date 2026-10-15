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

"""Per-follower kinematics/IK profiles for :class:`~.xr_controller.XRController`.

Selected from the ``--robot.*`` config passed to ``XRController.__init__`` as ``robot_config``
(see ``make_teleoperator_from_config``) -- the teleoperator and the robot are otherwise
constructed independently by ``lerobot-teleoperate``/``lerobot-record``.
"""

from __future__ import annotations

import os
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.constants import HF_LEROBOT_HOME

from .xr_controller_processor import MapXRControllerActionToRobotAction

# Per-frame EE rate limit [m]: an over-limit step raises out of the loop (EEBoundsAndSafety's
# default raise_on_jump). At FPS=30, 0.1 m/frame caps EE speed at ~3 m/s -- well above the ~0.05 m
# a fast hand produces at clutch_position_scale=0.5, so it only trips on a real tracking glitch.
# Position only; orientation is neither bounded nor rate-limited.
_MAX_EE_STEP_M = 0.1

# Soft-orientation IK weight: small but nonzero so the wrist follows the hand while position
# dominates (the 5-DOF SO-101 cannot realize an arbitrary orientation). 0.0 = position-only.
_IK_ORIENTATION_WEIGHT = 0.01


def _ensure_so101_urdf() -> str:
    """Return the cached SO-101 URDF path, fetching the ``so101`` folder (URDF + meshes) from
    the public ``lerobot/robot-urdfs`` HF bucket into the LeRobot cache on first use."""
    dest_dir = HF_LEROBOT_HOME / "robot-urdfs" / "so101"
    urdf_path = dest_dir / "so101_new_calib.urdf"
    # Completeness marker written only after a FULL sync: the URDF file alone is not a
    # completeness signal (an interrupted first sync can leave the meshes it references
    # missing, which the URDF's mere existence would then hide forever). Re-syncing is
    # idempotent and repairs a partial cache; delete the folder to force a re-download.
    marker = dest_dir / ".sync_complete"
    if not marker.exists():
        from huggingface_hub import sync_bucket

        sync_bucket("hf://buckets/lerobot/robot-urdfs/so101", str(dest_dir), quiet=True)
        marker.touch()
    return str(urdf_path)


# Seeed publishes the reBot DevArm description as a ROS package (URDF + 30 STL meshes, ~64 MB)
# rather than on the HF bucket the SO-101 uses. MuJoCo Menagerie's seeed_rebot_devarm MJCF is
# derived from this same URDF and validates against it, but placo needs the URDF itself.
_REBOT_RS_URDF_REPO = "https://raw.githubusercontent.com/Seeed-Projects/reBot-Isaacsim/main"
_REBOT_RS_URDF_PKG = "urdf/00-arm-rs_asm-v3"
_REBOT_RS_URDF_FILE = "00-arm-rs_asm-v3.urdf"
# placo resolves mesh paths relative to the URDF and cannot expand `package://`, so the cached
# copy is rewritten to point at a sibling meshes/ folder.
_REBOT_RS_PACKAGE_URI = f"package://{Path(_REBOT_RS_URDF_PKG).name}/meshes/"


def _ensure_rebot_b601_rs_urdf() -> str:
    """Return the cached reBot B601-RS URDF path, downloading it and its meshes on first use.

    ``REBOT_RS_URDF`` overrides with a local path (a checkout of the Seeed package), skipping
    the download entirely.
    """
    override = os.environ.get("REBOT_RS_URDF", "").strip()
    if override:
        return override

    dest_dir = HF_LEROBOT_HOME / "robot-urdfs" / "rebot_b601_rs"
    urdf_path = dest_dir / _REBOT_RS_URDF_FILE
    # Same completeness-marker rule as the SO-101 fetch above: an interrupted first download
    # leaves meshes missing, which the URDF's mere existence would hide forever.
    marker = dest_dir / ".sync_complete"
    if marker.exists():
        return str(urdf_path)

    mesh_dir = dest_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    urdf_url = f"{_REBOT_RS_URDF_REPO}/{_REBOT_RS_URDF_PKG}/urdf/{_REBOT_RS_URDF_FILE}"
    with urllib.request.urlopen(urdf_url, timeout=60) as response:  # nosec B310
        urdf_text = response.read().decode()

    for mesh in sorted(set(re.findall(r'filename="([^"]+)"', urdf_text))):
        name = Path(mesh).name
        mesh_url = f"{_REBOT_RS_URDF_REPO}/{_REBOT_RS_URDF_PKG}/meshes/{name}"
        with urllib.request.urlopen(mesh_url, timeout=120) as response:  # nosec B310
            (mesh_dir / name).write_bytes(response.read())

    urdf_path.write_text(urdf_text.replace(_REBOT_RS_PACKAGE_URI, "meshes/"))
    marker.touch()
    return str(urdf_path)


@dataclass(frozen=True)
class RobotProfile:
    """Everything the XR clutch -> IK pipeline needs to know about one follower.

    Keyed by :func:`robot_profile_key` in :data:`ROBOT_PROFILES`. Adding an arm is a profile
    entry, not a code change.
    """

    urdf: Callable[[], str]
    """Returns a local URDF path, fetching and caching it on first use."""

    urdf_joint_names: list[str]
    """URDF joint names IK solves for -- must match the URDF's own joint names, which are NOT
    necessarily the robot's action-feature names (they agree for the SO-101, not for the reBot,
    whose URDF uses ``jointN``). Positionally aligned with the ARM portion of ``motor_names``
    (same order, gripper excluded here -- ``RobotKinematics`` slices the leading joints and
    passes any extra (gripper) entries in ``motor_names`` through unsolved)."""

    motor_names: list[str]
    """The robot's own action-feature names (``{name}.pos``), gripper last. This is what
    ``get_action()``'s output, ``send_feedback()``'s input, and ``reset_pose`` below are keyed
    by -- distinct from ``urdf_joint_names`` because a robot's action names need not match its
    URDF's. No robot object is available to read ``robot.action_features`` (the teleoperator and
    the robot are constructed independently), so this has to be static, per-profile data."""

    ee_frame: str
    """URDF frame IK drives to."""

    ee_bounds: dict[str, list[float]]
    """Backstop box [m] in the robot base frame; sized from a URDF reach sweep, not a guess."""

    reset_pose: dict[str, float]
    """Reset target, keyed like ``motor_names``."""

    gripper_open: float
    """Follower action value at fully OPEN (trigger released)."""

    gripper_close: float
    """Follower action value at fully CLOSED (trigger pulled)."""

    max_ee_step_m: float = _MAX_EE_STEP_M
    orientation_weight: float = _IK_ORIENTATION_WEIGHT

    clutch_position_scale: float | None = None
    """Controller-to-EE translation gain for this arm, or ``None`` to keep
    ``XRControllerConfig``'s own default (0.5, sized to the SO-101's reach). Belongs to the arm
    because the right gain follows from its reach."""

    raise_on_ee_jump: bool = True
    """``False`` rate-limits an over-limit frame and warns instead of raising out of the loop."""

    preview_arm: str | None = None
    """Which of Isaac Teleop's ``viz.robot.PREVIEW_ARMS`` this arm is previewed as in the
    headset, or ``None`` for no twin. The preview holds this arm's own ``reset_pose`` and
    puts its own gripper on the operator's hand, so the key must name THIS robot -- a near
    fit shows the operator a different machine."""

    home_orientation_from_measured: bool = False
    """Re-seed the clutch home from the measured EE while disengaged, so a sagging arm does not
    kick on engage. Only the ORIENTATION half is new (the latch already takes position from the
    measured input), and it is safe only where measured-minus-commanded is sag ALONE. Converged
    IK leaves the 6-DOF reBot 0.0 deg short of a commanded orientation at ``orientation_weight``
    1.0, but the 5-DOF SO-101 8.8 deg short (17.9 worst) at 0.01: feeding that back would move
    the hand-to-arm orientation mapping by that much on every re-clutch."""


def robot_profile_key(robot_type: str, motor_family: str | None = None) -> str:
    """ROBOT_PROFILES key: a follower type, refined by its motor family for followers (like the
    reBot B601) that share one type across hardware variants."""
    return f"{robot_type}:{motor_family}" if motor_family else robot_type


ROBOT_PROFILES: dict[str, RobotProfile] = {
    # SO-101/SO-100 share the SO-101 URDF, whose joint names are the motor names.
    "so101_follower": RobotProfile(
        urdf=_ensure_so101_urdf,
        # On the SO-101 the URDF joint names ARE the motor names (includes the gripper; the
        # solver has always been handed all six).
        urdf_joint_names=[
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        motor_names=[
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        ee_frame="gripper_frame_link",
        # Sized to the arm's reachable envelope (URDF FK sweep over all joint limits; max reach
        # 0.545 m). The z floor is the tabletop: base_link's collision geometry bottoms out at
        # z=-0.0024, so 0.0 is the table plus ~2 mm.
        ee_bounds={"min": [-0.35, -0.45, 0.0], "max": [0.50, 0.45, 0.55]},
        # This IS isaacteleop's Q_HOME_DEG for the so101 preview arm, joint for joint: the
        # twin holds the pose the hardware parks at, so the operator sees the arm hold what
        # it will hold. Move one and move the other.
        reset_pose={
            "shoulder_pan": 0.0,
            "shoulder_lift": -45.0,
            "elbow_flex": 45.0,
            "wrist_flex": 90.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
        gripper_open=100.0,
        gripper_close=0.0,
        preview_arm="so101",
    ),
    robot_profile_key("rebot_b601_follower", motor_family="rs"): RobotProfile(
        urdf=_ensure_rebot_b601_rs_urdf,
        # The URDF's own joint names -- NOT the robot's action names (motor_names below).
        urdf_joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        motor_names=[
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw",
            "wrist_roll",
            "gripper",
        ],
        ee_frame="gripper_end",
        # 60k-sample FK sweep over the URDF joint limits: max reach 0.911 m, envelope
        # x/y within +-0.77, z in [-0.372, 0.907]. Kept well inside that, with the z floor at
        # the arm's own base plane (tabletop).
        ee_bounds={"min": [-0.45, -0.55, 0.0], "max": [0.65, 0.55, 0.70]},
        # The sit-down pose calibration zeroes the arm at, lifted 10/15 deg off the shoulder and
        # elbow endpoints (both travel one way only, from 0) so the IK is not seeded sitting on
        # two joint limits. FK puts the EE at [0.297, 0.0, 0.304] m, 8.6 cm above sit-down.
        # Also isaacteleop's Q_HOME_REBOT_DEG (jointN maps to these names positionally, as
        # urdf_joint_names above does) -- the twin holds what the hardware parks at.
        reset_pose={
            "shoulder_pan": 0.0,
            "shoulder_lift": -5.0,
            "elbow_flex": -10.0,
            "wrist_flex": 0.0,
            "wrist_yaw": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
        # RS_PROFILE.joint_limits["gripper"] is in public degrees: -270 open, 0 closed.
        # The RS gripper is driven by a force-limited impedance torque
        # (gripper_torque_limit), so commanding the full travel bounds grip force rather
        # than jaw position.
        gripper_open=-270.0,
        gripper_close=0.0,
        # 1:1 translation. The device default 0.5 is sized to the SO-101's 0.545 m reach; this
        # arm reaches 0.911 m, so halving a hand sweep leaves most of the workspace unreachable
        # without re-clutching. Provisional: one headset session, no reach-to-the-bounds sweep.
        clutch_position_scale=1.0,
        # Worst per-frame EE step measured over a headset session: 35 mm at
        # clutch_position_scale=0.5, 90 mm at 1.0 -- 2.7 m/s at 30 Hz. 120 mm sits above that
        # with little slack on purpose: over-limit frames rate-limit and warn rather than raise
        # (raise_on_ee_jump below), so a tight bound costs a warning, not a session.
        max_ee_step_m=0.12,
        # 6-DOF: unlike the SO-101 the wrist can actually realize a commanded orientation.
        orientation_weight=1.0,
        # Safe here for that same reason, and needed: the arm sags tens of degrees of EE pitch
        # while disengaged, which the IK used to reconcile in one frame on engage.
        home_orientation_from_measured=True,
        # RS only: the B601 also ships as a Damiao build with the same joint topology and
        # different geometry, and Isaac Teleop's model is the RobStride one. That build has
        # no profile here, so it correctly gets no twin rather than a near fit.
        preview_arm="rebot_devarm_rs",
    ),
}
ROBOT_PROFILES["so100_follower"] = ROBOT_PROFILES["so101_follower"]


def build_xr_joint_pipeline(
    profile: RobotProfile, kinematics
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    """Absolute base-frame EE target -> joint targets in the follower's own convention.

    ``rename -> bounds/rate-limit -> IK -> motor convention``. Every step between the first and
    the last works in URDF convention, so the observation handed to the pipeline (which is the
    IK seed) must already be rebased there.
    """
    steps = [
        MapXRControllerActionToRobotAction(
            gripper_open=profile.gripper_open, gripper_close=profile.gripper_close
        ),
        EEBoundsAndSafety(
            end_effector_bounds=profile.ee_bounds,
            max_ee_step_m=profile.max_ee_step_m,
            raise_on_jump=profile.raise_on_ee_jump,
        ),
        InverseKinematicsEEToJoints(
            kinematics=kinematics,
            # The full action list (gripper included): InverseKinematicsEEToJoints maps the
            # solver's output (sized to len(profile.urdf_joint_names)) back onto action keys
            # positionally and special-cases "gripper" to take ee.gripper_pos instead of a
            # solved value, rather than being handed the URDF's own (possibly different) names.
            motor_names=profile.motor_names,
            initial_guess_current_joints=True,
            orientation_weight=profile.orientation_weight,
        ),
    ]
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=steps,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

# !/usr/bin/env python

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

"""Teleoperate an SO-101 follower arm with an XR (VR) controller via Isaac Teleop.

This mirrors ``examples/phone_to_so100/teleoperate.py`` but swaps the phone for
an XR controller. The device is a thin reader: it emits the **raw** controller
grip pose (already rebased into the robot base frame), the squeeze, and the
trigger. ALL the calibration lives here in the loop — a small :class:`Clutch`
latches the controller origin on engage and drives the EE from the delta, so the
device carries no per-frame state::

    XRController.get_action()                       # raw base-frame grip_pos/grip_quat + squeeze + trigger
      -> Clutch.rebase(grip_pos, grip_quat)          # ee_pose = engage-relative delta applied to the EE home (pos + orient)
      -> MapXRControllerActionToRobotAction          # ee.x/y/z = abs pos, ee.w* = abs orient rotvec, ee.gripper_pos = f(trigger)
      -> EEBoundsAndSafety                           # workspace clip + per-frame jump clamp (position only)
      -> InverseKinematicsEEToJoints(ow=small)       # soft-orientation Placo IK (passes ee.gripper_pos -> gripper.pos)

Squeeze (and hold) the controller grip past ``clutch_threshold`` to engage; on the
engage edge the clutch latches its origin to the current controller pose and its
home to the last commanded EE pose, so the arm does not jump in position OR
orientation. The clutch rebases BOTH position and orientation (engage-relative
base-frame deltas); the orientation target is fed to the IK with a small weight
(``IK_ORIENTATION_WEIGHT``) so the wrist follows the hand while position dominates
(the 5-DOF SO-101 cannot fully realize an arbitrary orientation). The analog
trigger drives an absolute ``ee.gripper_pos`` jaw target.

Startup / safety contract: by default the script slews all joints to their URDF
origin (arm joints to 0°, gripper to 100 = fully open) over ``--reset-duration``
seconds before entering the loop.  Pass ``--no-reset-to-origin`` to skip this slew
and keep the arm exactly where it is.  After the slew (or if skipped) the clutch
seeds its home from the arm's MEASURED pose (FK of the joints read right after the
slew), so the seeded home equals the post-reset position and the first engage is
jump-free.  The robot is commanded ONLY while the clutch is engaged; while
disengaged the loop re-sends the measured joints (an explicit hold), and releasing
the clutch freezes it in place.

NOTE: EEBoundsAndSafety clamps (not raises) on a per-frame jump > max_ee_step_m; the
clutch's no-teleport keeps frames small, but set a generous bound for bring-up.

Requires the ``isaac-teleop`` extra (``isaacteleop``) and an OpenXR runtime.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.isaac_teleop import (
    MapXRControllerActionToRobotAction,
    XRController,
    XRControllerConfig,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation

FPS = 30

# Per-frame EE rate limit [m]. EEBoundsAndSafety (raise_on_jump=False below) clamps
# any per-frame position change above this instead of raising, so MAX_EE_STEP_M is a
# safety rate limit, not a crash threshold: at FPS=30, 0.1 m/frame caps EE speed at
# ~3 m/s, which deliberate teleop rarely exceeds while still absorbing controller
# tracking glitches as a single slow frame. (Only the per-frame change is bounded;
# the absolute target can still be far — that is what end_effector_bounds clips.)
MAX_EE_STEP_M = 0.1

# Orientation weight for the IK. Small but nonzero: the controller's (clutch-rebased)
# orientation is fed to the solver as a soft target so the wrist follows the hand,
# but position still dominates. The SO-101 is 5-DOF and CANNOT realize an arbitrary
# 3-DOF orientation, so the wrist tracks orientation only partially by design — turn
# this up to favor orientation over position, down (or 0.0) for position-only.
IK_ORIENTATION_WEIGHT = 0.05

# CloudXR device-profile env file passed to the launcher (see default.env next to
# this script). Resolved absolutely so it loads regardless of the working dir.
CLOUDXR_ENV_FILE = str(Path(__file__).parent / "default.env")

# Optional file written by record_reset_pose.py.  When present its values take
# priority over RESET_ORIGIN_DEG.
RESET_POSE_FILE = Path(__file__).parent / "reset_pose.json"

# Default duration [s] for the startup reset-to-origin slew.
RESET_DURATION_S = 5.0

# Reset target in each motor's native units.
# Arm joint values are the Lab SO-101 stack-task init pose (see Lab repo:
# source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/so101/
# stack_joint_pos_env_cfg.py, _SO101_STACK_INIT_JOINT_POS) converted from URDF
# radians to degrees via np.rad2deg.  This mid-range pose (elbow/wrist bent)
# avoids the boundary singularity of a fully-extended 5-DOF arm.
# Assumes standard calibration where 0° = URDF 0 rad (homing pose).
# Gripper uses MotorNormMode.RANGE_0_100; 100 = fully open (safe for teleop start).
RESET_ORIGIN_DEG: dict[str, float] = {
    "shoulder_pan":  float(np.rad2deg(0.0)),
    "shoulder_lift": float(np.rad2deg(-0.6)),
    "elbow_flex":    float(np.rad2deg(0.8)),
    "wrist_flex":    float(np.rad2deg(0.6)),
    "wrist_roll":    float(np.rad2deg(0.0)),
    "gripper":       100.0,
}


class Clutch:
    """Engage-relative clutch for both position AND orientation.

    Mirrors Isaac Teleop's ``SO101ClutchRetargeter`` but lives in this loop so the
    device can stay a thin raw-pose reader. Clutching is the same idea for both
    channels — latch an origin on engage, then track the base-frame delta from it —
    applied independently to position and orientation. State:

    - ``_last_commanded_pos`` / ``_last_commanded_rot``: the EE pose the loop last
      commanded; held while disengaged so the arm freezes where it was left.
    - ``_home_pos`` / ``_home_rot``: latched on the engage edge — the EE pose the
      per-frame delta is applied to.
    - ``_origin_pos`` / ``_origin_rot``: latched on the engage edge — the controller
      pose the per-frame delta is measured against.

    Each engaged frame :meth:`rebase` returns::

        pos = home_pos + (grip_pos - origin_pos)     # 1:1 controller -> EE translation
        rot = (R_ctrl @ R_origin^-1) @ R_home        # base-frame delta, left-composed

    On the engage edge ``grip_pos == origin_pos`` and ``R_ctrl == R_origin``, so the
    output is exactly the home pose (== the last commanded pose), i.e. no teleport in
    position OR orientation. The orientation delta is expressed in the base frame
    (left multiply), so rotating the hand 30° about base Z rotates the EE 30° about
    base Z — matching the position convention the operator sees in the room. A
    mid-task re-clutch latches a fresh home/origin, so the EE resumes from where it
    was left and tracks the new delta.

    NOTE: ``_home_rot`` is the last *commanded* orientation, not the achieved one. On
    the 5-DOF SO-101 the arm cannot fully realize an arbitrary orientation, so the
    commanded and achieved wrist orientation differ — but the commanded signal is
    continuous across a re-clutch, so there is still no jump.
    """

    def __init__(self, home_base_T_ee: np.ndarray):  # noqa: N803
        # Seed the held pose from the arm's measured startup EE pose so the first
        # engage latches home there (no jump on the first squeeze).
        home = np.asarray(home_base_T_ee, dtype=float)
        self._last_commanded_pos = home[:3, 3].copy()
        self._last_commanded_rot = Rotation.from_matrix(home[:3, :3])
        self._home_pos = self._last_commanded_pos.copy()
        self._home_rot = self._last_commanded_rot
        self._origin_pos = np.zeros(3, dtype=float)
        self._origin_rot = Rotation.from_quat(np.array([0.0, 0.0, 0.0, 1.0]))

    def engage(self, grip_pos: np.ndarray, grip_quat: np.ndarray) -> None:
        """Latch the engage home (where the arm is now) and controller origin."""
        self._home_pos = self._last_commanded_pos.copy()
        self._home_rot = self._last_commanded_rot
        self._origin_pos = np.asarray(grip_pos, dtype=float).copy()
        self._origin_rot = Rotation.from_quat(np.asarray(grip_quat, dtype=float))

    def rebase(self, grip_pos: np.ndarray, grip_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the absolute base-frame EE target ``(pos [m], quat [xyzw])`` for this frame."""
        pos = self._home_pos + (np.asarray(grip_pos, dtype=float) - self._origin_pos)
        rot_ctrl = Rotation.from_quat(np.asarray(grip_quat, dtype=float))
        rot = (rot_ctrl * self._origin_rot.inv()) * self._home_rot
        self._last_commanded_pos = pos.copy()
        self._last_commanded_rot = rot
        return pos, rot.as_quat()


def _load_reset_target(motor_names: list[str]) -> dict[str, float]:
    """Return reset targets: reset_pose.json if present, else RESET_ORIGIN_DEG."""
    if RESET_POSE_FILE.exists():
        saved = json.loads(RESET_POSE_FILE.read_text())
        # Fill any missing motors from the fallback dict.
        return {name: float(saved.get(name, RESET_ORIGIN_DEG.get(name, 0.0))) for name in motor_names}
    return {name: RESET_ORIGIN_DEG.get(name, 0.0) for name in motor_names}


def move_to_origin(robot, motor_names: list[str], duration_s: float = RESET_DURATION_S) -> None:
    """Linearly slew all joints from their current positions to the reset target.

    Target source priority: reset_pose.json (recorded by record_reset_pose.py)
    > RESET_ORIGIN_DEG (Lab-derived hardcoded fallback).
    """
    obs = robot.get_observation()
    start = {name: float(obs[f"{name}.pos"]) for name in motor_names}
    target = _load_reset_target(motor_names)
    source = "reset_pose.json" if RESET_POSE_FILE.exists() else "hardcoded defaults"
    print(f"Reset target source: {source}")
    n_steps = max(1, int(duration_s * FPS))
    print(f"Resetting to origin over {duration_s:.1f} s ({n_steps} steps)…")
    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        action = {f"{name}.pos": start[name] + alpha * (target[name] - start[name]) for name in motor_names}
        robot.send_action(action)
        precise_sleep(1.0 / FPS)
    print("Reset complete.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port the SO-101 follower arm is connected to (default: /dev/ttyACM0).",
    )
    parser.add_argument(
        "--id",
        type=str,
        default="so101_follower_arm",
        help="Device id for the SO-101 follower arm (selects its calibration; default: so101_follower_arm).",
    )
    parser.add_argument(
        "--reset-to-origin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Slew all joints to URDF origin before entering the teleop loop (default: True). "
             "Pass --no-reset-to-origin to keep the arm exactly where it is at startup.",
    )
    parser.add_argument(
        "--reset-duration",
        type=float,
        default=RESET_DURATION_S,
        metavar="SEC",
        help=f"Duration in seconds for the reset-to-origin slew (default: {RESET_DURATION_S}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    robot_config = SO100FollowerConfig(port=args.port, id=args.id, use_degrees=True)

    # SO100Follower is the shared SO-100/SO-101 follower class: so_follower
    # registers the same class under both "so100_follower" and "so101_follower".
    # Here it is configured for SO-101 (see the so101_new_calib.urdf below).
    robot = SO100Follower(robot_config)
    motor_names = list(robot.bus.motors.keys())

    # Loads ./SO101/so101_new_calib.urdf relative to this folder. Run
    # `python download_assets.py` from this directory first to fetch the URDF and
    # its meshes from the SO-ARM100 repo:
    # https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )

    # Connect the robot FIRST so the slew and clutch-home seed can use live joint
    # readings. With --reset-to-origin (default) the arm is smoothly moved to URDF
    # origin before the teleop loop starts; the clutch home is then seeded from the
    # post-slew measured pose so the first engage is jump-free. Pass
    # --no-reset-to-origin to skip the slew entirely.
    robot.connect()
    if args.reset_to_origin:
        move_to_origin(robot, motor_names, args.reset_duration)

    obs0 = robot.get_observation()
    q_measured_deg = np.array([float(obs0[f"{name}.pos"]) for name in motor_names], dtype=float)
    home_base_T_ee = kinematics_solver.forward_kinematics(q_measured_deg)  # noqa: N806

    # Engage-relative clutch, seeded with the measured startup EE pose so the first
    # squeeze latches its home there (no jump on engage). Owns ALL the calibration
    # (position AND orientation) that used to live in the Isaac Teleop retargeters.
    clutch = Clutch(home_base_T_ee)

    teleop_config = XRControllerConfig(
        hand_side="right",
        clutch_threshold=0.5,
        cloudxr_env_file=CLOUDXR_ENV_FILE,
    )
    teleop_device = XRController(teleop_config)

    # Post-processing: rebased EE pose action -> joint action. The clutch (above)
    # turns the raw controller grip pose into an absolute base-frame ee_pose; these
    # steps map it to joint targets.
    xr_to_robot_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            # Stateless rename: ee.x/y/z = the clutch's absolute base-frame target,
            # ee.w* = 0 (orientation free), ee.gripper_pos = (1 - closedness) * 100.
            MapXRControllerActionToRobotAction(),
            # Clip to the workspace + RATE-LIMIT each frame. raise_on_jump=False:
            # an over-limit step (e.g. a transient XR controller tracking glitch)
            # is clamped to MAX_EE_STEP_M and warned, NOT raised -- a crash mid-loop
            # would leave the arm uncontrolled. A glitch is absorbed as one slow
            # frame; a target that is *persistently* out of reach will warn every
            # frame (investigate base_T_anchor, not this clamp).
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=MAX_EE_STEP_M,
                raise_on_jump=False,
            ),
            # Soft-orientation IK (orientation_weight=IK_ORIENTATION_WEIGHT): the
            # clutch-rebased controller orientation is fed as a target so the wrist
            # follows the hand, but the weight is small so position dominates — the
            # SO-101 is 5-DOF and cannot realize an arbitrary 3-DOF orientation.
            # initial_guess_current_joints=True: seed each solve from the MEASURED
            # joints so the IK tracks physical reality and stays robust to external
            # moves (e.g. the hold while disengaged), rather than warm-starting from
            # the previous solution.
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=motor_names,
                initial_guess_current_joints=True,
                orientation_weight=IK_ORIENTATION_WEIGHT,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Robot is already connected (above, for the measured-home seed). Connect teleop.
    teleop_device.connect()

    if not teleop_device.is_connected:
        raise ValueError("Teleop is not connected!")

    _prev_enabled = False

    print("Starting teleop loop. Squeeze and move the controller to teleoperate the robot...")
    while True:
        t0 = time.perf_counter()

        robot_obs = robot.get_observation()
        xr_action = teleop_device.get_action()

        grip_pos = np.asarray(xr_action["grip_pos"], dtype=float)
        grip_quat = np.asarray(xr_action["grip_quat"], dtype=float)
        squeeze = float(xr_action["squeeze"])
        trigger = float(xr_action["trigger"])
        enabled = squeeze > teleop_config.clutch_threshold

        # Compute once per frame; _prev_enabled is updated at the very bottom of
        # the loop.
        _is_engage_frame = enabled and not _prev_enabled

        # On the engage edge, latch the clutch home (current arm EE) and the
        # controller origin so the per-frame delta starts at zero (no jump).
        if _is_engage_frame:
            clutch.engage(grip_pos, grip_quat)

        # SAFETY GATE: command the robot ONLY while the clutch is engaged. While
        # disengaged, re-send the MEASURED joints (an explicit hold) so the arm
        # stays exactly where it is — launching the script (clutch released) never
        # moves it, and releasing the clutch mid-session freezes it in place.
        if enabled:
            # Rebase the raw grip pose (position AND orientation) onto the EE, then
            # run the post-processing pipeline (rename -> bounds -> IK). closedness
            # from the trigger.
            ee_pos, ee_quat = clutch.rebase(grip_pos, grip_quat)
            ee_action = {
                "ee_pose": np.concatenate([ee_pos, ee_quat]).astype(np.float32),
                "closedness": trigger,
            }
            joint_action = xr_to_robot_joints_processor((ee_action, robot_obs))
        else:
            joint_action = {f"{name}.pos": float(robot_obs[f"{name}.pos"]) for name in motor_names}

        _prev_enabled = enabled

        _ = robot.send_action(joint_action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()

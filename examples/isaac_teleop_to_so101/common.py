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

"""Shared device + control-loop infrastructure for the Isaac Teleop -> SO-101 examples.

Both entry points — ``teleoperate.py`` (drive the arm live) and ``record.py`` (drive +
save a LeRobot dataset) — build a per-device :class:`Device` bundle here and run the same
read -> (maybe command) -> hold-when-idle -> sleep loop. ``--teleop.type`` selects the Isaac
input device (``xr_controller`` | ``so101_leader``); :func:`build_device` connects the
follower, dispatches to the right ``setup_*``, and runs its pre-loop ``startup``.

Each frame a device-specific ``compute`` returns the joint action, or ``None`` to mean
"device idle -> hold at the measured pose" (XR clutch disengaged, or leader stream stale).
Per-device ``setup_*`` builds that closure plus a ``startup`` (pre-loop slew / warm-up) and
``cleanup`` (reap / disconnect); the outer loop is branchless.

``xr`` device
-------------
The XR device is a thin reader: it emits the **raw** controller grip pose (already
rebased into the robot base frame), the squeeze, and the trigger. ALL the calibration
lives here in the loop — a small :class:`Clutch` latches the controller origin on engage
and drives the EE from the delta, so the device carries no per-frame state::

    XRController.get_action()                       # raw base-frame grip_pos/grip_quat + squeeze + trigger
      -> Clutch.rebase(grip_pos, grip_quat)          # ee_pose = engage-relative delta applied to the EE home (pos + orient)
      -> MapXRControllerActionToRobotAction          # ee.x/y/z = abs pos, ee.w* = abs orient rotvec, ee.gripper_pos = f(trigger)
      -> EEBoundsAndSafety                           # workspace clip + per-frame jump clamp (position only)
      -> InverseKinematicsEEToJoints(ow=small)       # soft-orientation Placo IK (passes ee.gripper_pos -> gripper.pos)

Squeeze (and hold) the controller grip past ``clutch_threshold`` to engage; on the
engage edge the clutch latches its origin to the current controller pose and its home to
the last commanded EE pose, so the arm does not jump in position OR orientation. The
clutch rebases BOTH position and orientation (engage-relative base-frame deltas); the
orientation target is fed to the IK with a small weight (``IK_ORIENTATION_WEIGHT``) so
the wrist follows the hand while position dominates (the 5-DOF SO-101 cannot fully
realize an arbitrary orientation). The analog trigger drives an absolute
``ee.gripper_pos`` jaw target.

XR startup / safety contract: by default the script slews all joints to a default reset
pose (a mid-range arm pose with the gripper open) over ``--reset_duration`` seconds
before entering the loop. Pass ``--reset_to_origin=false`` to skip this slew and keep the
arm exactly where it is. After the slew (or if skipped) the clutch seeds its home from
the arm's MEASURED pose (FK of the joints read right after the slew), so the seeded home
equals the post-reset position and the first engage is jump-free. The robot is commanded
ONLY while the clutch is engaged; while disengaged the loop re-sends the measured joints
(an explicit hold), and releasing the clutch freezes it in place.

NOTE: EEBoundsAndSafety clamps (not raises) on a per-frame jump > max_ee_step_m; the
clutch's no-teleport keeps frames small, but set a generous bound for bring-up.

``leader`` device
-----------------
The input is a back-drivable SO-101 *leader arm* whose six joint angles are streamed by
Isaac Teleop's native ``so101_leader`` plugin over the OpenXR tensor transport. Because
the leader and follower share the SO-101 kinematics, the control law is a direct 1:1
joint mirror -- no clutch, no IK, no URDF::

    so101_leader plugin ──(JointStateOutput over OpenXR)──▶ SO101LeaderArm.get_action()
                                                                  │  rad2deg + gripper->RANGE_0_100
                                                                  ▼
                                                          robot.send_action({joint}.pos)

:class:`SO101LeaderArm` does the unit conversion internally, so the loop is a thin
read->send mirror. Pieces that must be running:

* **CloudXR runtime** -- auto-launched by ``SO101LeaderArm.connect()`` (the shared
  ``IsaacTeleopTeleoperator`` base; first launch may prompt for the EULA and take ~30s).
  Opt out with ``--teleop.auto_launch_cloudxr=false`` if you run CloudXR externally.
* **so101_leader plugin** -- the C++ device that reads the physical leader's servos and
  pushes ``JointStateOutput``. Either start it yourself (``so101_leader_plugin <port>``)
  or pass ``--launch_plugin <path>`` (optionally with ``--teleop.port <port>``) to have
  the entry point spawn it AFTER CloudXR is up (so it inherits the runtime env). With no
  ``--teleop.port`` the plugin runs its synthetic trajectory -- a no-hardware dry run.
  When a port is set, the plugin's tick->radian calibration is inferred from ``--teleop.id``
  and forwarded as the plugin's third positional arg: the LeRobot-format JSON at
  ``HF_LEROBOT_CALIBRATION/teleoperators/so_leader/<id>.json`` (the same file the serial
  SO-101 leader uses). If it does not exist the script warns and the plugin uses its
  built-in defaults; calibrate via ``lerobot-calibrate --teleop.type=so101_leader
  --teleop.id=<id>`` or the plugin's ``calibrate`` subcommand.

Leader startup safety: by default the follower is smoothly slewed from its current pose
to the leader's first reading over ``--align_duration`` seconds (``--align=false`` to
skip), so the arm does not snap when the 1:1 mirror begins. While the leader is not
streaming the follower is held at its measured pose.

Requires the ``isaac-teleop`` extra (``isaacteleop``) and an OpenXR runtime.
"""

import atexit
import json
import logging
import os
import select
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.so_follower import SOFollowerConfig  # noqa: F401  (registers so101_follower)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.isaac_teleop import (
    IsaacTeleopConfig,
    MapXRControllerActionToRobotAction,
    SO101LeaderArm,
    SO101LeaderArmConfig,
    XRController,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation

# Terminal control for the stdin keyboard backend is Unix-only (absent on Windows).
try:
    import termios

    _HAS_TERMIOS = True
except ImportError:  # pragma: no cover - Windows
    _HAS_TERMIOS = False

FPS = 30

# CloudXR device-profile env file passed to the launcher (see default.env next to this
# script). Resolved absolutely so it loads regardless of the working dir. Shared by both
# devices.
CLOUDXR_ENV_FILE = str(Path(__file__).parent / "default.env")


class LoopConfig(Protocol):
    """Structural type for the loop/launch knobs ``build_device`` + the ``setup_*`` read.

    Both ``TeleoperateConfig`` and ``RecordConfig`` satisfy this (a knob is ignored for the
    device it does not apply to); typing against the Protocol keeps ``common`` decoupled from
    either entry point's concrete config.
    """

    teleop: IsaacTeleopConfig
    robot: RobotConfig
    launch_plugin: str | None
    reset_to_origin: bool
    reset_duration: float
    align: bool
    align_duration: float


# A per-device bundle returned by setup_xr / setup_leader and consumed by the one shared
# loop. ``compute(obs) -> RobotAction | None`` returns None to mean "idle -> hold at the
# measured pose"; ``startup`` runs the pre-loop slew/warm-up; ``cleanup`` reaps/disconnects.
@dataclass(frozen=True)
class Device:
    compute: Callable[[RobotObservation | None], RobotAction | None]
    startup: Callable[[], None]
    cleanup: Callable[[], None]


def hold_action(obs: RobotObservation, motor_names: list[str]) -> dict[str, float]:
    """Re-send the measured joints — the explicit hold when a device is idle."""
    return {f"{name}.pos": float(obs[f"{name}.pos"]) for name in motor_names}


def slew(
    robot,
    motor_names: list[str],
    target_fn: Callable[[], dict[str, float]],
    duration_s: float,
) -> None:
    """Linearly slew all joints from their current measured pose toward a target.

    ``target_fn`` is called EACH step and returns the per-joint target in motor units.
    XR passes a constant closure (the fixed reset pose); the leader passes a LIVE re-read
    (``lambda: leader_now``) so at ``alpha == 1`` the follower lands on the leader's
    *current* pose and the handoff to the 1:1 mirror is continuous even if the operator
    keeps moving the leader during the ramp.
    """
    obs = robot.get_observation()
    start = {name: float(obs[f"{name}.pos"]) for name in motor_names}
    n_steps = max(1, int(duration_s * FPS))
    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        target = target_fn()
        action = {f"{name}.pos": start[name] + alpha * (target[name] - start[name]) for name in motor_names}
        robot.send_action(action)
        precise_sleep(1.0 / FPS)


# ============================================================================
# XR controller device
# ============================================================================

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
IK_ORIENTATION_WEIGHT = 0.01

# Default duration [s] for the startup reset-to-origin slew.
RESET_DURATION_S = 5.0

# Optional file written by override_reset_pose.py. When present its values take priority
# over RESET_ORIGIN_DEG.
RESET_POSE_FILE = Path(__file__).parent / "reset_pose.json"

# Reset target in each motor's native units (arm joints in degrees, gripper in
# MotorNormMode.RANGE_0_100 where 100 = fully open, 0 = fully closed). These are an
# empirically recorded comfortable pose (elbow/wrist bent) that avoids the boundary
# singularity of a fully-extended 5-DOF arm; they assume standard calibration where
# 0° = URDF 0 rad (homing pose). Override per-setup by back-driving the arm and running
# override_reset_pose.py, which writes reset_pose.json (it takes priority over these).
RESET_ORIGIN_DEG: dict[str, float] = {
    "shoulder_pan": -4.0,
    "shoulder_lift": -103.0,
    "elbow_flex": 97.0,
    "wrist_flex": 78.0,
    "wrist_roll": -65.0,
    "gripper": 0.0,
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

        pos = home_pos + (grip_pos - origin_pos)  # 1:1 controller -> EE translation
        rot = (R_ctrl @ R_origin ^ -1) @ R_home  # base-frame delta, left-composed

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


# CloudXR web client URL + the WSS-proxy/cert port (Isaac Teleop quick start, step 5).
_CLOUDXR_WEB_CLIENT_URL = "https://nvidia.github.io/IsaacTeleop/client"
_CLOUDXR_WSS_PORT = 48322
# How often to re-print the connection hint while waiting for the headset [s].
_XR_CONNECT_REMINDER_S = 15.0
# Virtual / bridge / USB-gadget interfaces a headset can't reach over the network — skip
# by name prefix (``docker0``, compose ``br-*``, ``veth*``, libvirt ``virbr*``, and the
# Tegra USB device-mode bridge ``l4tbr0``).
_SKIP_IFACE_PREFIXES = ("docker", "br-", "veth", "virbr", "l4tbr")


def _primary_ipv4() -> str | None:
    """The workstation's primary outbound IPv4 (the default-route interface).

    Standard UDP-socket trick: ``connect()`` on a datagram socket selects the egress
    interface WITHOUT sending any packets, then ``getsockname()`` reports its IP.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def _candidate_ipv4s() -> list[tuple[str, str]]:
    """Return ``[(interface, ipv4), ...]`` the headset might reach this workstation at.

    Lists each interface's IPv4 (via ``psutil`` when available), dropping loopback
    (127.x) and link-local (169.254.x) addresses plus virtual/bridge/USB-gadget
    interfaces (see ``_SKIP_IFACE_PREFIXES``), with the primary outbound interface
    first. Falls back to just the primary IP when ``psutil`` is unavailable.
    """
    primary = _primary_ipv4()
    found: list[tuple[str, str]] = []
    try:
        import psutil

        for iface, addrs in psutil.net_if_addrs().items():
            if iface.startswith(_SKIP_IFACE_PREFIXES):
                continue
            for addr in addrs:
                if addr.family != socket.AF_INET:
                    continue
                ip = addr.address
                if ip.startswith("127.") or ip.startswith("169.254."):
                    continue
                found.append((iface, ip))
    except Exception:
        if primary:
            found.append(("default", primary))
    found.sort(key=lambda t: t[1] != primary)  # primary outbound interface first
    return found


def _print_xr_connect_help() -> None:
    """Print how to connect the headset to this workstation over CloudXR."""
    ips = _candidate_ipv4s()
    print("\n" + "=" * 76)
    print("Connect your XR headset to this workstation over NVIDIA CloudXR:")
    print(f"  1. In the headset, open the CloudXR web client:  {_CLOUDXR_WEB_CLIENT_URL}")
    print("  2. Enter this workstation's IP address:")
    if ips:
        for iface, ip in ips:
            print(f"        {ip:<15}  ({iface})")
        if len(ips) > 1:
            print("     (use the address on the same network as your headset)")
    else:
        print("        <could not determine — check `hostname -I` / `ip addr`>")
    print(f"  3. Accept the self-signed cert at https://<that-ip>:{_CLOUDXR_WSS_PORT}/ , then Connect.")
    print("=" * 76 + "\n")


def _wait_for_xr_controller(teleop_device: XRController) -> None:
    """Block until the XR controller is tracked (headset connected + controllers live).

    Prints connection instructions, then polls ``get_action()`` until
    :attr:`XRController.is_tracking`, re-printing a reminder every
    ``_XR_CONNECT_REMINDER_S`` seconds. User-paced; ``Ctrl-C`` aborts (no hard timeout —
    donning a headset and connecting is operator-driven). Mirrors :func:`_wait_for_leader`.
    """
    _print_xr_connect_help()
    print("Waiting for the headset controllers to start streaming…  (Ctrl-C to abort)")
    last_reminder = time.time()
    while True:
        teleop_device.get_action()  # steps the session; updates is_tracking
        if teleop_device.is_tracking:
            print("Headset connected — controllers are streaming.")
            return
        if time.time() - last_reminder >= _XR_CONNECT_REMINDER_S:
            print("…still waiting for the headset to connect (Ctrl-C to abort).")
            last_reminder = time.time()
        time.sleep(1.0 / FPS)


def setup_xr(cfg: LoopConfig, robot, motor_names: list[str]) -> Device:
    """Build the XR controller device bundle (clutch + soft-orientation IK pipeline)."""
    # Loads ./SO101/so101_new_calib.urdf relative to this folder. Run
    # `python download_assets.py` from this directory first to fetch the URDF and
    # its meshes from the SO-ARM100 repo:
    # https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )

    teleop_config = cfg.teleop  # XRControllerConfig (selected via --teleop.type=xr_controller)
    teleop_device = XRController(teleop_config)

    # Post-processing: rebased EE pose action -> joint action. The clutch (below)
    # turns the raw controller grip pose into an absolute base-frame ee_pose; these
    # steps map it to joint targets.
    xr_to_robot_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            # Stateless rename: ee.x/y/z = the clutch's absolute base-frame position,
            # ee.wx/wy/wz = the clutch's absolute base-frame orientation rotvec (fed to the
            # IK at IK_ORIENTATION_WEIGHT), ee.gripper_pos = (1 - closedness) * 100.
            MapXRControllerActionToRobotAction(),
            # Clip to the workspace + RATE-LIMIT each frame. raise_on_jump=False:
            # an over-limit step (e.g. a transient XR controller tracking glitch)
            # is clamped to MAX_EE_STEP_M and warned, NOT raised -- a crash mid-loop
            # would leave the arm uncontrolled. A glitch is absorbed as one slow
            # frame; a target that is *persistently* out of reach will warn every
            # frame (investigate base_T_anchor, not this clamp).
            # Workspace clip: the z floor is 0.0 (the table plane) so a stray target
            # cannot drive the EE below the table; x/y stay at the loose [-1,1]m box.
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=MAX_EE_STEP_M,
                raise_on_jump=False,
            ),
            # Soft-orientation IK (orientation_weight=IK_ORIENTATION_WEIGHT): the
            # clutch-rebased controller orientation is fed as a target so the wrist
            # follows the hand, but the weight is small so position dominates — the
            # SO-101 is 5-DOF and cannot realize an arbitrary 3-DOF orientation.
            # initial_guess_current_joints=False: warm-start each solve from the
            # PREVIOUS IK solution rather than re-seeding from the measured joints, so
            # the joint trajectory stays continuous frame-to-frame.
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=motor_names,
                initial_guess_current_joints=False,
                orientation_weight=IK_ORIENTATION_WEIGHT,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # The clutch is built in startup() (after the optional reset slew, seeded from the
    # post-slew MEASURED pose) and shared with compute() via nonlocal.
    clutch: Clutch | None = None
    prev_enabled = False

    def startup() -> None:
        nonlocal clutch
        # Bring up CloudXR + the OpenXR session FIRST, then wait for the operator to don
        # the headset and connect BEFORE moving the arm — so the reset slew happens while
        # they are watching in VR (mirrors the leader waiting for its plugin to stream).
        teleop_device.connect()
        if not teleop_device.is_connected:
            raise ValueError("Teleop is not connected!")
        _wait_for_xr_controller(teleop_device)

        if cfg.reset_to_origin:
            target = _load_reset_target(motor_names)
            source = "reset_pose.json" if RESET_POSE_FILE.exists() else "hardcoded defaults"
            print(f"Reset target source: {source}")
            print(f"Resetting to origin over {cfg.reset_duration:.1f} s…")
            slew(robot, motor_names, lambda: target, cfg.reset_duration)
            print("Reset complete.")

        # Seed the clutch home from the arm's MEASURED pose (FK of the joints read right
        # after the slew, or right now under --reset_to_origin=false). Runs UNCONDITIONALLY
        # so the first engage is jump-free either way.
        obs0 = robot.get_observation()
        q_measured_deg = np.array([float(obs0[f"{name}.pos"]) for name in motor_names], dtype=float)
        home_base_T_ee = kinematics_solver.forward_kinematics(q_measured_deg)  # noqa: N806
        clutch = Clutch(home_base_T_ee)

        print("Starting teleop loop. Squeeze and move the controller to teleoperate the robot...")

    def compute(robot_obs: RobotObservation | None) -> RobotAction | None:
        nonlocal prev_enabled
        assert clutch is not None  # set in startup(), which runs before compute()
        xr_action = teleop_device.get_action()
        grip_pos = np.asarray(xr_action["grip_pos"], dtype=float)
        grip_quat = np.asarray(xr_action["grip_quat"], dtype=float)
        squeeze = float(xr_action["squeeze"])
        trigger = float(xr_action["trigger"])
        enabled = squeeze > teleop_config.clutch_threshold

        # On the engage edge, latch the clutch home (current arm EE) and the controller
        # origin so the per-frame delta starts at zero (no jump).
        is_engage_frame = enabled and not prev_enabled
        if is_engage_frame:
            clutch.engage(grip_pos, grip_quat)
        prev_enabled = enabled

        # SAFETY GATE: command the robot ONLY while the clutch is engaged. While
        # disengaged, return None so the shared loop re-sends the MEASURED joints (an
        # explicit hold) — launching the script (clutch released) never moves the arm,
        # and releasing the clutch mid-session freezes it in place.
        if not enabled:
            return None

        # Rebase the raw grip pose (position AND orientation) onto the EE, then run the
        # post-processing pipeline (rename -> bounds -> IK). closedness from the trigger.
        ee_pos, ee_quat = clutch.rebase(grip_pos, grip_quat)
        ee_action = {
            "ee_pose": np.concatenate([ee_pos, ee_quat]).astype(np.float32),
            "closedness": trigger,
        }
        return xr_to_robot_joints_processor((ee_action, robot_obs))

    return Device(compute=compute, startup=startup, cleanup=teleop_device.disconnect)


# ============================================================================
# SO-101 leader arm device
# ============================================================================

# Default duration [s] for the startup alignment slew (follower current -> leader first pose).
ALIGN_DURATION_S = 3.0

# How long to wait for the leader plugin to start streaming before aligning / looping.
LEADER_WARMUP_TIMEOUT_S = 20.0

# Calibration subdir under HF_LEROBOT_CALIBRATION/teleoperators/. The plugin (NOT the LeRobot
# device) converts the leader's servo ticks to radians, so it needs the SAME tick->radian
# calibration the serial SO-101 leader uses. lerobot-calibrate stores that under the serial
# leader's name -- SO101Leader.name == "so_leader" (the shared SO-100/SO-101 leader dir) -- so the
# Isaac flow reuses that file rather than maintaining its own under the device's own name.
SO_LEADER_CALIBRATION_NAME = "so_leader"


def _leader_calibration_path(cfg: LoopConfig) -> Path | None:
    """Infer the LeRobot-format calibration JSON the launched plugin should read, or None.

    Path convention (mirrors the serial SO-101 leader): ``HF_LEROBOT_CALIBRATION /
    teleoperators / so_leader / {--teleop.id}.json`` (or ``--teleop.calibration_dir`` if set).
    Returns the path only when it exists; otherwise returns None so the plugin falls back to
    its built-in defaults, warning when an id was given but no file was found. With no
    ``--teleop.id`` the path cannot be inferred (returns None silently).
    """
    if not cfg.teleop.id:
        return None
    calib_dir = cfg.teleop.calibration_dir or (
        HF_LEROBOT_CALIBRATION / TELEOPERATORS / SO_LEADER_CALIBRATION_NAME
    )
    calib_path = Path(calib_dir) / f"{cfg.teleop.id}.json"
    if calib_path.is_file():
        return calib_path
    print(
        f"WARNING: no leader calibration at {calib_path}; the plugin will use built-in defaults. "
        f"Calibrate with the serial leader (`lerobot-calibrate --teleop.type=so101_leader "
        f"--teleop.id={cfg.teleop.id}`) or the plugin's `calibrate` subcommand."
    )
    return None


def _wait_for_leader(teleop: SO101LeaderArm, timeout_s: float) -> dict[str, float]:
    """Poll the leader until it streams a live frame; return that frame's ``{joint}.pos``.

    Raises ``SystemExit`` if no live frame arrives within ``timeout_s`` (plugin not pushing,
    wrong ``--teleop.collection_id``, or CloudXR not up).
    """
    print(f"Waiting up to {timeout_s:.0f}s for the so101_leader plugin to stream…")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        action = teleop.get_action()
        if teleop.is_tracking:
            print("Leader is streaming.")
            return action
        time.sleep(1.0 / FPS)
    raise SystemExit(
        f"FAILED: leader did not stream within {timeout_s:.0f}s. Is the so101_leader plugin "
        "running and pushing (check --teleop.collection_id)? Is CloudXR up?"
    )


def _maybe_launch_plugin(cfg: LoopConfig) -> subprocess.Popen | None:
    """Spawn the so101_leader plugin if ``--launch_plugin <path>`` was given (after connect())."""
    if cfg.launch_plugin is None:
        return None
    if not Path(cfg.launch_plugin).exists():
        raise SystemExit(
            f"plugin binary not found: {cfg.launch_plugin} (build it in the IsaacTeleop repo first)"
        )
    leader_port = cfg.teleop.port  # SO101LeaderArmConfig.port, forwarded to the plugin
    backend = f"leader on {leader_port}" if leader_port else "synthetic trajectory"
    print(f"launching plugin: {cfg.launch_plugin} ({backend})")
    # Positional args: [device_path] [collection_id] [calibration_file]. Empty device_path ->
    # synthetic backend. The calibration is the leader's tick->radian map; only real hardware
    # needs it, so it is inferred (from --teleop.id) and appended only when a port is set.
    argv = [cfg.launch_plugin, leader_port, cfg.teleop.collection_id]
    if leader_port:
        calib_path = _leader_calibration_path(cfg)
        if calib_path is not None:
            argv.append(str(calib_path))
            print(f"  leader calibration: {calib_path}")
    # Spawned after connect() so it inherits the CloudXR runtime env (XR_RUNTIME_JSON, ...).
    proc = subprocess.Popen(argv)
    time.sleep(1.5)  # let it create its OpenXR session and start pushing
    return proc


def setup_leader(cfg: LoopConfig, robot, motor_names: list[str]) -> Device:
    """Build the SO-101 leader arm device bundle (1:1 joint mirror)."""
    teleop_config = cfg.teleop  # SO101LeaderArmConfig (selected via --teleop.type=so101_leader)
    teleop = SO101LeaderArm(teleop_config)

    plugin_proc: subprocess.Popen | None = None

    def startup() -> None:
        nonlocal plugin_proc
        # connect() auto-launches CloudXR (unless opted out); spawn the plugin AFTER so it
        # inherits the runtime env. The plugin is reaped in cleanup().
        teleop.connect()
        plugin_proc = _maybe_launch_plugin(cfg)

        if not teleop.is_connected:
            raise ValueError("Teleop is not connected!")

        # Block until the leader streams a live frame (clear error if it never does).
        _wait_for_leader(teleop, LEADER_WARMUP_TIMEOUT_S)

        if cfg.align:
            print(f"Aligning follower to leader over {cfg.align_duration:.1f}s…")

            # Re-read the LIVE leader pose ONCE per step (slew calls this each step) so
            # alpha=1 lands on the leader's current pose; reading once per step keeps every
            # joint of the target from a single coherent frame.
            def _leader_target() -> dict[str, float]:
                leader_now = teleop.get_action()
                return {name: float(leader_now[f"{name}.pos"]) for name in motor_names}

            slew(robot, motor_names, _leader_target, cfg.align_duration)
            print("Alignment complete.")

        print(
            "Starting joint-mirror loop. Back-drive the leader to teleoperate the follower… (Ctrl-C to stop)"
        )

    def compute(robot_obs: RobotObservation | None) -> RobotAction | None:
        leader_action = teleop.get_action()
        # Hold the follower at its measured pose when the leader drops out (stale stream),
        # rather than commanding a held-last (possibly old) target: return None so the shared
        # loop re-sends the measured joints.
        if not teleop.is_tracking:
            return None
        return leader_action

    def cleanup() -> None:
        if plugin_proc is not None:
            plugin_proc.terminate()
            try:
                plugin_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                plugin_proc.kill()
        teleop.disconnect()

    return Device(compute=compute, startup=startup, cleanup=cleanup)


# ============================================================================
# Shared setup
# ============================================================================


def build_device(cfg: LoopConfig) -> tuple:
    """Connect the follower, build the selected Isaac device, and run its pre-loop startup.

    Shared by ``teleoperate.py`` and ``record.py``. Defaults the CloudXR profile, connects the
    SO-101 follower FIRST (so the startup slew / clutch-home seed can read live joints),
    dispatches on ``--teleop.type`` (``so101_leader`` -> :func:`setup_leader`, else
    :func:`setup_xr`), then runs ``device.startup()`` (XR reset slew / leader align + warm-up)
    BEFORE returning — matching the original order, where startup runs before any dataset
    creation in ``record.py``.

    Returns ``(robot, device, motor_names)``.
    """
    # Default the CloudXR input profile to this example's default.env unless the user overrode
    # it via --teleop.cloudxr_env_file.
    if cfg.teleop.cloudxr_env_file is None:
        cfg.teleop.cloudxr_env_file = CLOUDXR_ENV_FILE

    # so_follower registers the same follower class under both "so100_follower" and
    # "so101_follower"; here it is configured for SO-101 (see the so101_new_calib.urdf the xr
    # device loads). The degree-based pipeline relies on --robot.use_degrees (default True).
    robot = make_robot_from_config(cfg.robot)
    # Connect the follower FIRST so the startup slew and clutch-home seed can use live joint
    # readings.
    robot.connect()
    motor_names = list(robot.bus.motors.keys())

    # Dispatch on the parsed device config type (the registry only yields these two).
    if isinstance(cfg.teleop, SO101LeaderArmConfig):
        device = setup_leader(cfg, robot, motor_names)
    else:
        device = setup_xr(cfg, robot, motor_names)

    device.startup()
    return robot, device, motor_names


# ============================================================================
# Keyboard control (stdin backend; works over SSH / Wayland / headless-with-a-tty)
# ============================================================================

# Backends for init_keyboard_listener(), selectable via LEROBOT_KEYBOARD_BACKEND.
_VALID_KEYBOARD_BACKENDS = {"auto", "stdin", "pynput", "none"}


def _stdin_is_tty() -> bool:
    """True when stdin is an interactive terminal we can read keys from."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


class _StdinKeyboardListener:
    """Read arrow / Esc keys from a raw-mode stdin TTY (SSH- and Wayland-friendly).

    Drop-in for the pynput listener in ``lerobot.common.control_utils``: same ``.stop()``
    method and the same effect on the shared ``events`` dict. Where the global pynput/X
    listener can't see keystrokes (over SSH, under Wayland, or headless-with-a-tty), stdin
    still receives them. Puts the terminal in non-canonical, no-echo mode — KEEPING signals
    so Ctrl-C still raises ``KeyboardInterrupt`` — polls stdin on a daemon thread, and ALWAYS
    restores the original terminal mode on :meth:`stop` and at interpreter exit (so a crash
    or Ctrl-C never leaves the shell with echo off).
    """

    def __init__(self, events: dict):
        self._events = events
        self._fd = sys.stdin.fileno()
        self._old_term = termios.tcgetattr(self._fd)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._restored = False
        # Belt-and-braces: restore the tty even if stop() is never reached (crash / hard exit).
        atexit.register(self._restore)

    def start(self) -> None:
        new = termios.tcgetattr(self._fd)
        # Clear ICANON (line buffering) + ECHO; KEEP ISIG so Ctrl-C / Ctrl-Z still signal.
        new[3] &= ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(self._fd, termios.TCSADRAIN, new)
        self._thread = threading.Thread(target=self._run, name="isaac-stdin-keys", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
                if not ready:
                    continue
                data = os.read(self._fd, 1024)
            except (OSError, ValueError):
                break  # fd closed / stdin gone
            if data:
                self._dispatch(data)

    def _dispatch(self, data: bytes) -> None:
        # Read the whole available buffer and walk it: an arrow key arrives as the 3 bytes
        # ``ESC [ C/D`` together, while a real Esc is a lone ESC — matching the full sequence
        # avoids mistaking Esc for the start of an arrow.
        events = self._events
        i, n = 0, len(data)
        while i < n:
            if data[i] != 0x1B:  # only ESC-initiated keys are meaningful here
                i += 1
                continue
            seq = data[i : i + 3]
            if seq == b"\x1b[C":  # Right arrow
                print("Right arrow pressed. Ending the episode early...")
                events["exit_early"] = True
                i += 3
            elif seq == b"\x1b[D":  # Left arrow
                print("Left arrow pressed. Re-recording the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
                i += 3
            elif data[i : i + 2] == b"\x1b[":  # other CSI sequence (Up/Down/...) -> ignore
                i += 3 if n - i >= 3 else n
            else:  # lone ESC (not the start of an arrow) -> stop recording
                print("Escape pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
                i += 1

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._restore()

    def _restore(self) -> None:
        if self._restored:
            return
        self._restored = True
        with suppress(Exception):
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)


def _new_events() -> dict:
    return {"exit_early": False, "rerecord_episode": False, "stop_recording": False}


def init_keyboard_listener():
    """Keyboard listener for the recording shortcuts, with a stdin backend for SSH / headless.

    Drop-in for ``lerobot.common.control_utils.init_keyboard_listener`` but prefers a stdin
    (termios) backend so the Right / Left / Esc shortcuts work over SSH, under Wayland, and on
    headless boxes that have a TTY — where the upstream global pynput/X listener silently
    captures nothing. Backend selection (override with ``LEROBOT_KEYBOARD_BACKEND`` ∈
    {auto, stdin, pynput, none}, default auto):

    - interactive TTY -> the stdin backend in this module;
    - otherwise (GUI launch, no tty) -> delegate to the upstream pynput / headless listener;
    - ``none`` -> headless (returns ``None``).

    Returns ``(listener, events)`` where ``listener`` has ``.stop()`` (or is ``None`` when
    headless), and ``events`` is a dict with bool flags ``exit_early`` / ``rerecord_episode`` /
    ``stop_recording`` set by the key presses.
    """
    choice = os.environ.get("LEROBOT_KEYBOARD_BACKEND", "auto").strip().lower()
    if choice not in _VALID_KEYBOARD_BACKENDS:
        logging.warning("Unknown LEROBOT_KEYBOARD_BACKEND=%r; using 'auto'.", choice)
        choice = "auto"

    stdin_ok = _HAS_TERMIOS and _stdin_is_tty()

    if choice == "none":
        return None, _new_events()
    if choice in ("auto", "stdin") and stdin_ok:
        events = _new_events()
        listener = _StdinKeyboardListener(events)
        listener.start()
        logging.info(
            "Keyboard control via stdin (terminal): Right = end episode early, Left = re-record, Esc = stop."
        )
        return listener, events
    if choice == "stdin":  # forced stdin but no usable TTY -> headless
        return None, _new_events()

    # auto without a TTY, or choice == 'pynput': defer to the upstream pynput / headless listener.
    from lerobot.common.control_utils import init_keyboard_listener as _upstream_listener

    return _upstream_listener()

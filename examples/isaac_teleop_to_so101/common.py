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

Consumed by ``teleoperate.py`` (drive the arm live) and ``record.py`` (drive + save a
LeRobot dataset). Both build a per-device :class:`Device` bundle here and run the same
branchless loop: read -> (maybe command) -> hold-when-idle -> sleep. :func:`build_device`
connects the SO-101 follower, dispatches on ``--teleop.type`` (``xr_controller`` |
``so101_leader``) to the matching ``setup_*``, and runs its pre-loop ``startup``.

A :class:`Device` bundles three closures: ``compute(obs) -> RobotAction | None`` (the
per-frame action, or ``None`` to hold at the measured pose when the device is idle — XR
clutch disengaged or leader stream stale), ``startup`` (pre-loop slew / warm-up), and
``cleanup`` (reap / disconnect). The two devices:

* ``xr_controller`` — a thin :class:`XRController` reader whose raw grip pose an in-loop
  :class:`Clutch` turns into an EE target for LeRobot's Cartesian IK pipeline.
* ``so101_leader`` — a back-drivable SO-101 leader arm mirrored 1:1 into the follower
  (no clutch, no IK).

Requires the ``isaac-teleop`` extra (``isaacteleop``) and an OpenXR runtime.

The user-facing guide — pipeline diagrams, the clutch/engage model, the startup safety
contracts, and the ``so101_leader`` plugin setup — is ``docs/source/isaac_teleop.mdx``;
this docstring only orients a reader of the source.
"""

import json
import logging
import socket
import subprocess
import sys
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
    Clutch,
    IsaacTeleopConfig,
    MapXRControllerActionToRobotAction,
    SO101LeaderArm,
    SO101LeaderArmConfig,
    XRController,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, HF_LEROBOT_HOME, TELEOPERATORS
from lerobot.utils.robot_utils import precise_sleep

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

def _ensure_so101_urdf() -> str:
    """Return the cached SO-101 URDF path, fetching the whole ``so101`` folder (URDF + meshes) from the public ``lerobot/robot-urdfs`` HF bucket into the LeRobot cache on first use and reusing it after."""
    dest_dir = HF_LEROBOT_HOME / "robot-urdfs" / "so101"
    urdf_path = dest_dir / "so101_new_calib.urdf"
    if not urdf_path.exists():
        from huggingface_hub import sync_bucket

        sync_bucket("hf://buckets/lerobot/robot-urdfs/so101", str(dest_dir), quiet=True)
    return str(urdf_path)

# Default duration [s] for the startup reset-to-origin slew.
RESET_DURATION_S = 5.0

# Optional cached file written by override_reset_pose.py. When present it takes priority over RESET_ORIGIN_DEG.
RESET_POSE_FILE = str(HF_LEROBOT_HOME / "reset_poses" / "{robot_name}" / "{robot_id}.json")

# Reset target in each motor's native units (arm joints in degrees, gripper in
# MotorNormMode.RANGE_0_100 where 100 = fully open, 0 = fully closed). These are an
# empirically recorded comfortable pose (elbow/wrist bent) that avoids the boundary
# singularity of a fully-extended 5-DOF arm; they assume standard calibration where
# 0° = URDF 0 rad (homing pose). Override per-setup by back-driving the arm and running
# override_reset_pose.py, which writes the per-arm reset pose file (it takes priority over these).
RESET_ORIGIN_DEG: dict[str, float] = {
    "shoulder_pan": -4.0,
    "shoulder_lift": -103.0,
    "elbow_flex": 97.0,
    "wrist_flex": 78.0,
    "wrist_roll": -65.0,
    "gripper": 0.0,
}


def _load_reset_target(reset_pose_file: Path, motor_names: list[str]) -> dict[str, float]:
    """Return reset targets: the saved reset pose if present, else RESET_ORIGIN_DEG."""
    if reset_pose_file.exists():
        saved = json.loads(reset_pose_file.read_text())
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
    kinematics_solver = RobotKinematics(
        urdf_path=_ensure_so101_urdf(),
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
            reset_pose_file = Path(RESET_POSE_FILE.format(robot_name=robot.name, robot_id=robot.id))
            target = _load_reset_target(reset_pose_file, motor_names)
            source = str(reset_pose_file) if reset_pose_file.exists() else "hardcoded defaults"
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
    follower FIRST (so the startup slew / clutch-home seed can read live joints), dispatches on
    ``--teleop.type`` (``so101_leader`` -> :func:`setup_leader`, else :func:`setup_xr`), then
    runs ``device.startup()`` (XR reset slew / leader align + warm-up) BEFORE returning —
    matching the original order, where startup runs before any dataset creation in ``record.py``.

    On success returns ``(robot, device, motor_names)`` with the follower connected and the
    device warmed up. If any step after ``connect()`` fails (or is interrupted), the follower
    is disconnected before the error propagates, so a failed setup never leaks the connection.

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
    # Everything after connect() can fail (device setup, the ~30s CloudXR startup, or a
    # Ctrl-C while donning the headset). build_device runs OUTSIDE the callers' try/finally,
    # so on any failure disconnect the follower here — otherwise the connection leaks.
    device: Device | None = None
    try:
        # Joint names in action order. Robot-agnostic: every LeRobot robot advertises its
        # motors as ``{name}.pos`` action features, so this works without assuming a ``.bus``
        # (letting non-bus robots plug in as the integration grows beyond the SO-101).
        motor_names = [key.removesuffix(".pos") for key in robot.action_features if key.endswith(".pos")]

        # Dispatch on the parsed device config type (the registry only yields these two).
        if isinstance(cfg.teleop, SO101LeaderArmConfig):
            device = setup_leader(cfg, robot, motor_names)
        else:
            device = setup_xr(cfg, robot, motor_names)

        device.startup()
    except BaseException:
        # Reap a partially-started teleop device (half-open session / spawned plugin) if it
        # got that far, then always disconnect the follower before propagating.
        if device is not None:
            with suppress(Exception):
                device.cleanup()
        robot.disconnect()
        raise

    return robot, device, motor_names


# ============================================================================
# Keyboard control
# ============================================================================


def init_keyboard_listener():
    """``(listener, events)`` for the recording shortcuts — terminal-first (SSH-friendly).

    This example is normally driven from a terminal, often over SSH, with the arm/headset
    in front of the operator rather than the workstation's console. Upstream's
    ``init_keyboard_listener`` prefers pynput's GLOBAL listener whenever a local X display
    is present (``DISPLAY`` set, non-Wayland) — but that captures the workstation console,
    not the SSH terminal, so the shortcuts silently do nothing over SSH. So whenever stdin
    is a TTY we use upstream's stdlib :class:`TerminalKeyListener` directly (it reads the
    controlling terminal, restores it on exit, and decodes the same keys). With no TTY
    (GUI launch / piped) we defer to upstream, which selects pynput or a headless no-op.

    Controls (both backends): Right / ``n`` end the episode early, Left / ``r`` re-record,
    Esc / ``q`` stop recording. Returns ``(listener, events)`` where ``listener`` has
    ``.stop()`` (or is ``None`` when headless) and ``events`` holds the ``exit_early`` /
    ``rerecord_episode`` / ``stop_recording`` flags the key presses set.
    """
    if not (sys.stdin is not None and sys.stdin.isatty()):
        # No controlling terminal: defer to upstream (pynput on a GUI, else headless no-op).
        from lerobot.utils.keyboard_input import init_keyboard_listener as _upstream

        return _upstream()

    from lerobot.utils.keyboard_input import TerminalKeyListener, apply_recording_control

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    # Same key -> control mapping upstream's init uses; n/r/q are the arrow/Esc equivalents
    # that survive laggy SSH/VNC links where escape sequences can split.
    def on_key(name: str) -> None:
        key = name.lower()
        if key in ("right", "n"):
            apply_recording_control("right", events)
        elif key in ("left", "r"):
            apply_recording_control("left", events)
        elif key in ("esc", "q"):
            apply_recording_control("esc", events)

    listener = TerminalKeyListener(on_key)
    listener.start()
    logging.info(
        "Keyboard control via terminal — keep this terminal focused: "
        "Right/n = end episode early, Left/r = re-record, Esc/q = stop."
    )
    return listener, events

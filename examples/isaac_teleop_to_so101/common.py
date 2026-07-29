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

"""Shared device + control-loop infrastructure for the Isaac Teleop -> SO-101 examples.

Consumed by ``teleoperate.py`` and ``record.py``, which both build a per-device
:class:`Device` bundle and run the same loop: read -> (maybe command) -> hold-when-idle ->
sleep. A :class:`Device` bundles three closures: ``compute(obs) -> RobotAction | None``
(``None`` = hold at the measured pose while idle), ``startup``, and ``cleanup``. The devices:

* ``xr_controller`` — a thin :class:`XRController` whose raw grip pose an in-loop
  :class:`Clutch` turns into an EE target for LeRobot's Cartesian IK pipeline.
* ``so101_leader`` — a back-drivable leader arm mirrored 1:1 into the follower.

Requires the ``isaacteleop`` package and an OpenXR runtime (install instructions in this
folder's ``README.md``). User-facing guide: ``docs/source/isaac_teleop.mdx``.
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
from importlib.resources import files
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
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, HF_LEROBOT_HOME, TELEOPERATORS
from lerobot.utils.robot_utils import precise_sleep

from .isaac_teleop import (
    Clutch,
    IsaacTeleopConfig,
    MapXRControllerActionToRobotAction,
    SO101LeaderArm,
    SO101LeaderArmConfig,
    XRController,
)

# Fixed rate [Hz] for the teleoperate loop and the pre-loop slews / connect-wait poll sleeps.
FPS = 30

# CloudXR device-profile env file passed to the launcher (see default.env in this package).
CLOUDXR_ENV_FILE = str(files(__package__) / "default.env")


class LoopConfig(Protocol):
    """Structural type for the loop/launch knobs ``build_device`` and the ``setup_*`` read.

    Both ``TeleoperateConfig`` and ``RecordConfig`` satisfy it, keeping ``common`` decoupled
    from either entry point's concrete config.
    """

    teleop: IsaacTeleopConfig
    robot: RobotConfig
    launch_plugin: str | None
    reset_to_origin: bool
    reset_duration: float
    align: bool
    align_duration: float


# Per-device bundle consumed by the shared loop. ``compute`` returns None to mean
# "idle -> hold at the measured pose"; ``startup`` warms up; ``cleanup`` reaps/disconnects.
@dataclass(frozen=True)
class Device:
    compute: Callable[[RobotObservation | None], RobotAction | None]
    startup: Callable[[], None]
    cleanup: Callable[[], None]


def hold_action(obs: RobotObservation, motor_names: list[str]) -> dict[str, float]:
    """Re-send the measured joints — the explicit hold when a device is idle."""
    return {f"{name}.pos": float(obs[f"{name}.pos"]) for name in motor_names}


class HoldLatch:
    """Resolve the per-frame action, holding one LATCHED pose while the device is idle.

    Re-sending the freshly measured joints on every idle frame would ratchet the arm
    downward: under gravity the P-only servo settles below its goal by a steady-state
    error, so each re-command of the measurement lowers the goal by that error again.
    Latching the target once on the active->idle transition holds a fixed pose instead.
    """

    def __init__(self, motor_names: list[str]):
        self._motor_names = motor_names
        self._held: dict[str, float] | None = None

    def resolve(self, action: RobotAction | None, obs: RobotObservation) -> RobotAction:
        """Pass through an active action (clearing the latch); latch + hold when idle."""
        if action is not None:
            self._held = None
            return action
        if self._held is None:
            self._held = hold_action(obs, self._motor_names)
        return self._held


def slew(
    robot,
    motor_names: list[str],
    target_fn: Callable[[], dict[str, float]],
    duration_s: float,
) -> None:
    """Linearly slew all joints from their current measured pose toward a target.

    ``target_fn`` is called EACH step, so the leader can pass a live re-read (landing on its
    current pose at ``alpha == 1`` for a continuous handoff) while XR passes a constant.
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

# Per-frame EE rate limit [m]. With raise_on_jump=False, EEBoundsAndSafety clamps an
# over-limit step instead of raising, absorbing a tracking glitch as one slow frame. At
# FPS=30, 0.1 m/frame caps EE speed at ~3 m/s. (end_effector_bounds clips the absolute target.)
MAX_EE_STEP_M = 0.1

# Soft-orientation IK weight: small but nonzero so the wrist follows the hand while position
# dominates (the 5-DOF SO-101 cannot realize an arbitrary orientation). 0.0 = position-only.
IK_ORIENTATION_WEIGHT = 0.01


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


# Default duration [s] for the startup reset-to-origin slew.
RESET_DURATION_S = 5.0

# Optional cached file written by override_reset_pose.py. When present it takes priority over RESET_ORIGIN_DEG.
RESET_POSE_FILE = str(HF_LEROBOT_HOME / "reset_poses" / "{robot_name}" / "{robot_id}.json")

# Reset target in each motor's native units (arm joints in degrees, gripper RANGE_0_100,
# 100 = open). An empirically comfortable pose (elbow/wrist bent) avoiding the singularity of
# a fully-extended arm; assumes standard calibration. Override per-arm via override_reset_pose.py.
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


# CloudXR web client URL opened in the headset (Isaac Teleop quick start, step 5).
_CLOUDXR_WEB_CLIENT_URL = "https://nvidia.github.io/IsaacTeleop/client"
# WSS-proxy / self-signed-cert port the operator accepts in-browser before connecting.
_CLOUDXR_WSS_PORT = 48322
# How often to re-print the connection hint while waiting for the headset [s].
_XR_CONNECT_REMINDER_S = 15.0
# Virtual / bridge / USB-gadget interfaces a headset can't reach over the network — skip
# by name prefix (``docker0``, compose ``br-*``, ``veth*``, libvirt ``virbr*``, and the
# Tegra USB device-mode bridge ``l4tbr0``).
_SKIP_IFACE_PREFIXES = ("docker", "br-", "veth", "virbr", "l4tbr")


def _primary_ipv4() -> str | None:
    """The workstation's primary outbound IPv4, via the UDP-socket trick (``connect()`` on a
    datagram socket selects the egress interface without sending packets)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return None


def _candidate_ipv4s() -> list[tuple[str, str]]:
    """Return ``[(interface, ipv4), ...]`` the headset might reach this workstation at.

    Lists each interface's IPv4 via ``psutil`` (dropping loopback, link-local, and the
    virtual/bridge interfaces in ``_SKIP_IFACE_PREFIXES``), primary outbound first. Falls
    back to just the primary IP when ``psutil`` is unavailable.
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
    """Block until the XR controller is tracked, polling ``get_action()`` and re-printing a
    reminder every ``_XR_CONNECT_REMINDER_S``. User-paced; ``Ctrl-C`` aborts (no hard timeout).
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

    # The clutch (below) turns the raw grip pose into an absolute base-frame ee_pose; this
    # pipeline maps it to joint targets: rename -> bounds/rate-limit -> IK.
    xr_to_robot_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapXRControllerActionToRobotAction(),
            # raise_on_jump=False: an over-limit step (e.g. a tracking glitch) is clamped +
            # warned instead of raised, since a crash mid-loop would leave the arm uncontrolled.
            # z floor 0.0 keeps a stray target above the table; x/y stay at a loose [-1,1]m box.
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=MAX_EE_STEP_M,
                raise_on_jump=False,
            ),
            # initial_guess_current_joints=False: warm-start from the previous IK solution so
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
        # Connect and wait for the operator to don the headset BEFORE moving the arm, so the
        # reset slew happens while they are watching in VR.
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

        # Seed the clutch home from the arm's measured pose (FK of the current joints) so the
        # first engage is jump-free, whether or not a reset slew ran.
        obs0 = robot.get_observation()
        q_measured_deg = np.array([float(obs0[f"{name}.pos"]) for name in motor_names], dtype=float)
        home_base_T_ee = kinematics_solver.forward_kinematics(q_measured_deg)  # noqa: N806
        clutch = Clutch(home_base_T_ee)

        print("Starting teleop loop. Squeeze and move the controller to teleoperate the robot...")

    def compute(robot_obs: RobotObservation | None) -> RobotAction | None:
        nonlocal prev_enabled
        if clutch is None:  # set in startup(), which runs before compute()
            raise RuntimeError("compute() called before startup(); the clutch is not initialized")
        xr_action = teleop_device.get_action()
        grip_pos = np.asarray(xr_action["grip_pos"], dtype=float)
        grip_quat = np.asarray(xr_action["grip_quat"], dtype=float)
        squeeze = float(xr_action["squeeze"])
        trigger = float(xr_action["trigger"])
        enabled = squeeze > teleop_config.clutch_threshold

        # On the engage edge, latch the clutch home at the arm's MEASURED EE pose (FK of
        # the live joints) and the controller origin so the per-frame delta starts at zero.
        # Latching the last commanded pose instead would snap the arm back to it at full
        # servo speed if the arm moved while disengaged (gravity sag, external contact).
        is_engage_frame = enabled and not prev_enabled
        if is_engage_frame:
            q_measured = np.array([float(robot_obs[f"{name}.pos"]) for name in motor_names], dtype=float)
            measured_base_T_ee = kinematics_solver.forward_kinematics(q_measured)  # noqa: N806
            clutch.engage(grip_pos, grip_quat, measured_base_T_ee=measured_base_T_ee)
            # Re-anchor the pipeline state at the measured pose as well: EEBoundsAndSafety's
            # rate limiter and the IK warm start otherwise still reference the stale
            # pre-disengage command and would fight the fresh home for several frames.
            xr_to_robot_joints_processor.reset()
        prev_enabled = enabled

        # SAFETY GATE: command the robot ONLY while the clutch is engaged; otherwise return
        # None so the loop holds the measured joints (releasing the clutch freezes the arm).
        if not enabled:
            return None

        # Rebase the raw grip pose onto the EE, then run the pipeline. closedness = trigger.
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

# The plugin converts the leader's servo ticks to radians, so it reuses the serial SO-101
# leader's calibration, stored by lerobot-calibrate under SO101Leader.name == "so_leader".
SO_LEADER_CALIBRATION_NAME = "so_leader"


def _leader_calibration_path(cfg: LoopConfig) -> Path | None:
    """Infer the calibration JSON the launched plugin should read, or None.

    Path convention: ``HF_LEROBOT_CALIBRATION / teleoperators / so_leader / {--teleop.id}.json``
    (or ``--teleop.calibration_dir`` if set). Returns None (plugin falls back to defaults) when
    it does not exist, warning if an id was given, or when no ``--teleop.id`` is set.
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
    # synthetic backend. Calibration (only real hardware needs it) is appended when a port is set.
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

            # Re-read the live leader pose once per step so alpha=1 lands on its current pose
            # from a single coherent frame.
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
        # Hold the follower at its measured pose when the leader drops out (stale stream)
        # rather than commanding a possibly-old target.
        if not teleop.is_tracking:
            return None
        return leader_action

    def cleanup() -> None:
        # A plugin-reaping failure must not skip the session disconnect (and vice versa
        # the disconnect runs after the plugin stops pushing on it).
        try:
            if plugin_proc is not None:
                plugin_proc.terminate()
                try:
                    plugin_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    plugin_proc.kill()
        finally:
            teleop.disconnect()

    return Device(compute=compute, startup=startup, cleanup=cleanup)


# ============================================================================
# Shared setup
# ============================================================================


def build_device(cfg: LoopConfig) -> tuple:
    """Connect the follower, build the selected Isaac device, and run its pre-loop startup.

    Connects the follower FIRST (so the startup slew / clutch-home seed can read live joints),
    dispatches on ``--teleop.type``, then runs ``device.startup()`` before returning. On any
    failure after ``connect()`` the follower is disconnected so the connection never leaks.

    Returns ``(robot, device, motor_names)``.
    """
    # Default the CloudXR input profile to this example's default.env unless the user overrode
    # it via --teleop.cloudxr_env_file.
    if cfg.teleop.cloudxr_env_file is None:
        cfg.teleop.cloudxr_env_file = CLOUDXR_ENV_FILE

    # SO-101/SO-100 only (both share the SO-101 URDF), reject other followers.
    supported_robots = {"so101_follower", "so100_follower"}
    if cfg.robot.type not in supported_robots:
        raise ValueError(
            f"This example only supports SO-101/SO-100 followers ({sorted(supported_robots)}), "
            f"but got --robot.type={cfg.robot.type}."
        )

    # The degree-based pipeline relies on --robot.use_degrees (default True).
    robot = make_robot_from_config(cfg.robot)
    # Connect FIRST so the startup slew and clutch-home seed can read live joints.
    robot.connect()
    # Everything after connect() can fail; this runs outside the callers' try/finally, so
    # disconnect the follower on any failure to avoid leaking the connection.
    device: Device | None = None
    try:
        # Joint names in action order, read from {name}.pos action features (robot-agnostic).
        motor_names = [key.removesuffix(".pos") for key in robot.action_features if key.endswith(".pos")]

        if isinstance(cfg.teleop, SO101LeaderArmConfig):
            device = setup_leader(cfg, robot, motor_names)
        else:
            device = setup_xr(cfg, robot, motor_names)

        device.startup()
    except BaseException:
        # Reap a partially-started device, then always disconnect the follower.
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
    """Recording shortcuts, terminal-first so they work over SSH.

    Whenever stdin is a TTY we use the stdlib :class:`TerminalKeyListener` directly rather
    than upstream's pynput-first :func:`init_keyboard_listener`, whose global listener would
    capture the workstation console instead of this (often SSH) terminal. With no TTY we defer
    to upstream (pynput on a GUI, else headless no-op).
    """
    if not (sys.stdin is not None and sys.stdin.isatty()):
        from lerobot.utils.keyboard_input import init_keyboard_listener as _upstream

        return _upstream()

    from lerobot.utils.keyboard_input import TerminalKeyListener, apply_recording_control

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    # n/r/q are the arrow/Esc equivalents that survive escape-sequence splitting over laggy
    # SSH/VNC links. Case-insensitive so Shift+letter still works.
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

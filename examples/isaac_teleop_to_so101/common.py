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

Consumed by ``teleoperate.py`` and ``record.py``, which both build the :class:`Device`
bundle and run the same loop: read -> (maybe command) -> hold-when-idle -> sleep. A
:class:`Device` bundles ``compute(obs) -> RobotAction | None`` (``None`` = hold at the
measured pose while idle), ``startup``, ``cleanup``, plus ``engaged`` and ``reset``, which
both entry points use to send the arm home on declutch — ``record.py`` also ends the
recorded episode there. The device is an :class:`XRController` whose in-pipeline clutch
retargeter emits an absolute base-frame EE target for LeRobot's Cartesian IK pipeline.

Requires the ``isaacteleop`` package and an OpenXR runtime (install instructions in this
folder's ``README.md``). User-facing guide: ``docs/source/isaac_teleop.mdx``.
"""

import json
import logging
import os
import re
import socket
import sys
import time
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Protocol

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots import RobotConfig, make_robot_from_config

# ROBOT_PROFILES registrations live here rather than in each entry point.
from lerobot.robots.rebot_b601_follower import (  # noqa: F401  (registers rebot_b601_follower)
    RebotB601FollowerRobotConfig,
)
from lerobot.robots.so_follower import SOFollowerConfig  # noqa: F401  (registers so101_follower)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.robot_utils import precise_sleep

from .isaac_teleop import (
    MapXRControllerActionToRobotAction,
    XRController,
    XRControllerConfig,
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

    teleop: XRControllerConfig
    robot: RobotConfig
    reset_to_origin: bool
    reset_duration: float


# Device bundle consumed by the shared loop. ``compute`` returns None to mean
# "idle -> hold at the measured pose"; ``startup`` warms up; ``cleanup`` reaps/disconnects.
# ``engaged`` reports the clutch state of the frame ``compute`` last processed, and ``reset``
# returns the arm to its reset pose.
@dataclass(frozen=True)
class Device:
    compute: Callable[[RobotObservation | None], RobotAction | None]
    startup: Callable[[], None]
    cleanup: Callable[[], None]
    engaged: Callable[[], bool]
    reset: Callable[[], None]


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


class DeclutchLatch:
    """Detect the declutch that ends a segment: the first engaged -> disengaged transition.

    Arms on the first engaged frame, so a segment that starts disengaged (the operator has not
    squeezed yet) is not ended on frame 0. One instance per segment — a recorded episode, or one
    teleoperate squeeze; once it has fired it keeps reporting True.
    """

    def __init__(self):
        self._armed = False

    def update(self, engaged: bool) -> bool:
        """Feed this frame's clutch state; True once the operator has engaged and let go."""
        if engaged:
            self._armed = True
            return False
        return self._armed


def slew(
    robot,
    motor_names: list[str],
    target: dict[str, float],
    duration_s: float,
) -> None:
    """Linearly slew all joints from their current measured pose to ``target``."""
    obs = robot.get_observation()
    start = {name: float(obs[f"{name}.pos"]) for name in motor_names}
    n_steps = max(1, int(duration_s * FPS))
    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        action = {f"{name}.pos": start[name] + alpha * (target[name] - start[name]) for name in motor_names}
        robot.send_action(action)
        precise_sleep(1.0 / FPS)


# ============================================================================
# XR controller device
# ============================================================================

# Per-frame EE rate limit [m]: an over-limit step raises out of the loop (EEBoundsAndSafety's
# default raise_on_jump). At FPS=30, 0.1 m/frame caps EE speed at ~3 m/s -- well above the ~0.05 m
# a fast hand produces at clutch_position_scale=0.5, so it only trips on a real tracking glitch.
# Position only; orientation is neither bounded nor rate-limited.
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
    print(f"Fetching the reBot B601-RS description into {dest_dir} (~64 MB, first run only)…")

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

    Keyed by :func:`_robot_profile_key` in :data:`ROBOT_PROFILES`. Adding an arm is a profile
    entry, not a code change.
    """

    urdf: Callable[[], str]
    """Returns a local URDF path, fetching and caching it on first use."""

    urdf_joint_names: list[str]
    """URDF joint names driven by IK, ordered like the robot's action features (gripper last
    and excluded -- ``RobotKinematics`` slices the leading joints and passes the rest through)."""

    ee_frame: str
    """URDF frame IK drives to."""

    ee_bounds: dict[str, list[float]]
    """Backstop box [m] in the robot base frame; sized from a URDF reach sweep, not a guess."""

    reset_pose: dict[str, float]
    """Reset target in the follower's own action units. Overridden per-arm by override_reset_pose.py."""

    gripper_open: float
    """Follower action value at fully OPEN (trigger released)."""

    gripper_close: float
    """Follower action value at fully CLOSED (trigger pulled)."""

    max_ee_step_m: float = MAX_EE_STEP_M
    orientation_weight: float = IK_ORIENTATION_WEIGHT

    clutch_position_scale: float | None = None
    """Controller-to-EE translation gain for this arm, or ``None`` to keep
    ``XRControllerConfig``'s own default (0.5, sized to the SO-101's reach). Belongs to the arm
    because the right gain follows from its reach."""

    raise_on_ee_jump: bool = True
    """``False`` rate-limits an over-limit frame and warns instead of raising out of the loop."""

    home_orientation_from_measured: bool = False
    """Re-seed the clutch home from the measured EE while disengaged, so a sagging arm does not
    kick on engage. Only the ORIENTATION half is new (the latch already takes position from the
    measured input), and it is safe only where measured-minus-commanded is sag ALONE. Converged
    IK leaves the 6-DOF reBot 0.0 deg short of a commanded orientation at ``orientation_weight``
    1.0, but the 5-DOF SO-101 8.8 deg short (17.9 worst) at 0.01: feeding that back would move
    the hand-to-arm orientation mapping by that much on every re-clutch."""


def _robot_profile_key(robot_type: str, motor_family: str | None = None) -> str:
    """ROBOT_PROFILES key: ``--robot.type``, refined by ``--robot.motor_family`` for followers
    (like the reBot B601) that share one type across hardware variants."""
    return f"{robot_type}:{motor_family}" if motor_family else robot_type


ROBOT_PROFILES: dict[str, RobotProfile] = {
    # SO-101/SO-100 share the SO-101 URDF, whose joint names are the motor names.
    "so101_follower": RobotProfile(
        urdf=_ensure_so101_urdf,
        # Includes the gripper: on the SO-101 the URDF joint names ARE the motor names, and
        # the solver has always been handed all six.
        urdf_joint_names=[
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
        reset_pose={
            "shoulder_pan": 0.0,
            "shoulder_lift": -90.0,
            "elbow_flex": 90.0,
            "wrist_flex": 45.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
        gripper_open=100.0,
        gripper_close=0.0,
    ),
    _robot_profile_key("rebot_b601_follower", motor_family="rs"): RobotProfile(
        urdf=_ensure_rebot_b601_rs_urdf,
        urdf_joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        ee_frame="gripper_end",
        # 60k-sample FK sweep over the URDF joint limits: max reach 0.911 m, envelope
        # x/y within +-0.77, z in [-0.372, 0.907]. Kept well inside that, with the z floor at
        # the arm's own base plane (tabletop).
        ee_bounds={"min": [-0.45, -0.55, 0.0], "max": [0.65, 0.55, 0.70]},
        # The sit-down pose calibration zeroes the arm at, lifted 10/15 deg off the shoulder and
        # elbow endpoints (both travel one way only, from 0) so the IK is not seeded sitting on
        # two joint limits. FK puts the EE at [0.297, 0.0, 0.304] m, 8.6 cm above sit-down.
        # Negative because the follower's action space is its URDF's, not its motors' -- see
        # joint_directions below.
        reset_pose={
            "shoulder_pan": 0.0,
            "shoulder_lift": -5.0,
            "elbow_flex": -10.0,
            "wrist_flex": 0.0,
            "wrist_yaw": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
        # motor_family.RS_PROFILE.joint_limits["gripper"] endpoints through joint_directions:
        # calibration zeroes the arm with the jaw fully CLOSED, so motor 0 is closed and motor
        # 270 is open. The RS gripper is driven by a force-limited impedance torque
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
    ),
}
ROBOT_PROFILES["so100_follower"] = ROBOT_PROFILES["so101_follower"]


# Default duration [s] for the reset-to-origin slew (startup and every declutch): long enough to
# follow in VR, short enough not to stall the operator between segments.
RESET_DURATION_S = 2.0

# Optional cached file written by override_reset_pose.py. When present it takes priority over RESET_ORIGIN_DEG.
RESET_POSE_FILE = str(HF_LEROBOT_HOME / "reset_poses" / "{robot_name}" / "{robot_id}.json")


def _load_reset_target(
    reset_pose_file: Path, motor_names: list[str], profile_pose: dict[str, float]
) -> dict[str, float]:
    """Return reset targets: the saved reset pose if present, else the profile's."""
    if reset_pose_file.exists():
        saved = json.loads(reset_pose_file.read_text())
        # Fill any missing motors from the fallback dict.
        return {name: float(saved.get(name, profile_pose.get(name, 0.0))) for name in motor_names}
    return {name: profile_pose.get(name, 0.0) for name in motor_names}


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


def build_xr_joint_pipeline(
    profile: RobotProfile, motor_names: list[str], kinematics: RobotKinematics
) -> RobotProcessorPipeline:
    """Absolute base-frame EE target -> joint targets in the follower's own convention.

    ``rename -> bounds/rate-limit -> IK -> motor convention``. Every step between the first and
    the last works in URDF convention, so the observation handed to the pipeline (which is the
    IK seed) must already be rebased there. Split out of :func:`setup_xr` so the whole chain can
    be exercised without an XR runtime or an arm.
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
            motor_names=motor_names,
            initial_guess_current_joints=True,
            orientation_weight=profile.orientation_weight,
        ),
    ]
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=steps,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def setup_xr(cfg: LoopConfig, robot, motor_names: list[str], profile: RobotProfile) -> Device:
    """Build the XR controller device bundle (clutch + IK pipeline) for ``profile``'s arm."""
    kinematics_solver = RobotKinematics(
        urdf_path=profile.urdf(),
        target_frame_name=profile.ee_frame,
        joint_names=profile.urdf_joint_names,
    )

    teleop_config = cfg.teleop
    if profile.clutch_position_scale is not None:
        teleop_config.clutch_position_scale = profile.clutch_position_scale
        logging.info(f"{robot.name}: clutch_position_scale={teleop_config.clutch_position_scale}")
    teleop_device = XRController(teleop_config)

    xr_to_robot_joints_processor = build_xr_joint_pipeline(profile, motor_names, kinematics_solver)

    # Reset pose resolved once: the same target serves the startup slew and every
    # between-episode reset in record.py.
    reset_pose_file = Path(RESET_POSE_FILE.format(robot_name=robot.name, robot_id=robot.id))
    reset_target = _load_reset_target(reset_pose_file, motor_names, profile.reset_pose)

    # The clutch lives inside the device's retargeting pipeline. The loop tracks the engagement
    # so it can spot the engage edge — and, in record.py, the declutch that ends an episode; it
    # does NOT re-derive engagement itself, so the squeeze threshold is compared in exactly one
    # place (the retargeter). Holds this frame's value once compute() has run, the previous
    # frame's while it runs.
    clutch_engaged = False

    def _measured_base_T_ee(obs: RobotObservation) -> np.ndarray:  # noqa: N802
        """FK the measured joints to the arm's current base_T_ee."""
        q_measured = np.array([float(obs[f"{name}.pos"]) for name in motor_names], dtype=float)
        return kinematics_solver.forward_kinematics(q_measured)

    def reset() -> None:
        """Slew the arm to its reset pose and re-home the clutch there.

        Runs at startup, on every declutch in ``teleoperate.py``, and in every reset window
        between recorded episodes. The device holds STOPPED across the slew (stepped once so the
        state lands first) — the readiness interlock, not a formality: a squeeze mid-slew would
        otherwise latch the clutch against a home the arm has not reached yet. The clutch is then
        seeded from the post-slew MEASURED pose and the interlock released, so the next engage is
        jump-free.
        """
        nonlocal clutch_engaged
        teleop_device.stop()
        teleop_device.get_action()  # step once so STOPPED takes effect before the arm moves
        # Re-arm the engage edge: the rate limiter still references the pre-slew command, so the
        # pipeline must be reset on the next engaged frame even if the squeeze was never released.
        clutch_engaged = False

        if cfg.reset_to_origin:
            print(f"Resetting to origin over {cfg.reset_duration:.1f} s…")
            slew(robot, motor_names, reset_target, cfg.reset_duration)
            print("Reset complete.")

        teleop_device.set_home_base_T_ee(_measured_base_T_ee(robot.get_observation()))
        # Releases the readiness interlock; raises if the home was not seeded first.
        teleop_device.start()

    def engaged() -> bool:
        """Whether the clutch was engaged on the frame compute() last processed."""
        return clutch_engaged

    def startup() -> None:
        # Connect and wait for the operator to don the headset BEFORE moving the arm, so the
        # reset slew happens while they are watching in VR.
        teleop_device.connect()
        if not teleop_device.is_connected:
            raise ValueError("Teleop is not connected!")
        _wait_for_xr_controller(teleop_device)

        if cfg.reset_to_origin:
            source = str(reset_pose_file) if reset_pose_file.exists() else "hardcoded defaults"
            print(f"Reset target source: {source}")
        reset()

        print("Starting teleop loop. Squeeze and move the controller to teleoperate the robot...")

    def compute(robot_obs: RobotObservation | None) -> RobotAction | None:
        nonlocal clutch_engaged
        # Supply the arm's measured EE pose EVERY frame, from the observation the loop already
        # holds (never re-read the robot here — that would widen the skew and add a bus
        # transaction). The clutch consumes it only on the engage frame, but the loop cannot know
        # in advance which frame that is, and FK is ~0.005 ms.
        #
        # With no observation this frame there is no FK to send, so the clutch falls back to its
        # last commanded home. The same condition also gates the pipeline call below.
        if robot_obs is not None:
            teleop_device.set_measured_base_T_ee(_measured_base_T_ee(robot_obs))

        # The device MUST be stepped every frame, including while disengaged: the clutch observes
        # the release through this call, and skipping it would mean it never sees a falling edge,
        # so no re-clutch would ever fire again.
        xr_action = teleop_device.get_action()
        engaged = bool(xr_action["engaged"])
        trigger = float(xr_action["trigger"])

        # On the engage edge the clutch has just re-latched its home at the arm's measured EE
        # pose. Re-anchor the pipeline to match: EEBoundsAndSafety's rate limiter otherwise still
        # references the stale pre-disengage command and would clamp the first frames against it.
        # The edge is taken from the retargeter's own engagement, not re-derived from squeeze —
        # the latch can be deferred by a dropped or untrusted frame that the loop cannot observe.
        if engaged and not clutch_engaged:
            xr_to_robot_joints_processor.reset()
        clutch_engaged = engaged

        # Re-home the clutch off the arm's live pose while it is disengaged, so an arm that sags
        # there does not have the sag reconciled in one frame on the next engage. The latch takes
        # its home POSITION from the measured EE already, so this is the ORIENTATION half, which
        # is why it is per-arm -- see RobotProfile.home_orientation_from_measured. DISENGAGED
        # frames only: on an engaged-but-untracked one this would re-home and re-arm the latch
        # mid-segment, teleporting the mapping the operator is holding.
        if profile.home_orientation_from_measured and not engaged and robot_obs is not None:
            teleop_device.set_home_base_T_ee(_measured_base_T_ee(robot_obs))

        # SAFETY GATE: command the robot ONLY on a frame that is both engaged AND tracked;
        # otherwise return None so the loop holds the measured joints (releasing the clutch
        # freezes the arm).
        #
        # ``is_tracking`` is a SEPARATE condition from ``engaged`` on purpose. The device reads
        # the trigger outside the retargeting graph, so a partially-populated frame yields
        # trigger = 0.0 -- jaw fully OPEN -- while the clutch, which reads squeeze *inside* the
        # graph, stays engaged. Acting on that frame would drop a live grasp. Deleted code got
        # this for free because the loop thresholded the squeeze itself and the same failure
        # zeroed it; that coupling is gone. It is NOT folded into ``engaged`` above, because
        # ``engaged`` must stay the pure retargeter signal for the edge below to remain exactly
        # the clutch's latch frame.
        if not (engaged and teleop_device.is_tracking) or robot_obs is None:
            return None

        # The pose is already the clutch-rebased absolute EE target. closedness = trigger.
        ee_action = {
            "ee_pose": np.asarray(xr_action["ee_pose"], dtype=np.float32),
            "closedness": trigger,
        }
        return xr_to_robot_joints_processor((ee_action, robot_obs))

    return Device(
        compute=compute,
        startup=startup,
        cleanup=teleop_device.disconnect,
        engaged=engaged,
        reset=reset,
    )


# ============================================================================
# Shared setup
# ============================================================================


def build_device(cfg: LoopConfig) -> tuple:
    """Connect the follower, build the XR device, and run its pre-loop startup.

    Connects the follower FIRST (so the startup slew / clutch-home seed can read live joints),
    then runs ``device.startup()`` before returning. On any failure after ``connect()`` the
    follower is disconnected so the connection never leaks.

    Returns ``(robot, device, motor_names)``.
    """
    # Default the CloudXR input profile to this example's default.env unless the user overrode
    # it via --teleop.cloudxr_env_file.
    if cfg.teleop.cloudxr_env_file is None:
        cfg.teleop.cloudxr_env_file = CLOUDXR_ENV_FILE

    profile_key = _robot_profile_key(cfg.robot.type, getattr(cfg.robot, "motor_family", None))
    profile = ROBOT_PROFILES.get(profile_key)
    if profile is None:
        raise ValueError(
            f"No RobotProfile for --robot.type={cfg.robot.type} "
            f"(key={profile_key!r}). Supported: {sorted(ROBOT_PROFILES)}. Adding an arm is a "
            "ROBOT_PROFILES entry."
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

        device = setup_xr(cfg, robot, motor_names, profile)
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

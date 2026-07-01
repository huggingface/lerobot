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

"""Configuration dataclasses for NVIDIA Isaac Teleop-backed teleoperators.

:class:`IsaacTeleopConfig` holds the fields shared by every Isaac Teleop input
device (currently just the session ``app_name``); each device adds its own
config subclass (e.g. :class:`XRControllerConfig`, and future ``ManusConfig``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass(kw_only=True)
class IsaacTeleopConfig(TeleoperatorConfig):
    """Shared config for all Isaac Teleop-backed teleoperators.

    Subclassed per input device (XR controller, Manus gloves, hand tracking,
    ...). Abstract: register the concrete device subclasses, not this base.

    Its own draccus choice registry (own ``_choice_registry``, decoupled from the
    global :class:`TeleoperatorConfig` one) so a config field typed ``IsaacTeleopConfig``
    resolves ``--teleop.type`` against ONLY the Isaac devices. This lets them claim short,
    natural names (``xr_controller``, ``so101_leader``) without colliding with the global
    teleop registry — e.g. the serial ``so101_leader`` (``so_leader``) keeps its global key.
    Selected this way by ``examples/isaac_teleop_to_so101/teleoperate.py``; these devices
    drive a bespoke clutch/IK/align loop and are not routed through
    ``make_teleoperator_from_config``.
    """

    _choice_registry: ClassVar[dict] = {}

    app_name: str = "LeTeleop"
    """Application name for the OpenXR / Isaac Teleop session."""

    auto_launch_cloudxr: bool = True
    """Whether to auto-launch the NVIDIA CloudXR runtime on :meth:`connect`.

    When ``True`` (default), :meth:`connect` starts the CloudXR runtime so
    operators need not run ``python -m isaacteleop.cloudxr`` and ``source
    cloudxr.env`` by hand. Set ``False`` (or export
    ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1``, which takes precedence) when CloudXR
    is already running externally. The first launch blocks ~30s and, on a fresh
    machine, prompts for the CloudXR EULA on stdin (see
    :meth:`~lerobot.teleoperators.isaac_teleop.base.IsaacTeleopTeleoperator.connect`);
    EULA acceptance is deliberately not exposed here — accept it once via
    ``python -m isaacteleop.cloudxr --accept-eula``.
    """

    cloudxr_env_file: str | None = None
    """Optional CloudXR device-profile ``.env`` passed to ``CloudXRLauncher`` as input.

    This is an INPUT profile that selects the headset transport (e.g. an Apple
    Vision Pro profile); it is NOT the ``~/.cloudxr/run/cloudxr.env`` file the old
    manual flow told you to ``source`` (that is launcher OUTPUT, now handled
    automatically). ``None`` keeps the launcher's default auto-WebRTC profile,
    which matches today's manual default.
    """


# Static rebase from the OpenXR controller anchor frame into the robot base
# frame (X=Forward, Y=Left, Z=Up). Applied upstream of the SO-101 retargeters
# via Isaac Teleop's native ``ControllerTransform`` so the clutch/roll/gripper
# retargeters operate directly in the robot base frame. A proper rotation
# (det=+1): controller motion to the right maps to robot +X etc.
_DEFAULT_BASE_T_ANCHOR: list[list[float]] = [
    [0.0, 0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


@IsaacTeleopConfig.register_subclass("xr_controller")
@dataclass(kw_only=True)
class XRControllerConfig(IsaacTeleopConfig):
    """Config for Isaac Teleop XR (VR) controller teleoperation.

    Exposes the raw XR controller grip pose (base-frame), squeeze, and trigger
    via NVIDIA Isaac Teleop's ``ControllersSource``. There are no retargeters:
    the clutch (engage latch + delta rebase onto the EE) and the gripper mapping
    live in the owning loop (see
    :class:`~lerobot.teleoperators.isaac_teleop.teleop_xr_controller.XRController`
    and ``examples/isaac_teleop_to_so101/teleoperate.py``).

    Frames: :attr:`base_T_anchor` statically rebases the controller anchor frame
    into the robot base frame so the grip pose this device emits is already in the
    robot base frame.
    """

    hand_side: Literal["left", "right"] = "right"
    """Which controller hand to use."""

    clutch_threshold: float = 0.5
    """Squeeze value above which the owning loop's clutch engages.

    Mirrors the phone teleoperator's hold-to-enable button: while the controller
    squeeze is held above this threshold the owning loop drives the robot;
    releasing it freezes the robot and re-arms the engage origin. The device only
    reports the raw squeeze — the threshold is applied by the owning loop.
    """

    base_T_anchor: list[list[float]] = field(  # noqa: N815  (frameA_T_frameB transform-matrix convention)
        default_factory=lambda: _DEFAULT_BASE_T_ANCHOR
    )
    """Static 4x4 ``base_T_anchor`` transform [row-major] rebasing the OpenXR controller
    anchor frame into the robot base frame, applied to the controller grip pose in the device.

    Defaults to the OpenXR (X=Right, Y=Up, Z=Backward) -> robot (X=Forward, Y=Left, Z=Up)
    rotation. Kept as plain nested lists (not numpy) so the config stays serializable; the
    device materializes it to a float32 4x4 once at construction.
    """


# Provisional gripper open/close endpoints [rad] for the leader's gripper DOF, used to
# normalize the streamed gripper angle into the follower's RANGE_0_100 jaw target. These
# defaults are derived from the so101_leader plugin README's *example* calibration
# (home_ticks=2048, range 2000..3000 ticks; angle = (ticks - home) * 2*pi / 4096):
#   range_min 2000 ticks -> (2000-2048)*2*pi/4096 = -0.074 rad
#   range_max 3000 ticks -> (3000-2048)*2*pi/4096 = +1.460 rad
_DEFAULT_GRIPPER_OPEN_RAD = -0.074
_DEFAULT_GRIPPER_CLOSE_RAD = 1.460


@IsaacTeleopConfig.register_subclass("so101_leader")
@dataclass(kw_only=True)
class SO101LeaderArmConfig(IsaacTeleopConfig):
    """Config for an Isaac Teleop SO-101 *leader arm* (generic joint-space device).

    Drives a follower SO-101 by mirroring the leader's joint angles 1:1 (direct joint
    drive, like ``lerobot-teleoperate`` with the serial ``so101_leader``). The leader's
    joint state is streamed as a ``JointStateOutput`` FlatBuffer by NVIDIA Isaac Teleop's
    ``so101_leader`` plugin over the OpenXR tensor transport and read here via a
    ``JointStateSource`` (see
    :class:`~lerobot.teleoperators.isaac_teleop.teleop_so101_leader_arm.SO101LeaderArm`).

    Units: the plugin streams every joint -- gripper included -- in **radians**. The device
    converts the arm joints to degrees (``rad2deg``) and the gripper to the follower's
    RANGE_0_100 jaw target, so :meth:`SO101LeaderArm.get_action` returns follower-ready
    ``{joint}.pos`` values that can be sent straight to ``robot.send_action`` (no IK, no
    clutch, no retargeter on the LeRobot side).
    """

    port: str = ""
    """Serial port of the physical LEADER arm (e.g. ``/dev/ttyACM1``).

    The leader's servos are read by the native ``so101_leader`` *plugin*, not by this
    device (which only consumes the plugin's ``JointStateOutput`` stream over the OpenXR
    tensor transport). When the example script launches the plugin (its ``--launch_plugin``
    path), it forwards this port to the plugin as its device path. Empty (default) -> the
    plugin runs its synthetic trajectory (no leader hardware -> a dry run of the transport)."""

    collection_id: str = "so101_leader"
    """Tensor collection id the leader plugin pushes on; must match the running
    ``so101_leader`` plugin (its second positional arg, default ``"so101_leader"``)."""

    gripper_open_rad: float = _DEFAULT_GRIPPER_OPEN_RAD
    """Leader gripper angle [rad] at fully OPEN -> follower jaw 100. Provisional default;
    set from the plugin's ``calibrate`` subcommand. See ``_DEFAULT_GRIPPER_OPEN_RAD``."""

    gripper_close_rad: float = _DEFAULT_GRIPPER_CLOSE_RAD
    """Leader gripper angle [rad] at fully CLOSED -> follower jaw 0. Provisional default;
    set from the plugin's ``calibrate`` subcommand. See ``_DEFAULT_GRIPPER_CLOSE_RAD``."""

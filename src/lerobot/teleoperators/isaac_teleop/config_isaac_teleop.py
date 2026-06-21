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
config subclass (e.g. :class:`SO101LeaderArmConfig`, and future ``ManusConfig``).
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass(kw_only=True)
class IsaacTeleopConfig(TeleoperatorConfig):
    """Shared config for all Isaac Teleop-backed teleoperators.

    Subclassed per input device (XR controller, Manus gloves, hand tracking,
    ...). Abstract: register the concrete device subclasses, not this base.
    """

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


# Provisional gripper open/close endpoints [rad] for the leader's gripper DOF, used to
# normalize the streamed gripper angle into the follower's RANGE_0_100 jaw target. These
# defaults are derived from the so101_leader plugin README's *example* calibration
# (home_ticks=2048, range 2000..3000 ticks; angle = (ticks - home) * 2*pi / 4096):
#   range_min 2000 ticks -> (2000-2048)*2*pi/4096 = -0.074 rad
#   range_max 3000 ticks -> (3000-2048)*2*pi/4096 = +1.460 rad
_DEFAULT_GRIPPER_OPEN_RAD = -0.074
_DEFAULT_GRIPPER_CLOSE_RAD = 1.460


@TeleoperatorConfig.register_subclass("isaac_teleop_so101_leader")
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

    collection_id: str = "so101_leader"
    """Tensor collection id the leader plugin pushes on; must match the running
    ``so101_leader`` plugin (its second positional arg, default ``"so101_leader"``)."""

    gripper_open_rad: float = _DEFAULT_GRIPPER_OPEN_RAD
    """Leader gripper angle [rad] at fully OPEN -> follower jaw 100. Provisional default;
    set from the plugin's ``calibrate`` subcommand. See ``_DEFAULT_GRIPPER_OPEN_RAD``."""

    gripper_close_rad: float = _DEFAULT_GRIPPER_CLOSE_RAD
    """Leader gripper angle [rad] at fully CLOSED -> follower jaw 0. Provisional default;
    set from the plugin's ``calibrate`` subcommand. See ``_DEFAULT_GRIPPER_CLOSE_RAD``."""

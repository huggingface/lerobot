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

"""Configuration dataclasses for NVIDIA Isaac Teleop-backed teleoperators.

:class:`IsaacTeleopConfig` holds the fields shared by every Isaac Teleop input
device (currently just the session ``app_name``); each device adds its own
config subclass (e.g. :class:`XRControllerConfig`, and future ``ManusConfig``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


@TeleoperatorConfig.register_subclass("isaac_teleop_controller")
@dataclass(kw_only=True)
class XRControllerConfig(IsaacTeleopConfig):
    """Config for Isaac Teleop XR (VR) controller teleoperation.

    Produces an absolute end-effector pose (position + quaternion) and a
    gripper command from a VR controller via NVIDIA Isaac Teleop's
    ``ControllersSource``, ``Se3AbsRetargeter``, and ``GripperRetargeter``.
    """

    hand_side: Literal["left", "right"] = "right"
    """Which controller hand to use."""

    clutch_threshold: float = 0.5
    """Squeeze value above which the clutch (``enabled``) engages.

    Mirrors the phone teleoperator's hold-to-enable button: while the
    controller squeeze is held above this threshold the teleoperator drives
    the robot; releasing it freezes the robot and re-arms the origin latch.
    """

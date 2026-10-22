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

:class:`IsaacTeleopConfig` holds the shared fields; each device adds its own subclass
(e.g. :class:`XRControllerConfig`, :class:`SO101LeaderArmConfig`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass(kw_only=True)
class IsaacTeleopConfig(TeleoperatorConfig):
    """Shared config for all Isaac Teleop-backed teleoperators.

    Uses its own draccus ``_choice_registry`` (decoupled from the global
    :class:`TeleoperatorConfig` one) so ``--teleop.type`` on a field typed
    ``IsaacTeleopConfig`` resolves against ONLY the Isaac devices — letting them claim
    short names (``xr_controller``, ``so101_leader``) without colliding with the global
    registry. These devices are selected by the example scripts, not routed through
    ``make_teleoperator_from_config``.
    """

    _choice_registry: ClassVar[dict] = {}

    app_name: str = "LeTeleop"
    """Application name for the OpenXR / Isaac Teleop session."""

    auto_launch_cloudxr: bool = True
    """Auto-launch the CloudXR runtime on :meth:`connect`. Set ``False`` (or export
    ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1``, which wins) when CloudXR runs externally.
    """

    cloudxr_env_file: str | None = None
    """Optional CloudXR device-profile ``.env`` (an INPUT profile selecting the headset
    transport) passed to ``CloudXRLauncher``. ``None`` keeps the default auto-WebRTC profile.
    """


# Static rebase from the OpenXR controller anchor frame (X=Right, Y=Up, Z=Backward) into the
# robot base frame (X=Forward, Y=Left, Z=Up). A proper rotation (det=+1): controller motion
# forward -> robot +X, right -> robot -Y (i.e. rightward), up -> robot +Z.
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

    Exposes the raw base-frame grip pose, squeeze, and trigger via ``ControllersSource``.
    No retargeters: the clutch and gripper mapping live in the owning loop.
    """

    hand_side: str = "right"
    """Which controller hand to use: ``"left"`` or ``"right"``. A plain ``str`` (validated in
    ``__post_init__``) because draccus cannot decode ``Literal``-typed fields from the CLI."""

    clutch_threshold: float = 0.5
    """Squeeze value above which the owning loop's clutch engages (held-to-enable). The
    device reports only the raw squeeze; the threshold is applied by the loop."""

    base_T_anchor: list[list[float]] = field(  # noqa: N815  (frameA_T_frameB transform-matrix convention)
        # Fresh copy per instance: returning the module-level list itself would alias one
        # mutable matrix across every config.
        default_factory=lambda: [row.copy() for row in _DEFAULT_BASE_T_ANCHOR]
    )
    """Static 4x4 [row-major] transform rebasing the OpenXR controller anchor frame into
    the robot base frame. Defaults to OpenXR (X=Right, Y=Up, Z=Backward) -> robot
    (X=Forward, Y=Left, Z=Up). Plain nested lists so the config stays serializable.
    """

    def __post_init__(self):
        if self.hand_side not in ("left", "right"):
            raise ValueError(f"hand_side must be 'left' or 'right', got {self.hand_side!r}")


# Provisional gripper open/close endpoints [rad], normalizing the streamed gripper angle
# into the follower's RANGE_0_100 jaw target. Derived from the so101_leader plugin README's
# example calibration (home_ticks=2048, range 2000..3000; angle = (ticks-home)*2*pi/4096).
_DEFAULT_GRIPPER_OPEN_RAD = -0.074
_DEFAULT_GRIPPER_CLOSE_RAD = 1.460


@IsaacTeleopConfig.register_subclass("so101_leader")
@dataclass(kw_only=True)
class SO101LeaderArmConfig(IsaacTeleopConfig):
    """Config for an Isaac Teleop SO-101 *leader arm* (generic joint-space device).

    Mirrors the leader's joint angles 1:1 onto a follower SO-101. The leader state is
    streamed in radians by the native ``so101_leader`` plugin and read via a
    ``JointStateSource``; the device converts arm joints to degrees and the gripper to the
    follower's RANGE_0_100 jaw target (no IK/clutch/retargeter on the LeRobot side).
    """

    port: str = ""
    """Serial port of the physical LEADER arm (e.g. ``/dev/ttyACM1``), forwarded to the
    plugin (which reads the servos) when the example launches it. Empty -> the plugin runs
    its synthetic trajectory."""

    collection_id: str = "so101_leader"
    """Tensor collection id the leader plugin pushes on; must match the running
    ``so101_leader`` plugin (its second positional arg, default ``"so101_leader"``)."""

    gripper_open_rad: float = _DEFAULT_GRIPPER_OPEN_RAD
    """Leader gripper angle [rad] at fully OPEN -> follower jaw 100. Provisional default;
    set from the plugin's ``calibrate`` subcommand. See ``_DEFAULT_GRIPPER_OPEN_RAD``."""

    gripper_close_rad: float = _DEFAULT_GRIPPER_CLOSE_RAD
    """Leader gripper angle [rad] at fully CLOSED -> follower jaw 0. Provisional default;
    set from the plugin's ``calibrate`` subcommand. See ``_DEFAULT_GRIPPER_CLOSE_RAD``."""

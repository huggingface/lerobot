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

:class:`IsaacTeleopConfig` holds the session fields shared by every device;
:class:`XRControllerConfig` adds the XR controller's own knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass(kw_only=True)
class IsaacTeleopConfig(TeleoperatorConfig):
    """Shared config for all Isaac Teleop-backed teleoperators.

    Deliberately NOT a draccus choice type: the example scripts type their ``teleop`` field
    as the concrete device config, so ``--teleop.<field>`` parses as a plain nested dataclass
    with no ``--teleop.type`` selector. These devices are constructed by the example scripts,
    not routed through ``make_teleoperator_from_config``.
    """

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


@dataclass(kw_only=True)
class XRControllerConfig(IsaacTeleopConfig):
    """Config for Isaac Teleop XR (VR) controller teleoperation.

    Carries the clutch retargeter in-pipeline and emits an absolute base-frame EE pose; the
    gripper mapping stays in the owning loop.
    """

    hand_side: str = "right"
    """Which controller hand to use: ``"left"`` or ``"right"``. A plain ``str`` (validated in
    ``__post_init__``) because draccus cannot decode ``Literal``-typed fields from the CLI."""

    clutch_threshold: float = 0.5
    """Squeeze value above which the clutch engages (held-to-enable). Passed to the in-pipeline
    clutch retargeter, which is the only place the comparison happens; the loop reads engagement
    back off the device rather than re-deriving it."""

    clutch_position_scale: float = 0.5
    """Controller-to-EE translation gain the in-pipeline clutch retargeter applies to the
    engage-relative delta -- dimensionless, and unrelated to ``clutch_threshold`` above despite the shared
    default value. ``1.0`` is 1:1 motion. A comfortable operator arm sweep is ~0.7 m, roughly
    2x the SO-101's ~0.35 m reach, so 1:1 motion drives the commanded EE target outside the
    reachable envelope within a single engaged segment; the ``0.5`` default maps a full sweep
    inside reach (0.4 m of controller motion -> 0.2 m of EE motion). Translation only --
    orientation stays 1:1. See https://github.com/NVIDIA/IsaacTeleop/issues/733."""

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

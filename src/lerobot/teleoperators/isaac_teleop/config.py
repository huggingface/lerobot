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
:class:`XRControllerConfig` adds the XR controller's own knobs and registers as
``--teleop.type=isaac_teleop`` on the global :class:`TeleoperatorConfig` registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass(kw_only=True)
class IsaacTeleopConfig(TeleoperatorConfig):
    """Shared config for all Isaac Teleop-backed teleoperators."""

    app_name: str = "LeRobot Teleop"
    """Application name for the OpenXR / Isaac Teleop session."""

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

# Default duration [s] for the reset-to-origin slew (on connect and on every declutch): long
# enough to follow in VR, short enough not to stall the operator between segments.
_RESET_DURATION_S = 2.0


@TeleoperatorConfig.register_subclass("isaac_teleop")
@dataclass(kw_only=True)
class XRControllerConfig(IsaacTeleopConfig):
    """Config for Isaac Teleop XR (VR) controller teleoperation.

    Carries the clutch retargeter in-pipeline and emits an absolute base-frame EE pose;
    :class:`~lerobot.teleoperators.isaac_teleop.xr_controller.XRController` maps it to the
    follower's joint space internally (its ``RobotProfile``, derived from ``--robot.type`` --
    see ``XRController.__init__``), so ``get_action()`` returns a plain joint-space action
    usable with the stock ``lerobot-teleoperate`` / ``lerobot-record`` identity action processor.
    """

    hand_side: str = "right"
    """Which controller hand to use: ``"left"`` or ``"right"``. A plain ``str`` (validated in
    ``__post_init__``) because draccus cannot decode ``Literal``-typed fields from the CLI."""

    clutch_threshold: float = 0.5
    """Squeeze value above which the clutch engages (held-to-enable). Passed to the in-pipeline
    clutch retargeter, which is the only place the comparison happens; ``get_action`` reads
    engagement back off the device rather than re-deriving it."""

    clutch_position_scale: float = 0.5
    """Controller-to-EE translation gain the in-pipeline clutch retargeter applies to the
    engage-relative delta -- dimensionless, and unrelated to ``clutch_threshold`` above despite the shared
    default value. ``1.0`` is 1:1 motion. A comfortable operator arm sweep is ~0.7 m, roughly
    2x the SO-101's ~0.35 m reach, so 1:1 motion drives the commanded EE target outside the
    reachable envelope within a single engaged segment; the ``0.5`` default maps a full sweep
    inside reach (0.4 m of controller motion -> 0.2 m of EE motion). Translation only --
    orientation stays 1:1. Overridden per-arm by ``RobotProfile.clutch_position_scale`` unless
    explicitly set. See https://github.com/NVIDIA/IsaacTeleop/issues/733."""

    engage_gate: bool = True
    """Refuse the clutch's latch until the operator's wrist matches the pose it is about to
    latch. The clutch composes orientation as a **delta**, so a latch taken 40 deg off
    leaves the arm 40 deg off the hand for the whole engagement; this makes the alignment a
    precondition instead. An enable precondition, not a safety-rated stop -- it gates the
    latch only and can never drop a live engagement.

    ``False`` widens the band to 180 deg rather than removing the gate: the preview still
    holds the latch until the rate limiter is passing through, which is what stops an
    engagement revealing a tool already hundreds of milliseconds behind the hand. Needs the
    robot twin either way -- the gate is the twin's affordance, and green is how it speaks."""

    engage_enter_deg: float = 20.0
    """Wrist-alignment band below which the gate may open. Only the *relation* to
    ``engage_exit_deg`` is pinned -- no absolute value here is defensible without a headset
    on a real operator."""

    engage_exit_deg: float = 30.0
    """Band above which an open gate closes again. Must be >= ``engage_enter_deg``: the
    angle is recomputed every frame off a noisy controller, and an affordance strobing at
    display rate in a headset is worse than a wrong one."""

    engage_dwell_s: float = 0.1
    """How long alignment must hold before the gate goes green."""

    robot_twin: bool = True
    """Run Isaac Teleop's ``ClutchPreview`` in the headset: the follower the operator drags
    by hand while disengaged, swapped for a gripper locked to the hand once the clutch
    engages, with the safety harness recolouring it and the engage gate holding the latch.
    The same object ``examples/robot_viz`` runs. Needs a Linux ``isaacteleop`` built with
    ``-DBUILD_VIZ=ON``; warns and runs without it otherwise. Only arms whose ``RobotProfile``
    names a ``preview_arm`` get one regardless of this flag -- SO-101/SO-100 and the reBot
    B601-RS today."""

    twin_gl_device: int = -1
    """Which GPU to build the twin's OpenGL context on. ``-1`` takes the first that yields
    one, which is right on a single-GPU machine. On a multi-GPU host the context has to
    land on the card the compositor already picked, and nothing makes that happen by
    default: this indexes EGL devices and need not agree with CUDA's ordering."""

    align_frame_to_preview: bool = True
    """Measure the XR-to-robot yaw at each engage instead of trusting ``base_T_anchor``.

    ``base_T_anchor`` is a static claim that the operator faces the same way the arm does.
    Stand 90 deg off it and pushing the controller away from yourself moves the jaw 90 deg
    off what you meant -- the error is exactly the angle you are standing off. Only that one
    yaw is unknown: both frames are gravity-aligned, so the axis convention is fixed, and the
    translation cancels through an engage-relative clutch.

    The correspondence is the **preview arm's base against the real arm's base**, the one
    pair the operator can see both of. Turn the wrist until the virtual arm is parallel to
    the real one, then squeeze. Not the aim ray against the jaw: the preview's
    ``base_yaw_bias`` exists precisely to keep its JAW pointing where the controller does,
    so the aim ray carries no information about the arm's heading by construction.

    Re-measured on disengaged frames and frozen through the engagement. It costs no motion,
    because the yaw is applied to the DELTA from the latch rather than to the pose -- on the
    engage frame that delta is zero, so the latched pose comes back exactly, for any yaw.

    Needs the robot twin: without a preview there is nothing to align. ``False`` keeps
    ``base_T_anchor`` exactly as configured."""

    base_T_anchor: list[list[float]] = field(  # noqa: N815  (frameA_T_frameB transform-matrix convention)
        # Fresh copy per instance: returning the module-level list itself would alias one
        # mutable matrix across every config.
        default_factory=lambda: [row.copy() for row in _DEFAULT_BASE_T_ANCHOR]
    )
    """Static 4x4 [row-major] transform rebasing the OpenXR controller anchor frame into
    the robot base frame. Defaults to OpenXR (X=Right, Y=Up, Z=Backward) -> robot
    (X=Forward, Y=Left, Z=Up). Plain nested lists so the config stays serializable.
    """

    reset_to_origin: bool = True
    """Slew the arm to its reset pose on connect and on every declutch, re-homing the clutch
    there. ``False`` re-homes in place instead (no motion), which the clutch's engage-relative
    delta still requires -- see ``XRController._start_reset``."""

    reset_duration: float = _RESET_DURATION_S
    """Duration [s] of the reset-to-origin slew."""

    def __post_init__(self):
        if self.hand_side not in ("left", "right"):
            raise ValueError(f"hand_side must be 'left' or 'right', got {self.hand_side!r}")
        if not 0.0 < self.engage_enter_deg <= self.engage_exit_deg:
            raise ValueError(
                "require 0 < engage_enter_deg <= engage_exit_deg, got "
                f"{self.engage_enter_deg} and {self.engage_exit_deg}"
            )

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

"""The robot twin the operator sees, assembled out of ``isaacteleop.viz.robot``.

Nothing here reimplements the preview. ``ClutchPreview`` is Isaac Teleop's own -- the same
object ``examples/robot_viz`` runs -- and this module only builds the four things it binds
to and reports what could not be built. The scene, the drag, the thumbstick tuning, the
ghost, the harness colours, the phase machine and the engage gate all live in the library.

**One deliberate difference, and it is the only one:** ``owns_clutch_home=False``, set
where the preview is constructed. In ``robot_viz`` the preview arm is the only arm there
is, so it supplies the clutch's home every disengaged frame. Here a real follower is on the
other end, its home comes from measured forward kinematics, and a preview pushing its own
would command the hardware to wherever the operator waved.

Linux and an ``isaacteleop`` built with ``-DBUILD_VIZ=ON`` only; :func:`build_twin` returns
``None`` with a warning on anything else and the loop runs without a twin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: The preview is a right-handed gripper -- the leader meshes are handed, and ``robot_viz``
#: pins it in one constant. A left-handed session gets the gate and no twin.
PREVIEW_HAND_SIDE = "right"


@dataclass(frozen=True)
class TwinBundle:
    """The four objects ``ClutchPreview`` binds to, built before the session exists.

    Split from the preview itself because ``ClutchPreview`` also needs the clutch
    retargeter, which is not built until ``connect()`` assembles the graph -- whereas the
    scene must exist *before* it, since the twin creates the OpenXR session the trackers
    then share.
    """

    twin: Any
    """``SceneTwin`` -- also the session's ``joint_publisher``."""

    arm: Any
    """``PreviewArm``, the follower dragged by the controller while disengaged."""

    monitor: Any
    """``InterventionMonitor``, which recolours the ghost by rate-limiter band."""

    gate: Any
    """``EngageGate``, whose verdict reaches the clutch's ``ENGAGE_PERMITTED_INPUT``."""


def build_twin(config, preview_arm: str) -> TwinBundle | None:
    """Build ``preview_arm``'s twin for ``config``, or return ``None`` with a warning saying
    why not. ``preview_arm`` keys ``isaacteleop.viz.robot.PREVIEW_ARMS``, and comes from the
    follower's own :class:`RobotProfile`.

    Degrading rather than raising is deliberate: the scene backend needs a Linux
    ``isaacteleop`` built with ``-DBUILD_VIZ=ON``, and an operator whose wheel lacks it
    wants the arm to teleoperate, not a traceback. A checksum mismatch or a scene that does
    not compile still raises -- those are defects, not environments.
    """
    if config.hand_side != PREVIEW_HAND_SIDE:
        logger.warning(
            "Robot twin disabled: the preview is a %s-handed gripper and hand_side is %r.",
            PREVIEW_HAND_SIDE,
            config.hand_side,
        )
        return None
    try:
        import numpy as np
        from isaacteleop.viz.robot import (
            PREVIEW_ARMS,
            EngageGate,
            EngageGateConfig,
            InterventionMonitor,
            PreviewArm,
            SceneTwin,
        )

        try:
            profile = PREVIEW_ARMS[preview_arm]
        except KeyError:
            raise RuntimeError(
                f"robot twin: no preview arm named {preview_arm!r}. This isaacteleop "
                f"offers {sorted(PREVIEW_ARMS)}; a RobotProfile names one that is not "
                "there, which is a version skew, not an environment."
            ) from None

        # Before the renderer, which uploads geometry once: PreviewArm repoints geom
        # materials and poses its joints at construction. Not placed yet -- that waits for
        # the first head pose, inside ClutchPreview.before_step.
        twin = SceneTwin(profile.scene(), gl_device_index=config.twin_gl_device)
        arm = PreviewArm(twin, profile)
        monitor = InterventionMonitor(twin)
        # engage_gate=False widens the band rather than removing the node: ClutchPreview
        # binds to a gate unconditionally, and a band of 180 deg admits every wrist while
        # leaving the limiter conjunct below in place.
        enter = 180.0 if not config.engage_gate else config.engage_enter_deg
        exit_ = 180.0 if not config.engage_gate else config.engage_exit_deg
        gate = EngageGate(
            config=EngageGateConfig(
                enter_rad=float(np.radians(enter)),
                exit_rad=float(np.radians(exit_)),
                dwell_s=config.engage_dwell_s,
            ),
            # robot_viz's own extra conjunct: the ghost renders the limiter's output, so a
            # gate that goes green while the limiter is still catching up reveals a tool
            # tens of degrees and hundreds of milliseconds behind the hand.
            app_conjunct=("limiter", "still catching up"),
        )
    except ImportError as error:
        logger.warning(
            "Robot twin unavailable, running without it: %s. Either the wheel has no "
            "Televiz (it ships on Linux; a Windows build and a -DBUILD_VIZ=OFF build both "
            "lack it), or it predates PREVIEW_ARMS and so cannot preview a named arm. Pass "
            "--teleop.robot_twin=false to silence this.",
            error,
        )
        return None
    except OSError as error:
        # A first-run fetch with no route to GitHub. A checksum mismatch is a RuntimeError
        # and is deliberately NOT caught -- a substituted mesh draws as a broken arm.
        logger.warning(
            "Robot twin scene could not be fetched, running without it: %s. "
            "Pass --teleop.robot_twin=false to silence this.",
            error,
        )
        return None

    logger.info(
        "Robot twin built for %s (scene backend: MuJoCo %s)",
        profile.label,
        twin.backend_version,
    )
    return TwinBundle(twin=twin, arm=arm, monitor=monitor, gate=gate)

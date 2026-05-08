# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Smooth-move helpers for teleoperator and follower robot handovers.

Shared utilities for linearly interpolating motion of an actuated teleoperator
or a follower robot between a current and a target pose. Used during phase
transitions (e.g. DAgger handovers) to avoid jerky position changes.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.robots import Robot
    from lerobot.teleoperators import Teleoperator


def teleop_supports_feedback(teleop: Teleoperator) -> bool:
    """Return True when the teleop can receive position feedback (is actuated).
    TODO(Maxime): See if it is possible to unify this interface across teleops instead of duck-typing.
    """
    return (
        bool(teleop.feedback_features)
        and hasattr(teleop, "disable_torque")
        and hasattr(teleop, "enable_torque")
    )


def teleop_smooth_move_to(
    teleop: Teleoperator, target_pos: dict, duration_s: float = 2.0, fps: int = 30
) -> None:
    """Smoothly move an actuated teleop to ``target_pos`` via linear interpolation.

    Requires the teleoperator to support feedback
    (i.e. have non-empty ``feedback_features`` and implement ``disable_torque`` / ``enable_torque``).

    TODO(Maxime): This blocks up to ``duration_s`` seconds, during this time
    the follower robot doesn't receive new actions, this could be an issue on LeKiwi.
    """
    if not teleop_supports_feedback(teleop):
        raise ValueError(
            "teleop_smooth_move_to requires an actuated teleop with feedback support "
            "(non-empty feedback_features and enable_torque/disable_torque methods)."
        )
    if duration_s < 0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")

    teleop.enable_torque()
    current = teleop.get_action()
    steps = max(int(duration_s * fps), 1)

    for step in range(steps + 1):
        t = step / steps
        interp = {
            k: current[k] * (1 - t) + target_pos[k] * t if k in target_pos else current[k] for k in current
        }
        teleop.send_feedback(interp)
        time.sleep(1 / fps)


def follower_smooth_move_to(
    robot: Robot, current: dict, target: dict, duration_s: float = 1.0, fps: int = 30
) -> None:
    """Smoothly move the follower robot from ``current`` to ``target`` action.

    Used when the teleop is non-actuated: instead of driving the leader arm
    to the follower, we bring the follower to the teleop's current pose.
    Both ``current`` and ``target`` must be in robot-action key space.
    """
    if duration_s < 0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")

    steps = max(int(duration_s * fps), 1)

    for step in range(steps + 1):
        t = step / steps
        interp = {k: current[k] * (1 - t) + target[k] * t if k in target else current[k] for k in current}
        robot.send_action(interp)
        time.sleep(1 / fps)

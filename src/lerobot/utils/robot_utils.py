# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import logging
import math
import platform
import time
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from lerobot.robots import Robot

logger = logging.getLogger(__name__)

# The name endings of a robot's motor channels, the observations and actions a policy reads and writes.
MOTOR_CHANNEL_SUFFIXES = (".pos", ".vel")


class PositionedRobot(Protocol):
    """A robot, or a wrapper around one, that reports and takes joint positions."""

    def get_observation(self) -> dict[str, Any]: ...

    def send_action(self, action: dict[str, Any]) -> Any: ...


def precise_sleep(seconds: float, spin_threshold: float = 0.010, sleep_margin: float = 0.005):
    """
    Wait for `seconds` with better precision than time.sleep alone at the expense of more CPU usage.

    Parameters:
      - seconds: duration to wait
      - spin_threshold: if remaining <= spin_threshold -> spin; otherwise sleep (seconds). Default 10ms
      - sleep_margin: when sleeping leave this much time before deadline to avoid oversleep. Default 5ms

    Note:
        The default parameters are chosen to prioritize timing accuracy over CPU usage for the common 30 FPS use case.
    """
    if seconds <= 0:
        return
    if spin_threshold < 0:
        raise ValueError(f"spin_threshold must be >= 0, got {spin_threshold}")
    if sleep_margin < 0:
        raise ValueError(f"sleep_margin must be >= 0, got {sleep_margin}")

    system = platform.system()
    # On macOS and Windows the scheduler / sleep granularity can make
    # short sleeps inaccurate. Instead of burning CPU for the whole
    # duration, sleep for most of the time and spin for the final few
    # milliseconds to achieve good accuracy with much lower CPU usage.
    if system in ("Darwin", "Windows"):
        end_time = time.perf_counter() + seconds
        while True:
            remaining = end_time - time.perf_counter()
            if remaining <= 0:
                break
            # If there's more than a couple milliseconds left, sleep most
            # of the remaining time and leave a small margin for the final spin.
            if remaining > spin_threshold:
                # Sleep but avoid sleeping past the end by leaving a small margin.
                time.sleep(max(remaining - sleep_margin, 0))
            else:
                # Final short spin to hit precise timing without long sleeps.
                pass
    else:
        # On Linux time.sleep is accurate enough for most uses
        time.sleep(seconds)


def align_to_checkpoint_order(
    features: dict[str, type | tuple], policy_action_names: list[str] | None, *, what: str
) -> dict[str, type | tuple]:
    """Order ``features`` so its motor entries follow the checkpoint's joint order.

    One rule for both sides: the policy emits and consumes tensors in the order it was
    trained on, so whichever dict is about to be flattened into a tensor has to match it.
    Only the scalar motor entries take part; anything else (camera shapes, on the state
    side) keeps its relative order at the end.

    A set mismatch is left alone rather than forced: extra ``.vel`` channels or an
    uncommanded base mean the two describe different things, and the robot's own order is
    the only meaningful one. ``what`` names the side being aligned, so the warning says
    which of the two reordered.
    """
    if not policy_action_names:
        return features

    motor_names = [name for name, feature in features.items() if not isinstance(feature, tuple)]
    if set(motor_names) != set(policy_action_names) or motor_names == policy_action_names:
        return features

    logger.warning(
        "Robot %s order %s differs from checkpoint joint order %s; reordering %s",
        what,
        motor_names,
        policy_action_names,
        what,
    )
    reordered = {name: features[name] for name in policy_action_names}
    reordered.update({name: feature for name, feature in features.items() if name not in reordered})
    return reordered


def policy_features(
    robot: Robot, policy_action_names: list[str] | None, *, keep_all: bool = False
) -> tuple[dict[str, type | tuple], dict[str, type | tuple]]:
    """The robot's observation and action features a policy reads and writes, motors in the checkpoint's order.

    The observation keeps the cameras and the position and velocity channels, such as a mobile manipulator's
    base velocities; the action keeps the position and velocity channels. `keep_all` keeps every state and
    action component whatever its name, as a sim robot needs. Both follow `align_to_checkpoint_order`.
    """
    observation: dict[str, type | tuple] = {
        name: feature
        for name, feature in robot.observation_features.items()
        if isinstance(feature, tuple)
        or (feature is float and (keep_all or name.endswith(MOTOR_CHANNEL_SUFFIXES)))
    }
    action = {
        name: feature
        for name, feature in robot.action_features.items()
        if keep_all or name.endswith(MOTOR_CHANNEL_SUFFIXES)
    }
    return (
        align_to_checkpoint_order(observation, policy_action_names, what="state"),
        align_to_checkpoint_order(action, policy_action_names, what="action"),
    )


def return_to_position(
    robot: PositionedRobot, target: dict[str, float], duration_s: float = 3.0, fps: int = 50
) -> bool:
    """Move the robot's joints in a straight line from where they are to `target`, over `duration_s`.

    Returns `True` when the motion completed, `False` when it failed partway: the robot is then at an
    arbitrary pose, so callers must not report a completed reset.
    """
    try:
        current_obs = robot.get_observation()
        current_pos = {k: float(current_obs[k]) for k in target}
        if not all(math.isfinite(current_pos[k]) and math.isfinite(target[k]) for k in target):
            raise ValueError("Return movement requires finite current and target positions")
        steps = max(int(duration_s * fps), 1)
        for step in range(1, steps + 1):
            t = step / steps
            interp = {}
            for k in current_pos:
                interp[k] = current_pos[k] * (1 - t) + target[k] * t
            robot.send_action(interp)
            precise_sleep(1 / fps)
    except Exception as e:
        logger.warning("Could not return to the target position: %s", e)
        return False
    return True

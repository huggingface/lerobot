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

"""Teleoperate a follower arm from an NVIDIA Isaac Teleop XR (VR) controller.

``lerobot-teleoperate``-style CLI (draccus): ``--robot.*`` configures the follower,
``--teleop.*`` the XR controller. Any arm with a ``RobotProfile`` in
``common.ROBOT_PROFILES`` can be driven::

    python -m examples.isaac_teleop_to_so101.teleoperate --robot.type=so101_follower \
        --robot.port=/dev/ttyACM0 --robot.id=so101_follower_arm

    # reBot B601-RS (RobStride motors on SocketCAN), 6-DOF so IK tracks orientation fully
    python -m examples.isaac_teleop_to_so101.teleoperate --robot.type=rebot_b601_follower \
        --robot.motor_family=rs --robot.port=can0 --robot.can_adapter=socketcan --robot.id=rebot_arm

The pipeline, clutch/IK internals, and reset-pose behavior live in ``common.py``.

The clutch bounds a teleoperation segment: releasing the squeeze (having engaged it at least
once) slews the arm back to its reset pose and re-homes the clutch there, so the next squeeze
starts from a known pose. ``--reset_to_origin=false`` disables the slew (startup and every
declutch alike).

Requires the ``isaacteleop`` package and an OpenXR runtime (install instructions in this
folder's ``README.md``).
"""

import time
from dataclasses import dataclass, field

from lerobot.configs import parser
from lerobot.robots import RobotConfig
from lerobot.utils.robot_utils import precise_sleep

from .common import (
    FPS,
    RESET_DURATION_S,
    DeclutchLatch,
    HoldLatch,
    build_device,
)
from .isaac_teleop import XRControllerConfig


@dataclass
class TeleoperateConfig:
    """``lerobot-teleoperate``-style CLI for the Isaac Teleop -> SO-101 example.

    The fields below the configs are the loop knobs (not part of the device's config). Use
    ``--flag=false`` for booleans (draccus style).
    """

    # FOLLOWER arm (--robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=...).
    # Must have a RobotProfile entry in common.ROBOT_PROFILES.
    robot: RobotConfig
    # XR controller knobs (--teleop.<field>=...); all defaulted, so --teleop.* is optional.
    teleop: XRControllerConfig = field(default_factory=XRControllerConfig)

    # Slew all joints to a default reset pose before the loop AND on every declutch
    # (--reset_to_origin=false to keep the arm where it is). After the slew the clutch seeds its
    # home from the measured pose.
    reset_to_origin: bool = True
    # Duration [s] of the reset-to-origin slew.
    reset_duration: float = RESET_DURATION_S


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    robot, device, motor_names = build_device(cfg)
    hold = HoldLatch(motor_names)
    declutch = DeclutchLatch()
    try:
        while True:
            t0 = time.perf_counter()
            obs = robot.get_observation()
            # Idle (compute() -> None) holds the pose latched on the active->idle edge.
            action = hold.resolve(device.compute(obs), obs)

            # Releasing the clutch ends the segment: reset() slews the arm to its reset pose and
            # re-homes the clutch there. Both latches are stale across the slew — the held pose
            # predates it and the declutch has fired — so replace them.
            if declutch.update(device.engaged()):
                device.reset()
                hold = HoldLatch(motor_names)
                declutch = DeclutchLatch()
                continue

            robot.send_action(action)
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        # A failing device cleanup must not skip the follower disconnect (which is what
        # disables torque on the arm).
        try:
            device.cleanup()
        finally:
            robot.disconnect()


def main():
    teleoperate()


if __name__ == "__main__":
    main()

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

"""Teleoperate an SO-101 follower arm via NVIDIA Isaac Teleop.

``lerobot-teleoperate``-style CLI (draccus): ``--teleop.type`` selects the Isaac device
(``xr_controller`` | ``so101_leader``), ``--robot.*`` the follower::

    # XR (VR) controller: clutch + soft-orientation IK
    python -m examples.isaac_teleop_to_so101.teleoperate --robot.type=so101_follower \
        --robot.port=/dev/ttyACM0 --robot.id=so101_follower_arm --teleop.type=xr_controller

    # SO-101 leader arm: 1:1 joint mirror (real leader on /dev/ttyACM1)
    python -m examples.isaac_teleop_to_so101.teleoperate --robot.type=so101_follower \
        --robot.port=/dev/ttyACM0 --robot.id=so101_follower_arm --teleop.type=so101_leader \
        --teleop.port=/dev/ttyACM1 --teleop.id=so101_leader_arm \
        --launch_plugin=/code/Teleop/install/plugins/so101_leader/so101_leader_plugin

``--teleop.type`` resolves against the Isaac device registry (see :class:`IsaacTeleopConfig`),
distinct from the serial ``so101_leader``. The pipelines, clutch/IK/align internals, and
reset-pose behavior live in ``common.py``.

On the XR path the clutch bounds a teleoperation segment: releasing the squeeze (having engaged
it at least once) slews the arm back to its reset pose and re-homes the clutch there, so the
next squeeze starts from a known pose. ``--reset_to_origin=false`` disables the slew (startup
and every declutch alike). The leader arm has no clutch and is unaffected.

Requires the ``isaacteleop`` package and an OpenXR runtime (install instructions in this
folder's ``README.md``).
"""

import time
from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.robots import RobotConfig
from lerobot.robots.so_follower import SOFollowerConfig  # noqa: F401  (registers so101_follower)
from lerobot.utils.robot_utils import precise_sleep

from .common import (
    ALIGN_DURATION_S,
    FPS,
    RESET_DURATION_S,
    DeclutchLatch,
    HoldLatch,
    build_device,
)
from .isaac_teleop import IsaacTeleopConfig


@dataclass
class TeleoperateConfig:
    """``lerobot-teleoperate``-style CLI for the Isaac Teleop -> SO-101 example.

    The fields below are the loop/launch knobs (not part of either device's config); the
    ``[xr]`` / ``[leader]`` tags mark which device a knob applies to. Use ``--flag=false``
    for booleans (draccus style).
    """

    # Isaac Teleop input device + its knobs (--teleop.type=xr_controller|so101_leader,
    # then --teleop.<field>=...). Resolved against IsaacTeleopConfig's own choice registry.
    teleop: IsaacTeleopConfig
    # SO-101 FOLLOWER arm (--robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=...).
    robot: RobotConfig

    # [leader] Path to the so101_leader plugin binary to spawn AFTER CloudXR is up (it then
    # inherits the runtime env). None (default) -> assume the plugin already runs externally.
    # The leader's serial port is --teleop.port (forwarded to the plugin; empty -> synthetic).
    launch_plugin: str | None = None

    # [xr] Slew all joints to a default reset pose before the loop AND on every declutch
    # (--reset_to_origin=false to keep the arm where it is). After the slew the clutch seeds its
    # home from the measured pose.
    reset_to_origin: bool = True
    # [xr] Duration [s] of the reset-to-origin slew.
    reset_duration: float = RESET_DURATION_S

    # [leader] Slew the follower to the leader's first pose before mirroring (--align=false to
    # begin the 1:1 mirror immediately; the follower may snap).
    align: bool = True
    # [leader] Duration [s] of the startup alignment slew.
    align_duration: float = ALIGN_DURATION_S


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
            # predates it and the declutch has fired — so replace them. A device without a
            # clutch reports engaged forever and never gets here.
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

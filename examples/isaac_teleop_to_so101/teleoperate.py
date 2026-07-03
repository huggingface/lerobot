# !/usr/bin/env python

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

This mirrors ``examples/phone_to_so100/teleoperate.py`` but swaps the phone for an
Isaac Teleop input device. The CLI is ``lerobot-teleoperate``-style (draccus): a follower
``--robot.*`` and an input ``--teleop.*``, where ``--teleop.type`` selects the Isaac
device (``xr_controller`` | ``so101_leader``)::

    # XR (VR) controller: clutch + soft-orientation IK
    python -m examples.isaac_teleop_to_so101.teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
        --robot.id=so101_follower_arm --teleop.type=xr_controller

    # SO-101 leader arm: 1:1 joint mirror (real leader on /dev/ttyACM1)
    python -m examples.isaac_teleop_to_so101.teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
        --robot.id=so101_follower_arm --teleop.type=so101_leader \
        --teleop.port=/dev/ttyACM1 --teleop.id=so101_leader_arm \
        --launch_plugin=/code/Teleop/install/plugins/so101_leader/so101_leader_plugin

``--teleop.type`` resolves against the Isaac Teleop device registry (its own draccus
choice registry, see :class:`IsaacTeleopConfig`), so ``so101_leader`` here is the Isaac
leader, distinct from the serial ``so101_leader`` of ``lerobot-teleoperate``. Device
config knobs are ``--teleop.*`` (e.g. ``--teleop.clutch_threshold``,
``--teleop.collection_id``, ``--teleop.auto_launch_cloudxr=false``); loop knobs
(``--reset_to_origin=false``, ``--align=false``, ``--launch_plugin``) are top-level.
draccus uses ``--flag=false`` for booleans (no ``--no-*`` form).

The per-device pipelines, clutch/IK/align internals, and the reset-pose behavior all live
in ``common.py`` (shared with ``record.py``); see its module docstring for the full
architecture. ``record.py`` runs this same control loop while also saving a LeRobot dataset.

Requires the ``isaac-teleop`` extra (``isaacteleop``) and an OpenXR runtime.
"""

import time
from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.robots import RobotConfig
from lerobot.robots.so_follower import SOFollowerConfig  # noqa: F401  (registers so101_follower)
from lerobot.teleoperators.isaac_teleop import IsaacTeleopConfig
from lerobot.utils.robot_utils import precise_sleep

from .common import (
    ALIGN_DURATION_S,
    FPS,
    RESET_DURATION_S,
    build_device,
    hold_action,
)


@dataclass
class TeleoperateConfig:
    """``lerobot-teleoperate``-style CLI for the unified Isaac Teleop -> SO-101 example.

    ``--teleop.type`` selects the Isaac input device and ``--teleop.*`` its config knobs;
    ``--robot.*`` configures the SO-101 follower. The fields below are the loop/launch knobs
    that are not part of either device's config (``--flag=false`` for the booleans). The
    ``[xr]`` / ``[leader]`` tags mark which device a knob applies to; a knob is ignored for
    the other device.
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

    # [xr] Slew all joints to a default reset pose before the loop (--reset_to_origin=false to
    # keep the arm where it is). After the slew the clutch seeds its home from the measured pose.
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
    try:
        while True:
            t0 = time.perf_counter()
            obs = robot.get_observation()
            action = device.compute(obs)
            if action is None:  # idle -> hold at the measured pose
                action = hold_action(obs, motor_names)
            robot.send_action(action)
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        device.cleanup()
        robot.disconnect()


def main():
    teleoperate()


if __name__ == "__main__":
    main()

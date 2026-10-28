#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Hardware smoke test: wiggle each OpenArm joint by a few degrees, then close/open the Gripette.

python examples/smoke_test.py --port can0 --side right --gripette 172.18.7.107:50051
"""

import argparse
import time

import numpy as np

from lerobot.cameras.gripette import GripetteCameraConfig
from lerobot_robot_openarm_gripette import OpenArmGripetteFollower, OpenArmGripetteFollowerConfig

JOINTS = [f"joint_{i}.pos" for i in range(1, 8)]


def ramp(robot: OpenArmGripetteFollower, start: dict, goal: dict, duration: float = 1.0, hz: int = 50):
    for t in np.linspace(0, 1, int(duration * hz)):
        robot.send_action({k: start[k] + t * (goal[k] - start[k]) for k in goal})
        time.sleep(1 / hz)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="can0")
    parser.add_argument("--side", default="right")
    parser.add_argument("--gripette", default="172.18.7.107:50051")
    parser.add_argument("--id", default="my_follower")
    parser.add_argument("--amplitude", type=float, default=5.0, help="Joint excursion in degrees.")
    parser.add_argument("--close", type=float, default=0.8, help="Gripper close target in radians.")
    args = parser.parse_args()

    robot = OpenArmGripetteFollower(
        OpenArmGripetteFollowerConfig(
            id=args.id,
            port=args.port,
            side=args.side,
            max_relative_target=2.0,
            cameras={"gripette": GripetteCameraConfig(addr=args.gripette, width=648, height=486, fps=10)},
            gripette_addr=args.gripette,
        )
    )

    robot.connect()
    try:
        obs = robot.get_observation()
        home = {k: obs[k] for k in JOINTS}
        print("Arm (deg):", {k: round(v, 1) for k, v in home.items()})
        print(
            f"Gripper (rad): proximal={obs['proximal.pos']:.3f} distal={obs['distal.pos']:.3f}",
            "| camera:",
            obs["gripette"].shape,
        )
        input("Each joint will move a few degrees and back. Press ENTER to start...")

        for k in JOINTS:
            print(f"Moving {k}")
            target = {**home, k: home[k] + args.amplitude}
            ramp(robot, home, target)
            ramp(robot, target, home)

        for name, goal in [("close", args.close), ("open", 0.0)]:
            print(f"Gripper {name} -> {goal} rad")
            robot.send_action({**home, "proximal.pos": goal, "distal.pos": goal})
            time.sleep(1.5)
            obs = robot.get_observation()
            print(f"  proximal={obs['proximal.pos']:.3f} distal={obs['distal.pos']:.3f}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()

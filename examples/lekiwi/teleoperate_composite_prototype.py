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

"""Prototype the processor-based design from PR #4941's review.

Run without arguments for a hardware-free demonstration. For a LeKiwi client:
    uv run examples/lekiwi/teleoperate_composite_prototype.py --remote-ip <ip>
Add --leader-port <serial-port> to combine a standard keyboard and an SO-100/101 leader.
The original CLI implementation remains available for comparison in this prototype branch.
"""

import argparse
import time

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.processor.hold_joints_processor import HoldUncommandedJointsProcessorStep
from lerobot.processor.keyboard_base_processor import KeyboardToBaseActionProcessorStep
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.composite import CompositeTeleoperator
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig


def make_pipeline(
    joint_names: tuple[str, ...],
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    return RobotProcessorPipeline(
        steps=[KeyboardToBaseActionProcessorStep(), HoldUncommandedJointsProcessorStep(joint_names)],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def demonstrate() -> None:
    pipeline = make_pipeline(("arm_shoulder.pos", "arm_gripper.pos"))
    samples = [
        ("Keyboard only", {"w": None}, {"arm_shoulder.pos": 10.0, "arm_gripper.pos": 30.0}),
        ("Hold initial pose despite measurement drift", {"w": None}, {"arm_shoulder.pos": 9.0}),
        ("Leader command survives keyboard mapping", {"w": None, "arm_shoulder.pos": 20.0}, {}),
        ("Released keyboard stops base", {"arm_shoulder.pos": 20.0}, {}),
    ]
    for description, action, observation in samples:
        print(f"{description}:\n  {pipeline((action, observation))}")


def teleoperate(remote_ip: str, leader_port: str | None, fps: int) -> None:
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=remote_ip, id="prototype_lekiwi"))
    keyboard = KeyboardTeleop(KeyboardTeleopConfig(id="prototype_keyboard"))
    teleop: Teleoperator = keyboard
    if leader_port:
        leader = SO100Leader(SO100LeaderConfig(port=leader_port, id="prototype_leader"))
        teleop = CompositeTeleoperator({"": keyboard, "arm_": leader})
    pipeline = make_pipeline(tuple(key for key in robot.action_features if key.endswith(".pos")))

    try:
        robot.connect()
        teleop.connect()
        pipeline.reset()
        while keyboard.is_connected:
            start = time.perf_counter()
            observation = robot.get_observation()
            raw_action = teleop.get_action(observation=observation)
            action = pipeline((raw_action, observation))
            robot.send_action(action)
            time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if robot.is_connected:
                robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        finally:
            try:
                teleop.disconnect()
            finally:
                if robot.is_connected:
                    robot.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-ip")
    parser.add_argument("--leader-port")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.leader_port and not args.remote_ip:
        parser.error("--leader-port requires --remote-ip")
    if args.remote_ip:
        teleoperate(args.remote_ip, args.leader_port, args.fps)
    else:
        demonstrate()


if __name__ == "__main__":
    main()

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

"""Continuously hold a manually positioned OpenArm pose without policy inference."""

import argparse
import threading
import time

import numpy as np
from motorbridge import Controller

from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase
from lerobot_robot_openarm_gripette import (
    OpenArmGripetteFollower,
    OpenArmGripetteFollowerConfig,
)
from lerobot_robot_openarm_gripette.manual_start import prepare_manual_start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="can1")
    parser.add_argument("--id", default="my_follower")
    parser.add_argument("--can-fd", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frequency", type=float, default=50.0)
    parser.add_argument("--active-zero-torque", action="store_true")
    args = parser.parse_args()
    if args.frequency <= 0:
        parser.error("--frequency must be positive")

    robot = OpenArmGripetteFollower(
        OpenArmGripetteFollowerConfig(
            id=args.id,
            arm_config=OpenArmFollowerConfigBase(
                port=args.port,
                side="right",
                use_can_fd=args.can_fd,
            ),
            use_gripper_camera=False,
        )
    )
    arm = robot.arm
    stop = threading.Event()

    try:
        arm.bus._controller = (
            Controller.from_socketcanfd(args.port) if args.can_fd else Controller(channel=args.port)
        )
        for name, motor in arm.bus.motors.items():
            arm.bus._handles[name] = arm.bus._controller.add_damiao_motor(
                motor.id,
                motor.recv_id,
                motor.motor_type_str.lower().removeprefix("dm"),
            )
        target = prepare_manual_start(
            robot,
            np.ones(7),
            active_zero_torque=args.active_zero_torque,
            zero_torque_frequency_hz=args.frequency,
        )
        hold_action = {f"joint_{index}.pos": float(value) for index, value in enumerate(target, start=1)}
        print("Target joints (deg):", np.round(target, 2))

        def wait_for_stop() -> None:
            input("Torque is ON. Support the arm, then press ENTER to disable torque and exit...")
            stop.set()

        threading.Thread(target=wait_for_stop, daemon=True).start()
        period = 1.0 / args.frequency
        next_report = time.monotonic()
        while not stop.is_set():
            tick = time.monotonic()
            arm.send_action(hold_action)
            if tick >= next_report:
                current_observation = arm.get_observation()
                current = np.array(
                    [current_observation[f"joint_{index}.pos"] for index in range(1, 8)]
                )
                print(
                    f"max error={np.max(np.abs(current - target)):.2f} deg "
                    f"current={np.round(current, 1)}"
                )
                next_report = tick + 1.0
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    finally:
        if arm.bus.is_connected:
            arm.bus.disable_torque()
            arm.bus.disconnect()


if __name__ == "__main__":
    main()

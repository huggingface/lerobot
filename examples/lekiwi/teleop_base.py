#!/usr/bin/env python

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

"""Keyboard teleoperation for the LeKiwi mobile base only (no leader arm).

Run the host on the robot (e.g. Jetson) first, e.g.:
    uv run python -m lerobot.robots.lekiwi.lekiwi_host \
        --robot.id=my_awesome_kiwi --host.connection_time_s=99999

Then run this script on your laptop:
    uv run python examples/lekiwi/teleop_base.py

Controls (see LeKiwiClientConfig.teleop_keys):
    w/s : forward / backward
    a/d : strafe left / right
    z/x : rotate left / right
    r/f : speed up / slow down
    q   : quit
"""

import time

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import precise_sleep

# ---- Edit these ----
REMOTE_IP = "192.168.50.187"  # the robot host (Jetson) IP
FPS = 30
# --------------------


def main():
    # cameras={} -> wheels-only test, don't expect video frames from the host
    robot_config = LeKiwiClientConfig(remote_ip=REMOTE_IP, id="my_lekiwi", cameras={})
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    robot = LeKiwiClient(robot_config)
    keyboard = KeyboardTeleop(keyboard_config)

    print(f"Connecting to LeKiwi host at {REMOTE_IP} ...")
    robot.connect()
    keyboard.connect()

    if not robot.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or keyboard is not connected!")

    quit_key = robot.teleop_keys["quit"]
    print(__doc__.split("Controls")[1] if "Controls" in __doc__ else "")
    print("Starting base teleop loop. Focus THIS terminal window for key capture. Ctrl-C to stop.")

    try:
        while True:
            t0 = time.perf_counter()

            pressed_keys = keyboard.get_action()
            if quit_key in pressed_keys:
                print("Quit key pressed, stopping.")
                break

            base_action = robot._from_keyboard_to_base_action(pressed_keys)
            robot.send_action(base_action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        # Send an explicit stop, then disconnect (host watchdog also stops the base).
        try:
            robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        except Exception:
            pass
        keyboard.disconnect()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()

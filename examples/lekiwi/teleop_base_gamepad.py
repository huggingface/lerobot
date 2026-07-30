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

"""Drive the LeKiwi mobile base with a PlayStation DualShock (DS4/DS5) controller.

Unlike the keyboard teleop, a gamepad needs no macOS Accessibility permission and
gives smooth *proportional* speed from the analog sticks.

Run the host on the robot (e.g. Jetson) first:
    uv run python -m lerobot.robots.lekiwi.lekiwi_host \
        --robot.id=my_awesome_kiwi --host.connection_time_s=99999 --robot.cameras='{}'

Then on your laptop (controller plugged in / paired):
    uv run python examples/lekiwi/teleop_base_gamepad.py

Default controls (DualShock):
    Left stick   : translate  (up=forward, down=back, left/right=strafe)
    Right stick  : rotate      (left/right = turn left/right)
    L1 / R1      : slower / faster (3 speed levels)
    Options (or Ctrl-C) : quit

If your sticks/buttons are mapped differently, run once and read the
"Detected ... axes/buttons" dump printed on start, then tweak the AXIS_*/BTN_*
indices below.
"""

import time

import pygame

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import precise_sleep

# ---------------- Config: edit these ----------------
REMOTE_IP = "192.168.50.187"  # the robot host (Jetson) IP
FPS = 30
DEADZONE = 0.12  # ignore small stick noise near center

# Axis indices (SDL/pygame defaults for DS4/DS5)
AXIS_LEFT_X = 0  # strafe:   -1 = left,    +1 = right
AXIS_LEFT_Y = 1  # forward:  -1 = up,      +1 = down
AXIS_RIGHT_X = 2  # rotate:  -1 = left,    +1 = right

# Button indices (SDL/pygame defaults for DS4/DS5)
BTN_L1 = 9  # slow down a speed level
BTN_R1 = 10  # speed up a speed level
BTN_QUIT = 6  # Options/Share depending on model; Ctrl-C always works too

# Speed levels: (max linear m/s, max angular deg/s). Matches LeKiwiClient tiers.
SPEED_LEVELS = [
    (0.10, 30.0),  # slow
    (0.20, 60.0),  # medium
    (0.30, 90.0),  # fast
]
# ----------------------------------------------------


def _deadzone(v: float) -> float:
    return 0.0 if abs(v) < DEADZONE else v


def main():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No gamepad detected. Plug in / pair your DualShock and retry.")

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Detected controller: {joystick.get_name()}")
    print(f"  axes={joystick.get_numaxes()} buttons={joystick.get_numbuttons()} hats={joystick.get_numhats()}")
    print("Controls: left stick=translate, right stick=rotate, L1/R1=speed, Options/Ctrl-C=quit\n")

    robot_config = LeKiwiClientConfig(remote_ip=REMOTE_IP, id="my_lekiwi", cameras={})
    robot = LeKiwiClient(robot_config)
    print(f"Connecting to LeKiwi host at {REMOTE_IP} ...")
    robot.connect()
    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    speed_index = 0
    prev_l1 = prev_r1 = False

    try:
        while True:
            t0 = time.perf_counter()
            pygame.event.pump()  # refresh internal state

            # ---- speed level edge-triggered on L1/R1 ----
            l1 = joystick.get_numbuttons() > BTN_L1 and joystick.get_button(BTN_L1)
            r1 = joystick.get_numbuttons() > BTN_R1 and joystick.get_button(BTN_R1)
            if r1 and not prev_r1:
                speed_index = min(speed_index + 1, len(SPEED_LEVELS) - 1)
                print(f"speed -> level {speed_index} {SPEED_LEVELS[speed_index]}")
            if l1 and not prev_l1:
                speed_index = max(speed_index - 1, 0)
                print(f"speed -> level {speed_index} {SPEED_LEVELS[speed_index]}")
            prev_l1, prev_r1 = l1, r1

            if joystick.get_numbuttons() > BTN_QUIT and joystick.get_button(BTN_QUIT):
                print("Quit button pressed, stopping.")
                break

            xy_speed, theta_speed = SPEED_LEVELS[speed_index]

            lx = _deadzone(joystick.get_axis(AXIS_LEFT_X))
            ly = _deadzone(joystick.get_axis(AXIS_LEFT_Y))
            rx = _deadzone(joystick.get_axis(AXIS_RIGHT_X))

            # Map to LeKiwi body-frame velocities (see LeKiwiClient._from_keyboard_to_base_action):
            #   +x = forward, +y = left, +theta = rotate left
            x_vel = -ly * xy_speed  # stick up (negative) -> forward
            y_vel = -lx * xy_speed  # stick left (negative) -> +y (left)
            theta_vel = -rx * theta_speed  # stick right (positive) -> rotate right (negative)

            robot.send_action({"x.vel": x_vel, "y.vel": y_vel, "theta.vel": theta_vel})

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        try:
            robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        except Exception:
            pass
        robot.disconnect()
        pygame.quit()
        print("Disconnected.")


if __name__ == "__main__":
    main()

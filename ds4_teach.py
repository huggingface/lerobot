#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team and Antigravity. All rights reserved.
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

"""
Teaching utility for the SO-ARM101 follower arm using a DualShock 4 controller.
Move the arm into a desired position (via sticks or loose mode + manual posing),
then press [Options] to record it. Repeat for as many positions as needed.
Press [Cross] to save all recorded positions to a JSON file and exit.

Controller Mapping:
  Left Stick  X-axis  →  Joint 1 (shoulder_pan)
  Left Stick  Y-axis  →  Joint 2 (shoulder_lift)
  Right Stick Y-axis  →  Joint 3 (elbow_flex)
  Right Stick X-axis  →  Joint 4 (wrist_flex)
  L1 / R1             →  Joint 5 (wrist_roll)   dec / inc
  D-Pad Up / Down     →  Joint 6 (gripper)       open / close

Teaching Controls:
  [Options]   →  Record current motor positions as a new waypoint
  [Square]    →  Undo last recorded waypoint
  [Triangle]  →  Toggle Loose Mode (disable torque for manual arm posing)
  [Cross ✕]   →  Save all recorded positions to JSON and exit
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import pygame
from termcolor import colored

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.motors.motors_bus import MotorNormMode
from lerobot.utils.robot_utils import precise_sleep


# ── DS4 pygame axis / button indices (USB, macOS) ───────────────────────────
AXIS_LS_X     = 0
AXIS_LS_Y     = 1
AXIS_RS_X     = 2
AXIS_RS_Y     = 3

BTN_CROSS     = 0   # save & exit
BTN_CIRCLE    = 1
BTN_SQUARE    = 2   # undo last
BTN_TRIANGLE  = 3   # toggle loose mode
BTN_L1        = 9   # wrist_roll dec
BTN_R1        = 10  # wrist_roll inc
BTN_OPTIONS   = 6   # record position
BTN_DPAD_UP   = 11  # gripper open
BTN_DPAD_DOWN = 12  # gripper close

DEADZONE  = 0.12
MAX_SPEED = 80.0
DPAD_SPEED = 40.0
# ─────────────────────────────────────────────────────────────────────────────


def find_usb_ports():
    try:
        from serial.tools import list_ports
        return [port.device for port in list_ports.comports()]
    except ImportError:
        if platform.system() == "Windows":
            return [f"COM{i}" for i in range(1, 20)]
        else:
            return (
                [str(p) for p in Path("/dev").glob("tty.usbmodem*")]
                + [str(p) for p in Path("/dev").glob("ttyUSB*")]
            )


def get_motor_limits(follower, motor_name):
    motor = follower.bus.motors[motor_name]
    if not follower.calibration or motor_name not in follower.calibration:
        if motor.norm_mode == MotorNormMode.RANGE_M100_100:
            return -100.0, 100.0
        elif motor.norm_mode == MotorNormMode.RANGE_0_100:
            return 0.0, 100.0
        else:
            return -180.0, 180.0

    calib = follower.calibration[motor_name]
    min_ = calib.range_min
    max_ = calib.range_max

    if motor.norm_mode == MotorNormMode.RANGE_M100_100:
        return -100.0, 100.0
    elif motor.norm_mode == MotorNormMode.RANGE_0_100:
        return 0.0, 100.0
    elif motor.norm_mode == MotorNormMode.DEGREES:
        mid = (min_ + max_) / 2
        res = follower.bus.model_resolution_table[motor.model]
        max_res = res - 1
        norm_min = (min_ - mid) * 360 / max_res
        norm_max = (max_ - mid) * 360 / max_res
        return norm_min, norm_max
    else:
        return float("-inf"), float("inf")


def apply_deadzone(value: float, deadzone: float = DEADZONE) -> float:
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def get_ds4_inputs(joy: pygame.joystick.Joystick):
    pygame.event.pump()
    return {
        "ls_x":      apply_deadzone(joy.get_axis(AXIS_LS_X)),
        "ls_y":      apply_deadzone(joy.get_axis(AXIS_LS_Y)),
        "rs_x":      apply_deadzone(joy.get_axis(AXIS_RS_X)),
        "rs_y":      apply_deadzone(joy.get_axis(AXIS_RS_Y)),
        "dpad_up":   bool(joy.get_button(BTN_DPAD_UP)),
        "dpad_down": bool(joy.get_button(BTN_DPAD_DOWN)),
        "cross":     joy.get_button(BTN_CROSS),
        "square":    joy.get_button(BTN_SQUARE),
        "triangle":  joy.get_button(BTN_TRIANGLE),
        "l1":        joy.get_button(BTN_L1),
        "r1":        joy.get_button(BTN_R1),
        "options":   joy.get_button(BTN_OPTIONS),
    }


def print_ui(motors, targets, obs, limits, torque_enabled, inputs, waypoints, last_event):
    sys.stdout.write("\033[H")

    print(colored("=========================================================================", "cyan"))
    print(colored("  SO-ARM101  ·  DS4 TEACHING MODE", "cyan", attrs=["bold"]))
    print(colored("=========================================================================", "cyan"))

    torque_str = (
        colored(" [Torque Enabled] ", "green", attrs=["bold", "reverse"])
        if torque_enabled
        else colored(" [LOOSE MODE — pose arm freely] ", "red", attrs=["bold", "reverse"])
    )
    print(f"Status: {torque_str}")

    # Waypoint counter
    count = len(waypoints)
    count_str = colored(f"  {count} position{'s' if count != 1 else ''} recorded  ", "magenta", attrs=["bold", "reverse"])
    print(f"Waypoints: {count_str}")

    # Last event flash
    if last_event:
        print(colored(f"  ► {last_event}", "yellow", attrs=["bold"]))
    else:
        print()
    print()

    # Recorded positions preview (last 3)
    if waypoints:
        print(colored("Last Recorded Positions:", "white", attrs=["underline"]))
        preview = waypoints[-3:]
        for i, wp in enumerate(preview):
            idx = len(waypoints) - len(preview) + i + 1
            vals = "  ".join([f"{v:+7.2f}" for v in wp.values()])
            print(colored(f"  [{idx:>3}]  ", "grey") + vals)
        print()

    # Stick input
    def bar(val, width=8):
        filled = int(abs(val) * width)
        arrow = "►" if val >= 0 else "◄"
        if val >= 0:
            return "[" + " " * width + arrow + "█" * filled + "]"
        else:
            return "[" + "█" * filled + arrow + " " * width + "]"

    print(colored("Live Stick Input:", "white", attrs=["underline"]))
    print(f"  LS X (pan)         {bar(inputs['ls_x'])}  {inputs['ls_x']:+.2f}")
    print(f"  LS Y (lift)        {bar(inputs['ls_y'])}  {inputs['ls_y']:+.2f}")
    print(f"  RS Y (elbow)       {bar(inputs['rs_y'])}  {inputs['rs_y']:+.2f}")
    print(f"  RS X (wrist_flex)  {bar(inputs['rs_x'])}  {inputs['rs_x']:+.2f}")

    l1_str = colored("[L1]", "green" if inputs["l1"] else "grey")
    r1_str = colored("[R1]", "green" if inputs["r1"] else "grey")
    du_str = colored("[▲]", "green" if inputs["dpad_up"] else "grey")
    dd_str = colored("[▼]", "green" if inputs["dpad_down"] else "grey")
    print(f"  Wrist Roll         {l1_str} dec  {r1_str} inc")
    print(f"  Gripper            {du_str} open  {dd_str} close")
    print()

    # Telemetry table
    print(colored("-------------------------------------------------------------------------", "grey"))
    print(colored(
        f"{'MOTOR/JOINT':<15} | {'TARGET':>8} | {'PRESENT':>8} | {'MIN':>8} | {'MAX':>8} | {'STATUS':<8}",
        "white", attrs=["bold"]
    ))
    print(colored("-------------------------------------------------------------------------", "grey"))

    for motor in motors:
        t_val = targets[motor]
        o_val = obs.get(f"{motor}.pos", 0.0)
        l_min, l_max = limits[motor]

        status = colored("OK", "green")
        if abs(t_val - l_min) < 1e-2 or abs(t_val - l_max) < 1e-2:
            status = colored("CLAMPED", "yellow", attrs=["bold"])
        if not torque_enabled:
            status = colored("LOOSE", "red")

        print(f"{motor:<15} | {t_val:>8.2f} | {o_val:>8.2f} | {l_min:>8.2f} | {l_max:>8.2f} | {status:<8}")

    print(colored("-------------------------------------------------------------------------", "grey"))
    print(colored("Teaching Controls:", "white", attrs=["bold"]))
    print("  [Options]   → Record current position as new waypoint")
    print("  [Square]    → Undo last recorded waypoint")
    print("  [Triangle]  → Toggle loose mode (manually pose the arm)")
    print("  [Cross ✕]   → Save all waypoints to JSON and exit")
    print(colored("=========================================================================", "cyan"))
    sys.stdout.flush()


def save_waypoints(waypoints: list, output_dir: str) -> str:
    """Save waypoints list to a timestamped JSON file. Returns the file path."""
    filename = f"positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(waypoints, f, indent=2)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="DS4 teaching utility for SO-101 follower arm.")
    parser.add_argument("--port", type=str, default="/dev/tty.usbmodem5A7B2890981",
                        help="Serial port of follower arm")
    parser.add_argument("--id", type=str, default="my_awesome_follower_arm",
                        help="ID of follower arm calibration")
    parser.add_argument("--use-degrees", action="store_true", default=True,
                        help="Use degrees for control")
    args = parser.parse_args()

    # ── Pygame / DS4 init ────────────────────────────────────────────────────
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print(colored("No joystick/controller detected.", "red", attrs=["bold"]))
        print("Make sure the DS4 is connected via USB and try again.")
        sys.exit(1)

    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(colored(f"Controller detected: {joy.get_name()}", "green"))
    # ────────────────────────────────────────────────────────────────────────

    # ── Serial port validation ───────────────────────────────────────────────
    port = args.port
    if not os.path.exists(port):
        print(colored(f"Default port '{port}' not found.", "yellow"))
        print("Scanning for available USB ports...")
        available = find_usb_ports()
        if available:
            print("Found available ports:")
            for idx, dev in enumerate(available):
                print(f"  [{idx}] {dev}")
            try:
                user_choice = input(colored("Enter index to connect or press Enter to keep default: ", "cyan"))
                if user_choice.strip():
                    port = available[int(user_choice.strip())]
            except Exception:
                print("Invalid input, using default.")
        else:
            print(colored("No USB serial devices detected. Check arm power and connection.", "red"))
    # ────────────────────────────────────────────────────────────────────────

    print(colored(f"Connecting to SO Follower on port '{port}' with ID '{args.id}'...", "cyan"))

    follower_config = SO101FollowerConfig(
        port=port,
        id=args.id,
        use_degrees=args.use_degrees,
    )

    try:
        follower = SO101Follower(follower_config)
        follower.connect()
    except Exception as e:
        print(colored(f"\nFailed to connect to the follower robot: {e}", "red", attrs=["bold"]))
        print("Please verify the USB connection, motor power supply, and calibration.")
        sys.exit(1)

    # ── Prevent snap-to-stale-goal on startup ────────────────────────────────
    print(colored("Syncing motor goal positions to present positions...", "cyan"))
    follower.bus.disable_torque()
    _init_obs = follower.get_observation()
    _init_sync = {f"{m}.pos": _init_obs[f"{m}.pos"] for m in follower.bus.motors.keys()}
    follower.send_action(_init_sync)
    follower.bus.enable_torque()
    # ────────────────────────────────────────────────────────────────────────

    # ── Limits & initial targets ─────────────────────────────────────────────
    motors = list(follower.bus.motors.keys())
    limits = {motor: get_motor_limits(follower, motor) for motor in motors}

    if "wrist_roll" in follower.calibration:
        calib = follower.calibration["wrist_roll"]
        motor_obj = follower.bus.motors["wrist_roll"]
        if motor_obj.norm_mode == MotorNormMode.DEGREES:
            mid = (calib.range_min + calib.range_max) / 2.0
            max_res = follower.bus.model_resolution_table[motor_obj.model] - 1
            norm_min = (calib.range_min - mid) * 360 / max_res
            norm_max = (calib.range_max - mid) * 360 / max_res
            limits["wrist_roll"] = (norm_min, norm_max)
    if "wrist_roll" in limits:
        l_min, l_max = limits["wrist_roll"]
        limits["wrist_roll"] = (max(l_min, -180.0), min(l_max, 180.0))

    obs = follower.get_observation()
    targets = {motor: obs[f"{motor}.pos"] for motor in motors}
    # ────────────────────────────────────────────────────────────────────────

    # ── State ────────────────────────────────────────────────────────────────
    waypoints = []          # list of dicts, one per recorded position
    torque_enabled = True
    last_event = ""         # message shown on UI for one frame

    prev_cross    = False
    prev_square   = False
    prev_triangle = False
    prev_options  = False

    fps = 30
    dt  = 1.0 / fps

    # Output dir = same folder as this script
    output_dir = os.path.dirname(os.path.abspath(__file__))
    # ────────────────────────────────────────────────────────────────────────

    # Clear screen
    print("\033[2J\033[H", end="")

    inputs = {
        "ls_x": 0.0, "ls_y": 0.0, "rs_x": 0.0, "rs_y": 0.0,
        "dpad_up": False, "dpad_down": False,
        "cross": False, "square": False, "triangle": False,
        "l1": False, "r1": False, "options": False,
    }
    print_ui(motors, targets, obs, limits, torque_enabled, inputs, waypoints, last_event)

    try:
        while True:
            t0 = time.perf_counter()
            last_event = ""

            inputs = get_ds4_inputs(joy)

            # ── Edge-detected button events ──────────────────────────────────

            # Cross ✕ → save & exit
            if inputs["cross"] and not prev_cross:
                raise KeyboardInterrupt

            # Options → record current position
            if inputs["options"] and not prev_options:
                snapshot = {motor: round(obs.get(f"{motor}.pos", targets[motor]), 4)
                            for motor in motors}
                waypoints.append(snapshot)
                last_event = f"Recorded position [{len(waypoints)}]"

            # Square → undo last
            if inputs["square"] and not prev_square:
                if waypoints:
                    waypoints.pop()
                    last_event = f"Undid last — {len(waypoints)} position(s) remaining"
                else:
                    last_event = "Nothing to undo"

            # Triangle → toggle torque
            if inputs["triangle"] and not prev_triangle:
                if torque_enabled:
                    follower.bus.disable_torque()
                    torque_enabled = False
                    last_event = "Loose mode ON — pose arm freely, then record"
                else:
                    follower.bus.enable_torque()
                    torque_enabled = True
                    last_event = "Torque re-enabled"

            prev_cross    = inputs["cross"]
            prev_square   = inputs["square"]
            prev_triangle = inputs["triangle"]
            prev_options  = inputs["options"]

            # ── Observation ──────────────────────────────────────────────────
            obs = follower.get_observation()

            # ── Loose mode: track physical position ──────────────────────────
            if not torque_enabled:
                for motor in motors:
                    targets[motor] = obs.get(f"{motor}.pos", targets[motor])
            else:
                # Analog stick movement
                targets["shoulder_pan"]  += inputs["ls_x"] * MAX_SPEED * dt
                targets["shoulder_lift"] += inputs["ls_y"] * MAX_SPEED * dt
                targets["elbow_flex"]    += inputs["rs_y"] * MAX_SPEED * dt
                targets["wrist_flex"]    += inputs["rs_x"] * MAX_SPEED * dt

                if inputs["l1"]:
                    targets["wrist_roll"] -= MAX_SPEED * dt
                if inputs["r1"]:
                    targets["wrist_roll"] += MAX_SPEED * dt

                if inputs["dpad_up"]:
                    targets["gripper"] -= DPAD_SPEED * dt
                if inputs["dpad_down"]:
                    targets["gripper"] += DPAD_SPEED * dt

                # Clamp to limits
                for motor in motors:
                    l_min, l_max = limits[motor]
                    targets[motor] = max(l_min, min(l_max, targets[motor]))

                follower.send_action({f"{motor}.pos": targets[motor] for motor in motors})

            # Render UI
            print_ui(motors, targets, obs, limits, torque_enabled, inputs, waypoints, last_event)

            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\n")
    finally:
        try:
            follower.disconnect()
        except Exception:
            pass
        pygame.quit()

        # Save waypoints if any were recorded
        if waypoints:
            filepath = save_waypoints(waypoints, output_dir)
            print(colored(f"Saved {len(waypoints)} position(s) to:", "green", attrs=["bold"]))
            print(colored(f"  {filepath}", "cyan"))
        else:
            print(colored("No positions recorded — nothing saved.", "yellow"))

        print(colored("Goodbye!", "green", attrs=["bold"]))


if __name__ == "__main__":
    main()
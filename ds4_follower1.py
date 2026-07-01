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
Interactive utility to control a SO-ARM100/101 follower arm using a DualShock 4
controller connected via USB. Analog stick deflection controls joint speed —
push further for faster movement. Smoothing and variable speed make control
precise and natural. Includes safe calibration clamping and a live dashboard.

Controller Mapping:
  Left Stick  X-axis  →  Joint 1 (shoulder_pan)   left/right
  Left Stick  Y-axis  →  Joint 2 (shoulder_lift)  push UP = arm goes UP
  Right Stick Y-axis  →  Joint 3 (elbow_flex)     push UP = elbow raises
  L1 / R1             →  Joint 4 (wrist_flex)     L1 = pitch down, R1 = pitch up
  Right Stick X-axis  →  Joint 5 (wrist_roll)     twist wrist left/right
  D-Pad Up / Down     →  Joint 6 (gripper)        open / close

Speed Modes (toggle any time):
  [Square]    →  SLOW mode  (~25 deg/s)  — fine positioning
  [Circle]    →  FAST mode  (~60 deg/s)  — coarse movement
  Default speed is NORMAL (~35 deg/s)

Special Controls:
  [Options]   →  Go to home position (midpoints of mechanical ranges)
  [Triangle]  →  Toggle Loose Mode (disable/enable motor torque)
  [Cross ✕]   →  Gracefully disconnect and exit
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import pygame
from termcolor import colored

# Import lerobot components
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.motors.motors_bus import MotorNormMode
from lerobot.utils.robot_utils import precise_sleep


# ── DS4 pygame axis / button indices (USB, macOS/Linux) ─────────────────────
AXIS_LS_X    = 0   # Left stick horizontal   (-1 = left,  +1 = right)
AXIS_LS_Y    = 1   # Left stick vertical     (-1 = up,    +1 = down)  ← pygame convention
AXIS_RS_X    = 2   # Right stick horizontal  (-1 = left,  +1 = right)
AXIS_RS_Y    = 3   # Right stick vertical    (-1 = up,    +1 = down)  ← pygame convention

BTN_CROSS    = 0   # ✕  — exit
BTN_CIRCLE   = 1   # fast mode
BTN_SQUARE   = 2   # slow mode
BTN_TRIANGLE = 3   # toggle loose mode
BTN_L1       = 9   # wrist_flex pitch down
BTN_R1       = 10  # wrist_flex pitch up
BTN_OPTIONS  = 6   # home
BTN_DPAD_UP  = 11  # gripper open  (macOS USB: dpad reported as buttons)
BTN_DPAD_DOWN= 12  # gripper close

# ── Speed settings ───────────────────────────────────────────────────────────
SPEED_SLOW   = 40.0   # deg/s — fine positioning
SPEED_NORMAL = 60.0   # deg/s — default
SPEED_FAST   = 90.0   # deg/s — coarse movement
DPAD_SPEED   = 30.0   # deg/s — gripper (digital)
L1R1_SPEED   = 30.0   # deg/s — wrist_flex (digital)

# ── Feel tweaks ──────────────────────────────────────────────────────────────
DEADZONE     = 0.15   # slightly wider deadzone → less drift at rest
SMOOTH_ALPHA = 0.18   # EMA smoothing weight (lower = smoother, more lag)
                      # 0.18 gives ~2-3 frames of lag which feels natural
# ─────────────────────────────────────────────────────────────────────────────


def find_usb_ports():
    """List potential serial ports on the host."""
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
    """Dynamically compute safe limits in normalized action space."""
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
    """Zero out small stick values and rescale the remainder to 0–1."""
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def get_ds4_inputs(joy: pygame.joystick.Joystick):
    """
    Read and return a clean snapshot of all relevant DS4 inputs.
    Y-axes are INVERTED here so that push-up = positive throughout the script.
    """
    pygame.event.pump()

    return {
        # Analog axes — deadzone applied, Y axes inverted so UP = positive
        "ls_x":      apply_deadzone( joy.get_axis(AXIS_LS_X)),
        "ls_y":      apply_deadzone(-joy.get_axis(AXIS_LS_Y)),   # ← inverted
        "rs_x":      apply_deadzone( joy.get_axis(AXIS_RS_X)),
        "rs_y":      apply_deadzone(-joy.get_axis(AXIS_RS_Y)),   # ← inverted
        # D-pad
        "dpad_up":   bool(joy.get_button(BTN_DPAD_UP)),
        "dpad_down": bool(joy.get_button(BTN_DPAD_DOWN)),
        # Buttons
        "cross":    bool(joy.get_button(BTN_CROSS)),
        "triangle": bool(joy.get_button(BTN_TRIANGLE)),
        "square":   bool(joy.get_button(BTN_SQUARE)),
        "circle":   bool(joy.get_button(BTN_CIRCLE)),
        "l1":       bool(joy.get_button(BTN_L1)),
        "r1":       bool(joy.get_button(BTN_R1)),
        "options":  bool(joy.get_button(BTN_OPTIONS)),
    }


def smooth(current: float, target_raw: float) -> float:
    """
    Exponential moving average: blends new raw target toward current.
    Kills jitter from stick noise without adding noticeable lag.
    """
    return SMOOTH_ALPHA * target_raw + (1.0 - SMOOTH_ALPHA) * current


def print_ui(motors, targets, obs, limits, torque_enabled, speed_mode, inputs):
    """Draw a flicker-free terminal dashboard."""
    sys.stdout.write("\033[H")

    print(colored("=========================================================================", "cyan"))
    print(colored("  SO-ARM101 FOLLOWER  ·  DUALSHOCK 4 CONTROLLER", "cyan", attrs=["bold"]))
    print(colored("=========================================================================", "cyan"))

    torque_str = (
        colored(" [Torque Enabled] ", "green", attrs=["bold", "reverse"])
        if torque_enabled
        else colored(" [LOOSE MODE / Torque Disabled] ", "red", attrs=["bold", "reverse"])
    )
    speed_color = {"SLOW": "blue", "NORMAL": "green", "FAST": "red"}[speed_mode]
    speed_str = colored(f" [{speed_mode} MODE] ", speed_color, attrs=["bold", "reverse"])
    print(f"Status: {torque_str}  {speed_str}")
    print()

    # Live stick visualisation
    def bar(val, width=10):
        filled = int(abs(val) * width)
        arrow = "▲" if val >= 0 else "▼"
        if val >= 0:
            return "[" + " " * width + arrow + "█" * filled + "]"
        else:
            return "[" + "█" * filled + arrow + " " * width + "]"

    def bar_h(val, width=10):
        filled = int(abs(val) * width)
        arrow = "►" if val >= 0 else "◄"
        if val >= 0:
            return "[" + " " * width + arrow + "█" * filled + "]"
        else:
            return "[" + "█" * filled + arrow + " " * width + "]"

    print(colored("Live Stick Input  (▲=up/right is positive):", "white", attrs=["underline"]))
    print(f"  LS X  → pan        {bar_h(inputs['ls_x'])}  {inputs['ls_x']:+.2f}")
    print(f"  LS Y  → lift       {bar(inputs['ls_y'])}  {inputs['ls_y']:+.2f}  (▲ = arm up)")
    print(f"  RS Y  → elbow      {bar(inputs['rs_y'])}  {inputs['rs_y']:+.2f}  (▲ = elbow up)")
    print(f"  RS X  → wrist roll {bar_h(inputs['rs_x'])}  {inputs['rs_x']:+.2f}")

    l1_str = colored("[L1] pitch↓", "green" if inputs["l1"] else "white")
    r1_str = colored("[R1] pitch↑", "green" if inputs["r1"] else "white")
    du_str = colored("[▲] open",    "green" if inputs["dpad_up"]   else "white")
    dd_str = colored("[▼] close",   "green" if inputs["dpad_down"] else "white")
    print(f"  Wrist flex  {l1_str}  {r1_str}")
    print(f"  Gripper     {du_str}  {dd_str}")
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
    print(colored("Controls:", "white", attrs=["bold"]))
    print("  LS  X/Y     → pan / lift arm         RS  X/Y → wrist roll / elbow")
    print("  L1 / R1     → wrist pitch down / up  D-pad ▲▼ → gripper open / close")
    print("  [Square]    → SLOW mode  [Circle] → FAST mode  (default = NORMAL)")
    print("  [Options]   → Home all joints         [Triangle] → Toggle torque")
    print("  [Cross ✕]   → Exit")
    print(colored("=========================================================================", "cyan"))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="DS4 controller utility for SO-101 follower arm.")
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

    # Wrist roll: override with calibration-derived range, then cap to ±180
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
        safe_min, safe_max = -180.0, 180.0
        l_min, l_max = limits["wrist_roll"]
        limits["wrist_roll"] = (max(l_min, safe_min), min(l_max, safe_max))

    obs = follower.get_observation()
    targets = {motor: obs[f"{motor}.pos"] for motor in motors}

    # Soft out-of-range warning only — no forced move at startup
    for motor in motors:
        l_min, l_max = limits[motor]
        current = targets[motor]
        if current < l_min or current > l_max:
            print(colored(
                f"[WARN] {motor} present position {current:.2f} is outside "
                f"configured limits [{l_min:.2f}, {l_max:.2f}]. "
                "Limits enforced on next input.",
                "yellow", attrs=["bold"]
            ))
    # ────────────────────────────────────────────────────────────────────────

    # ── Button edge-detection state ──────────────────────────────────────────
    prev_triangle = False
    prev_options  = False
    prev_cross    = False
    # ────────────────────────────────────────────────────────────────────────

    # Clear screen
    print("\033[2J\033[H", end="")

    torque_enabled = True
    speed_mode     = "NORMAL"   # "SLOW" | "NORMAL" | "FAST"

    fps = 30
    dt  = 1.0 / fps

    # Blank inputs for first frame
    inputs = {
        "ls_x": 0.0, "ls_y": 0.0, "rs_x": 0.0, "rs_y": 0.0,
        "dpad_up": False, "dpad_down": False,
        "cross": False, "triangle": False,
        "square": False, "circle": False,
        "l1": False, "r1": False, "options": False,
    }
    print_ui(motors, targets, obs, limits, torque_enabled, speed_mode, inputs)

    try:
        while True:
            t0 = time.perf_counter()

            # Read controller
            inputs = get_ds4_inputs(joy)

            # ── Speed mode ───────────────────────────────────────────────────
            if inputs["square"]:
                speed_mode = "SLOW"
            elif inputs["circle"]:
                speed_mode = "FAST"
            else:
                speed_mode = "NORMAL"

            max_speed = {"SLOW": SPEED_SLOW, "NORMAL": SPEED_NORMAL, "FAST": SPEED_FAST}[speed_mode]

            # ── Edge-detected button events ──────────────────────────────────

            # Cross ✕ → exit
            if inputs["cross"] and not prev_cross:
                raise KeyboardInterrupt

            # Options → home
            if inputs["options"] and not prev_options:
                for motor in motors:
                    l_min, l_max = limits[motor]
                    targets[motor] = (l_min + l_max) / 2.0

            # Triangle → toggle torque
            if inputs["triangle"] and not prev_triangle:
                if torque_enabled:
                    follower.bus.disable_torque()
                    torque_enabled = False
                else:
                    follower.bus.enable_torque()
                    torque_enabled = True

            prev_cross    = inputs["cross"]
            prev_options  = inputs["options"]
            prev_triangle = inputs["triangle"]

            # ── Observation ──────────────────────────────────────────────────
            obs = follower.get_observation()

            # ── Loose mode: track physical position to prevent snap ──────────
            if not torque_enabled:
                for motor in motors:
                    targets[motor] = obs.get(f"{motor}.pos", targets[motor])

            else:
                # ── Compute raw desired targets from stick input ──────────────
                #
                # MAPPING (all Y axes already inverted in get_ds4_inputs):
                #   LS X       → shoulder_pan   (right = positive)
                #   LS Y       → shoulder_lift  (up stick = arm rises)
                #   RS Y       → elbow_flex     (up stick = elbow raises)
                #   RS X       → wrist_roll     (right = roll right)
                #   L1 / R1    → wrist_flex     (L1 = pitch down, R1 = pitch up)
                #   D-pad ▲▼  → gripper        (up = open, down = close)

                raw = dict(targets)  # copy current smoothed targets as base

                raw["shoulder_pan"]  += inputs["ls_x"] * max_speed * dt
                raw["shoulder_lift"] += inputs["ls_y"] * max_speed * dt   # up stick = positive = arm up
                raw["elbow_flex"]    -= inputs["rs_y"] * max_speed * dt   # up stick = elbow down (inverted)
                raw["wrist_roll"]    += inputs["rs_x"] * max_speed * dt

                # L1/R1 — wrist pitch, digital at a gentler fixed speed
                wrist_speed = L1R1_SPEED if speed_mode == "NORMAL" else max_speed
                if inputs["l1"]:
                    raw["wrist_flex"] -= wrist_speed * dt   # L1 = pitch down
                if inputs["r1"]:
                    raw["wrist_flex"] += wrist_speed * dt   # R1 = pitch up

                # D-pad — gripper, digital
                gripper_speed = DPAD_SPEED if speed_mode == "NORMAL" else max_speed * 0.5
                if inputs["dpad_up"]:
                    raw["gripper"] -= gripper_speed * dt    # open
                if inputs["dpad_down"]:
                    raw["gripper"] += gripper_speed * dt    # close

                # ── Apply EMA smoothing ──────────────────────────────────────
                # Smoothing is applied to analog-controlled joints only.
                # Digital (L1/R1, dpad) joints skip smoothing so buttons feel crisp.
                for motor in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_roll"]:
                    targets[motor] = smooth(targets[motor], raw[motor])

                # Digital joints: apply directly (no smoothing, feels more deliberate)
                targets["wrist_flex"] = raw["wrist_flex"]
                targets["gripper"]    = raw["gripper"]

                # ── Clamp all joints to mechanical limits ────────────────────
                for motor in motors:
                    l_min, l_max = limits[motor]
                    targets[motor] = max(l_min, min(l_max, targets[motor]))

                # ── Send action ──────────────────────────────────────────────
                follower.send_action({f"{motor}.pos": targets[motor] for motor in motors})

            # Render UI
            print_ui(motors, targets, obs, limits, torque_enabled, speed_mode, inputs)

            # Timing
            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nExiting. Disconnecting arm...")
    finally:
        try:
            follower.disconnect()
        except Exception:
            pass
        pygame.quit()
        print(colored("Goodbye!", "green", attrs=["bold"]))


if __name__ == "__main__":
    main()

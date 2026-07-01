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
Interactive utility to control a SO-ARM100/101 follower arm using keyboard input.
No leader arm or complex URDF required. Includes safe calibration clamping and a live dashboard.

Keys vertical columns:
- Joint 1 (shoulder_pan):  [Q] Dec (-)  |  [A] Inc (+)
- Joint 2 (shoulder_lift): [W] Dec (-)  |  [S] Inc (+)
- Joint 3 (elbow_flex):    [E] Dec (-)  |  [D] Inc (+)
- Joint 4 (wrist_flex):    [R] Dec (-)  |  [F] Inc (+)
- Joint 5 (wrist_roll):    [T] Dec (-)  |  [G] Inc (+)
- Joint 6 (gripper):       [Y] Dec (-)  |  [H] Inc (+)

Special Controls:
- [Space]: Go to home positions (midpoints of mechanical ranges)
- [L]:     Toggle Loose Mode (Disable/Enable motor torque dynamically for manual motion checks)
- [+]/[=]: Increase jog speed sensitivity (degrees/units per second)
- [-]:     Decrease jog speed sensitivity (degrees/units per second)
- [Esc]:   Gracefully disconnect and exit
"""

import argparse
import logging
import os
import platform
import sys
import time
import threading
from pathlib import Path

from termcolor import colored

# Import lerobot components
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.motors.motors_bus import MotorNormMode
from lerobot.utils.robot_utils import precise_sleep

# Check pynput availability
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


# Thread safety & global inputs
active_keys = set()
global_events = []
input_lock = threading.Lock()

def on_press(key):
    global global_events
    with input_lock:
        if hasattr(key, "char") and key.char is not None:
            char = key.char.lower()
            if char in ["+", "=", "-", "l"]:
                global_events.append(char)
            else:
                active_keys.add(char)
        else:
            if key == keyboard.Key.space:
                global_events.append("space")
            elif key == keyboard.Key.esc:
                global_events.append("esc")

def on_release(key):
    with input_lock:
        if hasattr(key, "char") and key.char is not None:
            active_keys.discard(key.char.lower())


def find_usb_ports():
    """List potential serial ports on the host."""
    try:
        from serial.tools import list_ports
        return [port.device for port in list_ports.comports()]
    except ImportError:
        # Fallback system globbing if pyserial isn't directly exposed
        if platform.system() == "Windows":
            return [f"COM{i}" for i in range(1, 20)]
        else:
            return [str(p) for p in Path("/dev").glob("tty.usbmodem*")] + [str(p) for p in Path("/dev").glob("ttyUSB*")]


def get_motor_limits(follower, motor_name):
    """Dynamically compute safe limits in normalized action space."""
    motor = follower.bus.motors[motor_name]
    if not follower.calibration or motor_name not in follower.calibration:
        # Without calibration fallback safely
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
        model = motor.model
        res = follower.bus.model_resolution_table[model]
        max_res = res - 1
        norm_min = (min_ - mid) * 360 / max_res
        norm_max = (max_ - mid) * 360 / max_res
        return norm_min, norm_max
    else:
        return float("-inf"), float("inf")


def print_ui(motors, targets, obs, limits, torque_enabled, speed, active_cols):
    """Draw a gorgeous, flicker-free terminal dashboard."""
    # Move cursor to top left of terminal
    sys.stdout.write("\033[H")
    
    # Header block
    print(colored("=========================================================================", "cyan"))
    print(colored("  SO-ARM101 FOLLOWER KEYBOARD CONTROLLER & TESTING TELEMETRY", "cyan", attrs=["bold"]))
    print(colored("=========================================================================", "cyan"))
    
    # State indicator
    torque_str = colored(" [Torque Enabled] ", "green", attrs=["bold", "reverse"]) if torque_enabled else colored(" [LOOSE MODE / Torque Disabled] ", "red", attrs=["bold", "reverse"])
    speed_str = colored(f" {speed:4.1f} units/s ", "yellow", attrs=["bold"])
    print(f"Status: {torque_str}  |  Jog Sensitivity: {speed_str}  (Adjust with [+/-])")
    print()

    # Mappings Guide
    print(colored("Keyboard Jog Layout Columns:", "white", attrs=["underline"]))
    print("  " + "  ".join([colored(f"Col {i+1}", "grey") for i in range(6)]))
    print("  " + "  ".join([colored(f"[{c[0].upper()}]", "red" if c[0] in active_cols else "yellow") + "/" + colored(f"[{c[1].upper()}]", "green" if c[1] in active_cols else "yellow") for c in [("q","a"), ("w","s"), ("e","d"), ("r","f"), ("t","g"), ("y","h")]]))
    print("   Pan   Lift   Elbow  WristF  WristR  Gripper")
    print()

    # Telemetry Table
    print(colored("-------------------------------------------------------------------------", "grey"))
    print(colored(f"{'MOTOR/JOINT':<15} | {'TARGET':>8} | {'PRESENT':>8} | {'MIN LIMIT':>10} | {'MAX LIMIT':>10} | {'STATUS':<8}", "white", attrs=["bold"]))
    print(colored("-------------------------------------------------------------------------", "grey"))
    
    for motor in motors:
        t_val = targets[motor]
        o_val = obs.get(f"{motor}.pos", 0.0)
        l_min, l_max = limits[motor]
        
        # Determine status flag
        status = colored("OK", "green")
        if abs(t_val - l_min) < 1e-2 or abs(t_val - l_max) < 1e-2:
            status = colored("CLAMPED", "yellow", attrs=["bold"])
        if not torque_enabled:
            status = colored("LOOSE", "red")

        print(f"{motor:<15} | {t_val:>8.2f} | {o_val:>8.2f} | {l_min:>10.2f} | {l_max:>10.2f} | {status:<8}")
        
    print(colored("-------------------------------------------------------------------------", "grey"))
    print(colored("Controls Guide:                                                          ", "white", attrs=["bold"]))
    print("  - [Space]: Move all joints to calibrated midpoints (Homed)")
    print("  - [L]    : Toggle Torque on/off (manual adjustment & telemetry inspection)")
    print("  - [Esc]  : Gracefully shutdown and exit")
    print(colored("=========================================================================", "cyan"))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Keyboard control and testing utility for SO-101 follower arm.")
    parser.add_argument("--port", type=str, default="/dev/tty.usbmodem5A7B2890981", help="Serial port of follower arm")
    parser.add_argument("--id", type=str, default="my_awesome_follower_arm", help="ID of follower arm calibration")
    parser.add_argument("--use-degrees", action="store_true", default=True, help="Use degrees for control")
    args = parser.parse_args()

    # Validate pynput
    if not PYNPUT_AVAILABLE:
        print(colored("Error: 'pynput' library is required to listen to keyboard events.", "red", attrs=["bold"]))
        print("Please install it with:")
        print(colored("  uv sync --extra hardware", "yellow", attrs=["bold"]))
        print("or:")
        print(colored("  uv pip install pynput", "yellow", attrs=["bold"]))
        sys.exit(1)

    # Port presence validation
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
            print(colored("No USB serial devices detected. Please make sure the arm is powered and connected.", "red"))

    print(colored(f"Connecting to SO Follower on port '{port}' with ID '{args.id}'...", "cyan"))
    
    follower_config = SO101FollowerConfig(
        port=port,
        id=args.id,
        use_degrees=args.use_degrees
    )

    try:
        follower = SO101Follower(follower_config)
        follower.connect()
    except Exception as e:
        print(colored(f"\nFailed to connect to the follower robot: {e}", "red", attrs=["bold"]))
        print("Please verify the USB connection, motor power supply, and that you calibrated the arm first.")
        sys.exit(1)

    # ── Prevent snap-to-stale-goal on startup ───────────────────────────────
    # When connect() enables torque the motor will immediately drive toward
    # whatever goal position is sitting in its register from the last session.
    # Fix: disable torque, read the TRUE present position, write that back as
    # the new goal, THEN re-enable torque so there is zero commanded movement.
    print(colored("Syncing motor goal positions to present positions (prevents startup snap)...", "cyan"))
    follower.bus.disable_torque()
    _init_obs = follower.get_observation()
    _init_sync = {f"{motor}.pos": _init_obs[f"{motor}.pos"]
                  for motor in follower.bus.motors.keys()}
    follower.send_action(_init_sync)
    follower.bus.enable_torque()
    # ────────────────────────────────────────────────────────────────────────

    # Initialize key mappings
    # Mapping of decrement/increment keys to motor names
    key_maps = {
        "q": ("shoulder_pan", -1),
        "a": ("shoulder_pan", 1),
        "w": ("shoulder_lift", -1),
        "s": ("shoulder_lift", 1),
        "e": ("elbow_flex", -1),
        "d": ("elbow_flex", 1),
        "r": ("wrist_flex", -1),
        "f": ("wrist_flex", 1),
        "t": ("wrist_roll", -1),
        "g": ("wrist_roll", 1),
        "y": ("gripper", -1),
        "h": ("gripper", 1)
    }

    # Fetch limits and initial positions
    motors = list(follower.bus.motors.keys())
    limits = {motor: get_motor_limits(follower, motor) for motor in motors}
    # Override wrist_roll limits using calibration data for accurate mechanical range
    if "wrist_roll" in follower.calibration:
        calib = follower.calibration["wrist_roll"]
        motor_obj = follower.bus.motors["wrist_roll"]
        if motor_obj.norm_mode == MotorNormMode.DEGREES:
            mid = (calib.range_min + calib.range_max) / 2.0
            max_res = follower.bus.model_resolution_table[motor_obj.model] - 1
            norm_min = (calib.range_min - mid) * 360 / max_res
            norm_max = (calib.range_max - mid) * 360 / max_res
            limits["wrist_roll"] = (norm_min, norm_max)

    # Ensure joint 5 (wrist_roll) does not exceed safe mechanical limits.
    # Example safe range: -150 to +150 degrees (adjust as needed).
    if "wrist_roll" in limits:
        safe_min, safe_max = -180.0, 180.0
        l_min, l_max = limits["wrist_roll"]
        # Clamp the configured limits to the safe range.
        limits["wrist_roll"] = (max(l_min, safe_min), min(l_max, safe_max))
    
    obs = follower.get_observation()
    targets = {motor: obs[f"{motor}.pos"] for motor in motors}
    # Goal is already synced by the startup block above; no send needed here.

    # ── Soft out-of-range warning (no forced move at startup) ───────────────
    # Do NOT hard-clamp targets here.  A hard clamp would immediately command
    # movement toward the limit boundary, which is exactly the snap behaviour
    # we just fixed.  Instead warn the user; the per-tick clamp in the main
    # loop will gently enforce limits the moment the user jogs that joint.
    for motor in motors:
        l_min, l_max = limits[motor]
        current = targets[motor]
        if current < l_min or current > l_max:
            print(colored(
                f"[WARN] {motor} present position {current:.2f} is outside "
                f"configured limits [{l_min:.2f}, {l_max:.2f}]. "
                "Limits enforced on next jog — do NOT press Space until arm is safe.",
                "yellow", attrs=["bold"]
            ))
    # ────────────────────────────────────────────────────────────────────────

    # Keyboard Listener setup
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Clear screen initially
    print("\033[2J\033[H", end="")

    torque_enabled = True
    jog_speed = 25.0  # Units/Degrees per second
    fps = 30
    dt = 1.0 / fps

    print_ui(motors, targets, obs, limits, torque_enabled, jog_speed, [])

    try:
        while True:
            t0 = time.perf_counter()

            # Process keyboard events
            current_active = set()
            events = []
            with input_lock:
                current_active = set(active_keys)
                events = list(global_events)
                global_events.clear()

            # Process global events
            for event in events:
                if event == "esc":
                    raise KeyboardInterrupt
                elif event == "space":
                    # Move to calibrated home midpoint
                    for motor in motors:
                        l_min, l_max = limits[motor]
                        targets[motor] = (l_min + l_max) / 2.0
                elif event == "l":
                    # Toggle torque
                    if torque_enabled:
                        follower.bus.disable_torque()
                        torque_enabled = False
                    else:
                        follower.bus.enable_torque()
                        torque_enabled = True
                elif event in ["+", "="]:
                    jog_speed = min(100.0, jog_speed + 5.0)
                elif event == "-":
                    jog_speed = max(5.0, jog_speed - 5.0)

            # Get current observation
            obs = follower.get_observation()

            # If torque is disabled (loose mode), update targets to track physical position
            # This prevents sudden snaps when re-enabling torque
            if not torque_enabled:
                for motor in motors:
                    targets[motor] = obs.get(f"{motor}.pos", targets[motor])
            else:
                # Update targets based on pressed keys
                for key in current_active:
                    if key in key_maps:
                        motor, direction = key_maps[key]
                        targets[motor] += direction * jog_speed * dt
                        
                # Clamp targets to mechanical limits
                for motor in motors:
                    l_min, l_max = limits[motor]
                    targets[motor] = max(l_min, min(l_max, targets[motor]))

                # Send action to follower
                action = {f"{motor}.pos": targets[motor] for motor in motors}
                follower.send_action(action)

            # Render UI
            print_ui(motors, targets, obs, limits, torque_enabled, jog_speed, current_active)

            # Control loop timing
            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nExiting. Disconnecting arm...")
    finally:
        # Stop keyboard listener
        listener.stop()
        
        # Safe disconnect (disables torque)
        try:
            follower.disconnect()
        except Exception:
            pass
        
        print(colored("Goodbye!", "green", attrs=["bold"]))


if __name__ == "__main__":
    main()

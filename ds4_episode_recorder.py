#!/usr/bin/env python
"""
DS4 Follower Arm — Data Collection for LeRobot VLA Training
============================================================
DS4 controls the arm as before. Keyboard manages episode lifecycle.

Keyboard Controls:
  [SPACE]   →  Start recording an episode
  [S]       →  Save the current episode
  [D]       →  Discard the current episode
  [R]       →  Redo  (discard + immediately restart recording)
  [Q]       →  Quit and push dataset to hub
"""

import argparse
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import pygame
import torch
from termcolor import colored

# Keyboard input (cross-platform, non-blocking)
import readchar

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.motors.motors_bus import MotorNormMode
from lerobot.utils.robot_utils import precise_sleep


# ── DS4 axis / button indices (unchanged) ────────────────────────────────────
AXIS_LS_X    = 0
AXIS_LS_Y    = 1
AXIS_RS_X    = 2
AXIS_RS_Y    = 3
BTN_CROSS    = 0
BTN_TRIANGLE = 3
BTN_L1       = 9
BTN_R1       = 10
BTN_OPTIONS  = 6
BTN_DPAD_UP  = 11
BTN_DPAD_DOWN= 12

DEADZONE     = 0.12
MAX_SPEED    = 80.0
DPAD_SPEED   = 40.0
FPS          = 30
IMAGE_SIZE   = (224, 224)
# ─────────────────────────────────────────────────────────────────────────────


# ── Camera discovery ──────────────────────────────────────────────────────────
def find_cameras(max_index: int = 8) -> list[dict]:
    """
    Probe VideoCapture indices and return working ones.
    On macOS, Continuity Camera (iPhone) usually appears at index 1 or 2.
    """
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                # Try to read a backend-provided name (macOS AVFoundation gives this)
                backend = cap.getBackendName()
                found.append({"index": idx, "backend": backend})
            cap.release()
    return found


def pick_camera() -> int:
    """
    Scan for cameras and let the user pick one interactively.
    Returns the chosen VideoCapture index.
    """
    print(colored("\nScanning for available cameras...", "cyan"))
    cams = find_cameras()

    if not cams:
        print(colored("No cameras found! Check permissions.", "red"))
        sys.exit(1)

    print(colored(f"Found {len(cams)} camera(s):\n", "green"))
    for cam in cams:
        label = ""
        if cam["index"] == 0:
            label = "  ← likely built-in FaceTime"
        elif cam["index"] >= 1:
            label = "  ← possibly iPhone (Continuity Camera)"
        print(f"  [{cam['index']}]  backend={cam['backend']}{label}")

    print()
    choice = input(colored("Enter camera index to use: ", "cyan")).strip()
    try:
        return int(choice)
    except ValueError:
        print("Invalid input, defaulting to 0.")
        return 0
# ─────────────────────────────────────────────────────────────────────────────


# ── Non-blocking keyboard listener ───────────────────────────────────────────
class KeyboardListener:
    """
    Runs in a background thread. Sets boolean flags when keys are pressed.
    Main loop reads and clears these flags each tick.
    Requires `readchar` (pip install readchar).
    """
    def __init__(self):
        self._flags = {
            "start":   False,   # SPACE
            "save":    False,   # S
            "discard": False,   # D
            "redo":    False,   # R
            "quit":    False,   # Q
        }
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        key_map = {
            " ":  "start",
            "s":  "save",
            "S":  "save",
            "d":  "discard",
            "D":  "discard",
            "r":  "redo",
            "R":  "redo",
            "q":  "quit",
            "Q":  "quit",
        }
        while True:
            try:
                key = readchar.readchar()
                action = key_map.get(key)
                if action:
                    with self._lock:
                        self._flags[action] = True
            except Exception:
                break

    def consume(self) -> dict:
        """Return current flags and reset them all to False."""
        with self._lock:
            snapshot = dict(self._flags)
            for k in self._flags:
                self._flags[k] = False
        return snapshot
# ─────────────────────────────────────────────────────────────────────────────


# ── DS4 / motor helpers (from your original script, unchanged) ────────────────
def find_usb_ports():
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports()]
    except ImportError:
        import platform
        if platform.system() == "Windows":
            return [f"COM{i}" for i in range(1, 20)]
        return (
            [str(p) for p in Path("/dev").glob("tty.usbmodem*")]
            + [str(p) for p in Path("/dev").glob("ttyUSB*")]
        )


def get_motor_limits(follower, motor_name):
    motor = follower.bus.motors[motor_name]
    if not follower.calibration or motor_name not in follower.calibration:
        if motor.norm_mode == MotorNormMode.RANGE_M100_100: return -100.0, 100.0
        elif motor.norm_mode == MotorNormMode.RANGE_0_100:  return   0.0, 100.0
        else:                                                return -180.0, 180.0
    calib = follower.calibration[motor_name]
    if motor.norm_mode == MotorNormMode.RANGE_M100_100: return -100.0, 100.0
    elif motor.norm_mode == MotorNormMode.RANGE_0_100:  return   0.0, 100.0
    elif motor.norm_mode == MotorNormMode.DEGREES:
        mid     = (calib.range_min + calib.range_max) / 2
        max_res = follower.bus.model_resolution_table[motor.model] - 1
        return (calib.range_min - mid)*360/max_res, (calib.range_max - mid)*360/max_res
    return float("-inf"), float("inf")


def apply_deadzone(value: float, deadzone: float = DEADZONE) -> float:
    if abs(value) < deadzone: return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def get_ds4_inputs(joy: pygame.joystick.Joystick) -> dict:
    pygame.event.pump()
    return {
        "ls_x":      apply_deadzone(joy.get_axis(AXIS_LS_X)),
        "ls_y":      apply_deadzone(joy.get_axis(AXIS_LS_Y)),
        "rs_x":      apply_deadzone(joy.get_axis(AXIS_RS_X)),
        "rs_y":      apply_deadzone(joy.get_axis(AXIS_RS_Y)),
        "dpad_up":   bool(joy.get_button(BTN_DPAD_UP)),
        "dpad_down": bool(joy.get_button(BTN_DPAD_DOWN)),
        "cross":     bool(joy.get_button(BTN_CROSS)),
        "triangle":  bool(joy.get_button(BTN_TRIANGLE)),
        "l1":        bool(joy.get_button(BTN_L1)),
        "r1":        bool(joy.get_button(BTN_R1)),
        "options":   bool(joy.get_button(BTN_OPTIONS)),
    }
# ─────────────────────────────────────────────────────────────────────────────


# ── Episode states ────────────────────────────────────────────────────────────
class State:
    IDLE      = "IDLE"        # waiting for SPACE to begin
    RECORDING = "RECORDING"   # actively buffering frames
    REVIEW    = "REVIEW"      # episode done, awaiting save/discard decision
# ─────────────────────────────────────────────────────────────────────────────


# ── Terminal UI ───────────────────────────────────────────────────────────────
def print_ui(motors, targets, obs, limits, torque_enabled, inputs,
             state, episode_idx, frame_idx, saved_count, total_episodes):

    sys.stdout.write("\033[H")

    state_colors = {
        State.IDLE:      ("white",  "IDLE  — press [SPACE] to start recording"),
        State.RECORDING: ("green",  f"RECORDING — frame {frame_idx:04d}   press [SPACE] to finish"),
        State.REVIEW:    ("yellow", "REVIEW — [S] save   [D] discard   [R] redo"),
    }
    color, state_msg = state_colors[state]

    print(colored("=========================================================================", "cyan"))
    print(colored("  SO-ARM101  ·  DS4 Control  ·  LeRobot Data Collection", "cyan", attrs=["bold"]))
    print(colored("=========================================================================", "cyan"))

    torque_str = (
        colored(" [Torque ON]  ", "green",  attrs=["bold", "reverse"]) if torque_enabled
        else colored(" [LOOSE MODE] ", "red", attrs=["bold", "reverse"])
    )
    print(f"Torque : {torque_str}     Episode: {episode_idx+1}/{total_episodes}    Saved: {saved_count}")
    print(colored(f"  ● {state_msg}", color, attrs=["bold"]))
    print()

    def bar(val, width=8):
        filled = int(abs(val) * width)
        arrow = "►" if val >= 0 else "◄"
        if val >= 0: return "[" + " "*width + arrow + "█"*filled + "]"
        else:        return "[" + "█"*filled + arrow + " "*width + "]"

    print(colored("Sticks:", "white", attrs=["underline"]))
    print(f"  LS X (pan)    {bar(inputs['ls_x'])}  {inputs['ls_x']:+.2f}")
    print(f"  LS Y (lift)   {bar(inputs['ls_y'])}  {inputs['ls_y']:+.2f}")
    print(f"  RS Y (elbow)  {bar(inputs['rs_y'])}  {inputs['rs_y']:+.2f}")
    print(f"  RS X (wrist)  {bar(inputs['rs_x'])}  {inputs['rs_x']:+.2f}")

    l1 = colored("[L1]", "green" if inputs["l1"] else "grey")
    r1 = colored("[R1]", "green" if inputs["r1"] else "grey")
    du = colored("[▲]",  "green" if inputs["dpad_up"]   else "grey")
    dd = colored("[▼]",  "green" if inputs["dpad_down"] else "grey")
    print(f"  WristRoll {l1} dec  {r1} inc   Gripper {du} open  {dd} close")
    print()

    print(colored("─────────────────────────────────────────────────────────────────────────", "grey"))
    print(colored(f"{'JOINT':<15} | {'TARGET':>8} | {'PRESENT':>8} | {'MIN':>7} | {'MAX':>7} | STATUS", "white", attrs=["bold"]))
    print(colored("─────────────────────────────────────────────────────────────────────────", "grey"))
    for motor in motors:
        t_val = targets[motor]
        o_val = obs.get(f"{motor}.pos", 0.0)
        l_min, l_max = limits[motor]
        status = colored("OK", "green")
        if abs(t_val - l_min) < 1e-2 or abs(t_val - l_max) < 1e-2:
            status = colored("CLAMPED", "yellow", attrs=["bold"])
        if not torque_enabled:
            status = colored("LOOSE", "red")
        print(f"{motor:<15} | {t_val:>8.2f} | {o_val:>8.2f} | {l_min:>7.2f} | {l_max:>7.2f} | {status}")

    print(colored("─────────────────────────────────────────────────────────────────────────", "grey"))
    print(colored("DS4: [Options]=Home  [Triangle]=Torque toggle", "white"))
    print(colored("Keys: [SPACE]=Start/Finish  [S]=Save  [D]=Discard  [R]=Redo  [Q]=Quit", "white"))
    print(colored("=========================================================================", "cyan"))
    sys.stdout.flush()
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",          default="/dev/tty.usbmodem5A7B2890981")
    parser.add_argument("--id",            default="my_awesome_follower_arm")
    parser.add_argument("--repo-id",       required=True,
                        help="e.g. 'yourname/pen-in-stand'")
    parser.add_argument("--num-episodes",  type=int, default=50)
    parser.add_argument("--camera-index",  type=int, default=-1,
                        help="Camera index (-1 = interactive picker)")
    args = parser.parse_args()

    # ── Camera setup ─────────────────────────────────────────────────────────
    cam_idx = args.camera_index if args.camera_index >= 0 else pick_camera()
    cam = cv2.VideoCapture(cam_idx)
    if not cam.isOpened():
        print(colored(f"Could not open camera {cam_idx}", "red"))
        sys.exit(1)
    # Warm up — first few frames from Continuity Camera can be blank
    for _ in range(5):
        cam.read()
    print(colored(f"Camera {cam_idx} ready.", "green"))

    # ── Pygame / DS4 ─────────────────────────────────────────────────────────
    pygame.init(); pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print(colored("No controller detected.", "red")); sys.exit(1)
    joy = pygame.joystick.Joystick(0); joy.init()
    print(colored(f"Controller: {joy.get_name()}", "green"))

    # ── Arm ──────────────────────────────────────────────────────────────────
    cfg      = SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True)
    follower = SO101Follower(cfg)
    follower.connect()

    motors = list(follower.bus.motors.keys())
    limits = {m: get_motor_limits(follower, m) for m in motors}

    # Wrist roll safety cap (same as your original)
    if "wrist_roll" in follower.calibration:
        calib    = follower.calibration["wrist_roll"]
        mobj     = follower.bus.motors["wrist_roll"]
        if mobj.norm_mode == MotorNormMode.DEGREES:
            mid     = (calib.range_min + calib.range_max) / 2.0
            max_res = follower.bus.model_resolution_table[mobj.model] - 1
            n_min   = (calib.range_min - mid)*360/max_res
            n_max   = (calib.range_max - mid)*360/max_res
            limits["wrist_roll"] = (max(n_min, -180.0), min(n_max, 180.0))

    # Sync goal to present position (no startup snap)
    follower.bus.disable_torque()
    init_obs = follower.get_observation()
    follower.send_action({f"{m}.pos": init_obs[f"{m}.pos"] for m in motors})
    follower.bus.enable_torque()

    obs     = follower.get_observation()
    targets = {m: obs[f"{m}.pos"] for m in motors}

    # ── LeRobot dataset ───────────────────────────────────────────────────────
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features={
            "observation.images.top": {
                "dtype": "image",
                "shape": (3, *IMAGE_SIZE),
                "names": ["channel", "height", "width"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(motors),),
                "names": motors,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(motors),),
                "names": motors,
            },
        },
        robot_type="so101",
    )

    # ── Keyboard listener ─────────────────────────────────────────────────────
    kb = KeyboardListener()

    # ── Main loop state ───────────────────────────────────────────────────────
    state          = State.IDLE
    episode_idx    = 0
    saved_count    = 0
    frame_idx      = 0
    episode_buffer = []          # list of dicts, one per frame
    torque_enabled = True
    prev_triangle  = False
    prev_options   = False

    dt = 1.0 / FPS
    print("\033[2J\033[H", end="")

    inputs = {k: 0.0 if k in ("ls_x","ls_y","rs_x","rs_y") else False
              for k in ("ls_x","ls_y","rs_x","rs_y","dpad_up","dpad_down",
                        "cross","triangle","l1","r1","options")}

    try:
        while episode_idx < args.num_episodes:
            t0 = time.perf_counter()

            # ── Read inputs ───────────────────────────────────────────────────
            inputs = get_ds4_inputs(joy)
            keys   = kb.consume()

            # ── Global DS4 controls (always active) ───────────────────────────

            # Options → home
            if inputs["options"] and not prev_options:
                for m in motors:
                    l_min, l_max = limits[m]
                    targets[m] = (l_min + l_max) / 2.0

            # Triangle → toggle torque
            if inputs["triangle"] and not prev_triangle:
                if torque_enabled:
                    follower.bus.disable_torque(); torque_enabled = False
                else:
                    follower.bus.enable_torque();  torque_enabled = True

            prev_options  = inputs["options"]
            prev_triangle = inputs["triangle"]

            # ── Observation ───────────────────────────────────────────────────
            obs = follower.get_observation()

            # ── Loose mode: track real position ──────────────────────────────
            if not torque_enabled:
                for m in motors:
                    targets[m] = obs.get(f"{m}.pos", targets[m])

            else:
                # ── Velocity control from sticks ──────────────────────────────
                targets["shoulder_pan"]  += inputs["ls_x"] * MAX_SPEED * dt
                targets["shoulder_lift"] += inputs["ls_y"] * MAX_SPEED * dt
                targets["elbow_flex"]    += inputs["rs_y"] * MAX_SPEED * dt
                targets["wrist_flex"]    += inputs["rs_x"] * MAX_SPEED * dt
                if inputs["l1"]: targets["wrist_roll"] -= MAX_SPEED * dt
                if inputs["r1"]: targets["wrist_roll"] += MAX_SPEED * dt
                if inputs["dpad_up"]:   targets["gripper"] -= DPAD_SPEED * dt
                if inputs["dpad_down"]: targets["gripper"] += DPAD_SPEED * dt

                for m in motors:
                    l_min, l_max = limits[m]
                    targets[m] = max(l_min, min(l_max, targets[m]))

                follower.send_action({f"{m}.pos": targets[m] for m in motors})

            # ── Camera frame (grabbed every tick regardless of state) ─────────
            ret, frame = cam.read()
            if ret:
                frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, IMAGE_SIZE)
                image_tensor  = torch.from_numpy(frame_resized).permute(2, 0, 1)

            # ── State machine ─────────────────────────────────────────────────
            if state == State.IDLE:
                if keys["quit"]:
                    break
                if keys["start"]:
                    episode_buffer = []
                    frame_idx      = 0
                    state          = State.RECORDING

            elif state == State.RECORDING:
                # Buffer this frame
                if ret:
                    episode_buffer.append({
                        "observation.images.top": image_tensor.clone(),
                        "observation.state": torch.tensor(
                            [obs[f"{m}.pos"] for m in motors], dtype=torch.float32),
                        "action": torch.tensor(
                            [targets[m] for m in motors], dtype=torch.float32),
                        "timestamp": frame_idx / FPS,
                    })
                frame_idx += 1

                # SPACE finishes recording → go to review
                if keys["start"] and frame_idx > 5:
                    state = State.REVIEW

                if keys["quit"]:
                    break

            elif state == State.REVIEW:
                if keys["save"] or keys["start"]:
                    # Commit episode to dataset
                    for frame_data in episode_buffer:
                        dataset.add_frame(frame_data)
                    dataset.save_episode()
                    saved_count += 1
                    episode_idx += 1
                    episode_buffer = []
                    state = State.IDLE
                    print(colored(f"\n✓ Episode {episode_idx} saved ({frame_idx} frames)\n", "green"))

                elif keys["discard"]:
                    episode_buffer = []
                    state = State.IDLE
                    print(colored("\n✗ Episode discarded.\n", "yellow"))

                elif keys["redo"]:
                    # Discard and immediately start a fresh recording
                    episode_buffer = []
                    frame_idx      = 0
                    state          = State.RECORDING
                    print(colored("\n↺ Redo — recording started.\n", "cyan"))

                elif keys["quit"]:
                    break

            # ── Render UI ─────────────────────────────────────────────────────
            print_ui(motors, targets, obs, limits, torque_enabled, inputs,
                     state, episode_idx, frame_idx, saved_count, args.num_episodes)

            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.release()
        try: follower.disconnect()
        except Exception: pass
        pygame.quit()

    # ── Push dataset ──────────────────────────────────────────────────────────
    if saved_count > 0:
        print(colored(f"\nPushing {saved_count} episodes to hub...", "cyan"))
        dataset.push_to_hub()
        print(colored("Done! Dataset uploaded.", "green"))
    else:
        print(colored("No episodes saved, skipping hub upload.", "yellow"))


if __name__ == "__main__":
    main()
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

"""DepthDodgeController driver: MuJoCo G1 by default, the real G1 with --real.

Onboard (--real) everything runs on the robot's Jetson: joint commands go straight to DDS
rt/lowcmd through the Unitree SDK (no bridge) and perception comes from a RealSense writer
subprocess (``src/lerobot/cameras/depth_shm.py``) through shared memory (no ZMQ, no sockets).

Fast path (default with --real): the writer runs on the GPU and publishes *Theia tokens*
(197x192) plus the 64x96 policy depth per camera frame; this process only reads them,
samples at 25 Hz and runs perception+actor at 50 Hz on the CPU (~1 ms).

    conda activate lerobot && cd ~/lerobot_dodge
    PYTHONPATH=src python examples/unitree_g1/depth_dodge.py --real --ball_only                      # ball-only depth, GPU Theia
    PYTHONPATH=src python examples/unitree_g1/depth_dodge.py --real --ball_only --writer_args "--ball-max-range 2.8 --trace"
    PYTHONPATH=src python examples/unitree_g1/depth_dodge.py --real --cpu_theia                      # old path: raw depth, CPU Theia
    PYTHONPATH=src python examples/unitree_g1/depth_dodge.py --camera zmq://<robot_ip>:5555          # sim on the laptop
    PYTHONPATH=src python examples/unitree_g1/depth_dodge.py --onscreen                              # sim, black depth (ablation)

SPACE toggles sonic <-> dodge, q quits. Live camera view (with --ball_only): http://<robot-ip>:8080
"""

import argparse
import logging
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.controllers.depth_dodge import DEFAULT_REPO_ID, DEPTH_SHAPE, FILES
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=600.0)
ap.add_argument("--real", action="store_true", help="run onboard: direct DDS lowcmd/lowstate, shm perception")
ap.add_argument("--network_interface", default=None, help="DDS interface for --real (default: auto)")
ap.add_argument("--onscreen", action="store_true")
ap.add_argument("--end_effector", default="dummy", choices=["dummy", "dex1", "dex3"])
ap.add_argument("--band_seconds", type=float, default=1.0, help="sim: keep the elastic band on this long")
ap.add_argument(
    "--camera",
    default=None,
    help="'shm' (default with --real), 'zmq://host:port' for the sim on a laptop, or 'none' for all-black",
)
ap.add_argument(
    "--rs_python",
    default=os.path.expanduser("~/rs_pub_venv/bin/python"),
    help="interpreter with pyrealsense2 (+ onnxruntime-gpu for the fast path); spawns the shm writer",
)
ap.add_argument("--rs_res", default="480x270x60", help="RealSense depth WxHxFPS for the writer")
ap.add_argument(
    "--camera_hz", type=float, default=25.0, help="policy camera rate (train 25); writer frames are sampled"
)
ap.add_argument(
    "--mode", default="sonic", choices=["sonic", "dodge"], help="starting mode; SPACE toggles, q quits"
)
ap.add_argument(
    "--cpu_theia",
    action="store_true",
    help="run Theia in this process on the CPU (~40 ms) instead of in the writer on the GPU (~8 ms)",
)
ap.add_argument(
    "--ball_only",
    action="store_true",
    help="writer detects the orange ball in colour and publishes only its depth pixels (rest = far)",
)
ap.add_argument(
    "--view_port",
    type=int,
    default=8080,
    help="with --ball_only: live MJPEG of colour+detection / published depth / mask / Theia input; 0 = off",
)
ap.add_argument(
    "--writer_args",
    default="",
    help='extra args for depth_shm.py, e.g. "--ball-max-range 2.8 --min-circularity 0.3 --trace"',
)
args = ap.parse_args()
if args.camera is None:
    args.camera = "shm" if args.real else "none"
# Tokens from the writer (GPU Theia) only make sense with the shm writer.
use_percept = args.camera == "shm" and not args.cpu_theia


def keyboard_loop(controller, stop: threading.Event):
    """SPACE toggles sonic <-> dodge, q quits. Raw tty so it works over ssh."""
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop.is_set():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                if ch == " ":
                    controller.mode = "dodge" if controller.mode == "sonic" else "sonic"
                    print(f"\n>>> MODE: {controller.mode.upper()}\n", flush=True)
                elif ch in ("q", "\x03"):
                    stop.set()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# Policy camera contract: 64x96 pinhole, 45 deg vertical FOV, 0 deg pitch.
POLICY_VFOV = math.radians(45.0)
POLICY_HFOV = 2 * math.atan(DEPTH_SHAPE[1] / DEPTH_SHAPE[0] * math.tan(POLICY_VFOV / 2))  # ~63.7 deg


def make_cropper(info: dict | None, h: int, w: int):
    """Return f(depth_mm uint16 HxW) -> (64,96) float32 metres, matching the policy's FOV."""
    if info:
        fx, fy, cx, cy = info["fx"], info["fy"], info["cx"], info["cy"]
    else:  # D435i nominal (58 deg vFOV)
        fy = fx = (h / 2) / math.tan(math.radians(58.0) / 2)
        cx, cy = w / 2, h / 2
    half_h = fy * math.tan(POLICY_VFOV / 2)
    half_w = fx * math.tan(POLICY_HFOV / 2)
    y0, y1 = int(round(cy - half_h)), int(round(cy + half_h))
    x0, x1 = int(round(cx - half_w)), int(round(cx + half_w))
    y0, x0 = max(y0, 0), max(x0, 0)
    y1, x1 = min(y1, h), min(x1, w)
    print(f"depth crop rows {y0}:{y1} cols {x0}:{x1} of {h}x{w} -> {DEPTH_SHAPE}", flush=True)

    def crop(depth_mm: np.ndarray) -> np.ndarray:
        roi = depth_mm[y0:y1, x0:x1]
        # Nearest keeps holes as holes (0 = no return), like the sim's rendered depth.
        small = cv2.resize(roi, (DEPTH_SHAPE[1], DEPTH_SHAPE[0]), interpolation=cv2.INTER_NEAREST)
        return small.astype(np.float32) * 1e-3

    return crop


stats = {"cam_hz": 0.0, "valid": 0.0, "near": 0.0, "frames": 0, "theia_ms": 0.0, "lat_ms": 0.0, "stalls": 0}


def open_camera():
    """Raw depth. Return (read() -> (depth_mm HxW uint16, capture_time|None), intrinsics|None, close())."""
    if args.camera == "shm":
        from lerobot.cameras.depth_shm import DepthShmReader

        reader = DepthShmReader(wait_s=15.0)

        def read():
            depth, meta = reader.read(timeout_s=2.0)
            return depth, meta["t_capture"]

        return read, reader.intrinsics, reader.close

    host, _, port = args.camera.removeprefix("zmq://").partition(":")
    from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig

    cam = ZMQCamera(
        ZMQCameraConfig(
            server_address=host,
            port=int(port or 5555),
            camera_name="head_camera",
            use_rgb=False,
            use_depth=True,
            warmup_s=3,
        )
    )
    cam.connect()

    def read():
        return cam.async_read_depth(timeout_ms=2000)[:, :, 0], cam.latest_server_timestamp

    return read, cam.depth_info, cam.disconnect


def open_percept():
    """Tokens from the writer. Return (read() -> (tokens, depth_m, t_capture, theia_ms), close())."""
    from lerobot.cameras.depth_shm import PerceptShmReader

    reader = PerceptShmReader(wait_s=30.0)  # the writer warms up CUDA first

    def read():
        tokens, depth_m, meta = reader.read(timeout_s=2.0)
        return tokens, depth_m, meta["t_capture"], meta["theia_ms"]

    return read, reader.close


class Cadence:
    """Pick frames at a fixed rate from a faster stream, by capture time (not arrival time), so the
    ring covers 16/25 s like in training even though the writer runs at 60 Hz."""

    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self.next: float | None = None

    def take(self, t_capture: float) -> bool:
        if self.next is None:
            self.next = t_capture + self.period
            return True
        if t_capture + 1e-4 < self.next:
            return False
        # Advance to the slot containing this frame; skip slots if we fell behind (stall).
        self.next += self.period * max(1, math.floor((t_capture - self.next) / self.period) + 1)
        return True


def camera_loop(controller, stop: threading.Event):
    cadence = Cadence(args.camera_hz)
    if use_percept:
        read_percept, close = open_percept()
        tokens, depth_m, t_capture, _ = read_percept()
        print("camera connected: tokens from the writer (Theia onboard GPU)", flush=True)
        crop = None
    else:
        read, intrinsics, close = open_camera()
        first, _ = read()
        crop = make_cropper(intrinsics, *first.shape)
        print(f"camera connected: {first.shape}, intrinsics {intrinsics}", flush=True)
    last = time.monotonic()
    try:
        while not stop.is_set():
            try:
                if use_percept:
                    tokens, depth_m, t_capture, theia_ms = read_percept()
                else:
                    depth_mm, t_capture = read()
            except TimeoutError as e:
                # Camera stalled (USB hiccup / writer restarting its pipeline). Don't act on stale
                # tokens: show the policy an empty scene until frames come back.
                stats["stalls"] += 1
                print(
                    f"\n!!! camera stall #{stats['stalls']}: {e}; feeding black frames until it recovers\n",
                    flush=True,
                )
                for _ in range(controller.depth_ring.maxlen):
                    controller.observe_tokens(controller.black_tokens)
                cadence.next = None
                continue
            if t_capture is None:
                t_capture = time.time()
            if not cadence.take(t_capture):
                continue
            now = time.monotonic()
            if use_percept:
                controller.observe_tokens(tokens)
                stats["theia_ms"] = theia_ms
            else:
                depth_m = crop(depth_mm)
                t_theia = time.perf_counter()
                controller.observe_depth(depth_m)
                stats["theia_ms"] = 0.8 * stats["theia_ms"] + 0.2 * (time.perf_counter() - t_theia) * 1e3
            # capture (sensor global time) -> tokens in the ring; same clock onboard
            stats["lat_ms"] = 0.8 * stats["lat_ms"] + 0.2 * (time.time() - t_capture) * 1e3
            stats["cam_hz"] = 0.9 * stats["cam_hz"] + 0.1 / max(now - last, 1e-6)
            last = now
            valid = depth_m > 0
            stats["valid"] = valid.mean()
            stats["near"] = float(depth_m[valid].min()) if valid.any() else float("nan")
            stats["frames"] += 1
    finally:
        close()


writer_proc = None
if args.camera == "shm":
    w, h, fps = args.rs_res.split("x")
    script = str(Path(__file__).resolve().parents[2] / "src/lerobot/cameras/depth_shm.py")
    subprocess.run(["pkill", "-f", "depth_shm.py --width"], check=False)
    writer_cmd = [args.rs_python, script, "--width", w, "--height", h, "--fps", fps]
    if args.ball_only:
        writer_cmd += ["--ball-only"]
        if args.view_port:
            writer_cmd += ["--view-port", str(args.view_port)]
    if use_percept:
        from huggingface_hub import hf_hub_download

        writer_cmd += ["--theia", hf_hub_download(DEFAULT_REPO_ID, FILES["theia"])]
    writer_cmd += shlex.split(args.writer_args)
    writer_proc = subprocess.Popen(writer_cmd, stdin=subprocess.DEVNULL)
    print(f"spawned depth writer pid {writer_proc.pid}: {' '.join(writer_cmd)}", flush=True)

cfg = UnitreeG1Config(
    is_simulation=not args.real,
    onboard=args.real,
    network_interface=args.network_interface,
    controller="DepthDodgeController",
    sim_onscreen=args.onscreen if not args.real else None,
    sim_publish_images=False,
    end_effector=args.end_effector,
)
robot = UnitreeG1(cfg)
robot.controller.mode = args.mode
robot.connect()

stop = threading.Event()
if sys.stdin.isatty():
    threading.Thread(target=keyboard_loop, args=(robot.controller, stop), daemon=True).start()
print(f"\n>>> MODE: {robot.controller.mode.upper()}   (SPACE toggles sonic/dodge, q quits)\n", flush=True)
has_camera = args.camera != "none"
if has_camera:
    threading.Thread(target=camera_loop, args=(robot.controller, stop), daemon=True).start()

default = robot.controller.default_angles
names = [m.name[1:] for m in G1_29_JointIndex]
band = getattr(getattr(robot.sim_env, "sim_env", None), "elastic_band", None)
t0 = time.time()
released = band is None
try:
    while time.time() - t0 < args.seconds and not stop.is_set():
        time.sleep(1.0)
        if not released and time.time() - t0 >= args.band_seconds:
            band.enable = False
            released = True
            print("elastic band released", flush=True)
        if writer_proc is not None and writer_proc.poll() is not None:
            print(f"depth writer exited with {writer_proc.returncode}", flush=True)
            writer_proc = None
        obs = robot.get_observation()
        q = np.array([obs[f"{m.name}.q"] for m in G1_29_JointIndex])
        dev = np.abs(q - default)
        top = ", ".join(f"{names[i]} {dev[i]:.2f}" for i in np.argsort(dev)[::-1][:3])
        c = robot.controller
        tm = c.timing
        line = (
            f"[{c.mode:5s}] t={time.time() - t0:5.1f}s  "
            f"tick {1e3 / max(tm['period_ms'], 1e-3):4.1f}Hz (train 50) compute {tm['tick_ms']:4.1f}ms (budget 20) | "
        )
        if has_camera:
            line += (
                f"cam {stats['cam_hz']:4.1f}Hz (train 25) theia {stats['theia_ms']:4.1f}ms "
                f"ring {c.ring_span_s * 1e3:4.0f}ms (train 640) lat {stats['lat_ms']:3.0f}ms (train 0-40) "
                f"valid {stats['valid'] * 100:3.0f}% near {stats['near']:.2f}m | "
            )
        else:
            line += "cam none (black) | "
        line += (
            f"|q-def| {dev.max():.2f}rad |res| {np.abs(c.last_action).max():.1f} "
            f"quat=({obs['imu.quat.w']:.2f},{obs['imu.quat.x']:.2f},{obs['imu.quat.y']:.2f},{obs['imu.quat.z']:.2f}) top: {top}"
        )
        print(line, flush=True)
finally:
    stop.set()
    robot.disconnect()
    if writer_proc is not None:
        writer_proc.terminate()

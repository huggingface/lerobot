#!/usr/bin/env python3
"""
lerobot_gamepad_control_server.py  –  runs on server Raspberry Pi
================================================
Receives action dicts from the Windows gamepad client (or policy client)
over TCP and drives the SO-101 follower arm.

Usage:
    # Both arm and motors (default)
    lerobot-gamepad-control-server --host=192.168.2.72 --tcp-port=2222 --arm-port=/dev/ttyACM0 --robot-id=so101_follower

    # Arm only
    lerobot-gamepad-control-server --no-motors --host=192.168.2.72 --tcp-port=2222 --arm-port=/dev/ttyACM0 --robot-id=so101_follower

    # Motors only
    lerobot-gamepad-control-server --no-arm --host=192.168.2.72 --tcp-port=2222 --arm-port=/dev/ttyACM0 --robot-id=so101_follower

    # Custom serial port or TCP port
    lerobot-gamepad-control-server --host=192.168.2.72 --tcp-port=5555 --arm-port=/dev/ttyACM1 --robot-id=so101_follower

    # Record a dataset while teleoperating (arm only, motors are not recorded)
    lerobot-gamepad-control-server --host=192.168.2.72 --tcp-port=2222 --arm-port=/dev/ttyACM0 --robot-id=so101_follower \
        --repo-id=${HF_USER}/sock-grab \
        --task="Grab the sock" \
        --num-episodes=10 \
        --episode-time=60 \
        --reset-time=30 \
        --cameras="front:/dev/video0,base:/dev/video2" \
        --no-motors

    Recording episode flow (keyboard on Pi terminal):
        Enter        save episode and move to reset
        r + Enter    discard episode and rerecord
        q + Enter    stop recording session early

Wire protocol (shared with client):
    Each message = [4-byte little-endian uint32 length][UTF-8 JSON payload]
"""

import argparse
import collections
from contextlib import nullcontext as _nullcontext
import json
import select
import socket
import struct
import sys
import time
import threading
from pathlib import Path

import base64

import numpy as np

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None
    _CV2_AVAILABLE = False

# Dataset recording (lerobot)
try:
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.cameras.configs import Cv2Backends, Cv2Rotation
    RECORDING_AVAILABLE = True
except Exception as _rec_err:
    print(f"⚠️  lerobot dataset/camera import failed: {_rec_err}  — recording disabled")
    CODEBASE_VERSION = "v3.0"
    LeRobotDataset = None
    Cv2Backends = None
    Cv2Rotation = None
    RECORDING_AVAILABLE = False

# Arm control (lerobot SO-101)
try:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    ARM_AVAILABLE = True
except Exception as _arm_import_err:
    print(f"⚠️  lerobot import failed: {_arm_import_err}  — arm control disabled")
    SO101Follower = None
    SO101FollowerConfig = None
    ARM_AVAILABLE = False

# Motor control (gpiozero + lgpio for Raspberry Pi 5)
try:
    from gpiozero import Motor, Device
    from gpiozero.pins.lgpio import LGPIOFactory
    Device.pin_factory = LGPIOFactory()
    GPIO_MOTORS_AVAILABLE = True
except Exception as _gpio_err:
    print(f"⚠️  GPIO motor init failed: {_gpio_err}  — motor control disabled")
    GPIO_MOTORS_AVAILABLE = False

# Serial motor control (ESP32/Arduino USB bridge)
try:
    import serial
    SERIAL_MOTORS_AVAILABLE = True
except Exception as _serial_err:
    print(f"⚠️  serial import failed: {_serial_err}  — ESP32 motor control disabled")
    serial = None
    SERIAL_MOTORS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# Wire protocol helpers
# ──────────────────────────────────────────────────────────────

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed by remote")
        buf += chunk
    return buf


def recv_msg(sock: socket.socket) -> dict:
    header = _recv_exact(sock, 4)
    length = struct.unpack("<I", header)[0]
    raw = _recv_exact(sock, length)
    return json.loads(raw.decode("utf-8"))


def send_msg(sock: socket.socket, obj: dict) -> None:
    raw = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("<I", len(raw)) + raw)


# ──────────────────────────────────────────────────────────────
# Observability
# ──────────────────────────────────────────────────────────────

class ServerStats:
    """Per-session receive and execution metrics."""

    DISPLAY_INTERVAL = 2.0   # seconds between log lines
    HZ_WINDOW        = 60    # samples for rolling Hz estimate

    def __init__(self):
        self._action_times: collections.deque = collections.deque(maxlen=self.HZ_WINDOW)
        self._exec_times:   collections.deque = collections.deque(maxlen=self.HZ_WINDOW)
        self._last_action_t: float | None = None
        self._last_display_t = time.perf_counter()
        self._session_start  = time.perf_counter()
        self._total_actions  = 0
        self._total_pings    = 0

    def record_action(self, recv_t: float, exec_s: float) -> None:
        # recv_t = timestamp taken BEFORE robot.send_action() — marks when the
        #          message arrived, independent of how long execution took.
        # exec_s = how long robot.send_action() itself took (servo comms).
        if self._last_action_t is not None:
            self._action_times.append(recv_t - self._last_action_t)
        self._last_action_t = recv_t
        self._exec_times.append(exec_s)
        self._total_actions += 1

    def record_ping(self) -> None:
        self._total_pings += 1

    def maybe_print(self) -> None:
        now = time.perf_counter()
        if now - self._last_display_t < self.DISPLAY_INTERVAL:
            return
        self._last_display_t = now

        if len(self._action_times) >= 2:
            avg_inter = np.mean(self._action_times)
            recv_hz   = 1.0 / avg_inter if avg_inter > 0 else 0.0
            jitter_ms = np.std(self._action_times) * 1000.0
        else:
            recv_hz   = 0.0
            jitter_ms = 0.0

        exec_ms = np.mean(self._exec_times) * 1000.0 if self._exec_times else 0.0
        uptime   = now - self._session_start

        print(
            f"[server] recv={recv_hz:.1f}Hz  "
            f"exec={exec_ms:.2f}ms  "
            f"jitter={jitter_ms:.2f}ms  "
            f"actions={self._total_actions}  "
            f"pings={self._total_pings}  "
            f"up={uptime:.0f}s"
        )

    def print_summary(self) -> None:
        uptime   = time.perf_counter() - self._session_start
        exec_avg = np.mean(self._exec_times) * 1000.0 if self._exec_times else float("nan")
        print(f"\n[server] ── Session summary ──────────────────────────")
        print(f"[server]   Uptime:        {uptime:.1f}s")
        print(f"[server]   Total actions: {self._total_actions}")
        print(f"[server]   Total pings:   {self._total_pings}")
        print(f"[server]   Avg exec time: {exec_avg:.2f} ms")
        print(f"[server] ──────────────────────────────────────────────")


# ──────────────────────────────────────────────────────────────
# Motor controller
# ──────────────────────────────────────────────────────────────

class MotorController:
    """
    Wraps two gpiozero Motor objects.
    Receives speed values in [-1.0, 1.0]:
      positive → forward, negative → backward, 0 → stop
    """

    def __init__(
        self,
        # Motor 1 pins
        m1_forward: int = 16,
        m1_backward: int = 1,
        m1_enable: int = 12,
        # Motor 2 pins
        m2_forward: int = 5,
        m2_backward: int = 6,
        m2_enable: int = 13,
    ):
        if not GPIO_MOTORS_AVAILABLE:
            raise RuntimeError("gpiozero/lgpio not available — cannot create MotorController")
        self.motor1 = Motor(forward=m1_forward, backward=m1_backward, enable=m1_enable, pwm=True)
        self.motor2 = Motor(forward=m2_forward, backward=m2_backward, enable=m2_enable, pwm=True)
        print(f"✓ Motors initialised  M1=(fwd={m1_forward},bwd={m1_backward},en={m1_enable})  "
              f"M2=(fwd={m2_forward},bwd={m2_backward},en={m2_enable})")

    def set(self, m1: float, m2: float) -> None:
        """
        Set motor speeds. Values in [-1.0, 1.0].
        Positive = forward, negative = backward, 0 = stop (coast).
        """
        self._drive(self.motor1, m1)
        self._drive(self.motor2, m2)

    @staticmethod
    def _drive(motor, value: float) -> None:
        value = max(-1.0, min(1.0, value)) # clamp
        if abs(value) < 0.01:
            motor.stop()
        elif value > 0:
            motor.forward(value)
        else:
            motor.backward(-value)

    def stop(self) -> None:
        self.motor1.stop()
        self.motor2.stop()


class SerialMotorController:
    """
    Sends motor speeds to an ESP32/Arduino over USB serial.
    Protocol: one line per command, "M <m1> <m2>\\n", values in [-1.0, 1.0].
    """

    def __init__(self, port: str, baud: int = 115200):
        if not SERIAL_MOTORS_AVAILABLE:
            raise RuntimeError("pyserial not available — cannot create SerialMotorController")
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.1, write_timeout=0.1)
        time.sleep(2.0)  # allow ESP32 reset after opening USB serial
        self.set(0.0, 0.0)
        print(f"✓ ESP32 motor bridge initialised  port={port} baud={baud}")

    def set(self, m1: float, m2: float) -> None:
        m1 = max(-1.0, min(1.0, float(m1)))
        m2 = max(-1.0, min(1.0, float(m2)))
        self.ser.write(f"M {m1:.3f} {m2:.3f}\n".encode("ascii"))

    def stop(self) -> None:
        try:
            self.set(0.0, 0.0)
        except Exception:
            pass

    def close(self) -> None:
        self.stop()
        self.ser.close()


# ──────────────────────────────────────────────────────────────
# Robot Arm helpers
# ──────────────────────────────────────────────────────────────

JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

PRESET_SPEED_DEG_PER_SEC = 50.0   # max joint speed during preset interpolation
PRESET_STEP_HZ           = 50     # interpolation tick rate (steps per second)


def send_action_preset(robot_arm, action_dict: dict, lock=None, on_step=None) -> float:
    """
    Execute a preset action with smooth interpolation so no joint moves faster than PRESET_SPEED_DEG_PER_SEC.

    lock:    optional threading.Lock guarding the servo bus.  When provided the lock
             is held only during each send_action call (not during sleep), so the main
             loop can serve full_obs_request between steps.
    on_step: optional callable(step_cmd: dict, obs: dict) invoked after each
             send_action while the lock is still held.  Used by the dataset recorder
             to capture each interpolated step so presets don't create gaps in
             the training trajectory.

    Returns total wall time taken (for stats).
    """
    t0 = time.perf_counter()

    phase_target = {k: v for k, v in action_dict.items()}
    if not phase_target:
        return 0.0

    # Read current joint positions from the robot_arm.
    try:
        with lock if lock else _nullcontext():
            obs = robot_arm.get_observation()
        phase_start = {k: float(obs[k]) for k in phase_target if k in obs}
    except Exception:
        phase_start = {}

    # Fall back to jumping straight to target if observation unavailable.
    if not phase_start:
        with lock if lock else _nullcontext():
            robot_arm.send_action(phase_target)
        return time.perf_counter() - t0

    # Number of steps driven by the largest joint displacement.
    step_dt   = 1.0 / PRESET_STEP_HZ
    max_delta = max(abs(phase_target[k] - phase_start[k]) for k in phase_target)
    min_steps = max(1, int(np.ceil(max_delta / (PRESET_SPEED_DEG_PER_SEC * step_dt))))

    for i in range(1, min_steps + 1):
        alpha = i / min_steps
        step_cmd = {
            k: phase_start[k] + alpha * (phase_target[k] - phase_start[k])
            for k in phase_target
        }
        with lock if lock else _nullcontext():
            robot_arm.send_action(step_cmd)
            # Read observation and call on_step while the lock is still held —
            # avoids a second lock acquisition and gives on_step a consistent
            # (action, state) snapshot from the same serial transaction.
            if on_step is not None:
                try:
                    raw = robot_arm.get_observation()
                    obs_scalars = {
                        k: float(v) if hasattr(v, "item") else float(v)
                        for k, v in raw.items()
                        if not k.startswith("observation.image")
                    }
                except Exception:
                    obs_scalars = {}
                on_step(step_cmd, obs_scalars)
        time.sleep(step_dt)   # sleep ensures smooth and slow motion, sleep OUTSIDE lock so obs requests can slip through

    return time.perf_counter() - t0


def connect_robot_arm(arm_port: str, robot_id: str):
    """Connect to SO-101 arm. Returns robot instance or None on failure."""
    if not ARM_AVAILABLE:
        print("ℹ️  lerobot not available — arm disabled")
        return None
    try:
        print(f"  Connecting to SO-101 on {arm_port} (id={robot_id}) …")
        config = SO101FollowerConfig(port=arm_port, id=robot_id)
        robot_arm = SO101Follower(config)
        robot_arm.connect()
        print("✓ Arm connected.")
        return robot_arm
    except Exception as e:
        print(f"⚠️  Could not connect to arm: {e}  — arm control disabled")
        return None


def get_observation_dict(robot) -> dict:
    """Return all scalar observations as a plain Python dict (JSON-serialisable)."""
    obs = robot.get_observation()
    return {
        k: float(v) if hasattr(v, "item") else v
        for k, v in obs.items()
        if not k.startswith("observation.image")
    }


# ──────────────────────────────────────────────────────────────
# Dataset recording helpers
# ──────────────────────────────────────────────────────────────

def _parse_camera_spec(camera_spec: str) -> dict:
    """
    Parse lerobot-style camera spec into {name: {key: value}} dict.

    Input format (same as lerobot-record --robot.cameras):
        { front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30},
          base:  {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}

    Strategy: strip braces/whitespace, split on camera boundaries, then split
    each camera's key-value pairs. No regex, no JSON conversion needed.
    """
    spec = camera_spec.strip().strip("{}")
    cameras = {}

    # Split into per-camera blocks by finding "name: {....}" segments
    # We walk character-by-character to split on top-level commas (outside braces)
    blocks = []
    depth = 0
    current = ""
    for ch in spec:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            blocks.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        blocks.append(current.strip())

    for block in blocks:
        # Split "name: { key: val, key: val }" into name and inner dict
        colon_pos = block.index(":")
        name = block[:colon_pos].strip().strip("\"'")
        inner_raw = block[colon_pos+1:].strip()
        if not inner_raw.startswith("{"):
            cameras[name] = {"index_or_path": inner_raw.strip().strip("\"'")}
            continue

        inner = inner_raw.strip("{}")

        # Parse inner key:value pairs (same depth-aware split)
        cfg = {}
        pairs = []
        depth = 0
        current = ""
        for ch in inner:
            if ch == "{": depth += 1
            elif ch == "}": depth -= 1
            if ch == "," and depth == 0:
                pairs.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            pairs.append(current.strip())

        for pair in pairs:
            if ":" not in pair:
                continue
            k, v = pair.split(":", 1)
            cfg[k.strip().strip("\"'")] = v.strip().strip("\"'")

        cameras[name] = cfg

    return cameras


def _parse_cv2_rotation(value: str):
    raw = str(value).strip()
    if raw.upper().startswith("ROTATE_"):
        return Cv2Rotation[raw.upper()]
    return Cv2Rotation(int(raw))


def _coerce_cv2_backend(value: str) -> int:
    if Cv2Backends is None:
        return int(value)
    if str(value).lstrip("-").isdigit():
        return int(value)
    return int(Cv2Backends[str(value).upper()])


def _should_recreate_local_dataset(err: Exception) -> bool:
    msg = str(err)
    return (
        "must be tagged with a codebase version" in msg
        or "Repository Not Found" in msg
        or "404 Client Error" in msg
    )


def connect_cameras(camera_spec: str) -> dict:
    """
    Parse lerobot-style camera spec and connect each OpenCVCamera.
    Returns {name: OpenCVCamera}.

    Accepts the same format as lerobot-record --robot.cameras:
        "{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30},
           base:  {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}"
    """
    cameras = {}
    if not camera_spec or not RECORDING_AVAILABLE:
        return cameras

    try:
        parsed = _parse_camera_spec(camera_spec)
    except Exception as e:
        print(f"⚠️  Could not parse --cameras spec: {e}")
        return cameras

    for name, cfg_dict in parsed.items():
        try:
            raw_path = cfg_dict.get("index_or_path", "0")
            # Convert /dev/videoN → integer index (most reliable with lerobot)
            if str(raw_path).lstrip("-").isdigit():
                index = int(raw_path)
            elif str(raw_path).startswith("/dev/video"):
                index = int(raw_path.replace("/dev/video", ""))
            else:
                index = raw_path

            kwargs = {
                "index_or_path": index,
                "fourcc": "MJPG",
                "backend": Cv2Backends.V4L2,
            }
            if "width"  in cfg_dict: kwargs["width"]  = int(cfg_dict["width"])
            if "height" in cfg_dict: kwargs["height"] = int(cfg_dict["height"])
            if "fps"    in cfg_dict: kwargs["fps"]    = int(cfg_dict["fps"])
            if "fourcc" in cfg_dict: kwargs["fourcc"] = str(cfg_dict["fourcc"])
            if "backend" in cfg_dict: kwargs["backend"] = _coerce_cv2_backend(cfg_dict["backend"])
            if "warmup_s" in cfg_dict: kwargs["warmup_s"] = int(cfg_dict["warmup_s"])
            if "rotation" in cfg_dict:
                kwargs["rotation"] = _parse_cv2_rotation(cfg_dict["rotation"])
            elif name == "base":
                kwargs["rotation"] = Cv2Rotation.ROTATE_180

            cfg = OpenCVCameraConfig(**kwargs)
            cam = OpenCVCamera(cfg)
            cam.connect()
            cameras[name] = cam
            detail = f"{raw_path}"
            if cfg.rotation != Cv2Rotation.NO_ROTATION:
                detail += f", rotation={cfg.rotation.value}"
            detail += f", fourcc={cfg.fourcc}, backend={cfg.backend}"
            print(f"✓ Camera '{name}' connected ({detail})")
        except Exception as e:
            print(f"⚠️  Camera '{name}' failed: {e}")
    return cameras


def build_dataset(repo_id: str, fps: int, cameras: dict) -> "LeRobotDataset | None":
    """Create or resume a LeRobotDataset. repo_id must already be clean."""
    if not RECORDING_AVAILABLE:
        return None
    try:
        import shutil
        local_root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
        meta_file  = local_root / "meta" / "info.json"

        if local_root.exists():
            if meta_file.exists():
                print(f"  Resuming existing dataset at {local_root} …")
                try:
                    ds = LeRobotDataset(repo_id=repo_id, root=local_root, revision=CODEBASE_VERSION)
                    print(f"✓ Dataset '{repo_id}' resumed ({ds.num_episodes} existing episodes)")
                    return ds
                except Exception as resume_err:
                    if _should_recreate_local_dataset(resume_err):
                        print(f"  Existing local dataset is not resumable — recreating {local_root} …")
                        shutil.rmtree(local_root)
                    else:
                        raise
            else:
                print(f"  Removing incomplete dataset directory: {local_root}")
                shutil.rmtree(local_root)

        n = len(JOINT_KEYS)
        features = {
            # Packed state vector (6,) — actual reported joint positions
            "observation.state": {"dtype": "float32", "shape": (n,), "names": JOINT_KEYS},
            # Packed action vector (6,) — commanded joint positions (what lerobot-replay reads)
            "action":            {"dtype": "float32", "shape": (n,), "names": JOINT_KEYS},
        }
        for name in cameras:
            features[f"observation.images.{name}"] = {
                "dtype": "video", "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            }
        ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, root=local_root, features=features)
        print(f"✓ Dataset '{repo_id}' created  (local: {local_root})")
        return ds
    except Exception as e:
        print(f"⚠️  Dataset creation failed: {e}")
        return None


def kb_ready() -> bool:
    """Non-blocking check for keyboard input on stdin."""
    return select.select([sys.stdin], [], [], 0)[0] != []


class CameraBuffer:
    """Non-blocking camera snapshot helper.

    OpenCVCamera already owns a background capture thread. This wrapper peeks at
    that latest frame and refuses stale frames instead of silently resending the
    last good image forever.
    """

    def __init__(self, cameras: dict, max_age_ms: int = 1000, verbose: bool = False):
        self._cameras = cameras
        self._max_age_ms = max_age_ms
        self._verbose = verbose
        self._last_warn_t = {n: 0.0 for n in cameras}

    def start(self) -> None:
        return None

    def get_latest(self) -> dict:
        frames = {}
        for name, cam in self._cameras.items():
            try:
                frames[name] = cam.read_latest(max_age_ms=self._max_age_ms)
            except Exception as e:
                now = time.perf_counter()
                if self._verbose and now - self._last_warn_t[name] > 2.0:
                    print(f"[server] camera '{name}' has no fresh frame: {e}")
                    self._last_warn_t[name] = now
        return frames

    def stop(self) -> None:
        return None


# ──────────────────────────────────────────────────────────────
# Client handler
# ──────────────────────────────────────────────────────────────

def handle_client(
    conn: socket.socket,
    robot_arm,
    motors: "MotorController | None",
    verbose: bool,
    # ── recording (all None = teleoperation-only mode, original behaviour) ──
    dataset: "LeRobotDataset | None" = None,
    cameras: dict | None = None,
    cam_buffer: "CameraBuffer | None" = None,
    task: str = "",
    num_episodes: int = 0,
    episode_time_s: float = 60.0,
    reset_time_s: float = 30.0,
) -> None:
    stats        = ServerStats()
    cameras      = cameras or {}
    recording    = dataset is not None and num_episodes > 0

    # _arm_lock: guards the physical servo bus (serial port).
    #   Held only during each individual send_action / get_observation call (~5ms),
    #   released during sleep() between preset steps so obs requests can slip through.
    #   Prevents concurrent serial writes from the preset thread and the main loop.
    _arm_lock    = threading.Lock()

    # _preset_busy: signals that a preset MOTION SEQUENCE is in progress.
    #   Unlike _arm_lock (which is free during sleep), this stays set for the entire
    #   preset duration (seconds).  The main loop checks this — not _arm_lock — to
    #   decide whether to skip continuous arm commands, because _arm_lock being free
    #   between steps would otherwise let a conflicting position command sneak through.
    _preset_busy = threading.Event()

    # ── per-episode state ─────────────────────────────────────────────────
    current_mode  : str | None = None   # tracks "arm" | "motor" | None
    ep_frames     : list[dict] = []
    ep_start      : float      = 0.0
    episode_idx   : int        = 0
    in_episode    : bool       = False  # True while actively recording an episode

    def start_episode() -> None:
        nonlocal ep_frames, ep_start, in_episode
        ep_frames  = []
        ep_start   = time.perf_counter()
        in_episode = True
        print(f"\n{'─'*50}")
        print(f"  Episode {episode_idx + 1}/{num_episodes}  —  {task}")
        print(f"  Enter=save  r+Enter=rerecord  q+Enter=quit")
        print(f"{'─'*50}")

    def save_episode() -> bool:
        """Commit buffered frames to dataset. Returns True if saved."""
        nonlocal episode_idx, in_episode
        in_episode = False
        if not ep_frames:
            print("  No frames captured — discarding.")
            return False
        print(f"  Saving {len(ep_frames)} frames …", end=" ", flush=True)
        for frame in ep_frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        episode_idx += 1
        print(f"✓  ({episode_idx}/{num_episodes})")
        return True

    def discard_episode() -> None:
        nonlocal in_episode
        in_episode = False
        print(f"  Discarding episode ({len(ep_frames)} frames)")

    def do_reset_pause() -> None:
        """Pause between episodes; keep serving arm commands during the wait."""
        print(f"\n  Reset the environment — {reset_time_s:.0f}s  (Enter to skip)")
        end_t = time.perf_counter() + reset_time_s
        while time.perf_counter() < end_t:
            if kb_ready():
                sys.stdin.readline()
                break
            conn.settimeout(0.05)
            try:
                m = recv_msg(conn)
                # Forward arm/motor commands during reset but don't record them
                _dispatch_msg(m, record=False)
            except socket.timeout:
                pass
            except Exception:
                break
            finally:
                conn.settimeout(None)

    def check_keyboard() -> str:
        """Returns 'save', 'rerecord', 'quit', or '' if nothing pressed."""
        if not kb_ready():
            return ""
        ch = sys.stdin.readline().strip().lower()
        if ch == "":   return "save"
        if ch == "r":  return "rerecord"
        if ch == "q":  return "quit"
        return ""

    def _append_frame(action_src: dict, obs_src: dict) -> None:
        """Pack (action, obs, cameras) into a frame dict and append to ep_frames.

        action_src: {joint_key: float} — commanded positions for this step
        obs_src:    {joint_key: float} — reported positions read after the command;
                    falls back to action_src values if a key is missing
        """
        try:
            action_vals = np.array(
                [float(action_src.get(k, 0.0)) for k in JOINT_KEYS], dtype=np.float32
            )
            obs_vals = np.array(
                [float(obs_src.get(k, action_src.get(k, 0.0))) for k in JOINT_KEYS],
                dtype=np.float32,
            )
            frame: dict = {
                "task":              task,
                "action":            action_vals,
                "observation.state": obs_vals,
            }
            if cam_buffer is not None:
                for cam_name, img in cam_buffer.get_latest().items():
                    if img is not None:
                        frame[f"observation.images.{cam_name}"] = img
            ep_frames.append(frame)
        except Exception as _e:
            if verbose:
                print(f"[server] recording error: {_e}")

    def _dispatch_msg(msg: dict, record: bool) -> None:
        """
        Core message dispatcher — identical logic to the original handle_client,
        with recording hooks added into the action branch.
        `record=False` during reset/non-arm modes so those frames are skipped.
        """
        nonlocal current_mode, in_episode, episode_idx

        mtype = msg.get("type")

        # ── mode_request ─────────────────────────────────────────────────
        # Client tapped RB or LB — check if that mode is available and reply
        if mtype == "mode_request":
            requested = msg.get("mode")
            if requested == "arm":
                ok     = robot_arm is not None
                reason = "arm not connected" if not ok else ""
            elif requested == "motor":
                ok     = motors is not None
                reason = "motors not connected" if not ok else ""
            else:
                ok     = False
                reason = f"unknown mode '{requested}'"
            send_msg(conn, {"type": "mode_response", "mode": requested, "ok": ok, "reason": reason})
            if ok:
                current_mode = requested
            print(f"[server] mode_request '{requested}' → {'granted' if ok else f'denied ({reason})'}")

        # ── action (arm) ─────────────────────────────────────────────────
        elif mtype == "action":
            if robot_arm is not None:
                action_dict: dict = msg["action"]   # {joint_key: float, …}
                is_preset: bool   = msg.get("preset", False)
                recv_t = time.perf_counter()
                if is_preset:
                    if _preset_busy.is_set():
                        return   # ignore — preset already in progress, unsafe to stack
                    # Run in a background thread so the recv loop stays live and
                    # can respond to full_obs_request while the arm is moving.
                    def _preset_worker(ad=action_dict):
                        try:
                            def _record_step(step_cmd: dict, obs_scalars: dict) -> None:
                                # Called for every interpolated step of this preset so the
                                # dataset captures the full sweep trajectory.  Without this,
                                # consecutive training frames would show the arm teleporting
                                # across the preset motion, leaving inference without any
                                # data for those intermediate states.
                                if record and in_episode and current_mode == "arm":
                                    _append_frame(step_cmd, obs_scalars)

                            send_action_preset(
                                robot_arm, ad, lock=_arm_lock,
                                on_step=_record_step if recording else None,
                            )
                        finally:
                            _preset_busy.clear()
                            # Send final arm state so the client can re-sync
                            # current_position and avoid jumping on next joystick input.
                            try:
                                with _arm_lock:
                                    final_state = [float(robot_arm.get_observation().get(k, 0.0)) for k in JOINT_KEYS]
                                send_msg(conn, {"type": "preset_done", "state": final_state})
                            except Exception:
                                pass
                    _preset_busy.set()
                    threading.Thread(target=_preset_worker, daemon=True).start()
                    exec_s = 0.0
                else:
                    if _preset_busy.is_set():
                        return   # skip — don't send conflicting commands during preset
                    t0 = time.perf_counter()
                    with _arm_lock:
                        robot_arm.send_action(action_dict)
                    exec_s = time.perf_counter() - t0
                stats.record_action(recv_t, exec_s)
                stats.maybe_print()
                if verbose:
                    kind = "preset" if is_preset else "continuous"
                    vals = [f"{action_dict.get(k, 0):.1f}" for k in JOINT_KEYS]
                    print(f"[server] action({kind}) exec={exec_s*1000:.2f}ms  pos={vals}")

                # ── recording hook ────────────────────────────────────────
                # Only record continuous arm actions here.  Preset frames are
                # captured step-by-step inside _preset_worker via on_step so
                # the full sweep trajectory is in the dataset.
                if record and in_episode and current_mode == "arm" and not is_preset:
                    with _arm_lock:
                        raw_obs = robot_arm.get_observation()
                    _append_frame(action_dict, raw_obs)
            else:
                if verbose:
                    print("[server] action msg received but arm not connected — ignoring")

        # ── motor ────────────────────────────────────────────────────────
        elif mtype == "motor":
            if motors is not None:
                m1 = float(msg.get("motor1", 0.0))
                m2 = float(msg.get("motor2", 0.0))
                motors.set(m1, m2)
                if verbose:
                    print(f"[server] motor  m1={m1:+.2f}  m2={m2:+.2f}")
            else:
                if verbose:
                    print("[server] motor msg received but motors not connected — ignoring")

        # ── ping → pong ──────────────────────────────────────────────────
        elif mtype == "ping":
            stats.record_ping()
            send_msg(conn, {"type": "pong", "seq": msg.get("seq")})

        # ── full observation request (joint state + camera images) ───────
        elif mtype == "full_obs_request":
            resp: dict = {"type": "full_obs", "images": {}}
            if robot_arm is not None:
                with _arm_lock:
                    obs_dict = get_observation_dict(robot_arm)
                resp["state"] = [float(obs_dict.get(k, 0.0)) for k in JOINT_KEYS]
            if msg.get("images") and cam_buffer is not None and _CV2_AVAILABLE:
                for cam_name, img in cam_buffer.get_latest().items():
                    if img is not None:
                        try:
                            # OpenCVCamera returns RGB; imencode expects BGR
                            bgr = _cv2.cvtColor(img, _cv2.COLOR_RGB2BGR)
                            _, buf = _cv2.imencode(
                                ".jpg", bgr, [_cv2.IMWRITE_JPEG_QUALITY, 70]
                            )
                            resp["images"][cam_name] = base64.b64encode(buf).decode("ascii")
                        except Exception:
                            pass
            send_msg(conn, resp)

        # ── graceful disconnect ──────────────────────────────────────────
        elif mtype == "disconnect":
            print("[server] Client requested disconnect.")
            raise ConnectionError("client disconnect")

        else:
            print(f"[server] Unknown message type: {mtype!r}")

    # ── main loop ─────────────────────────────────────────────────────────
    if recording:
        print(f"\n[server] Recording mode: {num_episodes} episodes  task='{task}'")
        print(f"[server] Tap RB on the gamepad to enter ARM mode, then start teleoperating.")
        start_episode()

    try:
        while True:
            msg = recv_msg(conn)

            _dispatch_msg(msg, record=recording)

            if not recording:
                continue

            # ── episode stop checks (only while recording) ────────────────
            kb = check_keyboard()
            timed_out = in_episode and (time.perf_counter() - ep_start >= episode_time_s)

            if timed_out:
                print(f"\n  Episode time reached ({episode_time_s:.0f}s) — saving.")
                kb = "save"

            if kb == "save":
                save_episode()
                if episode_idx >= num_episodes:
                    print("\n[server] All episodes recorded.")
                    break
                do_reset_pause()
                start_episode()

            elif kb == "rerecord":
                discard_episode()
                start_episode()

            elif kb == "quit":
                discard_episode()
                print("[server] Recording stopped early by user.")
                break

    except ConnectionError as e:
        print(f"[server] Connection lost: {e}")
    finally:
        if motors is not None:
            motors.stop()
            if hasattr(motors, "close"):
                motors.close()
            print("[server] Motors stopped.")
        stats.print_summary()
        conn.close()
        print("[server] Client connection closed.")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SO-101 + motor TCP server (Raspberry Pi)\n"
                    "Run with arm only, motors only, or both — any combination is valid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # TCP
    parser.add_argument("--host",      default="0.0.0.0",      help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--tcp-port",  type=int, default=2222,  help="TCP port (default: 2222)")
    parser.add_argument("--verbose",   action="store_true",     help="Print every action/motor command")
    # Arm
    parser.add_argument("--no-arm",    action="store_true",     help="Disable arm (motors only)")
    parser.add_argument("--arm-port",   default="/dev/ttyACM0",  help="Serial port for SO-101 (default: /dev/ttyACM0)")
    parser.add_argument("--robot-id",  default="so101_follower")
    # Motors
    parser.add_argument("--no-motors", action="store_true",     help="Disable motors (arm only)")
    parser.add_argument("--motor-backend", choices=["gpio", "serial"], default="gpio",
                        help="Motor backend: Raspberry Pi GPIO or ESP32/Arduino USB serial bridge (default: gpio)")
    parser.add_argument("--motor-serial-port", default="/dev/ttyUSB0",
                        help="Serial port for --motor-backend=serial, e.g. /dev/ttyUSB0 or /dev/ttyACM0")
    parser.add_argument("--m1-fwd",    type=int, default=16,    help="Motor 1 forward GPIO / IN1 (default: 16, physical pin 36)")
    parser.add_argument("--m1-bwd",    type=int, default=1,     help="Motor 1 backward GPIO / IN2 (default: 1, physical pin 28)")
    parser.add_argument("--m1-en",     type=int, default=12,    help="Motor 1 enable GPIO / ENA (default: 12, physical pin 32)")
    parser.add_argument("--m2-fwd",    type=int, default=5,     help="Motor 2 forward GPIO / IN3 (default: 5, physical pin 29)")
    parser.add_argument("--m2-bwd",    type=int, default=6,     help="Motor 2 backward GPIO / IN4 (default: 6, physical pin 31)")
    parser.add_argument("--m2-en",     type=int, default=13,    help="Motor 2 enable GPIO / ENB (default: 13, physical pin 33)")
    # Dataset recording (all optional — omit --repo-id to run without recording)
    parser.add_argument("--repo-id",       default=None,
                        help="HuggingFace repo id, e.g. ${HF_USER}/sock-grab")
    parser.add_argument("--task",          default="Teleoperate the robot arm",
                        help="Task description saved in the dataset")
    parser.add_argument("--num-episodes",  type=int,   default=10,
                        help="Number of episodes to record (default: 10)")
    parser.add_argument("--episode-time",  type=float, default=60.0,
                        help="Max seconds per episode (default: 60)")
    parser.add_argument("--reset-time",    type=float, default=30.0,
                        help="Seconds between episodes for env reset (default: 30)")
    parser.add_argument("--fps",           type=int,   default=30,
                        help="Dataset FPS (default: 30)")
    parser.add_argument("--cameras",       default="",
                        help="Camera spec: 'name:/dev/videoN,name2:/dev/videoM'")
    parser.add_argument("--camera-max-age-ms", type=int, default=1000,
                        help="Drop camera frames older than this many ms (default: 1000)")
    parser.add_argument("--no-push",       action="store_true",
                        help="Skip uploading dataset to HuggingFace Hub")
    args = parser.parse_args()

    # Expand and validate repo_id once, up front
    if args.repo_id:
        import os
        args.repo_id = os.path.expandvars(args.repo_id).strip()
        if "/" not in args.repo_id:
            parser.error(
                f"--repo-id must be 'username/dataset-name', got: '{args.repo_id}'\n"
                f"  Make sure HF_USER is set: export HF_USER=$(hf auth whoami | awk -F': *' 'NR==1 {{print $2}}')"
            )
        if not args.task:
            parser.error("--task is required when --repo-id is given, e.g. --task=\"Grab the sock\"")


    print("  SO-101 + Motor TCP Server")
    print("=" * 55)

    # ── Arm ───────────────────────────────────────────────────
    robot_arm = None
    if args.no_arm:
        print("ℹ️  Arm:    DISABLED (--no-arm)")
    else:
        robot_arm = connect_robot_arm(args.arm_port, args.robot_id)
        if robot_arm is None:
            print("ℹ️  Arm:    DISABLED (connection failed)")
        else:
            print(f"✓  Arm:    ENABLED  ({args.arm_port})")

    # ── Motors ────────────────────────────────────────────────
    motors = None
    if args.no_motors:
        print("ℹ️  Motors: DISABLED (--no-motors)")
    elif args.motor_backend == "gpio" and not GPIO_MOTORS_AVAILABLE:
        print("ℹ️  Motors: DISABLED (gpiozero/lgpio not available)")
    elif args.motor_backend == "serial" and not SERIAL_MOTORS_AVAILABLE:
        print("ℹ️  Motors: DISABLED (pyserial not available)")
    else:
        try:
            if args.motor_backend == "serial":
                motors = SerialMotorController(args.motor_serial_port)
            else:
                motors = MotorController(
                    m1_forward=args.m1_fwd, m1_backward=args.m1_bwd, m1_enable=args.m1_en,
                    m2_forward=args.m2_fwd, m2_backward=args.m2_bwd, m2_enable=args.m2_en,
                )
            print(f"✓  Motors: ENABLED")
        except Exception as e:
            print(f"⚠️  Motors: DISABLED (init failed: {e})")

    # ── Cameras ───────────
    cameras    = {}
    cam_buffer = None
    if args.cameras:
        cameras = connect_cameras(args.cameras)
        if cameras:
            cam_buffer = CameraBuffer(cameras, max_age_ms=args.camera_max_age_ms, verbose=args.verbose)
            cam_buffer.start()
            print(f"✓  Camera buffer started ({len(cameras)} cameras)")
    else:
        print("ℹ️  Cameras: DISABLED (no --cameras given)")

    # ── Dataset recording ─────────────────────────────────────
    dataset = None
    if args.repo_id:
        if not RECORDING_AVAILABLE:
            print("⚠️  Recording: DISABLED (lerobot dataset not available)")
        else:
            dataset = build_dataset(args.repo_id, args.fps, cameras)
            if dataset:
                print(f"✓  Recording: ENABLED  → {args.repo_id}  "
                      f"({args.num_episodes} episodes)")
    else:
        print("ℹ️  Recording: DISABLED (no --repo-id given)")

    # ── Sanity check ──────────────────────────────────────────
    if robot_arm is None and motors is None:
        print("\n⚠️  WARNING: Neither arm nor motors are connected.")
        print("   The server will run but all incoming commands will be ignored.")

    # ── Start TCP server ──────────────────────────────────────
    print("=" * 55)
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.tcp_port))
    server_sock.listen(1)
    print(f"✓ Listening on {args.host}:{args.tcp_port} …")
    print("  Waiting for client …\n")

    try:
        while True:
            conn, addr = server_sock.accept()
            print(f"[server] Client connected from {addr}")
            handle_client(
                conn, robot_arm, motors, args.verbose,
                dataset        = dataset,
                cameras        = cameras,
                cam_buffer     = cam_buffer,
                task           = args.task,
                num_episodes   = args.num_episodes if args.repo_id else 0,
                episode_time_s = args.episode_time,
                reset_time_s   = args.reset_time,
            )
            # After all episodes are done, stop accepting new clients
            if dataset is not None:
                break
            print("[server] Ready for next client …")
    except KeyboardInterrupt:
        print("\n[server] Shutting down.")
    finally:
        if motors is not None:
            motors.stop()
            if hasattr(motors, "close"):
                motors.close()
        server_sock.close()
        if robot_arm is not None:
            robot_arm.disconnect()
        # ── Finalize and push dataset ─────────────────────────
        if dataset is not None:
            print("\n[server] Finalizing dataset …")
            try:
                dataset.finalize()
            except Exception as e:
                print(f"⚠️  Finalize failed: {e}")
            if not args.no_push:
                print("[server] Pushing to HuggingFace Hub …")
                try:
                    dataset.push_to_hub()
                    print(f"✓ Uploaded → https://huggingface.co/datasets/{args.repo_id}")
                except Exception as e:
                    print(f"⚠️  Upload failed: {e}")
                    local_root = Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id
                    print(f"   Push manually:")
                    print(f"   huggingface-cli upload {args.repo_id} {local_root} --repo-type dataset")
        if cam_buffer is not None:
            cam_buffer.stop()
        for cam in cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass
        print("[server] Clean exit.")


if __name__ == "__main__":
    main()

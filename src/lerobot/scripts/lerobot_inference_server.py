#!/usr/bin/env python3
"""
lerobot_inference_server.py  –  runs on Raspberry Pi
=====================================================
Connects the robot arm and cameras, then waits for a client to connect.

For each step the client sends:
  1. full_obs_request  → server replies with joint state + camera images
  2. action            → server moves the arm

That's it. No dataset recording, no episode tracking, no HuggingFace uploads.
Those belong in the training/data-collection workflow (lerobot_gamepad_control_server.py).

Usage:
    python lerobot_inference_server.py \\
        --arm-port /dev/ttyACM0 \\
        --robot-id so101_follower \\
        --cameras "{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 15}, \\
                     base:  {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 15}}"

Wire protocol:
    Each message = [4-byte little-endian uint32 length][UTF-8 JSON payload]
"""

import argparse
import base64
import collections
import json
import socket
import struct
import time

import numpy as np

# ── lerobot SO-101 arm ──────────────────────────────────────────────────────
try:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    ARM_AVAILABLE = True
except Exception as _arm_err:
    print(f"⚠️  lerobot arm import failed: {_arm_err}  — arm control disabled")
    SO101Follower = SO101FollowerConfig = None
    ARM_AVAILABLE = False

# ── lerobot cameras ─────────────────────────────────────────────────────────
try:
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.cameras.configs import Cv2Backends, Cv2Rotation
    CAMERAS_AVAILABLE = True
except Exception as _cam_err:
    print(f"⚠️  lerobot camera import failed: {_cam_err}  — cameras disabled")
    OpenCVCamera = OpenCVCameraConfig = Cv2Backends = Cv2Rotation = None
    CAMERAS_AVAILABLE = False

# ── OpenCV for JPEG encoding ─────────────────────────────────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  opencv-python not found — camera images will not be sent to client")

# ── GPIO motors (optional, Raspberry Pi only) ────────────────────────────────
try:
    from gpiozero import Motor, Device
    from gpiozero.pins.lgpio import LGPIOFactory
    Device.pin_factory = LGPIOFactory()
    GPIO_MOTORS_AVAILABLE = True
except Exception:
    GPIO_MOTORS_AVAILABLE = False

# Serial motor control (ESP32/Arduino USB bridge)
try:
    import serial
    SERIAL_MOTORS_AVAILABLE = True
except Exception as _serial_err:
    print(f"serial import failed: {_serial_err} - ESP32 motor control disabled")
    serial = None
    SERIAL_MOTORS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# Wire protocol
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
    raw    = _recv_exact(sock, length)
    return json.loads(raw.decode("utf-8"))


def send_msg(sock: socket.socket, obj: dict) -> None:
    raw = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("<I", len(raw)) + raw)


# ──────────────────────────────────────────────────────────────
# Observability
# ──────────────────────────────────────────────────────────────

class ServerStats:
    DISPLAY_INTERVAL = 2.0
    HZ_WINDOW        = 60

    def __init__(self):
        self._action_times: collections.deque = collections.deque(maxlen=self.HZ_WINDOW)
        self._exec_times:   collections.deque = collections.deque(maxlen=self.HZ_WINDOW)
        self._last_action_t: float | None = None
        self._last_display_t = time.perf_counter()
        self._session_start  = time.perf_counter()
        self._total_actions  = 0
        self._total_pings    = 0

    def record_action(self, recv_t: float, exec_s: float):
        if self._last_action_t is not None:
            self._action_times.append(recv_t - self._last_action_t)
        self._last_action_t = recv_t
        self._exec_times.append(exec_s)
        self._total_actions += 1

    def record_ping(self):
        self._total_pings += 1

    def maybe_print(self):
        now = time.perf_counter()
        if now - self._last_display_t < self.DISPLAY_INTERVAL:
            return
        self._last_display_t = now
        hz      = 1.0 / np.mean(self._action_times) if len(self._action_times) >= 2 else 0.0
        jitter  = np.std(self._action_times) * 1000.0 if len(self._action_times) >= 2 else 0.0
        exec_ms = np.mean(self._exec_times) * 1000.0 if self._exec_times else 0.0
        uptime  = now - self._session_start
        print(
            f"[server] recv={hz:.1f}Hz  exec={exec_ms:.2f}ms  "
            f"jitter={jitter:.2f}ms  actions={self._total_actions}  "
            f"pings={self._total_pings}  up={uptime:.0f}s"
        )

    def print_summary(self):
        uptime   = time.perf_counter() - self._session_start
        exec_avg = np.mean(self._exec_times) * 1000.0 if self._exec_times else float("nan")
        print(f"\n[server] ── Session summary ─────────────────────────")
        print(f"[server]   Uptime:        {uptime:.1f}s")
        print(f"[server]   Total actions: {self._total_actions}")
        print(f"[server]   Total pings:   {self._total_pings}")
        print(f"[server]   Avg exec time: {exec_avg:.2f} ms")
        print(f"[server] ────────────────────────────────────────────")


# ──────────────────────────────────────────────────────────────
# Motor controller  (optional)
# ──────────────────────────────────────────────────────────────

class MotorController:
    def __init__(self, m1_forward=16, m1_backward=1, m1_enable=12,
                       m2_forward=5, m2_backward=6, m2_enable=13):
        if not GPIO_MOTORS_AVAILABLE:
            raise RuntimeError("gpiozero/lgpio not available")
        self.motor1 = Motor(forward=m1_forward, backward=m1_backward, enable=m1_enable, pwm=True)
        self.motor2 = Motor(forward=m2_forward, backward=m2_backward, enable=m2_enable, pwm=True)
        print("✓ Motors initialised")

    def set(self, m1: float, m2: float):
        self._drive(self.motor1, m1)
        self._drive(self.motor2, m2)

    @staticmethod
    def _drive(motor, value: float):
        value = max(-1.0, min(1.0, value))
        if abs(value) < 0.01: motor.stop()
        elif value > 0:        motor.forward(value)
        else:                  motor.backward(-value)

    def stop(self):
        self.motor1.stop()
        self.motor2.stop()


class SerialMotorController:
    """
    Sends motor speeds to an ESP32/Arduino over USB serial.
    Protocol: one line per command, "M <m1> <m2>\\n", values in [-1.0, 1.0].
    """

    def __init__(self, port: str, baud: int = 115200):
        if not SERIAL_MOTORS_AVAILABLE:
            raise RuntimeError("pyserial not available")
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.1, write_timeout=0.1)
        time.sleep(2.0)  # allow ESP32 reset after opening USB serial
        self.set(0.0, 0.0)
        print(f"ESP32 motor bridge initialised  port={port} baud={baud}")

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
# Arm helpers
# ──────────────────────────────────────────────────────────────

JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

PRESET_SPEED_DEG_PER_SEC = 75.0
PRESET_STEP_HZ           = 75.0


def send_action_preset(robot_arm, action_dict: dict) -> float:
    t0 = time.perf_counter()
    try:
        obs         = robot_arm.get_observation()
        phase_start = {k: float(obs[k]) for k in action_dict if k in obs}
    except Exception:
        phase_start = {}
    if not phase_start:
        robot_arm.send_action(action_dict)
        return time.perf_counter() - t0
    step_dt   = 1.0 / PRESET_STEP_HZ
    max_delta = max(abs(action_dict[k] - phase_start[k]) for k in action_dict)
    steps     = max(1, int(np.ceil(max_delta / (PRESET_SPEED_DEG_PER_SEC * step_dt))))
    for i in range(1, steps + 1):
        alpha = i / steps
        robot_arm.send_action({
            k: phase_start[k] + alpha * (action_dict[k] - phase_start[k])
            for k in action_dict
        })
        time.sleep(step_dt)
    return time.perf_counter() - t0


def connect_robot_arm(arm_port: str, robot_id: str):
    if not ARM_AVAILABLE:
        print("ℹ️  lerobot not available — arm disabled")
        return None
    try:
        config    = SO101FollowerConfig(port=arm_port, id=robot_id)
        robot_arm = SO101Follower(config)
        robot_arm.connect()
        print("✓ Arm connected.")
        return robot_arm
    except Exception as e:
        print(f"⚠️  Could not connect to arm: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# Camera helpers
# ──────────────────────────────────────────────────────────────

def _parse_camera_spec(camera_spec: str) -> dict:
    spec   = camera_spec.strip().strip("{}")
    blocks = []
    depth, current = 0, ""
    for ch in spec:
        if ch == "{": depth += 1
        elif ch == "}": depth -= 1
        if ch == "," and depth == 0:
            blocks.append(current.strip()); current = ""
        else:
            current += ch
    if current.strip():
        blocks.append(current.strip())
    cameras = {}
    for block in blocks:
        colon = block.index(":")
        name  = block[:colon].strip().strip("\"'")
        inner_raw = block[colon+1:].strip()
        if not inner_raw.startswith("{"):
            cameras[name] = {"index_or_path": inner_raw.strip().strip("\"'")}
            continue

        inner = inner_raw.strip("{}")
        cfg   = {}
        pairs, depth, current = [], 0, ""
        for ch in inner:
            if ch == "{": depth += 1
            elif ch == "}": depth -= 1
            if ch == "," and depth == 0:
                pairs.append(current.strip()); current = ""
            else:
                current += ch
        if current.strip():
            pairs.append(current.strip())
        for pair in pairs:
            if ":" not in pair: continue
            k, v = pair.split(":", 1)
            cfg[k.strip().strip("\"'")] = v.strip().strip("\"'")
        cameras[name] = cfg
    return cameras


def _coerce_cv2_backend(value: str) -> int:
    if Cv2Backends is None:
        return int(value)
    if str(value).lstrip("-").isdigit():
        return int(value)
    return int(Cv2Backends[str(value).upper()])


def _parse_cv2_rotation(value: str):
    raw = str(value).strip()
    if raw.upper().startswith("ROTATE_"):
        return Cv2Rotation[raw.upper()]
    return Cv2Rotation(int(raw))


def connect_cameras(camera_spec: str) -> dict:
    if not camera_spec or not CAMERAS_AVAILABLE:
        return {}
    try:
        parsed = _parse_camera_spec(camera_spec)
    except Exception as e:
        print(f"⚠️  Could not parse --cameras spec: {e}")
        return {}
    cameras = {}
    for name, cfg_dict in parsed.items():
        try:
            raw_path = cfg_dict.get("index_or_path", "0")
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


def encode_image_jpeg(img: np.ndarray, quality: int = 70) -> str:
    if not CV2_AVAILABLE or img is None:
        return ""
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


class CameraBuffer:
    """Non-blocking camera snapshot helper.

    OpenCVCamera already owns a background capture thread. This wrapper peeks at
    the latest frame. If a camera drops out, the last good frame is reused while
    the wrapper tries to reconnect the camera in the background.
    """
    def __init__(self, cameras: dict, max_age_ms: int = 1000, verbose: bool = False):
        self._cameras = cameras
        self._max_age_ms = max_age_ms
        self._verbose = verbose
        self._last_warn_t = {n: 0.0 for n in cameras}
        self._last_reconnect_t = {n: 0.0 for n in cameras}
        self._last_frames = {n: self._blank_frame(cam) for n, cam in cameras.items()}

    def start(self):
        return None

    @staticmethod
    def _blank_frame(cam) -> np.ndarray:
        height = getattr(cam, "height", None) or getattr(cam, "capture_height", None) or 480
        width = getattr(cam, "width", None) or getattr(cam, "capture_width", None) or 640
        return np.zeros((int(height), int(width), 3), dtype=np.uint8)

    def _reconnect(self, name: str, cam) -> None:
        now = time.perf_counter()
        if now - self._last_reconnect_t[name] < 2.0:
            return
        self._last_reconnect_t[name] = now

        if self._verbose:
            print(f"[server] reconnecting camera '{name}'")

        try:
            cam.disconnect()
        except Exception:
            pass

        try:
            cam.connect(warmup=False)
        except Exception as e:
            if self._verbose:
                print(f"[server] camera '{name}' reconnect failed: {e}")

    def get_latest(self) -> dict:
        frames = {}
        for name, cam in self._cameras.items():
            try:
                frame = cam.read_latest(max_age_ms=self._max_age_ms)
                self._last_frames[name] = frame
                frames[name] = frame
            except Exception as e:
                now = time.perf_counter()
                if self._verbose and now - self._last_warn_t[name] > 2.0:
                    print(f"[server] camera '{name}' has no fresh frame: {e}")
                    self._last_warn_t[name] = now
                if self._last_frames[name] is not None:
                    frames[name] = self._last_frames[name]
                if cam.thread is None or not cam.thread.is_alive():
                    self._reconnect(name, cam)
        return frames

    def stop(self):
        return None


# ──────────────────────────────────────────────────────────────
# Client handler
# ──────────────────────────────────────────────────────────────

def handle_client(
    conn:       socket.socket,
    robot_arm,
    motors:     "MotorController | None",
    cam_buffer: "CameraBuffer | None",
    verbose:    bool,
    jpeg_quality: int = 70,
) -> None:
    stats        = ServerStats()
    current_mode = None

    try:
        while True:
            msg   = recv_msg(conn)
            mtype = msg.get("type")

            # ── full_obs_request ──────────────────────────────────────
            if mtype == "full_obs_request":
                state_vals: list[float] = []
                if robot_arm is not None:
                    try:
                        obs        = robot_arm.get_observation()
                        state_vals = [float(obs.get(k, 0.0)) for k in JOINT_KEYS]
                    except Exception as e:
                        if verbose:
                            print(f"[server] obs read error: {e}")

                # Client sets "images": false during chunk execution to skip
                # encoding — only the first step of each chunk needs fresh images.
                want_images = msg.get("images", True)
                images: dict[str, str] = {}
                if want_images and cam_buffer is not None:
                    for cam_name, img in cam_buffer.get_latest().items():
                        if img is not None:
                            images[cam_name] = encode_image_jpeg(img, jpeg_quality)

                send_msg(conn, {"type": "full_obs", "state": state_vals, "images": images})

            # ── action ───────────────────────────────────────────────
            elif mtype == "action":
                if robot_arm is not None:
                    action_dict: dict = msg["action"]
                    is_preset: bool   = msg.get("preset", False)
                    recv_t = time.perf_counter()
                    t0     = time.perf_counter()
                    try:
                        if is_preset:
                            send_action_preset(robot_arm, action_dict)
                        else:
                            robot_arm.send_action(action_dict)
                        exec_s = time.perf_counter() - t0
                        stats.record_action(recv_t, exec_s)
                        stats.maybe_print()
                        if stats._total_actions <= 3 or verbose:
                            vals = [f"{action_dict.get(k, 0):.2f}" for k in JOINT_KEYS]
                            print(f"[server] action #{stats._total_actions} exec={exec_s*1000:.2f}ms  pos={vals}")
                    except Exception as e:
                        print(f"[server] action error: {e}")
                else:
                    if verbose:
                        print("[server] action received but arm not connected — ignoring")

            # ── mode_request (kept for gamepad client compatibility) ──
            elif mtype == "mode_request":
                requested = msg.get("mode")
                if requested == "arm":
                    ok, reason = (True, "") if robot_arm else (False, "arm not connected")
                elif requested == "motor":
                    ok, reason = (True, "") if motors else (False, "motors not connected")
                else:
                    ok, reason = False, f"unknown mode '{requested}'"
                send_msg(conn, {"type": "mode_response", "mode": requested,
                                "ok": ok, "reason": reason})
                if ok:
                    current_mode = requested

            # ── motor ────────────────────────────────────────────────
            elif mtype == "motor":
                if motors is not None:
                    try:
                        motors.set(float(msg.get("motor1", 0.0)), float(msg.get("motor2", 0.0)))
                    except Exception as e:
                        print(f"[server] motor error: {e}")

            # ── ping → pong ──────────────────────────────────────────
            elif mtype == "ping":
                stats.record_ping()
                send_msg(conn, {"type": "pong", "seq": msg.get("seq")})

            # ── disconnect ───────────────────────────────────────────
            elif mtype == "disconnect":
                print("[server] Client requested disconnect.")
                break

            else:
                if verbose:
                    print(f"[server] Unknown message type: {mtype!r}")

    except ConnectionError as e:
        print(f"[server] Connection lost: {e}")
    except Exception as e:
        print(f"[server] Client handler error: {e}")
    finally:
        if motors is not None:
            motors.stop()
        stats.print_summary()
        conn.close()
        print("[server] Client connection closed.")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "SO-101 inference server (Raspberry Pi)\n\n"
            "Connects the arm and cameras, then serves observations and\n"
            "executes actions sent by the inference client.\n\n"
            "Example:\n"
            "  python lerobot_inference_server.py \\\n"
            "    --arm-port /dev/ttyACM0 \\\n"
            "    --cameras \"{ front: {type: opencv, index_or_path: /dev/video0,"
            " width: 640, height: 480, fps: 15}}\""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host",          default="0.0.0.0")
    parser.add_argument("--tcp-port",      type=int, default=2222)
    parser.add_argument("--verbose",       action="store_true")
    # Arm
    parser.add_argument("--no-arm",        action="store_true")
    parser.add_argument("--arm-port",      default="/dev/ttyACM0")
    parser.add_argument("--robot-id",      default="so101_follower")
    # Cameras
    parser.add_argument("--cameras",       default="",
                        help="lerobot-style camera spec")
    parser.add_argument("--image-quality", type=int, default=70,
                        help="JPEG quality sent to client (1-100, default: 70)")
    parser.add_argument("--camera-max-age-ms", type=int, default=1000,
                        help="Drop camera frames older than this many ms (default: 1000)")
    # Motors (optional)
    parser.add_argument("--no-motors",  action="store_true")
    parser.add_argument("--motor-backend", choices=["gpio", "serial"], default="gpio",
                        help="Motor backend: Raspberry Pi GPIO or ESP32/Arduino USB serial bridge (default: gpio)")
    parser.add_argument("--motor-serial-port", default="/dev/ttyUSB0",
                        help="Serial port for --motor-backend=serial, e.g. /dev/ttyUSB0 or /dev/ttyACM0")
    parser.add_argument("--m1-fwd",     type=int, default=16, help="Motor 1 forward GPIO / IN1 (default: 16, physical pin 36)")
    parser.add_argument("--m1-bwd",     type=int, default=1,  help="Motor 1 backward GPIO / IN2 (default: 1, physical pin 28)")
    parser.add_argument("--m1-en",      type=int, default=12, help="Motor 1 enable GPIO / ENA (default: 12, physical pin 32)")
    parser.add_argument("--m2-fwd",     type=int, default=5,  help="Motor 2 forward GPIO / IN3 (default: 5, physical pin 29)")
    parser.add_argument("--m2-bwd",     type=int, default=6,  help="Motor 2 backward GPIO / IN4 (default: 6, physical pin 31)")
    parser.add_argument("--m2-en",      type=int, default=13, help="Motor 2 enable GPIO / ENB (default: 13, physical pin 33)")
    args = parser.parse_args()

    print("  SO-101 Inference Server")
    print("=" * 45)

    # ── Arm ───────────────────────────────────────────────────
    robot_arm = None
    if args.no_arm:
        print("ℹ️  Arm:     DISABLED (--no-arm)")
    else:
        robot_arm = connect_robot_arm(args.arm_port, args.robot_id)
        if robot_arm is None:
            print("ℹ️  Arm:     DISABLED (connection failed)")
        else:
            print(f"✓  Arm:     ENABLED  ({args.arm_port})")

    # ── Cameras ───────────────────────────────────────────────
    cameras    = {}
    cam_buffer = None
    if args.cameras:
        cameras = connect_cameras(args.cameras)
        if cameras:
            cam_buffer = CameraBuffer(cameras, max_age_ms=args.camera_max_age_ms, verbose=args.verbose)
            cam_buffer.start()
            print(f"✓  Cameras: {list(cameras.keys())}  (JPEG quality={args.image_quality})")
    else:
        print("ℹ️  Cameras: NONE (no --cameras given)")

    # ── Motors (optional) ─────────────────────────────────────
    motors = None
    if args.no_motors:
        print("Motors:  DISABLED (--no-motors)")
    elif args.motor_backend == "gpio" and not GPIO_MOTORS_AVAILABLE:
        print("Motors:  DISABLED (gpiozero/lgpio not available)")
    elif args.motor_backend == "serial" and not SERIAL_MOTORS_AVAILABLE:
        print("Motors:  DISABLED (pyserial not available)")
    else:
        try:
            if args.motor_backend == "serial":
                motors = SerialMotorController(args.motor_serial_port)
            else:
                motors = MotorController(
                    m1_forward=args.m1_fwd, m1_backward=args.m1_bwd, m1_enable=args.m1_en,
                    m2_forward=args.m2_fwd, m2_backward=args.m2_bwd, m2_enable=args.m2_en,
                )
            print("✓  Motors:  ENABLED")
        except Exception as e:
            print(f"⚠️  Motors:  DISABLED ({e})")

    print("=" * 45)

    # ── TCP server ────────────────────────────────────────────
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
                conn, robot_arm, motors, cam_buffer,
                verbose=args.verbose,
                jpeg_quality=args.image_quality,
            )
            print("[server] Ready for next client …")
    except KeyboardInterrupt:
        print("\n[server] Shutting down.")
    finally:
        if motors:
            motors.stop()
            if hasattr(motors, "close"):
                motors.close()
        server_sock.close()
        if robot_arm:  robot_arm.disconnect()
        if cam_buffer: cam_buffer.stop()
        for cam in cameras.values():
            try: cam.disconnect()
            except Exception: pass
        print("[server] Clean exit.")


if __name__ == "__main__":
    main()

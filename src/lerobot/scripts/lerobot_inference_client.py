#!/usr/bin/env python3
"""
lerobot_inference_client.py  –  runs on Windows/Mac/Linux
==========================================================
Loads a trained lerobot policy, streams observations from the robot,
and sends actions back — indefinitely until you press Ctrl+C.


  SERVER (Raspberry Pi)  ←──── actions ──────  CLIENT (this machine)
                                                       │
                                                  policy.predict()
                                                       │
  SERVER (Raspberry Pi)  ────── obs/images ──►  CLIENT (this machine)
Usage:
    lerobot-inference-client \\
        --host 192.168.1.42 \\
        --policy-path ${HF_USER}/my_policy

    # Test connectivity without moving the robot:
    lerobot-inference-client \\
        --host 192.168.1.42 \\
        --policy-path ${HF_USER}/my_policy \\
        --dry-run

Wire protocol:
    Each message = [4-byte little-endian uint32 length][UTF-8 JSON payload]
"""

import argparse
import base64
import collections
import json
import logging
import platform
import socket
import struct
import time

import numpy as np

# ── Optional: lerobot policy ────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  torch not found — install PyTorch to run policy inference")

try:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.utils.control_utils import predict_action
    LEROBOT_AVAILABLE = True
except ImportError as _e:
    print(f"⚠️  lerobot not found: {_e}")
    LEROBOT_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False


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


def connect_to_server(host: str, port: int, retries: int = 10) -> socket.socket:
    for attempt in range(1, retries + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            print(f"✓ Connected to {host}:{port}")
            return sock
        except OSError as e:
            print(f"  Attempt {attempt}/{retries} failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"Could not connect to {host}:{port} after {retries} attempts")


# ──────────────────────────────────────────────────────────────
# Joint keys  (must match server)
# ──────────────────────────────────────────────────────────────

JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


# ──────────────────────────────────────────────────────────────
# Observation helpers
# ──────────────────────────────────────────────────────────────

def decode_image(b64_str: str) -> np.ndarray:
    """Decode a base64 JPEG string into an HWC uint8 numpy array."""
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    if CV2_AVAILABLE:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Fallback: return raw bytes as 1-D array (won't work with most policies)
    return arr


# ──────────────────────────────────────────────────────────────
# Policy wrapper
# ──────────────────────────────────────────────────────────────

class PolicyRunner:
    """
    Thin wrapper around a loaded lerobot PreTrainedPolicy.

    Uses lerobot's make_pre_post_processors + predict_action pipeline (the same
    path as lerobot-record --policy.path=...) so normalisation/denormalisation is
    always correct without any manual heuristics.

    predict(obs_dict) → action_dict  {joint_key: float}
    """

    def __init__(self, policy_path: str, dataset_repo_id: str, device: str = "cpu"):
        if not LEROBOT_AVAILABLE:
            raise RuntimeError("lerobot is not installed")
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is not installed")

        import os
        import torch
        policy_path = os.path.expandvars(policy_path)

        if device == "cuda" and not torch.cuda.is_available():
            print("  ⚠️  CUDA not available — falling back to cpu")
            device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            print("  ⚠️  MPS not available — falling back to cpu")
            device = "cpu"

        print(f"  Loading policy from '{policy_path}' (device={device}) …")
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = policy_path
        cfg.device = device

        print(f"  Loading dataset metadata from '{dataset_repo_id}' …")
        ds_meta = LeRobotDatasetMetadata(dataset_repo_id)

        self.policy = make_policy(cfg, ds_meta=ds_meta)
        self.policy.eval()
        self.device = torch.device(device)

        # Build preprocessor/postprocessor — handles input normalisation and
        # output denormalisation using dataset stats, exactly as lerobot-record does.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=policy_path,
            dataset_stats=ds_meta.stats,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        print(f"✓ Policy loaded  ({type(self.policy).__name__}  device={device})")

    @property
    def n_action_steps(self) -> int:
        """How many queue-pop steps between NN re-runs."""
        return getattr(getattr(self.policy, "config", None), "n_action_steps", 1)

    def reset(self):
        """Call at the start of each episode to clear recurrent / queued state."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        if hasattr(self.preprocessor, "reset"):
            self.preprocessor.reset()
        if hasattr(self.postprocessor, "reset"):
            self.postprocessor.reset()

    def predict(self, obs: dict) -> dict:
        """
        obs: dict as returned by get_obs()
            "observation.state"        → np.ndarray (6,)   degrees
            "observation.images.NAME"  → np.ndarray (H,W,3) uint8

        Returns:
            {joint_key: float}  degree-scale values ready to send to the robot
        """
        action = predict_action(
            observation=obs,
            policy=self.policy,
            device=self.device,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=(self.device.type == "cuda"),
        )
        raw = action.cpu().numpy().flatten()  # .cpu() required: numpy cannot access GPU memory
        if len(raw) != len(JOINT_KEYS):
            raise ValueError(f"Policy returned {len(raw)} values, expected {len(JOINT_KEYS)}")
        return {k: float(raw[i]) for i, k in enumerate(JOINT_KEYS)}


# ──────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────

class InferenceStats:
    DISPLAY_INTERVAL = 2.0

    def __init__(self, fps: int):
        self._target_dt     = 1.0 / fps
        self._loop_times:   collections.deque = collections.deque(maxlen=fps * 4)
        self._infer_times:  collections.deque = collections.deque(maxlen=fps * 4)
        self._send_times:   collections.deque = collections.deque(maxlen=fps * 4)
        self._obs_times:    collections.deque = collections.deque(maxlen=fps * 4)
        self._last_loop_t:  float | None      = None
        self._last_disp_t   = time.perf_counter()
        self._session_start = time.perf_counter()
        self._total_steps   = 0
        self._late_steps    = 0

    def record_loop(self, t: float):
        if self._last_loop_t is not None:
            dt = t - self._last_loop_t
            self._loop_times.append(dt)
            if dt > self._target_dt * 1.5:
                self._late_steps += 1
        self._last_loop_t = t
        self._total_steps += 1

    def record_obs(self, dt):   self._obs_times.append(dt)
    def record_infer(self, dt): self._infer_times.append(dt)
    def record_send(self, dt):  self._send_times.append(dt)

    def maybe_print(self, step: int):
        now = time.perf_counter()
        if now - self._last_disp_t < self.DISPLAY_INTERVAL:
            return
        self._last_disp_t = now
        hz       = 1.0 / np.mean(self._loop_times) if len(self._loop_times) >= 2 else 0.0
        obs_ms   = np.mean(self._obs_times)   * 1000.0 if self._obs_times   else 0.0
        infer_ms = np.mean(self._infer_times) * 1000.0 if self._infer_times else 0.0
        send_ms  = np.mean(self._send_times)  * 1000.0 if self._send_times  else 0.0
        late_pct = 100.0 * self._late_steps / max(self._total_steps, 1)
        uptime   = now - self._session_start
        print(
            f"\r[step={step}] Hz={hz:.1f}  "
            f"obs={obs_ms:.1f}ms  infer={infer_ms:.1f}ms  send={send_ms:.1f}ms  "
            f"late={late_pct:.1f}%  up={uptime:.0f}s   ",
            end="", flush=True,
        )

    def print_summary(self):
        uptime   = time.perf_counter() - self._session_start
        hz       = 1.0 / np.mean(self._loop_times) if len(self._loop_times) >= 2 else 0.0
        infer_ms = np.mean(self._infer_times) * 1000.0 if self._infer_times else float("nan")
        late_pct = 100.0 * self._late_steps / max(self._total_steps, 1)
        print(f"\n\n{'─'*55}")
        print(f"  Inference session summary")
        print(f"  Uptime:      {uptime:.1f}s")
        print(f"  Total steps: {self._total_steps}")
        print(f"  Avg Hz:      {hz:.1f}")
        print(f"  Avg infer:   {infer_ms:.1f} ms")
        print(f"  Late steps:  {self._late_steps}  ({late_pct:.1f}%)")
        print(f"{'─'*55}\n")


# ──────────────────────────────────────────────────────────────
# Inference loop
# ──────────────────────────────────────────────────────────────

def get_obs(sock: socket.socket, want_images: bool = True) -> dict:
    """Fetch one observation from the server."""
    send_msg(sock, {"type": "full_obs_request", "images": want_images})
    msg = recv_msg(sock)
    if msg.get("type") != "full_obs":
        raise RuntimeError(f"Unexpected response: {msg.get('type')}")
    obs = {"observation.state": np.array(msg.get("state", []), dtype=np.float32)}
    for cam_name, b64 in msg.get("images", {}).items():
        obs[f"observation.images.{cam_name}"] = decode_image(b64)
    return obs


def show_images(obs: dict) -> None:
    """Display all camera images from obs in cv2 windows (no-op if cv2 unavailable)."""
    if not CV2_AVAILABLE:
        return
    for key, img in obs.items():
        if key.startswith("observation.images.") and isinstance(img, np.ndarray):
            cam_name = key.split(".")[-1]
            cv2.imshow(f"cam: {cam_name}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


class ManualMotorController:
    """Reads an Xbox-style gamepad and sends differential drive motor commands."""

    DEADZONE = 0.12

    def __init__(self, sock: socket.socket, max_speed: float = 1.0, send_hz: float = 30.0):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame is not installed")

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("no gamepad detected")

        self.sock = sock
        self.max_speed = max(0.0, min(1.0, float(max_speed)))
        self.send_dt = 1.0 / send_hz if send_hz > 0 else 0.0
        self.last_send_t = 0.0
        self.last_cmd = (None, None)
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        if platform.system() == "Linux":
            self.axis_left_x = 0
            self.axis_lt = 2
            self.axis_rt = 5
        else:
            self.axis_left_x = 0
            self.axis_lt = 4
            self.axis_rt = 5
        print(f"Manual motors enabled: {self.joystick.get_name()}")

    @classmethod
    def _deadzone(cls, value: float) -> float:
        return 0.0 if abs(value) < cls.DEADZONE else value

    def _axis(self, index: int, default: float = 0.0) -> float:
        if index >= self.joystick.get_numaxes():
            return default
        return float(self.joystick.get_axis(index))

    def read_command(self) -> tuple[float, float]:
        pygame.event.pump()
        lt = (self._axis(self.axis_lt, -1.0) + 1.0) / 2.0
        rt = (self._axis(self.axis_rt, -1.0) + 1.0) / 2.0
        throttle = self._deadzone(rt - lt) * self.max_speed
        turn = self._deadzone(self._axis(self.axis_left_x)) * self.max_speed
        motor1 = max(-1.0, min(1.0, throttle + turn))
        motor2 = max(-1.0, min(1.0, throttle - turn))
        return motor1, motor2

    def maybe_send(self) -> None:
        now = time.perf_counter()
        if self.send_dt > 0 and now - self.last_send_t < self.send_dt:
            return

        motor1, motor2 = self.read_command()
        rounded = (round(motor1, 3), round(motor2, 3))
        if rounded == self.last_cmd:
            return

        send_msg(self.sock, {"type": "motor", "motor1": motor1, "motor2": motor2})
        self.last_cmd = rounded
        self.last_send_t = now

    def stop(self) -> None:
        try:
            send_msg(self.sock, {"type": "motor", "motor1": 0.0, "motor2": 0.0})
        except Exception:
            pass

    def close(self) -> None:
        self.stop()
        if PYGAME_AVAILABLE:
            pygame.joystick.quit()
            pygame.quit()


def run_inference(
    sock:      socket.socket,
    policy:    "PolicyRunner",
    fps:       int,
    dry_run:   bool,
    show_imgs: bool = False,
    image_refresh_hz: float = 10.0,
    manual_motors: "ManualMotorController | None" = None,
):
    """
    Inference loop — ACT's internal queue handles chunking transparently.

    policy.predict() calls select_action() which runs the NN only when its
    internal queue is empty (every n_action_steps steps), then pops pre-computed
    actions for subsequent steps.  No manual chunk loop needed here.
    """
    stats      = InferenceStats(fps)
    control_dt = 1.0 / fps
    step       = 0

    # Tell the server we want ARM mode
    send_msg(sock, {"type": "mode_request", "mode": "arm"})
    resp = recv_msg(sock)
    if resp.get("type") == "mode_response":
        if resp.get("ok"):
            print("✓ ARM mode granted\n")
        else:
            print(f"⚠️  ARM mode denied: {resp.get('reason')} — continuing anyway\n")

    policy.reset()

    n_action_steps = policy.n_action_steps
    print(f"Running inference at {fps} Hz  (n_action_steps={n_action_steps})")
    print("Press Ctrl+C to stop.\n")

    # Absolute scheduler: each action is due at a fixed wall-clock time so
    # obs + infer time is absorbed into the sleep rather than added on top.
    next_step_t = time.perf_counter()

    obs: dict | None = None
    next_image_t = time.perf_counter()
    image_refresh_dt = 1.0 / image_refresh_hz if image_refresh_hz > 0 else float("inf")

    try:
        while True:
            if manual_motors is not None and not dry_run:
                manual_motors.maybe_send()

            # ── 1. Observe ────────────────────────────────────────────
            # Fetch obs only when the NN will run (every n_action_steps).
            # Queue-pop steps don't use obs at all — reuse the cached one.
            if step % n_action_steps == 0:
                t0 = time.perf_counter()
                new_obs = get_obs(sock, want_images=True)
                elapsed = time.perf_counter() - t0
                next_image_t = time.perf_counter() + image_refresh_dt

                state = new_obs.get("observation.state", np.array([]))
                image_keys = [k for k in new_obs if k.startswith("observation.images.")]
                if len(state) == 0:
                    logging.warning("step=%d: server returned empty state — reusing cached obs", step)
                elif not image_keys:
                    logging.warning("step=%d: server returned no fresh images — reusing cached obs", step)
                else:
                    obs = new_obs
                    stats.record_obs(elapsed)
                    if show_imgs:
                        show_images(obs)
            elif show_imgs and time.perf_counter() >= next_image_t:
                # Debug display only. ACT action chunks intentionally reuse the
                # cached policy observation, but the preview should still move.
                try:
                    preview_obs = get_obs(sock, want_images=True)
                    show_images(preview_obs)
                except Exception as e:
                    logging.warning("step=%d: image preview refresh failed: %s", step, e)
                finally:
                    next_image_t = time.perf_counter() + image_refresh_dt

            if obs is None:
                logging.warning("step=%d: waiting for first complete observation", step)
                time.sleep(min(control_dt, 0.1))
                continue

            # ── 2. Predict next action ────────────────────────────────
            # ACT runs the NN only when its queue is empty (every n_action_steps
            # steps); all other calls just pop the next pre-computed action.
            t0 = time.perf_counter()
            action_dict = policy.predict(obs)
            stats.record_infer(time.perf_counter() - t0)

            # ── 3. Execute at target Hz ───────────────────────────────
            sleep_t = next_step_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            next_step_t += control_dt

            stats.record_loop(time.perf_counter())

            if dry_run:
                vals = [f"{action_dict[k]:6.2f}" for k in JOINT_KEYS]
                print(f"\r  [dry-run] step={step:5d}  {vals}", end="", flush=True)
            else:
                t0 = time.perf_counter()
                send_msg(sock, {"type": "action", "action": action_dict})
                stats.record_send(time.perf_counter() - t0)

            if manual_motors is not None and not dry_run:
                manual_motors.maybe_send()

            step += 1
            stats.maybe_print(step)

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        stats.print_summary()
        if manual_motors is not None:
            manual_motors.stop()
        try:
            send_msg(sock, {"type": "disconnect"})
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run lerobot policy inference against a robot on a Raspberry Pi",
    )
    parser.add_argument("--host",        required=True, help="Raspberry Pi IP address")
    parser.add_argument("--tcp-port",    type=int, default=2222)
    parser.add_argument("--policy-path", required=True,
                        help="HF repo id or local path of the trained policy")
    parser.add_argument("--dataset-repo-id", required=True,
                        help="HF repo id of the training dataset (e.g. user/my_dataset)")
    parser.add_argument("--device",      default="cpu",
                        help="Torch device: cpu, cuda, mps (default: cpu)")
    parser.add_argument("--fps",         type=int, default=30,
                        help="Control frequency in Hz (default: 30)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print actions without sending them to the robot")
    parser.add_argument("--show-images", action="store_true",
                        help="Display camera frames in cv2 windows during inference")
    parser.add_argument("--image-refresh-hz", type=float, default=10.0,
                        help="Preview refresh rate for --show-images (default: 10)")
    parser.add_argument("--motor-max-speed", type=float, default=1.0,
                        help="Scale manual motor commands in [0, 1] (default: 1.0)")
    parser.add_argument("--motor-send-hz", type=float, default=30.0,
                        help="Maximum manual motor command rate in Hz (default: 30)")
    args = parser.parse_args()

    policy = PolicyRunner(args.policy_path, dataset_repo_id=args.dataset_repo_id, device=args.device)

    print("\n" + "=" * 45)
    print("  LEROBOT INFERENCE CLIENT")
    print("=" * 45)
    if dry_run := args.dry_run:
        print("  ⚠️  DRY-RUN — actions will NOT be sent")
    print(f"  FPS:    {args.fps}")
    print(f"  Device: {args.device}")
    print("  Motors: manual gamepad control if available")
    print()

    sock = connect_to_server(args.host, args.tcp_port)
    manual_motors = None
    try:
        try:
            manual_motors = ManualMotorController(
                sock,
                max_speed=args.motor_max_speed,
                send_hz=args.motor_send_hz,
            )
        except RuntimeError as e:
            print(f"  Motors: manual control unavailable ({e})")
        run_inference(sock, policy, fps=args.fps, dry_run=dry_run,
                      show_imgs=args.show_images,
                      image_refresh_hz=args.image_refresh_hz,
                      manual_motors=manual_motors)
    finally:
        if manual_motors is not None:
            manual_motors.close()
        sock.close()
        print("✓ Socket closed.")


if __name__ == "__main__":
    main()

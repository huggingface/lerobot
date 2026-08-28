#!/usr/bin/env python3

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

"""
DDS-to-ZMQ bridge server for Unitree G1 robot.

This server runs on the robot and forwards:
- Robot state (LowState) from DDS to ZMQ (for remote clients)
- Robot commands (LowCmd) from ZMQ to DDS (from remote clients)

Uses JSON for secure serialization instead of pickle.
"""

import argparse
import base64
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import zmq
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.utils.crc import CRC

from lerobot.cameras.configs import ColorMode, Cv2Backends
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.zmq.image_server import ImageServer
from lerobot.robots.openarm_follower.config_openarm_follower import LEFT_DEFAULT_JOINTS_LIMITS

if TYPE_CHECKING:
    from lerobot.motors.damiao.damiao import DamiaoMotorsBus

# DDS topic names follow Unitree SDK naming conventions
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"  # action to robot
kTopicLowState = "rt/lowstate"  # observation from robot

LOWCMD_PORT = 6000
LOWSTATE_PORT = 6001
# Side-channel for gripper commands sent by the teleop laptop (exo R3/L3 clicks).
# The exo joystick buttons are only known laptop-side, so the robot object forwards
# them here as JSON {"L": 0/1, "R": 0/1}; see UnitreeG1._send_gripper_cmd.
GRIPPER_PORT = 6002
NUM_MOTORS = 35

# The hands on this G1 are OpenArm grippers, so their CAN identity and travel come from
# the OpenArm follower's config rather than being restated here: J8 on that bus, and the
# same open/close range. Only stiffness and which CAN port each side sits on vary per run.
GRIPPER_SEND_ID = 0x08
GRIPPER_RECV_ID = 0x18
GRIPPER_MOTOR_TYPE = "dm4310"
GRIPPER_OPEN_DEG, GRIPPER_CLOSE_DEG = LEFT_DEFAULT_JOINTS_LIMITS["gripper"]


@dataclass
class Gripper:
    """A single Damiao gripper that only writes to CAN when the open/close state changes."""

    name: str
    bus: "DamiaoMotorsBus"
    open_deg: float
    close_deg: float
    _last_cmd: str | None = None  # "open" | "close"

    def apply(self, want_close: bool) -> None:
        want = "close" if want_close else "open"
        if want == self._last_cmd:
            return
        target = self.close_deg if want_close else self.open_deg
        self.bus.write("Goal_Position", "gripper", target)
        self._last_cmd = want
        print(f"[gripper] {self.name} -> {want.upper()} ({target:.1f} deg)")


def build_gripper(
    name: str,
    port: str,
    *,
    kp: float,
    kd: float,
    open_deg: float = GRIPPER_OPEN_DEG,
    close_deg: float = GRIPPER_CLOSE_DEG,
) -> Gripper:
    """Connect one Damiao gripper. Gains are keyword-only: they are easy to swap by
    accident, and a wrong ``kp`` is the difference between a grip and a crushed object.
    """
    from lerobot.motors.damiao.damiao import DamiaoMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorNormMode

    motors = {
        "gripper": Motor(
            id=GRIPPER_SEND_ID,
            model=GRIPPER_MOTOR_TYPE,
            norm_mode=MotorNormMode.DEGREES,
            motor_type_str=GRIPPER_MOTOR_TYPE,
            recv_id=GRIPPER_RECV_ID,
        )
    }
    bus = DamiaoMotorsBus(port=port, motors=motors, use_can_fd=True)
    print(f"Connecting {name} gripper on {port}...")
    bus.connect(handshake=True)
    bus.write("Kp", "gripper", kp)
    bus.write("Kd", "gripper", kd)
    bus.write("Goal_Position", "gripper", open_deg)  # start open
    print(f"  {name}: connected, torque enabled, opened (kp={kp}, kd={kd}).")
    return Gripper(name, bus, open_deg, close_deg, _last_cmd="open")


def resolve_v4l2_device(device: str) -> int | str:
    """Resolve a V4L2 device to the index OpenCV can open, leaving anything else alone.

    ``by-path`` names are the right way to name a camera, but OpenCV's V4L2 backend
    captures by index only -- handed a filename it reports "backend is generally
    available but can't be used to capture by name" and fails to open. Following the
    symlink to ``/dev/videoN`` and passing ``N`` keeps the stable name in configs and
    still opens the device.
    """
    if device.lstrip("-").isdigit():
        return int(device)
    node = re.fullmatch(r"video(\d+)", Path(device).resolve().name)
    return int(node.group(1)) if node else device


def parse_camera_specs(spec: str, fps: int, width: int, height: int) -> dict[str, OpenCVCameraConfig]:
    """Parse a multi-camera spec string into camera configs.

    Format: comma-separated ``name=device[@WxH]`` entries, e.g.
    ``head_camera=/dev/video4,left_wrist=/dev/v4l/by-path/…-video-index0@1280x720``.
    ``device`` is an integer index or a device path, and ``@WxH`` overrides the
    default resolution for that camera alone (cameras on this robot don't agree on a
    common one). ``=`` and ``@`` are used as separators rather than ``:`` because
    stable ``by-path`` device names contain colons themselves, and those are the
    names worth using: bare ``/dev/videoN`` indices reshuffle on USB re-enumeration
    and ``by-id`` names collide when two cameras share a serial.
    """
    cameras: dict[str, OpenCVCameraConfig] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Invalid camera spec '{entry}', expected 'name=device[@WxH]'")
        name, device = (part.strip() for part in entry.split("=", 1))
        cam_width, cam_height = width, height
        if "@" in device:
            device, resolution = (part.strip() for part in device.rsplit("@", 1))
            cam_width, cam_height = (int(v) for v in resolution.lower().split("x"))
        if not device:
            raise ValueError(f"Invalid camera spec '{entry}', missing device")
        if name in cameras:
            raise ValueError(f"Duplicate camera name '{name}' in --cameras")
        cameras[name] = OpenCVCameraConfig(
            index_or_path=resolve_v4l2_device(device),
            fps=fps,
            width=cam_width,
            height=cam_height,
            color_mode=ColorMode.RGB,
            # MJPG lets these UVC cameras negotiate their full rate, and V4L2 because
            # the default FFMPEG backend is read-only for capture properties, so it
            # can't apply the format or resolution at all.
            fourcc="MJPG",
            backend=Cv2Backends.V4L2,
            # Several cameras sharing a USB bus can take seconds to yield a first frame.
            warmup_s=5,
        )
    if not cameras:
        raise ValueError("No cameras parsed from --cameras spec")
    return cameras


def lowstate_to_dict(msg: hg_LowState) -> dict[str, Any]:
    """Convert LowState SDK message to a JSON-serializable dictionary."""
    motor_states = []
    for i in range(NUM_MOTORS):
        temp = msg.motor_state[i].temperature
        avg_temp = float(sum(temp) / len(temp)) if isinstance(temp, list) else float(temp)
        motor_states.append(
            {
                "q": float(msg.motor_state[i].q),
                "dq": float(msg.motor_state[i].dq),
                "tau_est": float(msg.motor_state[i].tau_est),
                "temperature": avg_temp,
            }
        )

    return {
        "motor_state": motor_states,
        "imu_state": {
            "quaternion": [float(x) for x in msg.imu_state.quaternion],
            "gyroscope": [float(x) for x in msg.imu_state.gyroscope],
            "accelerometer": [float(x) for x in msg.imu_state.accelerometer],
            "rpy": [float(x) for x in msg.imu_state.rpy],
            "temperature": float(msg.imu_state.temperature),
        },
        # Encode bytes as base64 for JSON compatibility
        "wireless_remote": base64.b64encode(bytes(msg.wireless_remote)).decode("ascii"),
        "mode_machine": int(msg.mode_machine),
    }


def dict_to_lowcmd(data: dict[str, Any]) -> hg_LowCmd:
    """Convert dictionary back to LowCmd SDK message."""
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = data.get("mode_pr", 0)
    cmd.mode_machine = data.get("mode_machine", 0)

    for i, motor_data in enumerate(data.get("motor_cmd", [])):
        cmd.motor_cmd[i].mode = motor_data.get("mode", 0)
        cmd.motor_cmd[i].q = motor_data.get("q", 0.0)
        cmd.motor_cmd[i].dq = motor_data.get("dq", 0.0)
        cmd.motor_cmd[i].kp = motor_data.get("kp", 0.0)
        cmd.motor_cmd[i].kd = motor_data.get("kd", 0.0)
        cmd.motor_cmd[i].tau = motor_data.get("tau", 0.0)

    return cmd


def gripper_cmd_loop(
    gripper_sock: zmq.Socket,
    grippers: dict[str, Gripper],
    shutdown_event: threading.Event,
) -> None:
    """Receive gripper commands from the teleop laptop and apply them.

    Payload is JSON ``{"L": 0/1, "R": 0/1}`` where 1 = close, 0 = open. Only writes
    CAN when a gripper's state actually changes (handled by Gripper.apply).
    """
    while not shutdown_event.is_set():
        try:
            payload = gripper_sock.recv()
        except zmq.ContextTerminated:
            break
        except zmq.Again:
            continue
        try:
            cmd = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        print(f"[gripper] recv {cmd}")
        if "L" in grippers and "L" in cmd:
            grippers["L"].apply(bool(cmd["L"]))
        if "R" in grippers and "R" in cmd:
            grippers["R"].apply(bool(cmd["R"]))


def state_forward_loop(
    lowstate_sub: ChannelSubscriber,
    lowstate_sock: zmq.Socket,
    state_period: float,
    shutdown_event: threading.Event,
) -> None:
    """Read observation from DDS and forward to ZMQ clients."""
    last_state_time = 0.0

    while not shutdown_event.is_set():
        # read from DDS
        msg = lowstate_sub.Read()
        if msg is None:
            continue

        now = time.time()
        # optional downsampling (if robot dds rate > state_period)
        if now - last_state_time >= state_period:
            # Convert to dict and serialize with JSON
            state_dict = lowstate_to_dict(msg)
            payload = json.dumps({"topic": kTopicLowState, "data": state_dict}).encode("utf-8")
            try:
                # if no subscribers / tx buffer full, just drop
                lowstate_sock.send(payload, zmq.NOBLOCK)
            except zmq.Again:
                pass
            except zmq.ContextTerminated:
                # Context torn down during shutdown; exit the loop quietly.
                break
            last_state_time = now


def cmd_forward_loop(
    lowcmd_sock: zmq.Socket,
    lowcmd_pub_debug: ChannelPublisher,
    crc: CRC,
) -> None:
    """Receive commands from ZMQ and forward to DDS."""
    while True:
        try:
            payload = lowcmd_sock.recv()
        except zmq.ContextTerminated:
            break
        msg_dict = json.loads(payload.decode("utf-8"))

        topic = msg_dict.get("topic", "")
        cmd_data = msg_dict.get("data", {})

        # Reconstruct LowCmd object from dict
        cmd = dict_to_lowcmd(cmd_data)

        # recompute crc
        cmd.crc = crc.Crc(cmd)

        if topic == kTopicLowCommand_Debug:
            lowcmd_pub_debug.Write(cmd)


def main() -> None:
    """Main entry point for the robot server bridge."""
    parser = argparse.ArgumentParser(description="DDS-to-ZMQ bridge server for Unitree G1")
    parser.add_argument("--camera", action="store_true", help="Also launch camera server")
    parser.add_argument("--camera-device", type=int, default=4, help="Camera device ID (default: 4)")
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help=(
            "Multi-camera spec: comma-separated 'name=device[@WxH]', e.g. "
            "'head_camera=/dev/video4,left_wrist=/dev/v4l/by-path/…-video-index0@1280x720'. "
            "Implies --camera and overrides --camera-device."
        ),
    )
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera FPS (default: 30)")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument("--camera-port", type=int, default=5555, help="Camera ZMQ port (default: 5555)")
    parser.add_argument(
        "--camera-publish-size",
        default=None,
        help=(
            "Downscale every camera to 'WxH' before publishing, e.g. '640x480'. Useful when a "
            "camera only negotiates a resolution larger than the consumer needs."
        ),
    )
    # Grippers: identity and travel come from the OpenArm config; these are the knobs that
    # actually change between sessions (which CAN port per side, and squeeze strength).
    parser.add_argument("--grippers", action="store_true", help="Enable Damiao gripper control")
    parser.add_argument("--gripper-port-left", default="can1", help="CAN interface for LEFT gripper")
    parser.add_argument("--gripper-port-right", default="can0", help="CAN interface for RIGHT gripper")
    parser.add_argument("--gripper-kp", type=float, default=15.0, help="MIT position gain (stiffness)")
    parser.add_argument("--gripper-kd", type=float, default=0.5, help="MIT damping gain")
    parser.add_argument(
        "--gripper-open-deg", type=float, default=GRIPPER_OPEN_DEG, help="Gripper OPEN position (deg)"
    )
    parser.add_argument(
        "--gripper-close-deg",
        type=float,
        default=GRIPPER_CLOSE_DEG,
        help="Gripper CLOSE position (deg); raise it toward the open value for delicate objects",
    )
    args = parser.parse_args()

    # Optionally start camera server in background thread
    camera_thread = None
    camera_server = None
    if args.camera or args.cameras:
        spec = args.cameras or f"head_camera={args.camera_device}"
        cameras = parse_camera_specs(spec, args.camera_fps, args.camera_width, args.camera_height)
        publish_size = None
        if args.camera_publish_size:
            w, h = (int(v) for v in args.camera_publish_size.lower().split("x"))
            publish_size = (w, h)
        camera_server = ImageServer(
            cameras, fps=args.camera_fps, port=args.camera_port, publish_size=publish_size
        )
        camera_thread = threading.Thread(target=camera_server.run, daemon=True)
        camera_thread.start()
        cam_summary = ", ".join(
            f"{name}({cfg.index_or_path} {cfg.width}x{cfg.height})" for name, cfg in cameras.items()
        )
        print(f"Camera server started on port {args.camera_port}: {cam_summary}")

    # initialize DDS
    ChannelFactoryInitialize(0)

    # stop all active publishers on the robot
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    status, result = msc.CheckMode()
    while result is not None and "name" in result and result["name"]:
        msc.ReleaseMode()
        status, result = msc.CheckMode()
        time.sleep(1.0)

    crc = CRC()

    # initialize DDS publisher
    lowcmd_pub_debug = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
    lowcmd_pub_debug.Init()

    # initialize DDS subscriber
    lowstate_sub = ChannelSubscriber(kTopicLowState, hg_LowState)
    lowstate_sub.Init()

    # initialize ZMQ
    ctx = zmq.Context.instance()

    # receive commands from remote client
    lowcmd_sock = ctx.socket(zmq.PULL)
    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")

    # publish state to remote clients
    lowstate_sock = ctx.socket(zmq.PUB)
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")

    # Optionally connect Damiao grippers driven by exo R3/L3 (forwarded from the laptop)
    grippers: dict[str, Gripper] = {}
    gripper_sock = None
    if args.grippers:
        try:
            for side, port in (("L", args.gripper_port_left), ("R", args.gripper_port_right)):
                grippers[side] = build_gripper(
                    side,
                    port,
                    kp=args.gripper_kp,
                    kd=args.gripper_kd,
                    open_deg=args.gripper_open_deg,
                    close_deg=args.gripper_close_deg,
                )
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: gripper setup failed ({e}); continuing without grippers.")
            grippers = {}

    state_period = 0.002  # ~500 hz
    shutdown_event = threading.Event()

    # start observation forwarding in background thread
    t_state = threading.Thread(
        target=state_forward_loop,
        args=(lowstate_sub, lowstate_sock, state_period, shutdown_event),
    )
    t_state.start()

    # start gripper command listener (commands come from the teleop laptop)
    t_gripper = None
    if grippers:
        gripper_sock = ctx.socket(zmq.PULL)
        gripper_sock.bind(f"tcp://0.0.0.0:{GRIPPER_PORT}")
        t_gripper = threading.Thread(
            target=gripper_cmd_loop,
            args=(gripper_sock, grippers, shutdown_event),
        )
        t_gripper.start()
        print(f"Grippers enabled: listening for R3/L3 commands on port {GRIPPER_PORT}")

    print("bridge running (lowstate -> zmq, lowcmd -> dds)")

    # run command forwarding in main thread
    try:
        cmd_forward_loop(lowcmd_sock, lowcmd_pub_debug, crc)
    except KeyboardInterrupt:
        print("shutting down bridge...")
    finally:
        shutdown_event.set()
        # Stop the camera server first so it releases the V4L2 devices cleanly;
        # otherwise the daemon thread is killed on exit and the cameras stay wedged.
        if camera_server is not None:
            camera_server.stop()
        ctx.term()  # terminates blocking zmq.recv() calls
        t_state.join(timeout=2.0)
        if t_gripper is not None:
            t_gripper.join(timeout=2.0)
        if camera_thread is not None:
            camera_thread.join(timeout=3.0)
        for g in grippers.values():
            try:
                g.bus.disconnect(disable_torque=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  {g.name} gripper disconnect error: {exc}")


if __name__ == "__main__":
    main()

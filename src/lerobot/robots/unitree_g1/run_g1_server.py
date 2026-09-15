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
import contextlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import draccus
import zmq
from draccus.cfgparsing import parse_string
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.utils.crc import CRC

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Backends
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.zmq.image_server import ImageServer
from lerobot.motors.damiao.damiao import DamiaoMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode
from lerobot.robots.openarm_follower.config_openarm_follower import (
    DEFAULT_MOTOR_CONFIG,
    LEFT_DEFAULT_JOINTS_LIMITS,
)

# DDS topic names follow Unitree SDK naming conventions
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"  # action to robot
kTopicLowState = "rt/lowstate"  # observation from robot

LOWCMD_PORT = 6000
LOWSTATE_PORT = 6001
# Gripper commands arrive as JSON {"L": <closedness>, "R": <closedness>} where 0 is fully
# open and 1 fully closed; see UnitreeG1._send_gripper_cmd. Teleop only ever knows the exo's
# R3/L3 button state, so it sends the integers 0 and 1, while a policy sends the fraction in
# between -- the same field, read the same way.
GRIPPER_PORT = 6002
NUM_MOTORS = 35

# The hands on this G1 are OpenArm grippers, so their CAN identity and travel come from
# the OpenArm follower's config rather than being restated here: J8 on that bus, and the
# same open/close range. Only stiffness and which CAN port each side sits on vary per run.
GRIPPER_MOTOR = "gripper"
GRIPPER_SEND_ID, GRIPPER_RECV_ID, GRIPPER_MOTOR_TYPE = DEFAULT_MOTOR_CONFIG[GRIPPER_MOTOR]
GRIPPER_OPEN_DEG, GRIPPER_CLOSE_DEG = LEFT_DEFAULT_JOINTS_LIMITS[GRIPPER_MOTOR]

# Ignore command changes smaller than this so a policy streaming at 30 Hz does not flood the
# CAN bus with sub-degree corrections the gripper cannot resolve anyway.
GRIPPER_DEADBAND = 0.02


class Gripper(DamiaoMotorsBus):
    """One OpenArm gripper, positioned anywhere between fully open and fully closed.

    It *is* the bus rather than owning one: each hand is the only motor on its CAN
    interface, so there is nothing else for a wrapper to hold. What it adds on top is the
    closedness command. The motor takes a continuous ``Goal_Position``, so partial grips
    cost nothing extra and the binary open/close of the teleop path is just the two
    endpoints of the same command.

    Gains are keyword-only: they are easy to swap by accident, and a wrong ``kp`` is the
    difference between a grip and a crushed object.
    """

    def __init__(
        self,
        name: str,
        port: str,
        *,
        kp: float,
        kd: float,
        open_deg: float = GRIPPER_OPEN_DEG,
        close_deg: float = GRIPPER_CLOSE_DEG,
    ):
        super().__init__(
            port=port,
            motors={
                GRIPPER_MOTOR: Motor(
                    id=GRIPPER_SEND_ID,
                    model=GRIPPER_MOTOR_TYPE,
                    norm_mode=MotorNormMode.DEGREES,
                    motor_type_str=GRIPPER_MOTOR_TYPE,
                    recv_id=GRIPPER_RECV_ID,
                )
            },
            use_can_fd=True,
        )
        self.name = name
        self.kp = kp
        self.kd = kd
        self.open_deg = open_deg
        self.close_deg = close_deg
        self._last_closedness: float | None = None

    def connect(self, handshake: bool = True) -> None:
        """Connect the bus, apply the gains, and open the jaw before anything commands it."""
        print(f"Connecting {self.name} gripper on {self.port}...")
        super().connect(handshake=handshake)
        self.write("Kp", GRIPPER_MOTOR, self.kp)
        self.write("Kd", GRIPPER_MOTOR, self.kd)
        self.apply(0.0)
        print(f"  {self.name}: connected, torque enabled, opened (kp={self.kp}, kd={self.kd}).")

    def apply(self, closedness: float) -> None:
        """Drive the gripper to ``closedness``, 0 fully open .. 1 fully closed."""
        closedness = min(1.0, max(0.0, float(closedness)))
        if self._last_closedness is not None and abs(closedness - self._last_closedness) < GRIPPER_DEADBAND:
            return
        target = self.open_deg + closedness * (self.close_deg - self.open_deg)
        self.write("Goal_Position", GRIPPER_MOTOR, target)
        self._last_closedness = closedness
        print(f"[gripper] {self.name} -> {closedness:.2f} ({target:.1f} deg)")


def resolve_v4l2_device(device: int | str | Path) -> int | Path:
    """Resolve a V4L2 device to the index OpenCV can open, leaving anything else alone.

    ``by-path`` names are the right way to name a camera, but OpenCV's V4L2 backend
    captures by index only -- handed a filename it reports "backend is generally
    available but can't be used to capture by name" and fails to open. Following the
    symlink to ``/dev/videoN`` and passing ``N`` keeps the stable name in configs and
    still opens the device.
    """
    if isinstance(device, int):
        return device
    text = str(device)
    if text.lstrip("-").isdigit():
        return int(text)
    node = re.fullmatch(r"video(\d+)", Path(text).resolve().name)
    return int(node.group(1)) if node else Path(text)


def parse_camera_configs(spec: str) -> dict[str, CameraConfig]:
    """Parse ``--cameras`` into camera configs, in the syntax the ``lerobot-`` scripts use.

    Same shape as ``--robot.cameras`` everywhere else, so a working laptop-side camera
    block can be pasted here unchanged::

        "{ego_view: {type: opencv, index_or_path: /dev/v4l/by-path/…-video-index0,
          width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2, warmup_s: 5}}"

    ``by-path`` names are resolved to the ``/dev/videoN`` index here rather than in the
    config, so a config stays readable and survives USB re-enumeration.
    """
    cameras = draccus.decode(dict[str, CameraConfig], parse_string(spec))
    if not cameras:
        raise ValueError("No cameras parsed from --cameras")
    for config in cameras.values():
        if isinstance(config, OpenCVCameraConfig):
            config.index_or_path = resolve_v4l2_device(config.index_or_path)
    return cameras


def head_camera_config(device: int, fps: int, width: int, height: int) -> OpenCVCameraConfig:
    """Config for the camera the G1 ships with, for the ``--camera`` shorthand."""
    return OpenCVCameraConfig(
        index_or_path=device,
        fps=fps,
        width=width,
        height=height,
        color_mode=ColorMode.RGB,
        # MJPG lets these UVC cameras negotiate their full rate, and V4L2 because the
        # default FFMPEG backend is read-only for capture properties, so it can't apply
        # the format or resolution at all.
        fourcc="MJPG",
        backend=Cv2Backends.V4L2,
        # Several cameras sharing a USB bus can take seconds to yield a first frame.
        warmup_s=5,
    )


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

    Payload is JSON ``{"L": <closedness>, "R": <closedness>}``, 0 fully open .. 1 fully
    closed. Teleop's 0/1 integers are the endpoints of that range, so the older binary
    senders keep working unchanged. Only writes CAN on a real change (see Gripper.apply).
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
        for side in ("L", "R"):
            if side in grippers and side in cmd:
                grippers[side].apply(cmd[side])


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
            "Cameras to publish, in the same syntax as --robot.cameras elsewhere, e.g. "
            "'{ego_view: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, "
            "fps: 30, fourcc: MJPG, backend: V4L2}}'. Implies --camera and overrides "
            "--camera-device."
        ),
    )
    parser.add_argument(
        "--camera-fps", type=int, default=30, help="Publish rate, and --camera's capture rate"
    )
    parser.add_argument("--camera-width", type=int, default=640, help="Width for --camera")
    parser.add_argument("--camera-height", type=int, default=480, help="Height for --camera")
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
        cameras = (
            parse_camera_configs(args.cameras)
            if args.cameras
            else {
                "head_camera": head_camera_config(
                    args.camera_device, args.camera_fps, args.camera_width, args.camera_height
                )
            }
        )
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
            f"{name}({getattr(cfg, 'index_or_path', type(cfg).__name__)} {cfg.width}x{cfg.height})"
            for name, cfg in cameras.items()
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
                gripper = Gripper(
                    side,
                    port,
                    kp=args.gripper_kp,
                    kd=args.gripper_kd,
                    open_deg=args.gripper_open_deg,
                    close_deg=args.gripper_close_deg,
                )
                gripper.connect()
                grippers[side] = gripper
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: gripper setup failed ({e}); continuing without grippers.")
            # Release the side that did come up, or it holds torque with nothing driving it.
            for connected in grippers.values():
                with contextlib.suppress(Exception):
                    connected.disconnect(disable_torque=True)
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
                g.disconnect(disable_torque=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  {g.name} gripper disconnect error: {exc}")


if __name__ == "__main__":
    main()

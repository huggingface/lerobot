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
DDS-to-ZMQ server for the Unitree G1 robot. Two modes, both run ON the robot:

* bridge (default): forward raw robot state (LowState) DDS -> ZMQ and raw commands
  (LowCmd) ZMQ -> DDS. The controller runs on the laptop and streams lowcmd.

* onboard (``--onboard --controller NAME``): run the controller ONBOARD instead. Builds
  ``UnitreeG1(onboard=True, controller=NAME)`` so its control loop runs locally against
  DDS at full rate; the laptop only PUSHes compact high-level actions (e.g. the 64-D
  SONIC token) on :ACTION_PORT and reads back ``observation.state`` on :STATE_PORT.

Uses JSON for secure serialization instead of pickle.
"""

import argparse
import base64
import contextlib
import json
import signal
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
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
# Gripper commands arrive as JSON {"L": <closedness>, "R": <closedness>} where 0 is fully
# open and 1 fully closed; see UnitreeG1._send_gripper_cmd. Teleop only ever knows the exo's
# R3/L3 button state, so it sends the integers 0 and 1, while a policy sends the fraction in
# between -- the same field, read the same way. This file runs as a script, so the port is
# repeated here rather than imported; keep it in step with unitree_sdk2_socket.GRIPPER_PORT.
GRIPPER_PORT = 6002
NUM_MOTORS = 35

# Onboard high-level channels (serve_onboard_controller): compact actions in, state out.
ACTION_PORT = 6004
STATE_PORT = 6005

# The hands on this G1 are OpenArm grippers, so their CAN identity and travel come from
# the OpenArm follower's config rather than being restated here: J8 on that bus, and the
# same open/close range. Only stiffness and which CAN port each side sits on vary per run.
GRIPPER_SEND_ID = 0x08
GRIPPER_RECV_ID = 0x18
GRIPPER_MOTOR_TYPE = "dm4310"
GRIPPER_OPEN_DEG, GRIPPER_CLOSE_DEG = LEFT_DEFAULT_JOINTS_LIMITS["gripper"]


# Ignore command changes smaller than this so a policy streaming at 30 Hz does not flood the
# CAN bus with sub-degree corrections the gripper cannot resolve anyway.
GRIPPER_DEADBAND = 0.02


@dataclass
class Gripper:
    """A single Damiao gripper, positioned anywhere between fully open and fully closed.

    The motor takes a continuous ``Goal_Position``, so partial grips cost nothing extra; the
    binary open/close of the teleop path is just the two endpoints of the same command.
    """

    name: str
    bus: "DamiaoMotorsBus"
    open_deg: float
    close_deg: float
    _last_closedness: float | None = None

    def apply(self, closedness: float) -> None:
        """Drive the gripper to ``closedness``, 0 fully open .. 1 fully closed."""
        closedness = min(1.0, max(0.0, float(closedness)))
        if self._last_closedness is not None and abs(closedness - self._last_closedness) < GRIPPER_DEADBAND:
            return
        target = self.open_deg + closedness * (self.close_deg - self.open_deg)
        self.bus.write("Goal_Position", "gripper", target)
        self._last_closedness = closedness
        print(f"[gripper] {self.name} -> {closedness:.2f} ({target:.1f} deg)")


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
    return Gripper(name, bus, open_deg, close_deg, _last_closedness=0.0)


def build_grippers(args: argparse.Namespace) -> dict[str, Gripper]:
    """Connect both hands, or none: a half-connected pair is worse than no hands at all.

    A missing gripper is not fatal -- the arms are the point and the CAN bus is the flakiest
    thing on the robot -- so this degrades to an empty dict rather than taking the run down.
    """
    grippers: dict[str, Gripper] = {}
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
        return {}
    return grippers


def cameras_from_args(
    args: argparse.Namespace,
) -> tuple[dict[str, OpenCVCameraConfig] | None, tuple[int, int] | None]:
    """Build the camera map and publish size from the CLI camera flags."""
    if not (args.camera or args.cameras):
        return None, None
    spec = args.cameras or f"head_camera={args.camera_device}"
    cameras = parse_camera_specs(spec, args.camera_fps, args.camera_width, args.camera_height)
    publish_size = None
    if args.camera_publish_size:
        width, height = (int(v) for v in args.camera_publish_size.lower().split("x"))
        publish_size = (width, height)
    return cameras, publish_size


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
            index_or_path=int(device) if device.lstrip("-").isdigit() else device,
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


def start_camera_server(
    cameras: dict[str, OpenCVCameraConfig],
    *,
    fps: int,
    port: int,
    publish_size: tuple[int, int] | None = None,
) -> tuple[ImageServer, threading.Thread]:
    """Launch the ZMQ ImageServer in a background daemon thread (independent of DDS).

    Returns the server alongside its thread: shutdown has to call ``stop()`` on it, since a
    daemon thread killed at exit leaves the V4L2 devices wedged until the next reboot.
    """
    server = ImageServer(cameras, fps=fps, port=port, publish_size=publish_size)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    summary = ", ".join(
        f"{name}({cfg.index_or_path} {cfg.width}x{cfg.height})" for name, cfg in cameras.items()
    )
    print(f"Camera server started on port {port}: {summary}")
    return server, thread


def serve_onboard_controller(
    *,
    controller: str,
    cameras: dict[str, OpenCVCameraConfig] | None = None,
    camera_fps: int = 30,
    camera_port: int = 5555,
    publish_size: tuple[int, int] | None = None,
    grippers: dict[str, Gripper] | None = None,
    action_port: int = ACTION_PORT,
    state_port: int = STATE_PORT,
    state_fps: float = 30.0,
    stop: threading.Event | None = None,
) -> None:
    """Run the controller ONBOARD -- the single control path on the robot.

    Builds ``UnitreeG1(onboard=True, controller=...)`` so its control loop runs locally
    against DDS at full rate (the ``_controller_loop`` thread lives in UnitreeG1), then
    receives compact high-level actions from the laptop over ZMQ (:action_port), feeds
    them to the controller, and publishes ``observation.state`` (:state_port). Camera
    frames are streamed separately by the ImageServer. The controller NEVER runs on the
    laptop; the laptop thin-client only ships arm targets / tokens and reads back state.
    """
    # Imported lazily to avoid importing the heavy controller stack until we serve.
    from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
    from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

    if stop is None:
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())

    # Optional camera server (background daemon thread; independent of DDS).
    camera_server = None
    if cameras:
        camera_server, _ = start_camera_server(
            cameras, fps=camera_fps, port=camera_port, publish_size=publish_size
        )

    # The hands are on CAN, not on lowcmd, so they keep their own listener here rather than
    # arriving through the action socket -- same split as in bridge mode.
    if grippers:
        threading.Thread(target=gripper_cmd_loop, args=(grippers, stop), daemon=True).start()

    cfg = UnitreeG1Config(is_simulation=False, onboard=True, controller=controller, cameras={})
    robot = UnitreeG1(cfg)
    print(f"Connecting onboard robot (controller={controller})...")
    robot.connect()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)  # only ever act on the freshest command
    sock.setsockopt(zmq.RCVTIMEO, 200)  # keeps the loop responsive to the stop event
    sock.bind(f"tcp://0.0.0.0:{action_port}")
    print(f"Onboard controller live. Waiting for laptop actions on :{action_port} ...")

    state_sock = None
    if state_fps > 0:
        state_sock = ctx.socket(zmq.PUB)
        state_sock.setsockopt(zmq.SNDHWM, 2)
        state_sock.setsockopt(zmq.LINGER, 0)
        state_sock.bind(f"tcp://0.0.0.0:{state_port}")
        print(f"Publishing observation.state on :{state_port} at {state_fps:.0f} Hz")

        def publish_state() -> None:
            period = 1.0 / state_fps
            while not stop.is_set():
                t0 = time.time()
                obs = robot.get_observation()
                if obs:
                    # Forward every scalar proprio key (joint .q, IMU, SONIC token echo).
                    # Camera arrays are streamed separately by the ImageServer.
                    state = {
                        k: float(v)
                        for k, v in obs.items()
                        if isinstance(v, (bool, int, float, np.floating, np.integer))
                    }
                    with contextlib.suppress(zmq.Again):
                        state_sock.send_json(state, zmq.NOBLOCK)
                time.sleep(max(0.0, period - (time.time() - t0)))

        threading.Thread(target=publish_state, daemon=True).start()

    try:
        while not stop.is_set():
            try:
                payload = sock.recv()
            except zmq.Again:
                continue
            except zmq.ContextTerminated:
                break
            try:
                action = json.loads(payload.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                print(f"Dropping malformed action: {e}")
                continue
            robot.send_action(action)
    finally:
        print("Shutting down onboard controller...")
        stop.set()
        if state_sock is not None:
            with contextlib.suppress(Exception):
                state_sock.close(linger=0)
        if camera_server is not None:
            camera_server.stop()
        robot.disconnect()


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
    """Receive gripper commands and apply them.

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
    """Main entry point for the robot server."""
    parser = argparse.ArgumentParser(description="DDS-to-ZMQ server for Unitree G1")
    parser.add_argument("--camera", action="store_true", help="Also launch camera server")
    parser.add_argument(
        "--camera-device",
        default="4",
        help="Camera index or V4L2 path, e.g. 4 or /dev/video0 (default: 4)",
    )
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
    # Onboard mode: run the controller on the robot instead of bridging raw lowcmd.
    parser.add_argument(
        "--onboard",
        action="store_true",
        help="Run the controller ONBOARD (requires --controller) instead of the raw DDS bridge",
    )
    parser.add_argument(
        "--controller",
        default=None,
        metavar="NAME",
        help="[--onboard] controller to run onboard, e.g. SonicLowerBodyController",
    )
    args = parser.parse_args()

    cameras, publish_size = cameras_from_args(args)

    # --- Onboard mode: controller runs on the robot; laptop ships high-level actions. ---
    if args.onboard:
        if not args.controller:
            parser.error("--onboard requires --controller (e.g. --controller SonicLowerBodyController)")
        serve_onboard_controller(
            controller=args.controller,
            cameras=cameras,
            camera_fps=args.camera_fps,
            camera_port=args.camera_port,
            publish_size=publish_size,
            grippers=build_grippers(args) if args.grippers else None,
        )
        return

    # --- Bridge mode (default): forward raw lowstate/lowcmd; controller runs on laptop. ---
    # Optionally start camera server in background thread
    camera_server = camera_thread = None
    if cameras:
        camera_server, camera_thread = start_camera_server(
            cameras, fps=args.camera_fps, port=args.camera_port, publish_size=publish_size
        )

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
    grippers: dict[str, Gripper] = build_grippers(args) if args.grippers else {}
    gripper_sock = None

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

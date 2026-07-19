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
from typing import TYPE_CHECKING, Any

import zmq
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.utils.crc import CRC

from lerobot.cameras.zmq.image_server import ImageServer

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
    send_id: int,
    recv_id: int,
    motor_type: str,
    use_can_fd: bool,
    open_deg: float,
    close_deg: float,
    kp: float,
    kd: float,
) -> Gripper:
    from lerobot.motors.damiao.damiao import DamiaoMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorNormMode

    motors = {
        "gripper": Motor(
            id=send_id,
            model=motor_type,
            norm_mode=MotorNormMode.DEGREES,
            motor_type_str=motor_type,
            recv_id=recv_id,
        )
    }
    bus = DamiaoMotorsBus(port=port, motors=motors, use_can_fd=use_can_fd)
    print(f"Connecting {name} gripper on {port} (fd={use_can_fd})...")
    bus.connect(handshake=True)
    bus.write("Kp", "gripper", kp)
    bus.write("Kd", "gripper", kd)
    bus.write("Goal_Position", "gripper", open_deg)  # start open
    print(f"  {name}: connected, torque enabled, opened.")
    return Gripper(name, bus, open_deg, close_deg, _last_cmd="open")


def parse_camera_specs(spec: str, default_width: int, default_height: int) -> dict[str, dict]:
    """Parse a multi-camera spec string into an ImageServer `cameras` dict.

    Format: comma-separated ``name:device[:WxH[:FOURCC]]`` entries, e.g.
    ``head_camera:4,left_wrist:0,right_wrist:1,ego:2``. ``device`` may be an
    integer index or an explicit device path (e.g. ``/dev/video4``); the path form
    is more reliable when the bare integer index fails to open. The optional ``WxH``
    overrides the default resolution (e.g. ``left_wrist:0:640x480``). The optional
    ``FOURCC`` forces a pixel format (e.g. ``head_camera:/dev/video8:1280x720:YUYV``),
    which some cameras (e.g. RealSense color nodes) require before the resolution
    can be applied.

    The device token may itself contain colons — notably stable ``by-path`` names
    like ``/dev/v4l/by-path/platform-3610000.xhci-usb-0:2.2:1.0-video-index0``,
    which survive USB re-enumeration/unplug (unlike bare ``/dev/videoN`` indices or
    ``by-id`` names that collide when two cameras share a serial). The optional
    ``WxH`` and ``FOURCC`` are therefore parsed from the *right* so the colons in
    the device path are preserved.
    """
    wh_re = re.compile(r"\d+x\d+", re.IGNORECASE)
    fourcc_re = re.compile(r"[A-Za-z0-9]{4}")

    cameras: dict[str, dict] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(f"Invalid camera spec '{entry}', expected 'name:device[:WxH[:FOURCC]]'")
        name, rest = entry.split(":", 1)
        name = name.strip()
        tokens = [t.strip() for t in rest.split(":")]

        # Peel optional FOURCC then WxH off the right. FOURCC only appears after a
        # WxH, so require that pairing to avoid mistaking a device-path segment for
        # a pixel format. Real device-path tail segments (e.g. "1.0-video-index0")
        # won't match these strict patterns.
        fourcc = None
        if (
            len(tokens) >= 3
            and wh_re.fullmatch(tokens[-2])
            and fourcc_re.fullmatch(tokens[-1])
        ):
            fourcc = tokens.pop().upper()
        width, height = default_width, default_height
        if len(tokens) >= 2 and wh_re.fullmatch(tokens[-1]):
            w, h = tokens.pop().lower().split("x")
            width, height = int(w), int(h)

        raw_id = ":".join(tokens).strip()
        if not raw_id:
            raise ValueError(f"Invalid camera spec '{entry}', missing device")
        # Accept either an integer index or an explicit device path (e.g. /dev/video4).
        device_id: int | str = int(raw_id) if raw_id.lstrip("-").isdigit() else raw_id
        if name in cameras:
            raise ValueError(f"Duplicate camera name '{name}' in --cameras")
        cameras[name] = {"device_id": device_id, "shape": [height, width], "fourcc": fourcc}
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
            "Multi-camera spec 'name:device_id[:WxH]' comma-separated, e.g. "
            "'head_camera:4,left_wrist:0,right_wrist:1,ego:2'. Overrides --camera-device "
            "and implies --camera. Per-camera resolution optional (defaults to "
            "--camera-width/--camera-height)."
        ),
    )
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera FPS (default: 30)")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument("--camera-port", type=int, default=5555, help="Camera ZMQ port (default: 5555)")
    # Gripper control from wireless-remote R3/L3
    parser.add_argument(
        "--grippers", action="store_true", help="Enable Damiao gripper control from wireless remote R3/L3"
    )
    parser.add_argument("--gripper-port-left", default="can1", help="CAN interface for LEFT gripper")
    parser.add_argument("--gripper-port-right", default="can0", help="CAN interface for RIGHT gripper")
    parser.add_argument("--gripper-send-id", type=lambda x: int(x, 0), default=0x08, help="Motor send CAN id")
    parser.add_argument("--gripper-recv-id", type=lambda x: int(x, 0), default=0x18, help="Motor recv CAN id")
    parser.add_argument("--gripper-motor-type", default="dm4310", help="Damiao motor type")
    parser.add_argument("--gripper-open-deg", type=float, default=-65.0, help="Gripper OPEN position (deg)")
    parser.add_argument("--gripper-close-deg", type=float, default=0.0, help="Gripper CLOSE position (deg)")
    parser.add_argument("--gripper-kp", type=float, default=15.0, help="MIT position gain (stiffness)")
    parser.add_argument("--gripper-kd", type=float, default=0.5, help="MIT damping gain")
    parser.add_argument(
        "--gripper-no-fd", dest="gripper_fd", action="store_false", help="Classic CAN (non-FD adapter)"
    )
    parser.set_defaults(gripper_fd=True)
    args = parser.parse_args()

    # Optionally start camera server in background thread
    camera_thread = None
    camera_server = None
    if args.camera or args.cameras:
        if args.cameras:
            cameras = parse_camera_specs(args.cameras, args.camera_width, args.camera_height)
        else:
            cameras = {
                "head_camera": {
                    "device_id": args.camera_device,
                    "shape": [args.camera_height, args.camera_width],
                }
            }
        camera_config = {"fps": args.camera_fps, "cameras": cameras}
        camera_server = ImageServer(camera_config, port=args.camera_port)
        camera_thread = threading.Thread(target=camera_server.run, daemon=True)
        camera_thread.start()
        cam_summary = ", ".join(f"{name}(dev {c['device_id']})" for name, c in cameras.items())
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
            grippers["L"] = build_gripper(
                "L",
                args.gripper_port_left,
                args.gripper_send_id,
                args.gripper_recv_id,
                args.gripper_motor_type,
                args.gripper_fd,
                args.gripper_open_deg,
                args.gripper_close_deg,
                args.gripper_kp,
                args.gripper_kd,
            )
            grippers["R"] = build_gripper(
                "R",
                args.gripper_port_right,
                args.gripper_send_id,
                args.gripper_recv_id,
                args.gripper_motor_type,
                args.gripper_fd,
                args.gripper_open_deg,
                args.gripper_close_deg,
                args.gripper_kp,
                args.gripper_kd,
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

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
from typing import Any

import numpy as np
import zmq
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.utils.crc import CRC

from lerobot.cameras.zmq.image_server import ImageServer

# DDS topic names follow Unitree SDK naming conventions
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"  # action to robot
kTopicLowState = "rt/lowstate"  # observation from robot

LOWCMD_PORT = 6000
LOWSTATE_PORT = 6001
NUM_MOTORS = 35

# Onboard high-level channels (serve_onboard_controller): compact actions in, state out.
ACTION_PORT = 6004
STATE_PORT = 6005


def cameras_from_args(args: argparse.Namespace) -> dict:
    """Build the ImageServer camera map from the CLI camera flags.

    ``--camera-device`` takes an index (``4``) or a V4L2 path (``/dev/video0``). Some
    devices only open by path, so the path form is not just a convenience.
    """
    device = args.camera_device
    return {
        "head_camera": {
            "device_id": int(device) if str(device).isdigit() else device,
            "shape": [args.camera_height, args.camera_width],
        }
    }


def start_camera_server(cameras: dict, *, fps: int, port: int) -> threading.Thread:
    """Launch the ZMQ ImageServer in a background daemon thread (independent of DDS)."""
    server = ImageServer({"fps": fps, "cameras": cameras}, port=port)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    summary = ", ".join(f"{name}(dev {c['device_id']})" for name, c in cameras.items())
    print(f"Camera server started on :{port}: {summary}")
    return thread


def serve_onboard_controller(
    *,
    controller: str,
    cameras: dict | None = None,
    camera_fps: int = 30,
    camera_port: int = 5555,
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
    laptop; the laptop thin-client only ships tokens/axes and reads back state + frames.
    """
    # Imported lazily to avoid importing the heavy controller stack until we serve.
    from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
    from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

    if stop is None:
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())

    # Optional camera server (background daemon thread; independent of DDS).
    if cameras:
        start_camera_server(cameras, fps=camera_fps, port=camera_port)

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
            # if no subscribers / tx buffer full, just drop
            with contextlib.suppress(zmq.Again):
                lowstate_sock.send(payload, zmq.NOBLOCK)
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
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera FPS (default: 30)")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument("--camera-port", type=int, default=5555, help="Camera ZMQ port (default: 5555)")
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
        help="[--onboard] controller to run onboard, e.g. SonicWholeBodyController",
    )
    args = parser.parse_args()

    # --- Onboard mode: controller runs on the robot; laptop ships high-level actions. ---
    if args.onboard:
        if not args.controller:
            parser.error("--onboard requires --controller (e.g. --controller SonicWholeBodyController)")
        cameras = cameras_from_args(args) if args.camera else None
        serve_onboard_controller(
            controller=args.controller,
            cameras=cameras,
            camera_fps=args.camera_fps,
            camera_port=args.camera_port,
        )
        return

    # --- Bridge mode (default): forward raw lowstate/lowcmd; controller runs on laptop. ---
    # Optionally start camera server in background thread
    camera_thread = None
    if args.camera:
        camera_thread = start_camera_server(
            cameras_from_args(args), fps=args.camera_fps, port=args.camera_port
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

    state_period = 0.002  # ~500 hz
    shutdown_event = threading.Event()

    # start observation forwarding in background thread
    t_state = threading.Thread(
        target=state_forward_loop,
        args=(lowstate_sub, lowstate_sock, state_period, shutdown_event),
    )
    t_state.start()

    print("bridge running (lowstate -> zmq, lowcmd -> dds)")

    # run command forwarding in main thread
    try:
        cmd_forward_loop(lowcmd_sock, lowcmd_pub_debug, crc)
    except KeyboardInterrupt:
        print("shutting down bridge...")
    finally:
        shutdown_event.set()
        ctx.term()  # terminates blocking zmq.recv() calls
        t_state.join(timeout=2.0)
        if camera_thread is not None:
            camera_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()

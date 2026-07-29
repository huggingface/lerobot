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

Controller-negotiation handshake
--------------------------------
The first message from a client agrees on which controller the server will run onboard
(``serve_onboard_controller``); the controller NEVER runs on the laptop client.
Test the handshake in isolation (no DDS, runs on a laptop) in two terminals::

    # terminal A: handshake-only server
    python -m lerobot.robots.unitree_g1.run_g1_server --handshake-only

    # terminal B: client proposes a controller
    python -m lerobot.robots.unitree_g1.run_g1_server \\
        --handshake-client SonicWholeBodyController --sonic-token-action --server-ip 127.0.0.1

On the real robot, add ``--handshake`` to the normal bridge to require agreement first.
"""

import argparse
import base64
import contextlib
import json
import os
import re
import signal
import sys
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

# Controller-negotiation handshake (REQ/REP). The client's first message agrees on
# which controller the server will run before any control data flows.
HANDSHAKE_PORT = 6002
PROTOCOL_VERSION = 1

# Controllers that can run ONBOARD (must match g1_utils.make_locomotion_controller).
# ``None`` (a.k.a. "bridge") means no onboard controller: the laptop owns control and
# streams raw lowcmd over the ZMQ DDS bridge (the legacy run_g1_server behavior).
VALID_CONTROLLERS = (
    "GrootLocomotionController",
    "HolosomaLocomotionController",
    "SonicWholeBodyController",
)
# SONIC latent-token dimensionality (mirrors sonic_whole_body.TOKEN_DIM; kept local so
# the handshake can run without importing the heavy controller / onnxruntime).
TOKEN_DIM = 64
_BRIDGE_ALIASES = {"", "none", "null", "bridge", "raw"}


def _normalize_controller(name: str | None) -> str | None:
    """Map a requested controller name to a canonical value (or None for raw bridge)."""
    if name is None:
        return None
    low = str(name).strip().lower()
    if low in _BRIDGE_ALIASES:
        return None
    for c in VALID_CONTROLLERS:
        if c.lower() == low:
            return c
    raise ValueError(f"Unknown controller {name!r}. Available: {list(VALID_CONTROLLERS)} or 'bridge'")


def _capabilities(controller: str | None, sonic_token_action: bool) -> dict[str, Any]:
    """The interface the server advertises for an agreed controller."""
    caps: dict[str, Any] = {
        "controller": controller,
        "sonic_token_action": bool(sonic_token_action),
        "protocol": PROTOCOL_VERSION,
    }
    if controller is None:
        # Raw DDS bridge: the laptop runs the controller and streams lowcmd.
        caps["mode"] = "bridge"
        caps["lowcmd_port"] = LOWCMD_PORT
        caps["lowstate_port"] = LOWSTATE_PORT
    else:
        # Onboard: the controller runs here; the laptop ships compact high-level actions.
        caps["mode"] = "onboard"
        caps["action_port"] = ACTION_PORT
        caps["state_port"] = STATE_PORT
        if sonic_token_action:
            caps["action_space"] = "motion_token"
            caps["action_dim"] = TOKEN_DIM
    return caps


def negotiate_controller(sock: zmq.Socket, shutdown_event: threading.Event) -> dict[str, Any]:
    """Server side of the handshake: block on one REP socket until a client sends a
    valid ``hello``, then reply with the negotiated capabilities and return them.

    Rejects malformed / unknown-controller requests with an error reply and keeps
    waiting (a rejected client can retry). Honors ``shutdown_event`` so Ctrl-C works.
    """
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    while not shutdown_event.is_set():
        if not dict(poller.poll(timeout=200)):
            continue
        raw = sock.recv()
        try:
            hello = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            sock.send_json({"type": "error", "ok": False, "error": f"bad hello: {e}"})
            continue
        try:
            controller = _normalize_controller(hello.get("controller"))
        except ValueError as e:
            sock.send_json(
                {"type": "error", "ok": False, "error": str(e), "available": list(VALID_CONTROLLERS)}
            )
            continue
        reply = {"type": "welcome", "ok": True, **_capabilities(controller, hello.get("sonic_token_action", False))}
        sock.send_json(reply)
        return reply
    raise KeyboardInterrupt


def request_controller(
    server_ip: str,
    controller: str | None,
    *,
    sonic_token_action: bool = False,
    port: int = HANDSHAKE_PORT,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Client side of the handshake: propose a controller, return the server's agreed
    capabilities (or raise on rejection / timeout)."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    sock.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
    sock.connect(f"tcp://{server_ip}:{port}")
    hello = {
        "type": "hello",
        "controller": controller,
        "sonic_token_action": bool(sonic_token_action),
        "protocol": PROTOCOL_VERSION,
    }
    try:
        sock.send_json(hello)
        reply = sock.recv_json()
    except zmq.Again as e:
        raise TimeoutError(f"no handshake reply from {server_ip}:{port} within {timeout_s}s") from e
    finally:
        sock.close(linger=0)
    if not reply.get("ok"):
        raise RuntimeError(f"handshake rejected: {reply.get('error')} (available: {reply.get('available')})")
    return reply


def serve_onboard_controller(
    *,
    controller: str,
    sonic_token_action: bool,
    dds_interface: str | None = None,
    sim: bool = False,
    cameras: dict | None = None,
    camera_fps: int = 30,
    camera_port: int = 5555,
    action_port: int = ACTION_PORT,
    state_port: int = STATE_PORT,
    state_fps: float = 30.0,
    stop: threading.Event | None = None,
) -> None:
    """Run the negotiated controller ONBOARD -- the single control path on the robot.

    Builds ``UnitreeG1(onboard=True, controller=...)`` so the controller/balance loop runs
    locally against DDS at full rate (the 50 Hz ``_controller_loop`` thread lives in
    UnitreeG1), then receives compact high-level actions from the laptop over ZMQ
    (:action_port), decodes them via the controller, publishes ``observation.state``
    (:state_port), and optionally serves the ego camera. The controller NEVER runs on the
    laptop; the laptop (lerobot-rollout thin-client) only ships tokens/axes and reads back
    state + camera frames.
    """
    # Imported lazily: UnitreeG1 imports request_controller from this module, so a
    # top-level import here would be circular.
    from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
    from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

    if stop is None:
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())

    cfg = UnitreeG1Config(
        is_simulation=False,
        onboard=True,
        controller=controller,
        dds_interface=dds_interface,
        release_motion_control=not sim,
        physical_remote=not sim,
        sonic_token_action=sonic_token_action,
        cameras={},
    )

    # Optional camera server (background thread; independent of DDS).
    camera_server = None
    if cameras:
        camera_server = ImageServer({"fps": camera_fps, "cameras": cameras}, port=camera_port)
        threading.Thread(target=camera_server.run, daemon=True).start()
        cam_summary = ", ".join(f"{name}(dev {c['device_id']})" for name, c in cameras.items())
        print(f"Camera server started on :{camera_port}: {cam_summary}")

    robot = UnitreeG1(cfg)
    print(f"Connecting onboard robot (controller={controller}, token={sonic_token_action})...")
    robot.connect()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)  # only ever act on the freshest command
    sock.setsockopt(zmq.RCVTIMEO, 200)  # keeps the loop responsive to the stop event
    sock.bind(f"tcp://0.0.0.0:{action_port}")
    print(f"Onboard controller live. Waiting for laptop actions on :{action_port} ...")
    print("Type 'e' then Enter to STOP immediately (or Ctrl-C for graceful shutdown).")

    def estop_listener() -> None:
        for line in sys.stdin:
            if line.strip().lower() == "e":
                print("E-STOP ('e'): going passive NOW.")
                try:
                    robot._shutdown_event.set()  # stop the controller loop publishing
                    time.sleep(0.05)
                    robot._send_zero_torque()  # motors limp; nothing overwrites it now
                except Exception as e:  # noqa: BLE001
                    print(f"E-stop zero-torque failed: {e}")
                os._exit(0)  # immediate hard exit, no slow cleanup

    threading.Thread(target=estop_listener, daemon=True).start()

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
                    # Forward every scalar proprio key the robot exposes (29 joint .q, IMU,
                    # and the SONIC token echo: 64-D motion_token_state.*). Camera arrays are
                    # streamed separately by the ImageServer, so drop ndarrays here. This
                    # makes the laptop thin-client a pure relay.
                    state = {
                        k: float(v)
                        for k, v in obs.items()
                        if isinstance(v, (bool, int, float, np.floating, np.integer))
                    }
                    with contextlib.suppress(zmq.Again):
                        state_sock.send_json(state, zmq.NOBLOCK)
                time.sleep(max(0.0, period - (time.time() - t0)))

        threading.Thread(target=publish_state, daemon=True).start()
    else:
        print("observation.state PUB disabled (state_fps<=0)")

    n = 0
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

            n += 1
            if n % 60 == 0:
                print(f"Applied {n} actions")
    finally:
        print("Shutting down onboard controller...")
        stop.set()
        if state_sock is not None:
            with contextlib.suppress(Exception):
                state_sock.close(linger=0)
        if camera_server is not None:
            with contextlib.suppress(Exception):
                camera_server.stop()
        robot.disconnect()


def parse_camera_specs(spec: str, default_width: int, default_height: int) -> dict[str, dict]:
    """Parse a multi-camera spec string into an ImageServer ``cameras`` dict.

    Format: comma-separated ``name:device[:WxH[:FOURCC]]`` entries, e.g.
    ``head_camera:6,left_wrist:0``. ``device`` may be an integer index or an explicit
    device path (e.g. ``/dev/video6``), including stable ``by-path`` names like
    ``/dev/v4l/by-path/platform-...:2.1:1.3-video-index0`` which survive USB
    re-enumeration (unlike bare ``/dev/videoN`` indices). Because a by-path name
    itself contains colons, the optional ``WxH`` and ``FOURCC`` are parsed from the
    *right* so the device-path colons are preserved.
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

        fourcc = None
        if len(tokens) >= 3 and wh_re.fullmatch(tokens[-2]) and fourcc_re.fullmatch(tokens[-1]):
            fourcc = tokens.pop().upper()
        width, height = default_width, default_height
        if len(tokens) >= 2 and wh_re.fullmatch(tokens[-1]):
            w, h = tokens.pop().lower().split("x")
            width, height = int(w), int(h)

        raw_id = ":".join(tokens).strip()
        if not raw_id:
            raise ValueError(f"Invalid camera spec '{entry}', missing device")
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
    """Main entry point for the robot server bridge."""
    parser = argparse.ArgumentParser(description="DDS-to-ZMQ bridge server for Unitree G1")
    parser.add_argument("--camera", action="store_true", help="Also launch camera server")
    parser.add_argument("--camera-device", default="4",
                        help="Camera device: index or /dev/video path or by-path name (default: 4)")
    parser.add_argument("--cameras", default=None,
                        help="Multi-camera spec 'name:device[:WxH[:FOURCC]]', comma-separated. Overrides "
                             "--camera-device; device may be a by-path name to survive USB re-enumeration.")
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera FPS (default: 30)")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument("--camera-port", type=int, default=5555, help="Camera ZMQ port (default: 5555)")
    # Controller-negotiation handshake (first message agrees on the controller).
    parser.add_argument("--handshake", action="store_true",
                        help="Wait for a client to negotiate the controller before bridging")
    parser.add_argument("--handshake-port", type=int, default=HANDSHAKE_PORT,
                        help=f"Handshake REQ/REP port (default: {HANDSHAKE_PORT})")
    parser.add_argument("--handshake-only", action="store_true",
                        help="Run ONLY the handshake server (no DDS/cameras) to test negotiation")
    parser.add_argument("--handshake-client", default=None, metavar="CONTROLLER",
                        help="Act as a client: propose CONTROLLER (or 'bridge') to --server-ip and print the reply")
    parser.add_argument("--server-ip", default="127.0.0.1", help="[--handshake-client] server IP")
    parser.add_argument("--sonic-token-action", action="store_true",
                        help="[handshake] negotiate the 64-D SONIC token action interface")
    args = parser.parse_args()

    # --- Isolated handshake test paths (no DDS, safe to run on a laptop) ---
    if args.handshake_client is not None:
        controller = None if args.handshake_client.strip().lower() in _BRIDGE_ALIASES else args.handshake_client
        reply = request_controller(
            args.server_ip, controller,
            sonic_token_action=args.sonic_token_action, port=args.handshake_port,
        )
        print(json.dumps(reply, indent=2))
        return

    if args.handshake_only:
        ctx = zmq.Context.instance()
        rep = ctx.socket(zmq.REP)
        rep.bind(f"tcp://0.0.0.0:{args.handshake_port}")
        print(f"[handshake] server listening on :{args.handshake_port} (no DDS). Ctrl-C to stop.")
        shutdown = threading.Event()
        try:
            while True:
                reply = negotiate_controller(rep, shutdown)
                print(f"[handshake] agreed: controller={reply['controller']} mode={reply['mode']} "
                      f"sonic_token_action={reply['sonic_token_action']}")
        except KeyboardInterrupt:
            print("\n[handshake] stopping")
        finally:
            rep.close(linger=0)
            ctx.term()
        return

    # Controller-negotiation handshake: the client's first message agrees on the
    # controller, which we then run ONBOARD (the controller NEVER runs on the laptop).
    # Bridge/None falls through to the legacy raw DDS forward (deprecated laptop control).
    if args.handshake:
        ctx = zmq.Context.instance()
        hs = ctx.socket(zmq.REP)
        hs.bind(f"tcp://0.0.0.0:{args.handshake_port}")
        print(f"[handshake] waiting for client controller agreement on :{args.handshake_port} ...")
        shutdown = threading.Event()
        try:
            agreed = negotiate_controller(hs, shutdown)
        except KeyboardInterrupt:
            print("[handshake] interrupted before agreement; exiting")
            hs.close(linger=0)
            ctx.term()
            return
        hs.close(linger=0)
        if agreed["controller"] is not None:
            print(f"[handshake] running controller ONBOARD: {agreed['controller']} "
                  f"(sonic_token_action={agreed['sonic_token_action']})")
            cameras = None
            if args.camera or args.cameras:
                if args.cameras:
                    cameras = parse_camera_specs(args.cameras, args.camera_width, args.camera_height)
                else:
                    dev = args.camera_device
                    dev = int(dev) if str(dev).lstrip("-").isdigit() else dev
                    cameras = {"head_camera": {"device_id": dev, "shape": [args.camera_height, args.camera_width]}}
            serve_onboard_controller(
                controller=agreed["controller"],
                sonic_token_action=bool(agreed["sonic_token_action"]),
                cameras=cameras,
                camera_fps=args.camera_fps,
                camera_port=args.camera_port,
            )
            return
        print("[handshake] client selected raw DDS bridge (laptop owns control) -> legacy forward.")

    # Optionally start camera server in background thread
    camera_thread = None
    if args.camera or args.cameras:
        if args.cameras:
            cameras = parse_camera_specs(args.cameras, args.camera_width, args.camera_height)
        else:
            # Single camera; accept an int index or a device/by-path string.
            dev = args.camera_device
            dev = int(dev) if str(dev).lstrip("-").isdigit() else dev
            cameras = {"head_camera": {"device_id": dev, "shape": [args.camera_height, args.camera_width]}}
        camera_config = {"fps": args.camera_fps, "cameras": cameras}
        camera_server = ImageServer(camera_config, port=args.camera_port)
        camera_thread = threading.Thread(target=camera_server.run, daemon=True)
        camera_thread.start()
        cam_summary = ", ".join(f"{n}(dev {c['device_id']})" for n, c in cameras.items())
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
    shutdown_event = threading.Event()

    # receive commands from remote client
    lowcmd_sock = ctx.socket(zmq.PULL)
    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")

    # publish state to remote clients
    lowstate_sock = ctx.socket(zmq.PUB)
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")

    state_period = 0.002  # ~500 hz

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

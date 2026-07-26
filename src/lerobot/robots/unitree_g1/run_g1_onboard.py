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

"""Run the G1 locomotion / whole-body controller ONBOARD, driven by high-level actions
from a laptop.

The controller (GR00T / Holosoma / SONIC whole-body) runs on the robot itself against
local DDS, at full control rate. The laptop ships only the resulting high-level action
(arm joint targets + joystick axes + gripper flags, or a 64-D SONIC motion token) as
JSON over ZMQ. This process applies each action via ``UnitreeG1.send_action`` while the
onboard controller thread keeps the legs balanced / decodes the token.

This is the real-deploy counterpart to running ``lerobot-rollout`` on the laptop with
``--robot.is_simulation=false`` (the ZMQ *socket bridge*): there the 50 Hz lowcmd
crosses the network; here only compact high-level actions do, and the control loop stays
local to the robot. Pair with a laptop client that produces actions (exo teleop, or a
policy such as ``nepyope/sonic_walk`` emitting ``motion_token.{i}.pos``).

Besides receiving actions, this process publishes ``observation.state`` (29 joint ``.q``)
on a ZMQ PUB port so a laptop policy client has proprioception.

Safety: type ``e`` then Enter in this terminal to stop immediately (zero-torque + exit).
Ctrl-C does the normal graceful shutdown (kp ramp).

Examples (on the robot):

    # GR00T locomotion, arm targets from the laptop:
    python -m lerobot.robots.unitree_g1.run_g1_onboard --controller GrootLocomotionController

    # SONIC whole-body walk policy: laptop ships 64-D tokens, decoder runs here:
    python -m lerobot.robots.unitree_g1.run_g1_onboard \
        --controller SonicWholeBodyController --sonic-token-action \
        --cameras "head_camera:/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.3-video-index0:640x480"
"""

import argparse
import contextlib
import json
import logging
import os
import signal
import sys
import threading
import time

import numpy as np
import zmq

from lerobot.cameras.zmq.image_server import ImageServer
from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
from lerobot.robots.unitree_g1.run_g1_server import Gripper, build_gripper, parse_camera_specs
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("g1_onboard")

ACTION_PORT = 6004
STATE_PORT = 6005


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", default="GrootLocomotionController", help="Controller class name")
    p.add_argument("--dds-interface", default=None, help="DDS network interface (default: SDK default)")
    p.add_argument(
        "--sim",
        action="store_true",
        help="Attach to a DDS MuJoCo sim: skip MotionSwitcher + physical remote, default dds-interface 'lo'.",
    )
    p.add_argument(
        "--sonic-token-action",
        action="store_true",
        help="SONIC token interface: actions carry a 64-D motion_token.{i}.pos that the decoder consumes.",
    )
    p.add_argument("--action-port", type=int, default=ACTION_PORT, help="ZMQ PULL port for laptop actions")
    p.add_argument("--state-port", type=int, default=STATE_PORT, help="ZMQ PUB port for observation.state")
    p.add_argument("--state-fps", type=float, default=30.0, help="observation.state publish rate; <=0 disables")
    p.add_argument("--gravity-compensation", action="store_true", help="Enable arm gravity compensation")
    # Gripper control (Damiao over CAN).
    p.add_argument("--grippers", action="store_true", help="Drive Damiao grippers from action L3/R3 flags")
    p.add_argument("--gripper-port-left", default="can1", help="CAN interface for LEFT gripper")
    p.add_argument("--gripper-port-right", default="can0", help="CAN interface for RIGHT gripper")
    p.add_argument("--gripper-send-id", type=lambda x: int(x, 0), default=0x08, help="Motor send CAN id")
    p.add_argument("--gripper-recv-id", type=lambda x: int(x, 0), default=0x18, help="Motor recv CAN id")
    p.add_argument("--gripper-motor-type", default="dm4310", help="Damiao motor type")
    p.add_argument("--gripper-open-deg", type=float, default=-65.0, help="Gripper OPEN position (deg)")
    p.add_argument("--gripper-close-deg", type=float, default=0.0, help="Gripper CLOSE position (deg)")
    p.add_argument("--gripper-kp", type=float, default=15.0, help="MIT position gain (stiffness)")
    p.add_argument("--gripper-kd", type=float, default=0.5, help="MIT damping gain")
    p.add_argument("--gripper-no-fd", dest="gripper_fd", action="store_false", help="Classic CAN (non-FD)")
    p.set_defaults(gripper_fd=True)
    # Optional camera streaming (ZMQ) so the laptop policy client / viewer can connect.
    p.add_argument("--cameras", default=None, help="Camera spec 'name:device[:WxH[:FOURCC]]', comma-sep")
    p.add_argument("--camera-fps", type=int, default=30, help="Camera FPS")
    p.add_argument("--camera-port", type=int, default=5555, help="Camera ZMQ port")
    p.add_argument("--camera-width", type=int, default=640, help="Default camera width")
    p.add_argument("--camera-height", type=int, default=480, help="Default camera height")
    args = p.parse_args()

    dds_interface = args.dds_interface
    if args.sim and dds_interface is None:
        dds_interface = "lo"

    cfg = UnitreeG1Config(
        is_simulation=False,
        onboard=True,
        controller=args.controller,
        dds_interface=dds_interface,
        gravity_compensation=args.gravity_compensation,
        release_motion_control=not args.sim,
        physical_remote=not args.sim,
        sonic_token_action=args.sonic_token_action,
        cameras={},
    )

    # Optional camera server (background thread; independent of DDS/CAN).
    camera_server = None
    if args.cameras:
        cameras = parse_camera_specs(args.cameras, args.camera_width, args.camera_height)
        camera_server = ImageServer({"fps": args.camera_fps, "cameras": cameras}, port=args.camera_port)
        threading.Thread(target=camera_server.run, daemon=True).start()
        cam_summary = ", ".join(f"{name}(dev {c['device_id']})" for name, c in cameras.items())
        logger.info("Camera server started on :%d: %s", args.camera_port, cam_summary)

    robot = UnitreeG1(cfg)
    logger.info("Connecting onboard robot (controller=%s, token=%s)...", args.controller, args.sonic_token_action)
    robot.connect()

    grippers: dict[str, Gripper] = {}
    if args.grippers:
        for side, port in (("L", args.gripper_port_left), ("R", args.gripper_port_right)):
            grippers[side] = build_gripper(
                side, port, args.gripper_send_id, args.gripper_recv_id, args.gripper_motor_type,
                args.gripper_fd, args.gripper_open_deg, args.gripper_close_deg, args.gripper_kp, args.gripper_kd,
            )
        logger.info("Grippers enabled: L3 -> left, R3 -> right")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)  # only ever act on the freshest command
    sock.setsockopt(zmq.RCVTIMEO, 200)  # keeps the loop responsive to the stop event
    sock.bind(f"tcp://0.0.0.0:{args.action_port}")
    logger.info("Onboard controller live. Waiting for laptop actions on :%d ...", args.action_port)
    logger.info("Type 'e' then Enter to STOP immediately (or Ctrl-C for graceful shutdown).")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def estop_listener() -> None:
        for line in sys.stdin:
            if line.strip().lower() == "e":
                logger.warning("E-STOP ('e'): going passive NOW.")
                try:
                    robot._shutdown_event.set()  # stop the controller loop publishing
                    time.sleep(0.05)
                    robot._send_zero_torque()  # motors limp; nothing overwrites it now
                except Exception as e:  # noqa: BLE001
                    logger.warning("E-stop zero-torque failed: %s", e)
                os._exit(0)  # immediate hard exit, no slow cleanup

    threading.Thread(target=estop_listener, daemon=True).start()

    # Proprioception feedback: publish observation.state (29 joint .q) so a laptop
    # inference client can feed it to a policy. DDS stays local; only compact JSON
    # state crosses the network. (For a token policy the laptop closes the loop on the
    # token instead, but publishing joint state is harmless and useful for logging.)
    state_sock = None
    if args.state_fps > 0:
        state_sock = ctx.socket(zmq.PUB)
        state_sock.setsockopt(zmq.SNDHWM, 2)
        state_sock.setsockopt(zmq.LINGER, 0)
        state_sock.bind(f"tcp://0.0.0.0:{args.state_port}")
        logger.info("Publishing observation.state on :%d at %.0f Hz", args.state_port, args.state_fps)

        def publish_state() -> None:
            period = 1.0 / args.state_fps
            joint_names = [j.name for j in G1_29_JointIndex]
            while not stop.is_set():
                t0 = time.time()
                obs = robot.get_observation()
                if obs:
                    state = {f"{name}.q": float(obs.get(f"{name}.q", 0.0)) for name in joint_names}
                    with contextlib.suppress(zmq.Again):
                        state_sock.send_json(state, zmq.NOBLOCK)
                time.sleep(max(0.0, period - (time.time() - t0)))

        threading.Thread(target=publish_state, daemon=True).start()
    else:
        logger.info("observation.state PUB disabled (--state-fps<=0)")

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
                logger.warning("Dropping malformed action: %s", e)
                continue

            robot.send_action(action)

            if grippers:
                # L3 = remote.button.4 -> left, R3 = remote.button.0 -> right.
                if "L" in grippers and "remote.button.4" in action:
                    grippers["L"].apply(bool(action["remote.button.4"]))
                if "R" in grippers and "remote.button.0" in action:
                    grippers["R"].apply(bool(action["remote.button.0"]))

            n += 1
            if n % 60 == 0:
                axes = {k: round(float(action.get(k, 0.0)), 3) for k in ("remote.lx", "remote.ly", "remote.rx", "remote.ry")}
                logger.info("Applied %d actions | axes=%s", n, axes)
    finally:
        logger.info("Shutting down onboard controller...")
        stop.set()
        if state_sock is not None:
            with contextlib.suppress(Exception):
                state_sock.close(linger=0)
        if camera_server is not None:
            with contextlib.suppress(Exception):
                camera_server.stop()
        for g in grippers.values():
            with contextlib.suppress(Exception):
                g.bus.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()

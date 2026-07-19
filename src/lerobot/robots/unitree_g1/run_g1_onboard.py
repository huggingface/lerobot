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

"""Run the G1 locomotion controller ONBOARD, driven by high-level actions from a laptop.

The locomotion policy (GRoot / Holosoma) runs on the robot itself against local DDS, at
full control rate. The laptop reads the exos, runs IK, and ships only the resulting
action (arm joint targets + joystick axes + gripper flags) as JSON over ZMQ. This process
applies each action via ``UnitreeG1.send_action`` while the onboard controller thread
keeps the legs balanced.

Pair with ``run_g1_teleop_client.py`` (exo teleop) or ``infer_pi05_g1_onboard.py`` (policy
eval) on the laptop. Grippers (exo L3/R3 or policy flags) are driven directly over CAN here
when ``--grippers`` is passed; cameras are served over ZMQ when ``--cameras`` is passed.

Besides receiving actions, this process also publishes ``observation.state`` (29 joint ``.q``)
on a ZMQ PUB port so the laptop policy client has proprioception, and applies absolute base
height / torso orientation from the action dict (``groot.height`` / ``groot.rpy.*``) onto the
controller (the joystick interface can only nudge height, not set it absolutely).

Examples (on the robot):

    python -m lerobot.robots.unitree_g1.run_g1_onboard --controller GrootLocomotionController

    # with grippers (bring CAN up first, e.g. lerobot-setup-can --mode=setup --interfaces=can0,can1)
    python -m lerobot.robots.unitree_g1.run_g1_onboard --controller GrootLocomotionController \
        --grippers --gripper-port-left can1 --gripper-port-right can0
"""

import argparse
import json
import logging
import signal
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
    p.add_argument("--controller", default="GrootLocomotionController", help="Locomotion controller class")
    p.add_argument("--dds-interface", default=None, help="DDS network interface (default: SDK default)")
    p.add_argument("--action-port", type=int, default=ACTION_PORT, help="ZMQ port for laptop actions")
    p.add_argument(
        "--state-port",
        type=int,
        default=STATE_PORT,
        help="ZMQ PUB port for observation.state feedback (29 joint .q) to the inference client",
    )
    p.add_argument("--state-fps", type=float, default=60.0, help="observation.state publish rate (Hz)")
    p.add_argument(
        "--gravity-compensation",
        action="store_true",
        help="Enable arm gravity compensation (needs pinocchio/casadi on the robot)",
    )
    # Gripper control from exo L3/R3 (same wiring as run_g1_server.py).
    p.add_argument("--grippers", action="store_true", help="Drive Damiao grippers from exo L3/R3")
    p.add_argument("--gripper-port-left", default="can1", help="CAN interface for LEFT gripper")
    p.add_argument("--gripper-port-right", default="can0", help="CAN interface for RIGHT gripper")
    p.add_argument("--gripper-send-id", type=lambda x: int(x, 0), default=0x08, help="Motor send CAN id")
    p.add_argument("--gripper-recv-id", type=lambda x: int(x, 0), default=0x18, help="Motor recv CAN id")
    p.add_argument("--gripper-motor-type", default="dm4310", help="Damiao motor type")
    p.add_argument("--gripper-open-deg", type=float, default=-65.0, help="Gripper OPEN position (deg)")
    p.add_argument("--gripper-close-deg", type=float, default=0.0, help="Gripper CLOSE position (deg)")
    p.add_argument("--gripper-kp", type=float, default=15.0, help="MIT position gain (stiffness)")
    p.add_argument("--gripper-kd", type=float, default=0.5, help="MIT damping gain")
    p.add_argument(
        "--gripper-no-fd", dest="gripper_fd", action="store_false", help="Classic CAN (non-FD adapter)"
    )
    p.set_defaults(gripper_fd=True)
    # Optional camera streaming (so view_cameras.py can connect). Same spec format as
    # run_g1_server.py: 'name:device[:WxH[:FOURCC]]' comma-separated.
    p.add_argument("--cameras", default=None, help="Camera spec 'name:device[:WxH[:FOURCC]]', comma-sep")
    p.add_argument("--camera-fps", type=int, default=30, help="Camera FPS")
    p.add_argument("--camera-port", type=int, default=5555, help="Camera ZMQ port")
    p.add_argument("--camera-width", type=int, default=640, help="Default camera width")
    p.add_argument("--camera-height", type=int, default=480, help="Default camera height")
    args = p.parse_args()

    cfg = UnitreeG1Config(
        is_simulation=False,
        onboard=True,
        controller=args.controller,
        dds_interface=args.dds_interface,
        gravity_compensation=args.gravity_compensation,
        cameras={},
    )
    # Optional camera server (for view_cameras.py). Runs in a background thread and is
    # independent of DDS/CAN, so it coexists with the onboard controller and grippers.
    camera_server = None
    if args.cameras:
        cameras = parse_camera_specs(args.cameras, args.camera_width, args.camera_height)
        camera_server = ImageServer({"fps": args.camera_fps, "cameras": cameras}, port=args.camera_port)
        threading.Thread(target=camera_server.run, daemon=True).start()
        cam_summary = ", ".join(f"{name}(dev {c['device_id']})" for name, c in cameras.items())
        logger.info("Camera server started on :%d: %s", args.camera_port, cam_summary)

    robot = UnitreeG1(cfg)
    logger.info("Connecting onboard robot (controller=%s)...", args.controller)
    robot.connect()

    # Grippers: driven directly over CAN from the exo L3/R3 flags in each action
    # (L3 = remote.button.4 -> left, R3 = remote.button.0 -> right; pressed = close).
    grippers: dict[str, Gripper] = {}
    if args.grippers:
        for side, port in (("L", args.gripper_port_left), ("R", args.gripper_port_right)):
            grippers[side] = build_gripper(
                side,
                port,
                args.gripper_send_id,
                args.gripper_recv_id,
                args.gripper_motor_type,
                args.gripper_fd,
                args.gripper_open_deg,
                args.gripper_close_deg,
                args.gripper_kp,
                args.gripper_kd,
            )
        logger.info("Grippers enabled: L3 -> left, R3 -> right")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)  # only ever act on the freshest command
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.bind(f"tcp://0.0.0.0:{args.action_port}")
    logger.info("Onboard controller live. Waiting for laptop actions on :%d ...", args.action_port)

    # Proprioception feedback: publish observation.state (29 joint .q) so the
    # laptop-side inference client can feed it to the policy. DDS stays local to
    # the robot; only this compact JSON state crosses the network (like the
    # camera/action ZMQ channels).
    state_sock = ctx.socket(zmq.PUB)
    state_sock.setsockopt(zmq.SNDHWM, 2)
    state_sock.setsockopt(zmq.LINGER, 0)
    state_sock.bind(f"tcp://0.0.0.0:{args.state_port}")
    logger.info("Publishing observation.state on :%d at %.0f Hz", args.state_port, args.state_fps)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def publish_state() -> None:
        period = 1.0 / args.state_fps if args.state_fps > 0 else 0.0
        joint_names = [j.name for j in G1_29_JointIndex]
        while not stop.is_set():
            t0 = time.time()
            obs = robot.get_observation()
            if obs:
                state = {f"{name}.q": float(obs.get(f"{name}.q", 0.0)) for name in joint_names}
                try:
                    state_sock.send_json(state, zmq.NOBLOCK)
                except zmq.Again:
                    pass
            if period:
                time.sleep(max(0.0, period - (time.time() - t0)))

    threading.Thread(target=publish_state, daemon=True).start()

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

            # Absolute base height / torso orientation from the policy (not
            # expressible via the joystick interface, so set on the controller
            # directly). Teleop omits these keys, so this is a no-op there.
            if robot.controller is not None:
                height = action.get("groot.height")
                if height is not None:
                    robot.controller.groot_height_cmd = float(height)
                roll = action.get("groot.rpy.roll")
                pitch = action.get("groot.rpy.pitch")
                yaw = action.get("groot.rpy.yaw")
                if None not in (roll, pitch, yaw):
                    robot.controller.groot_orientation_cmd = np.array(
                        [float(roll), float(pitch), float(yaw)], dtype=np.float32
                    )

            if grippers:
                # L3 = remote.button.4 -> left, R3 = remote.button.0 -> right.
                if "L" in grippers and "remote.button.4" in action:
                    grippers["L"].apply(bool(action["remote.button.4"]))
                if "R" in grippers and "remote.button.0" in action:
                    grippers["R"].apply(bool(action["remote.button.0"]))

            n += 1
            if n % 60 == 0:
                axes = {
                    k: round(float(action.get(k, 0.0)), 3)
                    for k in ("remote.lx", "remote.ly", "remote.rx", "remote.ry")
                }
                btn = {k: action.get(k) for k in ("remote.button.0", "remote.button.4") if k in action}
                logger.info("Applied %d actions | axes=%s buttons=%s", n, axes, btn)
    finally:
        logger.info("Shutting down onboard controller...")
        stop.set()
        try:
            state_sock.close(linger=0)
        except Exception as e:  # noqa: BLE001
            logger.warning("State socket close failed: %s", e)
        if camera_server is not None:
            try:
                camera_server.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("Camera server stop failed: %s", e)
        for g in grippers.values():
            try:
                g.bus.disconnect()
            except Exception as e:  # noqa: BLE001
                logger.warning("Gripper %s disconnect failed: %s", g.name, e)
        robot.disconnect()


if __name__ == "__main__":
    main()

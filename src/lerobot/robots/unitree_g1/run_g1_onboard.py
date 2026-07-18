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

Pair with ``run_g1_teleop_client.py`` on the laptop. Grippers/cameras are handled
separately by ``run_g1_server.py`` and are intentionally out of scope here.

Example (on the robot):

    python -m lerobot.robots.unitree_g1.run_g1_onboard --controller GrootLocomotionController
"""

import argparse
import json
import logging
import signal
import threading

import zmq

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("g1_onboard")

ACTION_PORT = 6004


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", default="GrootLocomotionController", help="Locomotion controller class")
    p.add_argument("--dds-interface", default=None, help="DDS network interface (default: SDK default)")
    p.add_argument("--action-port", type=int, default=ACTION_PORT, help="ZMQ port for laptop actions")
    p.add_argument(
        "--gravity-compensation",
        action="store_true",
        help="Enable arm gravity compensation (needs pinocchio/casadi on the robot)",
    )
    args = p.parse_args()

    cfg = UnitreeG1Config(
        is_simulation=False,
        onboard=True,
        controller=args.controller,
        dds_interface=args.dds_interface,
        gravity_compensation=args.gravity_compensation,
        cameras={},
    )
    robot = UnitreeG1(cfg)
    logger.info("Connecting onboard robot (controller=%s)...", args.controller)
    robot.connect()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)  # only ever act on the freshest command
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.bind(f"tcp://0.0.0.0:{args.action_port}")
    logger.info("Onboard controller live. Waiting for laptop actions on :%d ...", args.action_port)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

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
        robot.disconnect()


if __name__ == "__main__":
    main()

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

"""Laptop-side thin client: read the exos, run IK, ship the action to the onboard G1.

This is the counterpart to ``run_g1_onboard.py``. The heavy IK (pinocchio/casadi) stays
on the laptop where it's set up; only the resulting action (arm joint targets + joystick
axes + gripper flags) is sent as JSON over ZMQ. The locomotion policy runs on the robot,
so nothing latency-critical crosses the network.

Example (on the laptop):

    export LD_PRELOAD="$HOME/Documents/miniconda3/envs/lerobot312/lib/libstdc++.so.6"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python -m lerobot.robots.unitree_g1.run_g1_teleop_client \
        --robot-ip 172.18.130.111 \
        --left-arm-port /dev/ttyACM1 --right-arm-port /dev/ttyACM0 \
        --teleop-id asdasd
"""

import argparse
import logging
import time

import numpy as np
import zmq

from lerobot.teleoperators.unitree_g1.config_unitree_g1 import (
    ExoskeletonArmPortConfig,
    UnitreeG1TeleoperatorConfig,
)
from lerobot.teleoperators.unitree_g1.unitree_g1 import UnitreeG1Teleoperator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger("g1_teleop_client")

ACTION_PORT = 6004


def _to_jsonable(action: dict) -> dict:
    """Cast numpy scalars to plain Python so json.dumps accepts the action."""
    out: dict = {}
    for k, v in action.items():
        if isinstance(v, (np.generic,)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot-ip", required=True, help="IP of the robot running run_g1_onboard.py")
    p.add_argument("--action-port", type=int, default=ACTION_PORT)
    p.add_argument("--left-arm-port", required=True, help="Serial port for the LEFT exo arm")
    p.add_argument("--right-arm-port", required=True, help="Serial port for the RIGHT exo arm")
    p.add_argument("--teleop-id", default="exo", help="Teleoperator id (for calibration files)")
    p.add_argument("--frozen-joints", default="", help="Comma-separated joints to freeze in IK")
    p.add_argument("--fps", type=float, default=60.0, help="Max action send rate")
    args = p.parse_args()

    cfg = UnitreeG1TeleoperatorConfig(
        id=args.teleop_id,
        left_arm_config=ExoskeletonArmPortConfig(port=args.left_arm_port),
        right_arm_config=ExoskeletonArmPortConfig(port=args.right_arm_port),
        frozen_joints=args.frozen_joints,
    )
    teleop = UnitreeG1Teleoperator(cfg)
    logger.info("Connecting exo teleoperator (L=%s, R=%s)...", args.left_arm_port, args.right_arm_port)
    teleop.connect()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.setsockopt(zmq.CONFLATE, 1)  # drop stale actions rather than queue them
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{args.robot_ip}:{args.action_port}")
    logger.info("Sending actions to %s:%d. Ctrl-C to stop.", args.robot_ip, args.action_port)

    # The normal lerobot-teleoperate loop calls teleop.send_feedback(robot_obs) each
    # iteration, which resets the RemoteController axes from the robot's (usually idle)
    # wireless-remote bytes. That reset is what keeps `wireless_active` False so the exo
    # thumb-sticks (set_from_exo) are read every frame. We have no robot feedback here,
    # so without this reset the first non-zero exo axis latches `wireless_active` True
    # and the sticks freeze. Reset the latched axes ourselves before every get_action.
    rc = teleop.remote_controller

    period = 1.0 / args.fps if args.fps > 0 else 0.0
    n = 0
    try:
        while True:
            t0 = time.time()
            rc.lx = rc.ly = rc.rx = rc.ry = 0.0
            rc.button = [0] * 16
            action = teleop.get_action()
            try:
                sock.send_json(_to_jsonable(action), zmq.NOBLOCK)
            except zmq.Again:
                pass
            n += 1
            if n % 60 == 0:
                axes = {
                    k: round(float(action.get(k, 0.0)), 3)
                    for k in ("remote.lx", "remote.ly", "remote.rx", "remote.ry")
                }
                logger.info("Sent %d actions | axes=%s", n, axes)
            if period:
                time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        logger.info("Stopping teleop client...")
    finally:
        with_suppress_disconnect(teleop)
        sock.close(linger=0)


def with_suppress_disconnect(teleop) -> None:
    try:
        teleop.disconnect()
    except Exception as e:  # noqa: BLE001
        logger.warning("teleop.disconnect() failed: %s", e)


if __name__ == "__main__":
    main()

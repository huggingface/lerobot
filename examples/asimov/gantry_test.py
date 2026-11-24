# !/usr/bin/env python

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

"""Staged Asimov 1 bring-up on the gantry: STAND, then legs and optionally one exo arm.

The robot must hang from its gantry, with Asimov Manager's E-Stop at hand. The script
sends STAND, waits until the robot is armed, then hands the joints to LeRobot:

* ``--legs policy``: LeRobot's locomotion policy balances the legs at zero velocity
  (``control_mode=whole_body``).
* ``--legs stand``: legs, waist and neck hold the measured STAND pose, with no
  balancing (``control_mode=arms``).

Both arms hold their measured pose unless ``--exo-arms`` hands one or both to the
exoskeleton; exo joint targets move at most ``--arm-speed`` rad/s. The exo joysticks
are never read, so the robot does not walk. Ctrl+C stops sending setpoints; Asimov
Edge DAMPs the robot about two seconds later.

    python examples/asimov/gantry_test.py --host <robot-ip>
    python examples/asimov/gantry_test.py --host <robot-ip> --legs stand --exo-arms right \
        --exo-id <calibration-id> --left-port /dev/ttyACM0 --right-port /dev/ttyACM1
"""

import argparse
import logging
import time
from pathlib import Path

from lerobot.robots.asimov import Asimov, AsimovConfig
from lerobot.robots.asimov.asimov import ARM_ACTIONS
from lerobot.teleoperators.asimov_exo import AsimovExoTeleoperator, AsimovExoTeleoperatorConfig
from lerobot.teleoperators.unitree_g1.config_unitree_g1 import ExoskeletonArmPortConfig

ARM_PREFIXES = {"none": (), "left": ("L_",), "right": ("R_",), "both": ("L_", "R_")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", required=True, help="Asimov's IP address")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--legs", choices=["policy", "stand"], default="policy")
    parser.add_argument("--exo-arms", choices=list(ARM_PREFIXES), default="none")
    parser.add_argument("--exo-id", default=None, help="exoskeleton calibration id")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="directory with <exo-id>_left.json and <exo-id>_right.json (default: the G1 calibration cache)",
    )
    parser.add_argument("--left-port", default="/dev/ttyACM0")
    parser.add_argument("--right-port", default="/dev/ttyACM1")
    parser.add_argument("--arm-speed", type=float, default=0.6, help="max exo arm target speed (rad/s)")
    return parser.parse_args()


def stand(host: str) -> None:
    robot = Asimov(AsimovConfig(host=host, control_mode="velocity"))
    robot.connect()
    try:
        print(f"Robot is in {robot._get_state().mode.name}; sending STAND.", flush=True)
        robot._client.stand(timeout=10.0)
        print("STAND reached and armed.", flush=True)
    finally:
        robot.disconnect()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prefixes = ARM_PREFIXES[args.exo_arms]
    exo_keys = [key for key in ARM_ACTIONS if key.startswith(prefixes)] if prefixes else []

    exo = None
    if exo_keys:
        exo = AsimovExoTeleoperator(
            AsimovExoTeleoperatorConfig(
                id=args.exo_id,
                calibration_dir=args.calibration_dir,
                control_mode="arms",
                left_arm_config=ExoskeletonArmPortConfig(port=args.left_port),
                right_arm_config=ExoskeletonArmPortConfig(port=args.right_port),
            )
        )
        exo.connect()
        print(f"Exoskeleton drives: {exo_keys}", flush=True)

    stand(args.host)

    policy = args.legs == "policy"
    robot = Asimov(
        AsimovConfig(
            host=args.host,
            control_mode="whole_body" if policy else "arms",
            robot_supported=True,
            max_linear_velocity=0.1,
            max_lateral_velocity=0.1,
            max_angular_velocity=10.0,
        )
    )
    robot.connect()
    observation = robot.get_observation()
    action = {key: float(observation[key]) for key in ARM_ACTIONS}
    if policy:
        action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
    max_step = args.arm_speed / args.fps
    print(
        "Legs: " + ("policy at zero velocity" if policy else "held in STAND") + ". Ctrl+C to stop.",
        flush=True,
    )

    start = last_print = time.monotonic()
    try:
        while True:
            if exo is not None:
                exo_action = exo.get_action()
                for key in exo_keys:
                    delta = float(exo_action[key]) - action[key]
                    action[key] += max(-max_step, min(max_step, delta))
            robot.send_action(action)
            now = time.monotonic()
            if now - last_print >= 1.0:
                last_print = now
                state = robot._get_state()
                print(
                    f"t={now - start:5.1f}s mode={state.mode.name} faulted={state.faulted} "
                    f"roll={state.euler[0]:+.2f} pitch={state.euler[1]:+.2f} "
                    f"alerts={[alert.name for alert in state.alerts]}",
                    flush=True,
                )
            time.sleep(1.0 / args.fps)
    except KeyboardInterrupt:
        print("Stopping; Asimov Edge DAMPs the robot about two seconds after the last setpoint.", flush=True)
    finally:
        robot.disconnect()
        if exo is not None:
            exo.disconnect()


if __name__ == "__main__":
    main()

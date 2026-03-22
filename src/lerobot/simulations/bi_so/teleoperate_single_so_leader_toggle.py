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

"""
Teleoperate the bimanual MuJoCo simulation with a single real SO leader arm.

The active simulated arm is selected in the MuJoCo viewer. Press `t` in the
viewer to toggle between arm 0 and arm 1. The single leader arm then controls
whichever arm is currently active.

Example:

```shell
python -m lerobot.simulations.bi_so.teleoperate_single_so_leader_toggle \
  --leader-port COM5 \
  --sim-root C:/Users/Ninja/AOSH/lerobot/sim \
  --launch-viewer \
  --hz 60
```
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from lerobot.robots.bi_so_follower_simulated.bi_so_follower_simulated import (
    MOTOR_NAMES,
    BiSOFollowerSimulated,
    BiSOFollowerSimulatedConfig,
)
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--leader-port", required=True, help="Serial port for the single SO leader arm.")
    parser.add_argument("--leader-id", default="single_so_leader")
    parser.add_argument("--leader-calibration-dir", default=None)
    parser.add_argument("--leader-use-degrees", action="store_true", default=True)
    parser.add_argument("--no-leader-use-degrees", dest="leader_use_degrees", action="store_false")
    parser.add_argument("--calibrate", action="store_true", default=True)
    parser.add_argument("--no-calibrate", dest="calibrate", action="store_false")

    parser.add_argument("--sim-root", default=None)
    parser.add_argument("--bridge-path", default=None)
    parser.add_argument("--xml-path", default=None)
    parser.add_argument("--bridge-factory-name", default="make_task2_bimanual_buses")
    parser.add_argument("--robot-dofs", type=int, default=6)
    parser.add_argument("--launch-viewer", action="store_true", default=False)
    parser.add_argument("--realtime", action="store_true", default=True)
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")
    parser.add_argument("--slowmo", type=float, default=1.0)
    parser.add_argument("--hz", type=float, default=60.0)

    return parser.parse_args()


def _build_sim_helper(args: argparse.Namespace) -> BiSOFollowerSimulated:
    cfg = BiSOFollowerSimulatedConfig(
        id="single_leader_toggle_sim",
        sim_root=None if args.sim_root is None else Path(args.sim_root),
        bridge_path=None if args.bridge_path is None else Path(args.bridge_path),
        xml_path=None if args.xml_path is None else Path(args.xml_path),
        bridge_factory_name=args.bridge_factory_name,
        robot_dofs=args.robot_dofs,
        realtime=args.realtime,
        slowmo=args.slowmo,
        launch_viewer=args.launch_viewer,
    )
    return BiSOFollowerSimulated(cfg)


def _leader_action_to_robot_array(action: dict[str, float], motor_names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([float(action[f"{motor_name}.pos"]) for motor_name in motor_names], dtype=np.float32)


def _active_arm_index(backend) -> int:
    try:
        return int(getattr(backend.sim, "active_arm", 0))
    except Exception:
        return 0


def main() -> int:
    args = _parse_args()
    init_logging()
    logger = logging.getLogger(__name__)

    sim_helper = _build_sim_helper(args)
    bridge_module = sim_helper._load_bridge_module()
    bridge_factory = getattr(bridge_module, sim_helper.config.bridge_factory_name)
    xml_path = sim_helper._resolve_xml_path()

    backend, buses = bridge_factory(
        xml_path=str(xml_path),
        robot_dofs=sim_helper.config.robot_dofs,
        render_size=None,
        realtime=sim_helper.config.realtime,
        slowmo=sim_helper.config.slowmo,
        launch_viewer=sim_helper.config.launch_viewer,
    )

    left_bus = buses.get("arm0")
    right_bus = buses.get("arm1")
    if left_bus is None or right_bus is None:
        raise RuntimeError("The simulation bridge must expose both `arm0` and `arm1`.")

    leader_cfg = SOLeaderTeleopConfig(
        id=args.leader_id,
        calibration_dir=None if args.leader_calibration_dir is None else Path(args.leader_calibration_dir),
        port=args.leader_port,
        use_degrees=args.leader_use_degrees,
    )
    leader = SOLeader(leader_cfg)

    previous_active_arm: int | None = None

    try:
        left_bus.connect()
        right_bus.connect()

        sim_helper._backend = backend
        sim_helper._left_bus = left_bus
        sim_helper._right_bus = right_bus
        sim_helper._gripper_ctrlrange_deg = sim_helper._read_gripper_ctrlrange_deg()

        leader.connect(calibrate=args.calibrate)

        logger.info("Single-leader toggle teleop started. Press `t` in the MuJoCo viewer to switch active arms.")

        while True:
            tick_start = time.perf_counter()

            active_arm = _active_arm_index(backend)
            if previous_active_arm != active_arm:
                logger.info(f"Active simulated arm: {active_arm}")
                previous_active_arm = active_arm

            leader_action = leader.get_action()
            robot_arm_action = _leader_action_to_robot_array(leader_action, MOTOR_NAMES)

            sim_arm_action = sim_helper._robot_to_sim_arm_units(active_arm, robot_arm_action)
            if active_arm == 0:
                left_bus.write(sim_arm_action)
            else:
                right_bus.write(sim_arm_action)

            dt_s = time.perf_counter() - tick_start
            precise_sleep(max((1.0 / float(args.hz)) - dt_s, 0.0))
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            leader.disconnect()
        except Exception:
            pass
        try:
            left_bus.disconnect()
        except Exception:
            pass
        try:
            right_bus.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Simple script to control a robot from teleoperation.

Example:

```shell
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
import draccus
import numpy as np
import rerun as rr

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up
from lerobot.common.utils.visualization_utils import _init_rerun

from .common.teleoperators import gamepad, koch_leader, so100_leader, so101_leader  # noqa: F401


last_wrong = {}
last_right = {}
mark  = {}

@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False

def diff(cur, nxt, names, THRESH):
    cur2 = cur.copy()
    nxt2 = nxt.copy()
    nxt2 = {key.removesuffix(".pos"): val for key, val in nxt2.items() if key.endswith(".pos")}
    cur2 = {key.removesuffix(".pos"): val for key, val in cur2.items() if key.endswith(".pos")}
    # print("-------------------")
    # print(nxt2)
    # print(cur2)
    is_wrong = {}
    for name in names:
        if abs(nxt2[name] - cur2[name]) > THRESH[name]:
            is_wrong[name] = True
        else:
            is_wrong[name] = False
    return is_wrong


def teleop_loop(
    teleop: Teleoperator, robot: Robot, fps: int, display_data: bool = False, duration: float | None = None
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        action = teleop.get_action()
        if display_data:
            observation = robot.get_observation()
            for obs, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation_{obs}", rr.Scalar(val))
                elif isinstance(val, np.ndarray):
                    rr.log(f"observation_{obs}", rr.Image(val), static=True)
            for act, val in action.items():
                if isinstance(val, float):
                    rr.log(f"action_{act}", rr.Scalar(val))

        THRESHOLD_TIME_LOCK = 0.375
        THRESHOLD_TIME_UNLOCK = 1
        THRESHOLD_DIFF = {'shoulder_pan': 2, 'shoulder_lift': 2, 'elbow_flex': 2, 'wrist_flex': 2, 'wrist_roll': 2, 'gripper': 1}

        motors = list(robot.bus.motors.keys())

        # pegar diff com o action e o get action
        cur_robot = robot.get_action()
        robot.send_action(action)
        is_wrong = diff(cur_robot, action, motors, THRESHOLD_DIFF)
        
        for motor in motors:
            if is_wrong[motor]:
                if mark[motor]:
                    continue
                else:
                    last_wrong[motor] = time.perf_counter()
                    mark[motor] = True
            else:
                last_right[motor] = time.perf_counter()
                mark[motor] = False

        #teleop.send_action(robot.get_action())
        motors = ["gripper"]
        for motor in motors:
            #print("-------------------------------")
            #print("motor is stalled", motor, robot.bus.is_stalled(motor))
            #robot.bus.is_stalled(motor) 

            #print("-------------------------------")
            #print("motor",motor, mark[motor])
            #print(time.perf_counter() - last_wrong[motor])
            #print(time.perf_counter() - last_right[motor])
            #print("torqued", teleop.bus.is_torqued(motor))

            if mark[motor] and time.perf_counter() - last_wrong[motor] > THRESHOLD_TIME_LOCK:
                # tem que ter isso
                # print("lock ", motor)
                teleop.bus.sync_write("Goal_Position",{motor: robot.get_action()[motor+".pos"]})
            else:
                # teleop.bus.is_torqued(motor) 
                #if time.perf_counter() - last_right[motor] > THRESHOLD_TIME_UNLOCK:
                # print("release ", motor)
                last_right[motor] = time.perf_counter()
                teleop.bus.disable_torque(motor)
                

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        # print("\n" + "-" * (display_len + 10))
        # print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        # for motor, value in action.items():
        #     print(f"{motor:<{display_len}} | {value:>7.2f}")
        # print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        #print(action)

        if duration is not None and time.perf_counter() - start >= duration:
            return

        # move_cursor_up(len(action) + 5)


@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()

    motors = list(robot.bus.motors.keys())
    for motor in motors:
        mark[motor] = False
        last_right[motor] = time.perf_counter()
        last_wrong[motor] = time.perf_counter()

    try:
        teleop_loop(teleop, robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    teleoperate()

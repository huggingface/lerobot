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

# 0 -> sinal vermelho
# 1 -> sinal amarelo
# 2 -> sinal verde
state = {}
stall = {}
pos = {}

start_time = 0

@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False

def green_light(motor):
    state[motor] = 2

def yellow_light(motor):
    state[motor] = 1

def red_light(motor):
    state[motor] = 0

def is_green(motor):
    return state[motor] == 2

def is_yellow(motor):
    return state[motor] == 1

def is_red(motor):
    return state[motor] == 0


def check_stall(robot,teleop,motors,THRESHOLD_CURRENT):
    action = robot.get_action()
    for motor in motors:
        if robot.bus.get_current(motor) > THRESHOLD_CURRENT[motor]:
            stall[motor] = True
        elif abs(action[motor+".pos"] - pos[motor]) > 0: 
            stall[motor] = False

def check_state(robot,teleop,motors,THRESHOLD_CURRENT):
    action = robot.get_action()
    for motor in motors:
        if is_green(motor):
            if stall[motor]:
                red_light(motor)
            pos[motor] = action[motor+".pos"]
        elif is_yellow(motor):
            if not stall[motor]:
                if teleop.bus.is_torqued:
                    teleop.bus.disable_torque(motor,5)
                green_light(motor)
            else:
                if robot.bus.get_current(motor) > THRESHOLD_CURRENT[motor]:
                    red_light(motor)
                elif teleop.bus.is_torqued:
                    teleop.bus.disable_torque(motor,5)
        else:
            teleop.bus.sync_write("Goal_Position",{motor:pos[motor]})
            teleop.bus.enable_torque(motor)
            if not robot.bus.get_current(motor) > THRESHOLD_CURRENT[motor] and not teleop.bus.get_current(motor) > 0:
                yellow_light(motor)

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

        THRESHOLD_CURRENT = {'shoulder_pan': 50, 'shoulder_lift': 100, 'elbow_flex': 75, 'wrist_flex': 50, 'wrist_roll': 50, 'gripper': 20}
        STARTUP_TIME = 5

        motors = list(robot.bus.motors.keys())

        # pegar diff com o action e o get action
        robot.send_action(action)

        if time.perf_counter() - start_time > STARTUP_TIME:
            check_stall(robot,teleop,motors,THRESHOLD_CURRENT)
            check_state(robot,teleop,motors,THRESHOLD_CURRENT)

        #print("stall",stall)
        #print("state",state)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        #print("\n" + "-LAELE-" * (display_len//3 + 1))
        #print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        #for motor, value in action.items():
        #    print(f"{motor:<{display_len}} | {value:>7.2f}")
        #print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        #move_cursor_up(len(action) + 5)

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

    for motor in list(robot.bus.motors.keys()):
        green_light(motor)
        pos[motor] = robot.get_action()[motor+".pos"]
        stall[motor] = False
    
    start_time = time.perf_counter()

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

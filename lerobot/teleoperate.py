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
from pathlib import Path
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Dict, Optional

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
from lerobot.common.utils.urdf_logger import URDFLogger
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up
from lerobot.common.utils.video_logger import VideoLogger
from lerobot.common.utils.visualization_utils import RerunRobotLogger, _init_rerun
from lerobot.common.constants import URDFS

from .common.teleoperators import gamepad, koch_leader, so100_leader, so101_leader  # noqa: F401


@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False

def teleop_loop(
    teleop: Teleoperator, robot: Robot, fps: int, robot_logger: Optional[RerunRobotLogger] = None, duration: float | None = None,
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()
        action = teleop.get_action()

        robot.send_action(action)

        if robot_logger is not None:
            robot_logger.log_all(sync_time=True)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start

        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        for motor, value in action.items():
            print(f"{motor:<{display_len}} | {value:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            break

        move_cursor_up(len(action) + 5)

def log_to_rerun(observation, action, video_loggers: Dict[str, VideoLogger], robot_urdf_logger: Optional[URDFLogger]):
    rr.set_time("log_time", duration=np.timedelta64(time.time_ns(), "ns"))

    if robot_urdf_logger is not None:
        robot_urdf_logger.log_joint_angles(observation)

    for obs, val in observation.items():
        if isinstance(val, float):
            rr.log(["observation", obs], rr.Scalars(val))
        elif isinstance(val, np.ndarray) and obs in video_loggers:
            video_loggers[obs].log_frame(val)

    for act, val in action.items():
        if isinstance(val, float):
            rr.log(["action", act], rr.Scalars(val))


@draccus.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()

    robot_logger = None
    if cfg.display_data:
        robot_logger = RerunRobotLogger(teleop=teleop, robot=robot, fps=cfg.fps)
        robot_logger.init()

    try:
        teleop_loop(teleop, robot, cfg.fps, robot_logger=robot_logger, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        pass
    finally:
        if robot_logger is not None:
            robot_logger.cleanup()
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    teleoperate()

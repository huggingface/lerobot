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
Teleoperates two robots simultaneously.

Example:

```shell
python -m lerobot.teleoperate_two_arm --config_path=./robot_config/two_arm_config.yaml
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import numpy as np
import rerun as rr

from lerobot.common.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
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
from lerobot.common.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.common.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.common.utils.visualization_utils import _init_rerun
from lerobot.configs import parser

from .common.teleoperators import koch_leader, so100_leader, so101_leader  # noqa: F401


@dataclass
class TeleoperateTwoArmConfig:
    robot1: RobotConfig
    robot2: RobotConfig
    teleop1: TeleoperatorConfig
    teleop2: TeleoperatorConfig
    # Limit the frames per second.
    fps: int = 30
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True


def teleoperate_loop(
    robot1: Robot,
    robot2: Robot,
    teleop1: Teleoperator,
    teleop2: Teleoperator,
    events: dict,
    fps: int,
    display_data: bool = False,
):
    start_time = time.perf_counter()
    while not events["stop_teleoperation"]:
        loop_start_time = time.perf_counter()

        observation1 = robot1.get_observation()
        observation2 = robot2.get_observation()
        # Combine observations, ensuring keys are unique or appropriately prefixed if necessary
        # For simplicity, we assume unique keys or that the user handles potential conflicts
        observation = {**{f"robot1_{k}": v for k,v in observation1.items()}, 
                       **{f"robot2_{k}": v for k,v in observation2.items()}}

        action1 = teleop1.get_action()
        action2 = teleop2.get_action()

        robot1.send_action(action1)
        robot2.send_action(action2)
        
        # Combine actions for display, ensuring keys are unique
        action = {**{f"robot1_{k}": v for k,v in action1.items()}, 
                  **{f"robot2_{k}": v for k,v in action2.items()}}


        if display_data:
            for obs_key, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation/{obs_key}", rr.Scalar(val))
                elif isinstance(val, np.ndarray) and val.ndim in [2, 3]: # Check for 2D (grayscale) or 3D (RGB) images
                    try:
                        rr.log(f"observation/{obs_key}", rr.Image(val), static=False)
                    except Exception as e:
                        logging.warning(f"Failed to log image {obs_key} to rerun: {e}")
                # Add more types if needed
            
            for act_key, val in action.items():
                 if isinstance(val, (float, int)):
                    rr.log(f"action/{act_key}", rr.Scalar(val))
                # Add more types if needed


        dt_s = time.perf_counter() - loop_start_time
        busy_wait(1 / fps - dt_s)

        if events["exit_early"]:
            events["exit_early"] = False # Reset for next potential use if loop continues
            break
    
    logging.info(f"Teleoperation loop ended. Total duration: {time.perf_counter() - start_time:.2f}s")


@parser.wrap()
def teleoperate(cfg: TeleoperateTwoArmConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        _init_rerun(session_name="teleoperate_two_arm")

    robot_one = make_robot_from_config(cfg.robot1)
    teleop_one = make_teleoperator_from_config(cfg.teleop1)
    
    robot_two = make_robot_from_config(cfg.robot2)
    teleop_two = make_teleoperator_from_config(cfg.teleop2)

    robot_one.connect()
    teleop_one.connect()
    robot_two.connect()
    teleop_two.connect()

    log_say("Starting teleoperation for two arms. Press 'q' to quit.", cfg.play_sounds)

    listener, events = init_keyboard_listener()
    events["stop_teleoperation"] = False # Initialize stop flag

    try:
        teleoperate_loop(
            robot1=robot_one,
            robot2=robot_two,
            teleop1=teleop_one,
            teleop2=teleop_two,
            events=events,
            fps=cfg.fps,
            display_data=cfg.display_data,
        )
    except KeyboardInterrupt:
        log_say("Teleoperation interrupted by user.", cfg.play_sounds)
    finally:
        log_say("Stopping teleoperation.", cfg.play_sounds, blocking=True)

        robot_one.disconnect()
        teleop_one.disconnect()
        robot_two.disconnect()
        teleop_two.disconnect()

        if not is_headless() and listener is not None:
            listener.stop()
            logging.info("Keyboard listener stopped.")

    log_say("Exiting.", cfg.play_sounds)


if __name__ == "__main__":
    teleoperate()
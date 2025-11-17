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
Helper to automatically recalibrate your device (robot or teleoperator) using current monitoring.

This script performs automatic calibration by detecting mechanical limits using current monitoring,
then setting homing offsets to center the detected ranges.

WARNING: This process involves moving the robot to find limits.
Ensure the robot arm is clear of obstacles and people during calibration.

Example:

```shell
python -m lerobot.auto_calibrate \
    --robot.type=sourccey_follower \
    --robot.port=/dev/tty.usbmodem58760431551
```
"""

import logging
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    hope_jr,
    koch_follower,
    lekiwi,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    sourccey,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
    sourccey,
)
from lerobot.teleoperators.sourccey.sourccey.bi_sourccey_leader.bi_sourccey_leader import BiSourcceyLeader
from lerobot.teleoperators.sourccey.sourccey.sourccey_leader.sourccey_leader import SourcceyLeader
from lerobot.utils.utils import init_logging


@dataclass
class AutoCalibrateConfig:
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None
    full_reset: bool = False
    arm: str | None = None  # "left", "right", or None for both

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")

        self.device = self.robot if self.robot else self.teleop

@draccus.wrap()
def auto_calibrate(cfg: AutoCalibrateConfig):
    """Automatically calibrate a robot or teleoperator using current monitoring."""
    init_logging()
    logging.info("Starting automatic calibration process...")
    logging.info(pformat(asdict(cfg)))

    # Create device instance
    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    elif isinstance(cfg.device, TeleoperatorConfig):
        device = make_teleoperator_from_config(cfg.device)
    else:
        raise ValueError(f"Unsupported device type: {type(cfg.device)}")

    try:
        # Connect without calibration (we'll do auto-calibration)
        device.connect(calibrate=False)

        # Check if device supports auto-calibration
        if hasattr(device, 'auto_calibrate'):
            device.auto_calibrate(full_reset=cfg.full_reset)
        else:
            logging.warning("Device does not support auto-calibration. Returning")

    except Exception as e:
        logging.error(f"Calibration failed: {e}")
        raise
    finally:
        device.disconnect()


if __name__ == "__main__":
    auto_calibrate()

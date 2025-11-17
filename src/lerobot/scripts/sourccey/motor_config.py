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
Helper to configure motor bus baud rate for your device (robot or teleoperator).

This script allows you to set the baud rate of the motor bus communication.
The default baud rate is 1,000,000 bits per second.

Example:

```shell
python -m lerobot.motor_config \
    --robot.type=sourccey_follower \
    --robot.port=/dev/tty.usbmodem58760431551 \
    --baud_rate=1000000
```

```shell
python -m lerobot.motor_config \
    --teleop.type=sourccey_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --baud_rate=2000000
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
class MotorConfigConfig:
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None
    baud_rate: int = 1_000_000

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")

        self.device = self.robot if self.robot else self.teleop

@draccus.wrap()
def motor_config(cfg: MotorConfigConfig):
    """Configure motor bus baud rate for a robot or teleoperator."""
    init_logging()
    logging.info("Starting motor configuration process...")
    logging.info(pformat(asdict(cfg)))

    # Create device instance
    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    elif isinstance(cfg.device, TeleoperatorConfig):
        device = make_teleoperator_from_config(cfg.device)
    else:
        raise ValueError(f"Unsupported device type: {type(cfg.device)}")

    try:
        # Connect without calibration (we just need to configure baud rate)
        device.connect(calibrate=False)

        # Check if device supports baud rate configuration
        if hasattr(device, 'set_baud_rate'):
            logging.info(f"Setting baud rate to {cfg.baud_rate}")
            device.set_baud_rate(cfg.baud_rate)
            logging.info("✅ Successfully configured motor bus baud rate")
        else:
            logging.warning("Device does not support baud rate configuration")

    except Exception as e:
        logging.error(f"Motor configuration failed: {e}")
        raise
    finally:
        device.disconnect()


if __name__ == "__main__":
    motor_config()

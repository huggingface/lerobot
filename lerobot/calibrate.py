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
Helper to recalibrate your device (robot or teleoperator).

Example:

```shell
python -m lerobot.calibrate \
    --device.type=so100_leader \
    --device.port=/dev/tty.usbmodem58760431551 \
    --device.id=blue
```
"""

import logging
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus

from lerobot.common.cameras import intel, opencv  # noqa: F401
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
)
from lerobot.common.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.common.utils.utils import init_logging

from .common.teleoperators import koch_leader, so100_leader  # noqa: F401


@dataclass
class CalibrateConfig:
    device: RobotConfig | TeleoperatorConfig


@draccus.wrap()
def calibrate(cfg: CalibrateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    elif isinstance(cfg.device, TeleoperatorConfig):
        device = make_teleoperator_from_config(cfg.device)

    device.connect(calibrate=False)
    device.calibrate()
    device.disconnect()


if __name__ == "__main__":
    calibrate()

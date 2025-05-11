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
Helper to set motor ids and baudrate.

Example:

```shell
python -m lerobot.setup_motors \
    --device.type=so100_leader \
    --device.port=/dev/tty.usbmodem575E0031751
```
"""

from dataclasses import dataclass

import draccus

from .common.robots import RobotConfig, koch_follower, make_robot_from_config, so100_follower  # noqa: F401
from .common.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
)

COMPATIBLE_DEVICES = [
    "koch_follower",
    "koch_leader",
    "so100_follower",
    "so100_leader",
]


@dataclass
class SetupConfig:
    device: RobotConfig | TeleoperatorConfig


@draccus.wrap()
def setup_motors(cfg: SetupConfig):
    if cfg.device.type not in COMPATIBLE_DEVICES:
        raise NotImplementedError

    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    else:
        device = make_teleoperator_from_config(cfg.device)

    device.setup_motors()


if __name__ == "__main__":
    setup_motors()

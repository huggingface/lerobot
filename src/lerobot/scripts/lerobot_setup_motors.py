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

Default: setup all the motors sequentially

```shell
lerobot-setup-motors \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem575E0031751
```

Optional: setup specific motors by passing a list of motor names (e.g. "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_pan", "gripper")
This is useful when you have to replace a motor and don't want to redo the setup for the existing motors.

```shell
lerobot-setup-motors \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem575E0031751 \
    --motors=[shoulder_pan,elbow_flex]
```
"""

from dataclasses import dataclass

import draccus

from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so_follower,
    koch_follower,
    lekiwi,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_so_leader,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_mini,
    so_leader,
)

COMPATIBLE_DEVICES = [
    "koch_follower",
    "koch_leader",
    "omx_follower",
    "omx_leader",
    "openarm_mini",
    "so100_follower",
    "so100_leader",
    "so101_follower",
    "so101_leader",
    "lekiwi",
]


@dataclass
class SetupConfig:
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None
    motors: list[str] | None = None

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")
        if self.motors is not None and len(self.motors) == 0:
            raise ValueError(
                "--motors flag is provided but the list is empty. Remove it or provide at least one motor name."
            )

        self.device = self.robot if self.robot else self.teleop


@draccus.wrap()
def setup_motors(cfg: SetupConfig):
    if cfg.device.type not in COMPATIBLE_DEVICES:
        raise NotImplementedError

    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    else:
        device = make_teleoperator_from_config(cfg.device)

    device.setup_motors(motors=cfg.motors)


def main():
    setup_motors()


if __name__ == "__main__":
    main()

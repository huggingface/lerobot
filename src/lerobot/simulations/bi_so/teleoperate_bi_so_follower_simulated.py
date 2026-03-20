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
Simulation-only entrypoint for teleoperating the bimanual SO MuJoCo wrapper.

This file intentionally avoids changing the core CLI registration files.
It imports the new simulated robot module up front so the existing
`lerobot_teleoperate` flow can discover and instantiate it through the
ChoiceRegistry/fallback factory.

Example:

```shell
python -m lerobot.simulations.bi_so.teleoperate_bi_so_follower_simulated ^
  --robot.type=bi_so_follower_simulated ^
  --robot.launch_viewer=true ^
  --robot.sim_root=C:/Users/Ninja/AOSH/lerobot/sim ^
  --teleop.type=bi_so_leader ^
  --teleop.left_arm_config.port=COM5 ^
  --teleop.right_arm_config.port=COM6 ^
  --teleop.id=bimanual_leader ^
  --fps=60
```
"""

from lerobot.robots import bi_so_follower_simulated  # noqa: F401
from lerobot.scripts.lerobot_teleoperate import main


if __name__ == "__main__":
    main()


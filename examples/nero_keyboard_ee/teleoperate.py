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

"""Keyboard end-effector teleoperation example for NERO.

This example reuses the standard teleoperate runtime and automatically activates
NERO keyboard EE -> IK -> joint command processing when:

- `--teleop.type=keyboard_ee`
- `--robot.type=nero_follower`

Run from repository root:

```bash
PYTHONPATH=./src /home/yuhang/miniconda3/envs/lerobot/bin/python examples/nero_keyboard_ee/teleoperate.py \
  --teleop.type=keyboard_ee \
  --robot.type=nero_follower \
  --robot.interface=socketcan \
  --robot.channel=can0 \
  --robot.firmeware_version=default \
  --robot.speed_percent=20 \
  --robot.keyboard_ee.enabled=true \
  --robot.keyboard_ee.urdf_path=/absolute/path/to/nero.urdf \
  --robot.keyboard_ee.target_frame_name=gripper \
  --robot.keyboard_ee.max_linear_step_m=0.01 \
  --robot.keyboard_ee.max_angular_step_rad=0.08 \
  --robot.keyboard_ee.gripper_delta_per_step=1.5 \
  --teleop.linear_step=0.004 \
  --teleop.angular_step=0.05 \
  --teleop.require_deadman=true \
  --fps=30 \
  --display_data=true
```

Key controls (base frame):
- Translation: arrows + left/right shift
- Rotation: i/k (wx), j/l (wy), u/o (wz)
- Gripper: x/z
- Deadman (default): hold space to enable motion
"""

from lerobot.scripts.lerobot_teleoperate import main


if __name__ == "__main__":
    main()

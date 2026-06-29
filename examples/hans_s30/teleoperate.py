# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Teleoperate the Hans Robot S30 using its built-in zero-force teaching mode.

The S30's gravity-compensation mode lets an operator manually guide the arm
while joint positions are streamed at ``FPS`` Hz and displayed via Rerun.

Edit the constants below to match your setup before running::

    python examples/hans_s30/teleoperate.py

Press Ctrl-C to stop.  The robot is automatically returned to position-control
mode on exit.
"""

import time

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.hans_s30 import HansS30, HansS30RobotConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ── User settings ────────────────────────────────────────────────────────────
ROBOT_IP = "192.168.115.11"
FPS = 30
# ─────────────────────────────────────────────────────────────────────────────


def main():
    camera_config = {
        "wrist_cam": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
    }

    robot_config = HansS30RobotConfig(
        ip=ROBOT_IP,
        port=10003,
        id="my_hans_s30",
        cameras=camera_config,
    )

    robot = HansS30(robot_config)
    robot.connect()

    init_rerun(session_name="hans_s30_teleoperate")

    try:
        print("Enabling zero-force teaching mode …  Press Ctrl-C to stop.")
        robot.enable_free_driver()

        while True:
            t0 = time.perf_counter()
            obs = robot.get_observation()
            log_rerun_data(observation=obs)
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    finally:
        robot.disable_free_driver()
        robot.disconnect()


if __name__ == "__main__":
    main()

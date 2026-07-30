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

"""Tailored LeKiwi teleoperation entrypoint (no data collection).

Same controls as `record_mytask.py`, but nothing is saved to a dataset — use this
to test/drive the robot freely.

Prerequisite: the host must be running on the LeKiwi (matching `REMOTE_IP` below).
Enable the Orbbec camera pan/tilt on the host too (`--robot.use_camera_head=true`):

    python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi --robot.use_camera_head=true

Then run this script on your laptop (with the leader arm connected):

    uv run python examples/lekiwi/teleop_mytask.py

Controls (keyboard focus on this terminal / rerun window):
  - Leader arm drives the follower arm.
  - w/a/s/d move the base, z/x rotate, r/f change speed.
  - i/k tilt the Orbbec camera up/down, j/l pan left/right.
  - Esc: stop teleoperation.
"""

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# ─── Session settings — edit these ──────────────────────────────────────────
REMOTE_IP = "192.168.50.187"  # LeKiwi host IP
ROBOT_ID = "my_awesome_kiwi"  # must match the id used to calibrate / run the host
LEADER_PORT = "/dev/tty.usbmodem5B8E1169971"  # <- verify with `lerobot-find-port`
LEADER_ID = "my_leader_arm"  # must match the id used to calibrate the leader

FPS = 30
CHUNK_SEC = 2  # how often to check for the Esc key (drive is continuous across chunks)
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # use_camera_head=True adds the Orbbec camera pan/tilt to the action/state; the host must run
    # with --robot.use_camera_head=true so its motor bus and features match.
    robot_config = LeKiwiClientConfig(remote_ip=REMOTE_IP, id=ROBOT_ID, use_camera_head=True)
    leader_arm_config = SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID)
    keyboard_config = KeyboardTeleopConfig()

    robot = LeKiwiClient(robot_config)
    leader_arm = SO101Leader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # The host must already be running on LeKiwi:
    #   python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_teleop")

    try:
        if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )

        log_say("Teleoperating. Press Esc to stop.")
        # Drive continuously with no dataset attached (nothing is recorded).
        # `record_loop` only returns on its own timer (or the right-arrow "exit_early"
        # event), so we run it in short chunks and check the Esc/"stop_recording"
        # flag between chunks to allow a clean stop.
        while not events["stop_recording"]:
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=[leader_arm, keyboard],
                control_time_s=CHUNK_SEC,
                display_data=True,
            )
    finally:
        log_say("Stop teleoperation")
        robot.disconnect()
        leader_arm.disconnect()
        keyboard.disconnect()
        listener.stop()


if __name__ == "__main__":
    main()

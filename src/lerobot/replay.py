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
Replays the actions of an episode from a dataset on a robot.

Example:

```shell
python -m lerobot.replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=aliberts/record-test \
    --dataset.episode=2
```

Biteleop example:
```shell
python -m lerobot.replay \
    --robot.type=so101_follower_t \
    --robot.port=/dev/tty.usbmodem58760432961 \
    --robot.id=follower_arm_torque \
    --dataset.repo_id=pepijn223/bilateral-wipe-large \
    --dataset.episode=10 \
    --biteleop=true
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import draccus

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    so101_follower_torque,
)
from lerobot.robots.so101_follower_torque import SO101FollowerT
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Use biteleop to replay the dataset
    biteleop: bool = False


@draccus.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])
    actions = dataset.hf_dataset.select_columns("action")

    if cfg.biteleop:
        if not isinstance(robot, SO101FollowerT):
            raise ValueError(
                "Bilateral teleoperation replay requires the robot to be of type SO101FollowerT."
            )
        log_say("Bilateral teleoperation replay enabled.", cfg.play_sounds)

    robot.connect()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)

    start_time_all = time.perf_counter()

    for idx in range(dataset.num_frames):
        start_loop_t = time.perf_counter()

        action_from_ds_array = actions[idx]["action"]
        action_from_ds = {}
        for i, name in enumerate(dataset.features["action"]["names"]):
            action_from_ds[name] = action_from_ds_array[i]

        # Bilateral teleoperation
        if cfg.biteleop:
            # Get current follower robot observation
            obs_f = robot.get_observation()
            pos_f = {j: obs_f[f"{j}.pos"] for j in robot.bus.motors}
            vel_f = {j: obs_f[f"{j}.vel"] for j in robot.bus.motors}
            tau_reaction_f = {j: obs_f[f"{j}.effort"] for j in robot.bus.motors}

            # Get target leader state from the dataset
            pos_l = {j: action_from_ds[f"{j}.pos"] for j in robot.bus.motors}
            vel_l = {j: action_from_ds[f"{j}.vel"] for j in robot.bus.motors}
            # The saved effort in dataset is -tau_reaction_l
            neg_tau_reaction_l = {j: action_from_ds[f"{j}.effort"] for j in robot.bus.motors}

            # Get control gains from the robot instance
            kp_gains = robot.kp_gains
            kd_gains = robot.kd_gains
            kf_gains = robot.kf_gains

            # Compute torque command for the follower robot
            tau_cmd_f = [
                (
                    kp_gains[j] * (pos_l[j] - pos_f[j])  # Position tracking
                    + kd_gains[j] * (vel_l[j] - vel_f[j])  # Velocity damping
                    + kf_gains[j] * (neg_tau_reaction_l[j] - tau_reaction_f[j])  # Force reflection
                )
                for j in robot.bus.motors
            ]

            # Format action with calculated torques and send to robot
            action_to_send = {f"{m}.effort": tau_cmd_f[i] for i, m in enumerate(robot.bus.motors)}
            robot.send_action(action_to_send)
        else:
            # Original logic for standard position-based replay
            robot.send_action(action_from_ds)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / dataset.fps - dt_s)

    total_time = time.perf_counter() - start_time_all
    actual_fps = idx / total_time if total_time > 0 else float("inf")
    logging.info(f"Average FPS achieved over episode: {actual_fps:.2f}")
    log_say(f"Average FPS achieved: {actual_fps:.2f}", cfg.play_sounds)

    robot.disconnect()


if __name__ == "__main__":
    replay()

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

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Examples:

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=<USER>/record-test \
    --dataset.episode=0
```

Example replay with bimanual so100:
```shell
lerobot-replay \
  --robot.type=bi_so_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.processor import (
    make_default_robot_action_processor,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1,
)
from lerobot.common.control_utils import smooth_follower_to_action
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
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
    # Root directory where the dataset will be stored (e.g. 'dataset/path'). If None, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Smoothly move to episode start (and back on exit). Set 0 to disable.
    smooth_handover_duration_s: float = 1.0
    smooth_handover_fps: int = 30


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    actions = dataset.select_columns(ACTION)

    robot.connect()

    # Capture pre-replay pose for optional soft return.
    pre_replay_pose = None
    first_processed = None
    if cfg.smooth_handover_duration_s > 0 and dataset.num_frames > 0:
        robot_obs = robot.get_observation()
        action_keys = getattr(robot, "action_features", {}) or {}
        pre_replay_pose = {
            k: robot_obs[k]
            for k in action_keys
            if k in robot_obs
        }
        if not pre_replay_pose:
            pre_replay_pose = {
                k: v for k, v in robot_obs.items() if isinstance(k, str) and k.endswith(".pos")
            }
        action_array0 = actions[0][ACTION]
        action0 = {
            name: action_array0[i] for i, name in enumerate(dataset.features[ACTION]["names"])
        }
        first_processed = robot_action_processor((action0, robot_obs))
        logging.info(
            "Smooth handover to episode start (%.2fs @ %d Hz)",
            cfg.smooth_handover_duration_s,
            cfg.smooth_handover_fps,
        )
        smooth_follower_to_action(
            robot,
            first_processed,
            duration_s=cfg.smooth_handover_duration_s,
            fps=cfg.smooth_handover_fps,
        )

    try:
        log_say("Replaying episode", cfg.play_sounds, blocking=True)
        for idx in range(dataset.num_frames):
            start_episode_t = time.perf_counter()

            action_array = actions[idx][ACTION]
            action = {}
            for i, name in enumerate(dataset.features[ACTION]["names"]):
                action[name] = action_array[i]

            robot_obs = robot.get_observation()

            processed_action = robot_action_processor((action, robot_obs))

            _ = robot.send_action(processed_action)

            dt_s = time.perf_counter() - start_episode_t
            precise_sleep(max(1 / dataset.fps - dt_s, 0.0))
    finally:
        if (
            cfg.smooth_handover_duration_s > 0
            and pre_replay_pose
            and robot.is_connected
        ):
            logging.info(
                "Smooth return to pre-replay pose (%.2fs @ %d Hz)",
                cfg.smooth_handover_duration_s,
                cfg.smooth_handover_fps,
            )
            try:
                smooth_follower_to_action(
                    robot,
                    pre_replay_pose,
                    duration_s=cfg.smooth_handover_duration_s,
                    fps=cfg.smooth_handover_fps,
                )
            except Exception:
                logging.exception("Smooth return to pre-replay pose failed; disconnecting anyway")
        robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()

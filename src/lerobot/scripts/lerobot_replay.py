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
"""

import logging
import time
import keyboard  # Import pour la touche Echap
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


@parser.wrap()
def replay(cfg: ReplayConfig):
    # --- INITIALISATION DU ROBOT ET DU DATASET (Lignes qui manquaient) ---
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    actions = dataset.select_columns(ACTION)
    # ---------------------------------------------------------------------

    robot.connect()

    try:
        
        keep_playing = True
        idpaly=0

        while keep_playing:
            idpaly+=1
            print(f"\n[INFO] Starting replay iteration {idpaly}...")
            for idx in range(dataset.num_frames):
                
                # --- Vérification de la touche Échap ---
                if keyboard.is_pressed('esc'):
                    print("\n[!] Touche Échap pressée. Arrêt du replay.")
                    keep_playing = False
                    break  # Sort de la boucle 'for'
                # ---------------------------------------

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
            
            # Si on n'a pas appuyé sur Échap, on attend avant de recommencer
            if keep_playing:
                time.sleep(1) 

    except KeyboardInterrupt:
        # Permet de quitter proprement avec Ctrl+C si Echap ne marche pas
        print("\n[!] Arrêt forcé via Ctrl+C.")
    finally:
        robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
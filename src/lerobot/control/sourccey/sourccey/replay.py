import time
from dataclasses import dataclass

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.sourccey.sourccey.sourccey import SourcceyClientConfig, SourcceyClient
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say
from lerobot.configs import parser
import logging
from pprint import pformat
from dataclasses import asdict
from lerobot.utils.utils import init_logging

@dataclass
class DatasetReplayConfig:
    dataset: str = "sourccey-001/sourccey-001__wave_hand-001"
    episode: int = 0
    fps: int = 30

@dataclass
class SourcceyReplayConfig:
    id: str = "sourccey"
    remote_ip: str = "192.168.1.237"
    dataset: DatasetReplayConfig = DatasetReplayConfig()

@parser.wrap()
def replay(cfg: SourcceyReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Intialize the robot config and the robot
    robot_config = SourcceyClientConfig(remote_ip=cfg.remote_ip, id=cfg.id)
    robot = SourcceyClient(robot_config)

    # Fetch the dataset to replay
    dataset = LeRobotDataset(cfg.dataset.dataset, episodes=[cfg.dataset.episode])

    # Filter dataset to only include frames from the specified episode since episodes are chunked in dataset V3.0
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.episode)
    actions = episode_frames.select_columns("action")

    # Connect to the robot
    robot.connect()

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print("Starting replay loop...")
    log_say(f"Replaying episode {cfg.dataset.episode}")
    for idx in range(len(episode_frames)):
        t0 = time.perf_counter()

        action = {
            name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
        }

        robot.send_action(action)

        busy_wait(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

    robot.disconnect()

def main():
    replay()

if __name__ == "__main__":
    main()

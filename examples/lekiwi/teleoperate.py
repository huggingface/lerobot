from dataclasses import asdict, dataclass
from pprint import pformat
import draccus
import rerun as rr
import time

from examples.lekiwi.utils import display_data
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.common.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.visualization_utils import _init_rerun

@dataclass
class ReplayConfig:
    dataset_path: str = "local/lekiwi_001_tape_a"
    episode: int = 0
    robot_ip: str = "192.168.1.204"
    robot_id: str = "lekiwi"
    display_data: bool = False
    rerun_session_name: str = "lekiwi_replay"


@draccus.wrap()
def replay(cfg: ReplayConfig):
    if cfg.display_data:
        _init_rerun(session_name=cfg.rerun_session_name)

    # Initialize robot
    robot_config = LeKiwiClientConfig(
        remote_ip=cfg.robot_ip,
        id=cfg.robot_id
    )
    robot = LeKiwiClient(robot_config)

    # Load dataset
    dataset = LeRobotDataset(cfg.dataset_path, episodes=[cfg.episode])

    # Connect to robot
    robot.connect()
    if not robot.is_connected:
        print("Failed to connect to robot")
        return

    try:
        for i, action_array in enumerate(dataset.hf_dataset["action"]):
            t0 = time.perf_counter()

            action = {name: float(action_array[i]) for i, name in enumerate(dataset.features["action"]["names"])}

            # Display data in Rerun if enabled
            if cfg.display_data:
                observation = robot.get_observation()
                display_data(observation, action)

            robot.send_action(action)

            # Maintain timing
            busy_wait(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nReplay stopped by user")
    finally:
        print("Cleaning up...")
        rr.rerun_shutdown()
        robot.disconnect()
        print("Replay ended")


if __name__ == "__main__":
    replay()

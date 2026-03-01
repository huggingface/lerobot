from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say
from lerobot.robots.meca.mecaconfig import MecaConfig
from lerobot.robots.meca.meca import Meca

import time

# Setup your Meca
meca_cfg = MecaConfig(ip="192.168.0.100")
robot = Meca(meca_cfg)
robot.connect()

# Load dataset
dataset = LeRobotDataset("dylanmcguir3/meca-needle-pick")
actions = dataset.hf_dataset.select_columns("action")

# Choose which episodes to replay
episodes_to_replay = [0, 1]   # put indices here
RESET_TIME_SEC = 5               # how long to wait during reset

for episode_idx in episodes_to_replay:
    log_say(f"Replaying episode {episode_idx}")
    episode = dataset.hf_dataset.filter(lambda e: e["episode_index"] == episode_idx)

    for idx in range(len(episode)):
        t0 = time.perf_counter()

        action = {
            name: float(episode[idx]["action"][i]) 
            for i, name in enumerate(dataset.features["action"]["names"])
        }
        print(f"Step {idx+1}/{len(episode)}, Action: {action}")
        robot.send_action(action)

        busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))

    # Reset between episodes
    robot.reset()
    # Optionally send robot to home pose
    # robot.reset()  # uncomment if you have a reset method

robot.disconnect()

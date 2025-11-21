from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1, G1_29_JointIndex
from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import time
config = UnitreeG1Config(
    motion_mode=False,
    simulation_mode=False
)

robot = UnitreeG1(config)
#print(robot.get_observation())
#robot.calibrate()
#print(robot.get_observation())
while True:
   observation = robot.get_observation()
   robot.send_action(observation)
   #print(robot.get_observation())
   time.sleep(0.01)
# print(observation)
# time.sleep(0.1)

# dataset = LeRobotDataset(repo_id='nepyope/unitree_box_push_30fps_actions_fix', root="", episodes=[0])

# actions = dataset.hf_dataset.select_columns("action")
# print(actions)
# episode_idx = 0
# episode_info = dataset.meta.episodes[episode_idx]

# from_idx = episode_info["dataset_from_index"]
# to_idx = episode_info["dataset_to_index"]


# for idx in range(from_idx, to_idx):
#             robot.send_action(actions[idx]["action"].numpy()[:14])
#             time.sleep(1.0 / 30)

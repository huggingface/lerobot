import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.common.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.common.utils.robot_utils import busy_wait

robot_config = LeKiwiClientConfig(remote_ip="172.18.134.136", id="lekiwi")
robot = LeKiwiClient(robot_config)

dataset = LeRobotDataset("pepijn223/lekiwi1749025613", episodes=[0])

robot.connect()

print("Replaying episode…")
for _, action_array in enumerate(dataset.hf_dataset["action"]):
    t0 = time.perf_counter()

    action = {name: float(action_array[i]) for i, name in enumerate(dataset.features["action"]["names"])}
    robot.send_action(action)

    busy_wait(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

print("Disconnecting LeKiwi Client")
robot.disconnect()

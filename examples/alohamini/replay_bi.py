"""
Usage:
  python examples/alohamini/replay_bi.py \
    --dataset liyitenga/alohamini_test1 \
    --episode 0 \
    --remote_ip 192.168.50.84
"""

import argparse
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.alohamini.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.alohamini.lekiwi_client import LeKiwiClient
from lerobot.utils.robot_utils import precise_sleep

parser = argparse.ArgumentParser(description="Replay a LeRobot dataset episode")
parser.add_argument(
    "--dataset", type=str, required=True, help="Dataset repo_id, e.g. liyitenga/record_20250914225057"
)
parser.add_argument("--episode", type=int, default=0, help="Episode index to replay (default 0)")
parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="LeKiwi host IP address")
parser.add_argument("--robot_id", type=str, default="lekiwi", help="Robot ID")


args = parser.parse_args()


robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id)
robot = LeKiwiClient(robot_config)


# dataset = LeRobotDataset("liyitenga/record_20250914225057", episodes=[EPISODE_IDX])
dataset = LeRobotDataset(args.dataset, episodes=[args.episode])
actions = dataset.hf_dataset.select_columns("action")
# print(f"Dataset loaded with id: {dataset.repo_id}, num_frames: {dataset.num_frames}")

robot.connect()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

# log_say(f"Replaying episode {args.episode} from {args.dataset}")
print(f"Replaying episode {args.episode} from {args.dataset}")
for idx in range(dataset.num_frames):
    t0 = time.perf_counter()

    action = {
        name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
    }

    print(f"replay_bi.action:{action}")
    robot.send_action(action)

    precise_sleep(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

robot.disconnect()

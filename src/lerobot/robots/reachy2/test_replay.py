from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.reachy2 import Reachy2Robot, Reachy2RobotConfig
from lerobot.utils.robot_utils import busy_wait
import time


# Create the robot configuration
robot_config = Reachy2RobotConfig(
    ip_address="192.168.0.199",
    id="reachy2-pvt02",
)

# Initialize the robot
robot = Reachy2Robot(robot_config)

# Create the dataset
dataset = LeRobotDataset(repo_id="glannuzel/store-rubiks-cube", episodes=[0])
actions = dataset.hf_dataset.select_columns("action")

# Connect the robot
robot.connect()

# Go smoothly to the first action
action_array = actions[0]["action"]
action = {}
for i, name in enumerate(dataset.features["action"]["names"]):
    action[name] = action_array[i].item()

neck_goal = [action["neck_roll.pos"], action["neck_pitch.pos"], action["neck_yaw.pos"]]
r_arm_goal = [action["r_shoulder_pitch.pos"],
              action["r_shoulder_roll.pos"],
              action["r_elbow_yaw.pos"],
              action["r_elbow_pitch.pos"],
              action["r_wrist_roll.pos"],
              action["r_wrist_pitch.pos"],
              action["r_wrist_yaw.pos"]]
l_arm_goal = [action["l_shoulder_pitch.pos"],
              action["l_shoulder_roll.pos"],
              action["l_elbow_yaw.pos"],
              action["l_elbow_pitch.pos"],
              action["l_wrist_roll.pos"],
              action["l_wrist_pitch.pos"],
              action["l_wrist_yaw.pos"]]

robot.reachy.head.goto(neck_goal)
robot.reachy.r_arm.goto(r_arm_goal)
robot.reachy.r_arm.gripper.goto(100.0, percentage=True)
robot.reachy.l_arm.gripper.goto(100.0, percentage=True)
robot.reachy.l_arm.goto(l_arm_goal, wait=True)

for idx in range(dataset.num_frames):
    start_episode_t = time.perf_counter()

    action_array = actions[idx]["action"]

    action = {}
    for i, name in enumerate(dataset.features["action"]["names"]):
        action[name] = action_array[i].item()

    robot.send_action(action)
    dt_s = time.perf_counter() - start_episode_t
    busy_wait(1 / dataset.fps - dt_s)

# Clean up
robot.disconnect()

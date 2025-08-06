from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.robots.reachy2 import Reachy2Robot, Reachy2RobotConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
# from lerobot.utils.visualization_utils import _init_rerun
from lerobot.record import record_loop
from reachy2_sdk import ReachySDK
from lerobot.utils.robot_utils import busy_wait
import numpy as np
import time


REACHY2_MOTORS = {
    "neck_yaw.pos": "head.neck.yaw",
    "neck_pitch.pos": "head.neck.pitch",
    "neck_roll.pos": "head.neck.roll",
    "r_shoulder_pitch.pos": "r_arm.shoulder.pitch",
    "r_shoulder_roll.pos": "r_arm.shoulder.roll",
    "r_elbow_yaw.pos": "r_arm.elbow.yaw",
    "r_elbow_pitch.pos": "r_arm.elbow.pitch",
    "r_wrist_roll.pos": "r_arm.wrist.roll",
    "r_wrist_pitch.pos": "r_arm.wrist.pitch",
    "r_wrist_yaw.pos": "r_arm.wrist.yaw",
    "r_gripper.pos": "r_arm.gripper",
    "l_shoulder_pitch.pos": "l_arm.shoulder.pitch",
    "l_shoulder_roll.pos": "l_arm.shoulder.roll",
    "l_elbow_yaw.pos": "l_arm.elbow.yaw",
    "l_elbow_pitch.pos": "l_arm.elbow.pitch",
    "l_wrist_roll.pos": "l_arm.wrist.roll",
    "l_wrist_pitch.pos": "l_arm.wrist.pitch",
    "l_wrist_yaw.pos": "l_arm.wrist.yaw",
    "l_gripper.pos": "l_arm.gripper",
    "l_antenna.pos": "head.l_antenna",
    "r_antenna.pos": "head.r_antenna",
}

# Create the robot configuration
robot_config = Reachy2RobotConfig(
    # ip_address="localhost",
    # ip_address="172.18.131.66",
    ip_address="192.168.0.199",
    id="reachy2-pvt02",
)

# Initialize the robot
robot = Reachy2Robot(robot_config)


reachy = ReachySDK(robot_config.ip_address)


# Create the dataset
dataset = LeRobotDataset(repo_id="glannuzel/grab_cube_2", episodes=[0])
actions = dataset.hf_dataset.select_columns("action")

# Connect the robot
robot.connect()

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

reachy.head.goto(neck_goal)
reachy.r_arm.goto(r_arm_goal)
reachy.l_arm.goto(l_arm_goal, wait=True)

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
# robot.disconnect()

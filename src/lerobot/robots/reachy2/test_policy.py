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
import numpy as np
import time


NUM_EPISODES = 2
FPS = 15
EPISODE_TIME_SEC = 4
TASK_DESCRIPTION = "Grab a cube with Reachy 2"


# Create the robot configuration
robot_config = Reachy2RobotConfig(
    ip_address="192.168.0.199",
    id="reachy2-pvt02",
    with_mobile_base=False,
)

# Initialize the robot
robot = Reachy2Robot(robot_config)
# Instantiate a client for starting and intermediate positions
# reachy = ReachySDK(robot_config.ip_address)

# Initialize the policy
policy = ACTPolicy.from_pretrained("pepijn223/grab_cube_2")

# Get initial dataset first episode
initial_dataset = LeRobotDataset(repo_id="glannuzel/grab_cube_2", episodes=[0])
actions = initial_dataset.hf_dataset.select_columns("action")

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="glannuzel/eval_grab_cube_2.1",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
# _init_rerun(session_name="recording")

# Connect the robot
robot.connect()

# Go to the initial pose
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
robot.reachy.r_arm.gripper.goto(100.0)
robot.reachy.l_arm.goto(l_arm_goal, wait=True)
time.sleep(1.0)

for episode_idx in range(NUM_EPISODES):
    log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

    # Run the policy inference loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        policy=policy,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=False,
    )

    # Set the robot back to the initial pose
    robot.reachy.head.goto(neck_goal)
    robot.reachy.r_arm.goto(r_arm_goal)
    robot.reachy.r_arm.gripper.goto(100.0)
    robot.reachy.l_arm.goto(l_arm_goal, wait=True)
    time.sleep(1.0)

    dataset.save_episode()

# Clean up
robot.disconnect()
dataset.push_to_hub()

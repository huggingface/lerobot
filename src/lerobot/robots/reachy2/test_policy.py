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

NUM_EPISODES = 5
FPS = 20
EPISODE_TIME_SEC = 10
TASK_DESCRIPTION = "Grab a cube in Mujoco simulation"


# Create the robot configuration
robot_config = Reachy2RobotConfig(
    ip_address="localhost",
    # ip_address="172.18.131.66",
    id="test_reachy",
)

# Initialize the robot
robot = Reachy2Robot(robot_config)

reachy = ReachySDK("localhost")
reachy.turn_on()
reachy.mobile_base.goto(-0.2, -0.3, 0, wait=True)
time.sleep(2)
reachy.r_arm.goto_posture("elbow_90", wait=True)
reachy.r_arm.gripper.open()
reachy.mobile_base.goto(0, -0.3, 0)

# Initialize the policy
policy = ACTPolicy.from_pretrained("pepijn223/grab_cube_simulation_2")

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="glannuzel/eval_grab_cube_simulation_2",
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

M = reachy.r_arm.get_default_posture_matrix("elbow_90")
np.round(M, 3)
first_pose = M.copy()
first_pose[0, 3] += 0.05
first_pose[1, 3] += 0.1

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

    reachy.r_arm.gripper.goto(100, percentage=True, wait=True)
    reachy.head.goto_posture()
    reachy.r_arm.goto(first_pose)
    reachy.r_arm.goto(M, wait=True)

    dataset.save_episode()

# Clean up
robot.disconnect()
# dataset.push_to_hub()
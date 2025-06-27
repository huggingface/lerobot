import time

from examples.lekiwi.utils import lekiwi_record_loop
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import hw_to_dataset_features
from lerobot.common.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.common.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.common.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.common.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.common.utils.utils import log_say
from lerobot.common.utils.visualization_utils import _init_rerun

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "My task description"

# Create the robot and teleoperator configurations
robot_config = LeKiwiClientConfig(remote_ip="172.18.134.136", id="lekiwi")
leader_arm_config = SO100LeaderConfig(port="/dev/tty.usbmodem585A0077581", id="my_awesome_leader_arm")
keyboard_config = KeyboardTeleopConfig()

robot = LeKiwiClient(robot_config)
leader_arm = SO100Leader(leader_arm_config)
keyboard = KeyboardTeleop(keyboard_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="<hf_username>/<dataset_repo_id>",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

leader_arm.connect()
keyboard.connect()
robot.connect()

_init_rerun(session_name="lekiwi_record")

if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
    exit()

for episode_idx in range(NUM_EPISODES):
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    # Run the record loop
    lekiwi_record_loop(
        robot=robot,
        fps=FPS,
        teleop_arm=leader_arm,
        teleop_keyboard=keyboard,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        log_data=True,
    )

    log_say("Reset environment")
    time.sleep(int(RESET_TIME_SEC))

# Clean up and upload to hub
dataset.save_episode()
dataset.push_to_hub()

robot.disconnect()
leader_arm.disconnect()
keyboard.disconnect()

"""
ALOHA Bimanual Recording Script

This script records episodes using ALOHA dual-arm system (ViperX followers + WidowX leaders).

Usage:
1. New dataset: Set RESUME = False
2. Resume/append: Set RESUME = True (will continue from existing episodes)

The script will:
- Record NUM_EPISODES new episodes
- Show progress with total episode count
- Push dataset to HuggingFace Hub when complete
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.record import record_loop
from lerobot.robots.aloha import Aloha, AlohaConfig
from lerobot.teleoperators.aloha_teleop import AlohaTeleop, AlohaTeleopConfig
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun

# Recording configuration
NUM_EPISODES = 0
FPS = 30
EPISODE_TIME_SEC = 200
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "First put the Hugging Face t shirt with both arms in the box, then place the hat with the right arm in the box."
REPO_ID = "pepijn223/aloha_box_2"
RESUME = True  # Set to True to resume/append to existing dataset

# Create camera configuration
camera_config = {
    "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
    "wrist_right": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS),
    "wrist_left": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=FPS),
}

# ALOHA Robot Configuration (dual ViperX followers)
aloha_robot_config = AlohaConfig(
    id="aloha",
    left_arm_port="/dev/tty.usbserial-FT89FM09",
    right_arm_port="/dev/tty.usbserial-FT891KBG",
    left_arm_max_relative_target=20.0,
    right_arm_max_relative_target=20.0,
    left_arm_use_degrees=True,
    right_arm_use_degrees=True,
    cameras=camera_config,
)

# ALOHA Teleoperator Configuration (dual WidowX leaders)
aloha_teleop_config = AlohaTeleopConfig(
    id="aloha_teleop",
    left_arm_port="/dev/tty.usbserial-FT891KPN",
    right_arm_port="/dev/tty.usbserial-FT89FM77",
    left_arm_gripper_motor="xl430-w250",
    right_arm_gripper_motor="xc430-w150",
    left_arm_use_degrees=True,
    right_arm_use_degrees=True,
)

# Initialize the robot and teleoperator
robot = Aloha(aloha_robot_config)
teleop = AlohaTeleop(aloha_teleop_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create or resume the dataset
if RESUME:
    print(f"Resuming existing dataset: {REPO_ID}")
    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=None,  # Use default root
    )

    # Start image writer for cameras
    if hasattr(robot, "cameras") and len(robot.cameras) > 0:
        dataset.start_image_writer(
            num_processes=0,  # Use threads only
            num_threads=4 * len(robot.cameras),  # 4 threads per camera
        )

    # Sanity check compatibility
    sanity_check_dataset_robot_compatibility(dataset, robot, FPS, dataset_features)
    print(f"Resumed dataset with {dataset.num_episodes} existing episodes")
else:
    print(f"Creating new dataset: {REPO_ID}")
    # Sanity check dataset name
    sanity_check_dataset_name(REPO_ID, None)

    # Create new dataset
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4 * len(robot.cameras),  # 4 threads per camera
    )

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
_init_rerun(session_name="aloha_recording")

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

episode_idx = 0
total_episodes_to_record = NUM_EPISODES
existing_episodes = dataset.num_episodes if RESUME else 0

while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    current_episode = existing_episodes + episode_idx + 1
    log_say(f"Recording episode {current_episode} (batch: {episode_idx + 1}/{NUM_EPISODES})")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()

# Summary
final_episodes = dataset.num_episodes
log_say(f"Dataset now contains {final_episodes} episodes total")

# Push to hub
dataset.push_to_hub()
log_say(f"Dataset '{REPO_ID}' pushed to HuggingFace Hub")

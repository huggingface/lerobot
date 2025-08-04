from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.robots.reachy2 import Reachy2Robot, Reachy2RobotConfig
from lerobot.teleoperators.reachy2_fake_teleoperator import Reachy2FakeTeleoperator, Reachy2FakeTeleoperatorConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.record import record_loop

import time

NUM_EPISODES = 35
FPS = 20
EPISODE_TIME_SEC = 10
RESET_TIME_SEC = 5
TASK_DESCRIPTION = "Grab a cube in Mujoco simulation"

# Create the robot and teleoperator configurations
robot_config = Reachy2RobotConfig(
    # ip_address="localhost",
    # ip_address="172.18.131.66",
    ip_address="192.168.0.200",
    id="test_reachy",
)
teleop_config = Reachy2FakeTeleoperatorConfig(
    # ip_address="172.18.131.66",
    ip_address="192.168.0.200",
)

# Initialize the robot and teleoperator
robot = Reachy2Robot(robot_config)
teleop = Reachy2FakeTeleoperator(teleop_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="glannuzel/grab_cube",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
# _init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    start_time = time.time()
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    print("########### RECORDING ###########")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=False,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")

        print("------------- RESETTING -------------")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=False,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    # episode_idx = NUM_EPISODES
    dataset.save_episode()
    episode_idx += 1
    print(time.time()-start_time)

# Clean up
log_say("Stop recording")
robot.disconnect()
dataset.push_to_hub()

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

from lerobot.teleoperators.ps4_joystick import PS4JoystickTeleop, PS4JoystickTeleopConfig
from lerobot.robots.assembling_sim import AssemblingSim, AssemblingSimCut, AssemblingSimConfig

import numpy as np

NUM_EPISODES = 50
FPS = 20
EPISODE_TIME_SEC = 25
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "My task description"

sim_config = AssemblingSimConfig(
    xml_path="scene.xml",
    sim_timestep=0.001,
    control_hz=20,
    mode="fast",   # "realtime" | "fast"
    max_episode_steps=1000,
    use_task_space=True,
    render_mode="all",   # None | "human" | "rgb_array" | "all"
    camera_names=["cam_front", "cam_side", "cam_gripper", "cam_state"],
    resolution=(224, 224),

    action_pos_scale=1000,
    action_angle_scale=100
)

teleop_config = PS4JoystickTeleopConfig(
    id="my_teleop_ps4_joystick",
    max_speed=0.05,
    max_rot_speed=0.5,
    deadzone=0.05,
    alpha=0.3,
    poll_rate=100,
    x_init=0.1,
    y_init=-0.65,
    z_init=0.37,
    roll_init=np.pi,
    pitch_init=0.0,
    yaw_init=np.pi/2
)

# Initialize the robot and teleoperator
robot = AssemblingSimCut(sim_config)
teleop = PS4JoystickTeleop(teleop_config)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id="local/ACT_assembling_sim_s3",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    video_backend="libx264",
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

# Create the required processors
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        robot.reset()
        teleop.reset()

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
# dataset
# dataset.save_episode
dataset.finalize()
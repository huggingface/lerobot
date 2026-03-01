from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop

# Import your robot + teleop
from lerobot.robots.meca.meca import Meca
from lerobot.robots.meca.mecaconfig import MecaConfig
from lerobot.teleoperators.omni.omni import OmniTeleoperator, OmniConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

# ------------------------
# Experiment parameters
# ------------------------
NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Microsurgery teleop task"

# ------------------------
# Configurations
# ------------------------

# Cameras (adapt indices to your setup)
camera_config = {
    "top": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
    "bottom": OpenCVCameraConfig(index_or_path=2, width=1280, height=720, fps=260),
}

# Meca robot config
meca_cfg = MecaConfig(
    ip="192.168.0.100",       # <-- Replace with your robot IP
    id="meca500",
    cameras=camera_config,
)

# Omni haptic device config
omni_cfg = OmniConfig()

# ------------------------
# Instantiate
# ------------------------
robot = Meca(meca_cfg)
teleop = OmniTeleoperator(omni_cfg)

# Dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}



# try:
#     dataset = LeRobotDataset("dylanmcguir3/meca-needle-pick-lr")
#     print("📂 Loaded existing dataset")
# except Exception:
    # Otherwise, create it fresh
dataset = LeRobotDataset.create(
    repo_id="dylanmcguir3/meca-needle-pick-lr",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)
print("📂 Created new dataset")

# ------------------------
# Setup utils
# ------------------------
_, events = init_keyboard_listener()
init_rerun(session_name="meca_teleop_recording")

# ------------------------
# Main recording loop
# ------------------------
robot.connect()
teleop.connect()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=robot.teleop_action_processor,
        robot_action_processor=robot.robot_action_processor,
        robot_observation_processor=robot.robot_observation_processor,
    )

    # Reset environment if needed
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Resetting environment...")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            robot_observation_processor=robot.robot_observation_processor,
            teleop_action_processor=robot.teleop_action_processor,
            robot_action_processor=robot.robot_action_processor,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# ------------------------
# Cleanup
# ------------------------
log_say("Stopping recording...")
robot.disconnect()
teleop.disconnect()
dataset.push_to_hub()

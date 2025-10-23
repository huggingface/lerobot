from pathlib import Path
import time

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy  # 👈 change here
from lerobot.robots.meca.mecaconfig import MecaConfig
from lerobot.robots.meca.meca import Meca
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.policies.factory import make_policy, make_pre_post_processors
import cv2


# -------------------------
# User settings
# -------------------------
NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "Microsurgery teleop task"

# 👇 Fill in your actual repos
HF_USER = "dylanmcguir3"
HF_MODEL_ID = f"{HF_USER}/needle-pick-act"        # ACT policy model
HF_DATASET_ID = f"{HF_USER}/meca-needle-pick"    # dataset to store evaluation rollouts

# -------------------------
# Robot + camera config
# -------------------------
camera_config = {
    "top": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
    "bottom": OpenCVCameraConfig(index_or_path=2, width=1280, height=720, fps=260),
}

robot_config = MecaConfig(ip="192.168.0.100", id="meca_eval_robot", cameras=camera_config)
robot = Meca(robot_config)

# -------------------------
# Load trained ACT policy
# -------------------------
policy = ACTPolicy.from_pretrained(Path("/home/dylan/LeRobot/lerobot/outputs/train/needle-pick-act-test/checkpoints/100000/pretrained_model"))

# -------------------------
# Dataset features
# -------------------------
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

dataset = LeRobotDataset.create(
    repo_id=HF_DATASET_ID + "-act-eval",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# -------------------------
# Keyboard + visualization
# -------------------------
_, events = init_keyboard_listener()
init_rerun(session_name="policy_eval_act")

# -------------------------
# Connect robot
# -------------------------
robot.connect()

# Pre/post processors for ACT policy
preprocessor, postprocessor = make_pre_post_processors(
    policy, 
    pretrained_path=Path("/home/dylan/LeRobot/lerobot/outputs/train/needle-pick-act-test/checkpoints/100000/pretrained_model"),
    dataset_stats=dataset.meta.stats
)

# -------------------------
# Evaluation loop
# -------------------------
for episode_idx in range(NUM_EPISODES):
    log_say(f"Running inference, recording ACT eval episode {episode_idx + 1}/{NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=robot.teleop_action_processor,
        robot_action_processor=robot.robot_action_processor,
        robot_observation_processor=robot.robot_observation_processor,
    )

    if cv2.waitKey(1) & 0xFF == ord("r"):
        robot.reset()

    dataset.save_episode()

# -------------------------
# Cleanup
# -------------------------
robot.disconnect()
dataset.push_to_hub()

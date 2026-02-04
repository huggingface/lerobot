import shutil

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.piper_dual.config_piper_dual import PIPERDualConfig
from lerobot.robots.piper_dual.piper_dual import PIPERDual
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 60
HF_MODEL_ID = "/home/droplab/workspace/lerobot_piper/outputs/train/act_pick_and_place_50/checkpoints/last/pretrained_model"
HF_DATASET_ID = "local/eval_recording_test"
TASK_DESCRIPTION = "Dual arm evaluation task"
POLICY_CHUNK_SIZE = 50
POLICY_N_ACTION_STEPS = 50

# Create the robot configuration
camera_config = {
    "left": OpenCVCameraConfig(index_or_path="/dev/video6", width=640, height=480, fps=FPS),
    "right": OpenCVCameraConfig(index_or_path="/dev/video12", width=640, height=480, fps=FPS),
    "middle": OpenCVCameraConfig(index_or_path="/dev/video0", width=640, height=480, fps=FPS),
}

robot_config = PIPERDualConfig(left_port="can_left", right_port="can_right", cameras=camera_config)

# Initialize the robot
robot = PIPERDual(robot_config)

# Initialize the policy
policy = ACTPolicy.from_pretrained(HF_MODEL_ID)
# Update policy config if needed, though from_pretrained usually handles it.
# The command line args --policy.chunk_size=50 --policy.n_action_steps=50 suggest we might need to override config,
# but usually checking the policy config is enough.
# Assuming the pretrained model has these configs or we trust the loaded policy.
# However, if we need to explicitly set them on the loaded policy object:
policy.config.chunk_size = POLICY_CHUNK_SIZE
policy.config.n_action_steps = POLICY_N_ACTION_STEPS


# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Check if the dataset already exists and delete it if it does
dataset_path = HF_LEROBOT_HOME / HF_DATASET_ID
if dataset_path.exists():
    shutil.rmtree(dataset_path)

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=HF_DATASET_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Connect the robot
robot.connect()

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=HF_MODEL_ID,
    dataset_stats=dataset.meta.stats,
)

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

for episode_idx in range(NUM_EPISODES):
    log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

    # Run the policy inference loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    dataset.save_episode()

# Clean up
robot.disconnect()
# dataset.push_to_hub=false (implicit by not calling it, but dataset.push_to_hub() call was at the end)
# The user specified --dataset.push_to_hub=false, so we simply DO NOT call dataset.push_to_hub()

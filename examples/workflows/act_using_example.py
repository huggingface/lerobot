import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_STR

device = torch.device("mps")  # or "cuda" or "cpu"

# # find ports using lerobot-find-port
# follower_port = "/dev/tty.usbmodem58760431631"

# # the robot ids are used the load the right calibration files
# follower_id = "follower_so100"

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20

# Robot and environment configuration
camera_config = {
    "side": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
    "up": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
}
# robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
# robot = SO100Follower(robot_cfg)
# robot.connect()

model_id = "fracapuano/robot_learning_tutorial_act_example_model"

dataset_id = "lerobot/svla_so101_pickplace"

dataset_metadata = LeRobotDatasetMetadata(dataset_id)
model = ACTPolicy.from_pretrained(model_id)

preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

obs = {
    # Motor positions (6 motors)
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
    # Camera images (C, H, W)
    "side": np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8),
    "up": np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8),
}

for _ in range(MAX_EPISODES):
    for _ in range(MAX_STEPS_PER_EPISODE):
        # obs = robot.get_observation()
        obs_frame = build_dataset_frame(dataset_metadata.features, obs, prefix=OBS_STR)
        obs_frame = {k: torch.from_numpy(v) for k, v in obs_frame.items()}

        obs = preprocess(obs_frame)

        action = model.select_action(obs)
        action = postprocess(action)
        # robot.send_action(action)

    print("Episode finished! Starting new episode...")

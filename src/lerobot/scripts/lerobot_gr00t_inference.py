#!/usr/bin/env python

"""
Run GR00T policy inference on real robot and record evaluation episodes.

Setup:
    1. Ensure checkpoint exists at CHECKPOINT_PATH
    2. Configure via environment variables
    3. Ensure robot is connected and calibrated

Usage:
    python lerobot_gr00t_inference.py

Environment Variables:
    CHECKPOINT_PATH: Path to checkpoint (default: /outputs/train/so101_gr00t_test/checkpoints/020000/pretrained_model)
    DATA_CONFIG: Data config name (default: so100_fronttop)
    EMBODIMENT_TAG: Embodiment tag (default: new_embodiment)
    DENOISING_STEPS: Denoising steps (default: 4)
    EVAL_DATASET_ID: Hugging Face repo for eval dataset (default: eval_so101_gr00t_test)
    NUM_EPISODES: Number of evaluation episodes (default: 5)
    TASK_DESCRIPTION: Task description for eval (default: "Pick up the cube")
"""

import os
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# Configuration constants
FPS = 30
EPISODE_TIME_SEC = 60

# Inference configuration - MODIFY THESE VALUES
# LeRobot saves checkpoints in: train/{run_name}/checkpoints/{step}/pretrained_model/
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", "/outputs/train/so101_gr00t_test/checkpoints/020000/pretrained_model"
)
DATA_CONFIG = os.environ.get("DATA_CONFIG", "so100_fronttop")  # Reference dataset for modality config
EMBODIMENT_TAG = os.environ.get("EMBODIMENT_TAG", "new_embodiment")
DENOISING_STEPS = int(os.environ.get("DENOISING_STEPS", "4"))
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "5"))
TASK_DESCRIPTION = os.environ.get("TASK_DESCRIPTION", "Stack the rubix cubes on top of each other")
EVAL_DATASET_ID = os.environ.get("EVAL_DATASET_ID", "eval_so101_gr00t_test")

# Camera and robot configuration
camera_config = {
    "front": OpenCVCameraConfig(index_or_path=0, width=1280, height=720, fps=FPS),
    "top": OpenCVCameraConfig(index_or_path=1, width=1920, height=1080, fps=FPS),
}
robot_config = SO100FollowerConfig(
    port="/dev/ttyACM1", id="so101_follower_arm", cameras=camera_config
)


def main():
    """Run GR00T inference on real robot and record evaluation episodes."""
    
    # Initialize the robot
    robot = SO101Follower(robot_config)
    
    # Load reference dataset to get modality config (LeRobot standard)
    reference_dataset = LeRobotDataset(DATA_CONFIG)
    modality_config = reference_dataset.meta.modality_config
    
    # Initialize policy with LeRobot GrootPolicy
    log_say(f"Loading GR00T policy from {CHECKPOINT_PATH}")
    policy = GrootPolicy.from_pretrained(CHECKPOINT_PATH)
    
    # Configure the dataset features for evaluation
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    
    # Create the evaluation dataset
    log_say(f"Creating evaluation dataset: {EVAL_DATASET_ID}")
    dataset = LeRobotDataset.create(
        repo_id=EVAL_DATASET_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )
    
    # Initialize the keyboard listener and rerun visualization
    _, events = init_keyboard_listener()
    init_rerun(session_name="groot_inference")
    
    # Connect the robot
    log_say("Connecting to robot...")
    robot.connect()
    
    # Create pre/post processors using LeRobot factory pattern
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=CHECKPOINT_PATH,
        dataset_stats=dataset.meta.stats,
    )
    
    # Run inference loop for multiple episodes
    for episode_idx in range(NUM_EPISODES):
        log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")
        
        # Run the policy inference loop
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
        )
        
        dataset.save_episode()
    
    # Clean up
    log_say("Evaluation complete. Disconnecting robot and pushing dataset to hub...")
    robot.disconnect()
    dataset.push_to_hub()
    log_say(f"Evaluation dataset pushed to {EVAL_DATASET_ID}")


if __name__ == "__main__":
    main()

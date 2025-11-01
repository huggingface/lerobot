#!/usr/bin/env python

"""
Evaluation script for real robot with policy.
Based on lerobot eval but adapted for physical robots.
"""

import logging
import time

import numpy as np
import torch
from tqdm import trange
from transformers import AutoTokenizer

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class
from lerobot.processor.factory import make_default_robot_observation_processor
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.utils import init_logging, log_say

# Configuration
POLICY_PATH = "helper2424/smolval_move_green_object_to_purple_plate"
DEVICE = "mps"
NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_S = 60
ROBOT_PORT = "/dev/tty.usbmodem58FA0834591"
ROBOT_ID = "so100_follower"


def evaluate_on_real_robot():
    """Run evaluation episodes on real robot."""

    init_logging()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create robot configuration
    camera_config = {
        "gripper": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
        "front": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS),
    }

    robot_config = SO100FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        cameras=camera_config,
        use_degrees=True,
    )

    # Initialize robot
    logger.info("Connecting to robot...")
    robot = SO100Follower(robot_config)
    robot.connect()

    # Load policy
    logger.info(f"Loading policy from {POLICY_PATH}")
    policy_class = get_policy_class("smolvla")
    policy = policy_class.from_pretrained(POLICY_PATH)
    policy = policy.to(DEVICE)
    policy.eval()

    # Load tokenizer for SmolVLA
    tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    task_description = "Move green small object into the purple platform"

    # Create observation processor
    robot_observation_processor = make_default_robot_observation_processor()

    # Get dataset features for observation processing
    dataset_features = hw_to_dataset_features(robot.observation_features, "observation")

    # Evaluation metrics
    episode_results = []

    log_say(f"Starting evaluation for {NUM_EPISODES} episodes")

    for episode_idx in trange(NUM_EPISODES, desc="Evaluation Episodes"):
        log_say(f"Episode {episode_idx + 1}/{NUM_EPISODES}")

        # Reset robot to initial position
        logger.info("Resetting robot position...")
        # Add your reset logic here if needed
        time.sleep(2)  # Wait for manual reset

        # Run episode
        episode_start = time.time()
        step_count = 0
        max_steps = int(EPISODE_TIME_S * FPS)

        while (time.time() - episode_start) < EPISODE_TIME_S and step_count < max_steps:
            step_start = time.time()

            # Get observation
            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)

            # Prepare observation for policy
            obs_with_policy_features = build_dataset_frame(
                dataset_features, obs_processed, prefix="observation"
            )

            # Convert to tensors
            for k, v in obs_with_policy_features.items():
                if isinstance(v, np.ndarray):
                    obs_with_policy_features[k] = torch.from_numpy(v).to(DEVICE)
                if k.startswith("observation.images"):
                    obs_with_policy_features[k] = obs_with_policy_features[k].type(torch.float32) / 255
                    obs_with_policy_features[k] = obs_with_policy_features[k].permute(2, 0, 1).unsqueeze(0)
                elif isinstance(obs_with_policy_features[k], torch.Tensor):
                    obs_with_policy_features[k] = obs_with_policy_features[k].unsqueeze(0)

            # Add task description and language tokens for SmolVLA
            obs_with_policy_features["task"] = "Move green small object into the purple platform"
            obs_with_policy_features["observation.language.tokens"] = (
                "Move green small object into the purple platform"
            )

            # Get action from policy
            with torch.no_grad():
                if hasattr(policy, "predict_action_chunk"):
                    # SmolVLA returns action chunks
                    action_chunk = policy.predict_action_chunk(obs_with_policy_features)
                    # Take the first action from the chunk
                    action = action_chunk[:, 0, :]
                elif hasattr(policy, "predict_action"):
                    action = policy.predict_action(obs_with_policy_features)
                elif hasattr(policy, "predict"):
                    action = policy.predict(obs_with_policy_features)
                else:
                    action = policy(obs_with_policy_features)

            # Convert action to robot format
            action = action.squeeze(0).cpu()
            action_dict = {key: action[i].item() for i, key in enumerate(robot.action_features)}

            # Send action to robot
            robot.send_action(action_dict)

            step_count += 1

            # Maintain FPS
            elapsed = time.time() - step_start
            if elapsed < 1.0 / FPS:
                time.sleep(1.0 / FPS - elapsed)

        # Episode complete
        episode_duration = time.time() - episode_start
        logger.info(
            f"Episode {episode_idx + 1} complete. Duration: {episode_duration:.2f}s, Steps: {step_count}"
        )

        # Get success from user
        success = input(f"Was episode {episode_idx + 1} successful? (y/n): ").lower() == "y"
        episode_results.append(
            {
                "episode": episode_idx + 1,
                "success": success,
                "duration": episode_duration,
                "steps": step_count,
            }
        )

    # Calculate metrics
    num_success = sum(r["success"] for r in episode_results)
    success_rate = num_success / NUM_EPISODES
    avg_duration = np.mean([r["duration"] for r in episode_results])

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Success Rate: {success_rate:.2%} ({num_success}/{NUM_EPISODES})")
    print(f"Average Duration: {avg_duration:.2f}s")
    print("\nEpisode Details:")
    for result in episode_results:
        status = "✓" if result["success"] else "✗"
        print(f"  Episode {result['episode']}: {status} ({result['duration']:.2f}s, {result['steps']} steps)")
    print("=" * 50)

    # Cleanup
    robot.disconnect()
    log_say("Evaluation complete!")


if __name__ == "__main__":
    evaluate_on_real_robot()

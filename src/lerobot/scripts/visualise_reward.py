#!/usr/bin/env python

"""
Script to analyze policy behavior on dataset episodes.
Runs policy inference on episodes and analyzes proprioceptive feature importance.
By default, analyzes all episodes in the dataset.
"""

import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
from reward_wrapper import ACTPolicyWithReward, create_reward_visualization_video
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def load_policy(
    policy_path: str, dataset_meta, policy_overrides: list = None
) -> tuple[torch.nn.Module, dict]:
    """Load and initialize a policy from checkpoint."""

    # Load regular LeRobot policy
    if policy_overrides:
        # Convert list of "key=value" strings to dict
        overrides = {}
        for override in policy_overrides:
            key, value = override.split("=", 1)
            overrides[key] = value
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path, **overrides)
    else:
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.pretrained_path = policy_path

    # NOTE: policy has to be an ACT policy for this to work
    policy = make_policy(policy_cfg, ds_meta=dataset_meta)
    policy = ACTPolicyWithReward(policy)

    return policy, policy_cfg


def prepare_observation_for_policy(
    frame: dict, device: torch.device, model_dtype: torch.dtype = torch.float32, debug: bool = False
) -> dict:
    """Convert dataset frame to policy observation format."""
    observation = {}

    for key, value in frame.items():
        if "image" in key:
            if debug:
                print(f"Processing {key}: original shape {value.shape}, dtype {value.dtype}")

            # Convert image to policy format: channel first, float32 in [0,1], with batch dimension
            if isinstance(value, torch.Tensor):
                # Remove any extra batch dimensions first
                while value.dim() > 3:
                    value = value.squeeze(0)

                # Now we should have 3D tensor in format (H, W, C) from camera
                if value.dim() != 3:
                    raise ValueError(f"Expected 3D tensor for {key} after squeezing, got shape {value.shape}")

                # Camera images from your robot are in (H, W, C) format, so we need to permute to (C, H, W)
                # Let's identify dimensions by size
                h, w, c = value.shape

                # Sanity check: channels should be 1 or 3
                if c not in [1, 3]:
                    # Maybe the format is actually (H, C, W) or (C, H, W)?
                    if h in [1, 3]:
                        # Format is (C, H, W) - already correct
                        pass
                    elif w in [1, 3]:
                        # Format is (H, W, C) but W is the channel dim - unusual
                        value = value.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                    else:
                        # Assume standard (H, W, C) and C is whatever it is
                        value = value.permute(2, 0, 1)
                else:
                    # Standard (H, W, C) format - convert to (C, H, W)
                    value = value.permute(2, 0, 1)

                if debug:
                    print(f"After permutation: {value.shape}")

                # Ensure float and normalize if needed
                if value.dtype != model_dtype:
                    value = value.type(model_dtype)

                # Normalize to [0, 1] if values are in [0, 255] range
                if value.max() > 1.0:
                    value = value / 255.0

                if debug:
                    print(
                        f"Final shape for {key}: {value.shape}, range: [{value.min():.3f}, {value.max():.3f}]"
                    )

            observation[key] = value.unsqueeze(0).to(device)  # Add batch dimension

        elif key in ["observation.state", "robot_state", "state"]:
            # Proprioceptive state
            if not isinstance(value, torch.Tensor):
                value = torch.from_numpy(value).type(model_dtype)
            observation[key] = value.unsqueeze(0).to(device)

    return observation


def analyze_episode(
    dataset: LeRobotDataset,
    policy,
    episode_id: int,
    device: torch.device,
    output_dir: str,
    model_dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Run policy inference on an episode and analyze proprioceptive importance.

    Returns:
        Dictionary containing analysis results
    """

    # Filter dataset to only include the specified episode
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_id)
    episode_length = len(episode_frames)

    if episode_length == 0:
        raise ValueError(f"Episode {episode_id} not found or is empty")

    print(f"Analyzing episode {episode_id} with {episode_length} frames")

    # Initialize reward tracking for ACTPolicyWithReward
    reward_data = []
    reward_images = []

    # Debug policy configuration
    if hasattr(policy, "config"):
        print("=== Policy Configuration ===")

        # Handle image features
        image_features = getattr(policy.config, "image_features", None)
        if image_features:
            if hasattr(image_features, "__iter__") and not isinstance(image_features, str):
                image_feature_names = [getattr(f, "name", str(f)) for f in image_features]
            else:
                image_feature_names = [str(image_features)]
            print(f"Image features: {image_feature_names}")
        else:
            print("Image features: None")

        # Handle robot state feature
        robot_state_feature = getattr(policy.config, "robot_state_feature", None)
        if robot_state_feature:
            robot_state_name = getattr(robot_state_feature, "name", str(robot_state_feature))
            print(f"Robot state feature: {robot_state_name}")
        else:
            print("Robot state feature: None")

        print(f"Env state feature: {getattr(policy.config, 'env_state_feature', 'None')}")
        print(f"Chunk size: {getattr(policy.config, 'chunk_size', 'None')}")
        print("=" * 30)

    # Process each frame
    # proprio_ratios = []
    timestamp_counter = 0
    for i in tqdm(range(episode_length), desc="Processing frames"):
        frame = dataset[episode_frames[i]["index"].item()]

        # Prepare observation for policy (with debug on first frame)
        observation = prepare_observation_for_policy(frame, device, model_dtype, debug=(i == 0))

        # Run policy inference
        with torch.inference_mode():
            if hasattr(policy, "select_action"):
                action, reward = policy.select_action(observation)

                reward_data.append({"step": timestamp_counter, "reward": reward})

                # Extract images for reward visualization
                reward_images_step = []
                for key in observation:
                    if "image" in key:
                        # Convert back to original format for reward visualization
                        img = observation[key].squeeze(0)  # Remove batch dim
                        img = img * 255  # Convert back to 0-255 range
                        img = img.permute(1, 2, 0)  # Convert from CHW to HWC
                        reward_images_step.append(img.cpu())
                reward_images.append(reward_images_step)

            else:
                # Fallback for other policy types
                action = policy(observation)

        timestamp_counter += 1

    # Generate output files
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")

    if reward_data and reward_images:
        output_filename_reward = "reward_visualization.mp4"
        create_reward_visualization_video(reward_images, reward_data, output_filename_reward, fps=20)

        # Print reward statistics
        rewards = [r["reward"] for r in reward_data]
        print("Reward Statistics:")
        print(f"  Mean: {np.mean(rewards):.3f}")
        print(f"  Std: {np.std(rewards):.3f}")
        print(f"  Min: {np.min(rewards):.3f}")
        print(f"  Max: {np.max(rewards):.3f}")
        print(f"  Final: {rewards[-1]:.3f}")

    print("Video encoding process finished.")


def main():
    parser = argparse.ArgumentParser(description="Analyze policy behavior on dataset episodes")
    parser.add_argument(
        "--dataset-repo-id", type=str, required=True, help="Repository ID of the dataset to analyze"
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        default=None,
        help="Episode ID to analyze (if not specified, analyzes all episodes)",
    )
    parser.add_argument("--policy-path", type=str, required=True, help="Path to the policy checkpoint")
    parser.add_argument(
        "--output-dir", type=str, default="./analysis_output", help="Directory to save analysis results"
    )
    parser.add_argument(
        "--policy-overrides", type=str, nargs="*", help="Policy config overrides in key=value format"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model data type",
    )

    args = parser.parse_args()

    # Set up device and dtype
    device = torch.device(args.device)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model_dtype = dtype_map[args.model_dtype]

    print(f"Loading dataset: {args.dataset_repo_id}")
    print(f"Policy path: {args.policy_path}")
    print(f"Using device: {device}")

    # Load dataset
    try:
        dataset = LeRobotDataset(args.dataset_repo_id)
        print(f"Dataset loaded successfully. Total episodes: {dataset.num_episodes}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Determine which episodes to analyze
    if args.episode_id is not None:
        # Single episode analysis
        if args.episode_id >= dataset.num_episodes:
            raise ValueError(
                f"Episode {args.episode_id} not found. Dataset has {dataset.num_episodes} episodes."
            )
        episodes_to_analyze = [args.episode_id]
        print(f"Target episode: {args.episode_id}")
    else:
        # All episodes analysis
        episodes_to_analyze = list(range(dataset.num_episodes))
        print(f"Will analyze all {dataset.num_episodes} episodes")

    # Load policy
    try:
        print("Loading policy...")
        policy, policy_cfg = load_policy(args.policy_path, dataset.meta, args.policy_overrides)

        if hasattr(policy, "model"):
            policy.model.eval()
            policy.model.to(device)
        elif hasattr(policy, "eval"):
            policy.eval()

        print("Policy loaded successfully")

    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    # Run analysis on all specified episodes
    failed_episodes = []

    for episode_id in tqdm(episodes_to_analyze, desc="Analyzing episodes"):
        try:
            print(f"\nStarting analysis of episode {episode_id}...")
            analyze_episode(
                dataset=dataset,
                policy=policy,
                episode_id=episode_id,
                device=device,
                output_dir=args.output_dir,
                model_dtype=model_dtype,
            )
            print(f"Episode {episode_id} analysis completed successfully")

        except Exception as e:
            print(f"Error analyzing episode {episode_id}: {e}")
            failed_episodes.append(episode_id)
            import traceback

            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()

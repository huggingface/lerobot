#!/usr/bin/env python
"""
Example of creating a RECAP dataset for OpenLoong robot.

RECAP (RL with Experience and Corrections via Advantage-conditioned Policies)
uses three types of data:
1. Demo data: Human demonstrations
2. Auto data: Autonomous policy rollouts
3. Intervention data: Human interventions/corrections

This example shows how to prepare data for training π*₀.₆ with OpenLoong.
"""

import numpy as np
import torch
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_json
from lerobot.robots.openloong import OpenLoongJointIndex


def create_mock_episode(
    episode_id: int,
    num_frames: int = 100,
    data_type: str = "demo",
    reward_range: tuple = (0, 1),
) -> dict:
    """
    Create a mock episode for testing.
    
    In practice, this would record actual robot data.
    
    Args:
        episode_id: Episode identifier
        num_frames: Number of frames in episode
        data_type: Type of data (demo, auto, intervention)
        reward_range: Range of rewards
        
    Returns:
        Episode data dictionary
    """
    frames = []
    
    for frame_idx in range(num_frames):
        # Mock joint positions (sine wave motion)
        joint_pos = []
        for i, joint in enumerate(OpenLoongJointIndex):
            # Create different patterns for different joints
            phase = (frame_idx / num_frames) * 2 * np.pi
            if "Knee" in joint.name:
                pos = 0.3 + 0.2 * np.sin(phase + i * 0.1)
            elif "Hip" in joint.name:
                pos = 0.1 * np.sin(phase + i * 0.1)
            else:
                pos = 0.0
            joint_pos.append(pos)
        
        # Mock actions (next joint positions)
        next_frame_idx = min(frame_idx + 1, num_frames - 1)
        next_phase = (next_frame_idx / num_frames) * 2 * np.pi
        joint_action = []
        for i, joint in enumerate(OpenLoongJointIndex):
            if "Knee" in joint.name:
                pos = 0.3 + 0.2 * np.sin(next_phase + i * 0.1)
            elif "Hip" in joint.name:
                pos = 0.1 * np.sin(next_phase + i * 0.1)
            else:
                pos = 0.0
            joint_action.append(pos)
        
        # Mock reward (progress-based for demo, lower for auto)
        if data_type == "demo":
            reward = reward_range[0] + (reward_range[1] - reward_range[0]) * (
                frame_idx / num_frames
            )
        elif data_type == "auto":
            # Auto data might have lower rewards
            reward = reward_range[0] + 0.5 * (reward_range[1] - reward_range[0]) * (
                frame_idx / num_frames
            )
        else:  # intervention
            # Interventions typically correct mistakes, so higher reward
            reward = reward_range[0] + 1.5 * (reward_range[1] - reward_range[0]) * (
                frame_idx / num_frames
            )
        
        # Done flag (True for last frame)
        done = frame_idx == num_frames - 1
        
        frame = {
            "observation.state": np.array(joint_pos, dtype=np.float32),
            "action": np.array(joint_action, dtype=np.float32),
            "reward": float(reward),
            "done": done,
            "data_type": data_type,
            "episode_id": episode_id,
            "frame_id": frame_idx,
        }
        frames.append(frame)
    
    return {
        "episode_id": episode_id,
        "data_type": data_type,
        "frames": frames,
        "total_reward": sum(f["reward"] for f in frames),
    }


def prepare_recap_dataset(
    num_demo_episodes: int = 10,
    num_auto_episodes: int = 20,
    num_intervention_episodes: int = 5,
    frames_per_episode: int = 100,
    output_dir: str = "./openloong_recap_dataset",
) -> str:
    """
    Prepare RECAP dataset with mixed data types.
    
    Args:
        num_demo_episodes: Number of demonstration episodes
        num_auto_episodes: Number of autonomous rollout episodes
        num_intervention_episodes: Number of intervention episodes
        frames_per_episode: Frames per episode
        output_dir: Output directory for dataset
        
    Returns:
        Path to created dataset
    """
    print("Preparing RECAP dataset for OpenLoong...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    episodes = []
    episode_id = 0
    
    # Create demo episodes
    print(f"Creating {num_demo_episodes} demonstration episodes...")
    for i in range(num_demo_episodes):
        episode = create_mock_episode(
            episode_id=episode_id,
            num_frames=frames_per_episode,
            data_type="demo",
            reward_range=(0.5, 1.0),
        )
        episodes.append(episode)
        episode_id += 1
    
    # Create auto episodes
    print(f"Creating {num_auto_episodes} autonomous episodes...")
    for i in range(num_auto_episodes):
        episode = create_mock_episode(
            episode_id=episode_id,
            num_frames=frames_per_episode,
            data_type="auto",
            reward_range=(0.0, 0.7),
        )
        episodes.append(episode)
        episode_id += 1
    
    # Create intervention episodes
    print(f"Creating {num_intervention_episodes} intervention episodes...")
    for i in range(num_intervention_episodes):
        episode = create_mock_episode(
            episode_id=episode_id,
            num_frames=frames_per_episode,
            data_type="intervention",
            reward_range=(0.6, 1.0),
        )
        episodes.append(episode)
        episode_id += 1
    
    # Save dataset info
    dataset_info = {
        "dataset_name": "openloong_recap",
        "total_episodes": len(episodes),
        "total_frames": sum(len(ep["frames"]) for ep in episodes),
        "data_type_counts": {
            "demo": num_demo_episodes,
            "auto": num_auto_episodes,
            "intervention": num_intervention_episodes,
        },
        "action_space": {
            "type": "joint_position",
            "dim": len(OpenLoongJointIndex),
        },
        "observation_space": {
            "state_dim": len(OpenLoongJointIndex),
            "has_imu": True,
            "has_camera": True,
        },
    }
    
    write_json(dataset_info, output_path / "dataset_info.json")
    
    # Save episodes
    for episode in episodes:
        episode_file = output_path / f"episode_{episode['episode_id']:04d}.npz"
        np.savez(
            episode_file,
            observation=np.array([f["observation.state"] for f in episode["frames"]]),
            action=np.array([f["action"] for f in episode["frames"]]),
            reward=np.array([f["reward"] for f in episode["frames"]]),
            done=np.array([f["done"] for f in episode["frames"]]),
            data_type=np.array([f["data_type"] for f in episode["frames"]]),
        )
    
    print(f"Dataset saved to: {output_path}")
    print(f"  Total episodes: {len(episodes)}")
    print(f"  Total frames: {dataset_info['total_frames']}")
    
    return str(output_path)


def compute_advantages(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """
    Compute advantages using GAE (Generalized Advantage Estimation).
    
    Args:
        rewards: Reward array
        gamma: Discount factor
        
    Returns:
        Advantage array
    """
    advantages = np.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = rewards[t + 1]
        
        delta = rewards[t] + gamma * next_value - rewards[t]
        gae = delta + gamma * 0.95 * gae  # lambda = 0.95
        advantages[t] = gae
    
    return advantages


def analyze_dataset(dataset_path: str):
    """
    Analyze RECAP dataset statistics.
    
    Args:
        dataset_path: Path to dataset directory
    """
    print(f"\nAnalyzing dataset: {dataset_path}")
    
    import json
    with open(Path(dataset_path) / "dataset_info.json") as f:
        info = json.load(f)
    
    print(f"Dataset: {info['dataset_name']}")
    print(f"Total episodes: {info['total_episodes']}")
    print(f"Total frames: {info['total_frames']}")
    
    print("\nData type distribution:")
    for data_type, count in info['data_type_counts'].items():
        print(f"  {data_type}: {count} episodes")
    
    # Load and analyze episodes
    all_rewards = {"demo": [], "auto": [], "intervention": []}
    
    for episode_file in sorted(Path(dataset_path).glob("episode_*.npz")):
        data = np.load(episode_file)
        data_type = str(data["data_type"][0])
        rewards = data["reward"]
        all_rewards[data_type].extend(rewards.tolist())
    
    print("\nReward statistics by data type:")
    for data_type, rewards in all_rewards.items():
        if rewards:
            print(f"  {data_type}:")
            print(f"    Mean: {np.mean(rewards):.4f}")
            print(f"    Std: {np.std(rewards):.4f}")
            print(f"    Min: {np.min(rewards):.4f}")
            print(f"    Max: {np.max(rewards):.4f}")


def main():
    """Run RECAP dataset preparation example."""
    # Create dataset
    dataset_path = prepare_recap_dataset(
        num_demo_episodes=5,
        num_auto_episodes=10,
        num_intervention_episodes=3,
        frames_per_episode=50,
        output_dir="./openloong_recap_example",
    )
    
    # Analyze dataset
    analyze_dataset(dataset_path)
    
    print("\nExample: Computing advantages for an episode")
    episode_file = Path(dataset_path) / "episode_0000.npz"
    data = np.load(episode_file)
    rewards = data["reward"]
    advantages = compute_advantages(rewards)
    print(f"  Rewards: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
    print(f"  Advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
    
    print("\nDataset preparation complete!")
    print(f"You can use this dataset to train π*₀.₆ RECAP with:")
    print(f"  python -m lerobot.train policy=pi_star_recap dataset.path={dataset_path}")


if __name__ == "__main__":
    main()

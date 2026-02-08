#!/usr/bin/env python3
"""
Prepare RECAP dataset for π*₀.₆ training

This script demonstrates how to convert robot data into LeRobot format with RECAP-specific fields.

RECAP supports three types of data:
1. Demonstrations (demo): Expert teleoperation
2. Autonomous (auto): Policy execution  
3. Interventions (intervention): Expert takeover during autonomy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from datasets import Dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub import push_dataset_to_hub


def create_episode_data(
    episode_id: int,
    data_type: str,  # "demo", "auto", "intervention"
    num_frames: int = 100,
    image_shape: tuple = (224, 224, 3),
    state_dim: int = 7,
    action_dim: int = 7,
) -> dict:
    """
    Create a single episode with RECAP metadata
    
    Args:
        episode_id: Unique episode identifier
        data_type: Type of data (demo/auto/intervention)
        num_frames: Number of frames in episode
        image_shape: (H, W, C) for images
        state_dim: Dimension of robot state
        action_dim: Dimension of actions
    Returns:
        Dictionary containing episode data
    """
    # Generate synthetic data (replace with real robot data)
    images = np.random.randint(0, 255, (num_frames, *image_shape), dtype=np.uint8)
    states = np.random.randn(num_frames, state_dim).astype(np.float32)
    actions = np.random.randn(num_frames, action_dim).astype(np.float32)
    
    # RECAP-specific fields
    rewards = np.random.randn(num_frames).astype(np.float32)
    dones = np.zeros(num_frames, dtype=bool)
    dones[-1] = True  # Last frame is done
    
    # Intervention mask (for intervention data type)
    # Indicates which timesteps were under expert control
    if data_type == "intervention":
        # Example: expert took over at timesteps 30-50
        intervention_mask = np.zeros(num_frames, dtype=bool)
        intervention_mask[30:50] = True
    else:
        intervention_mask = np.zeros(num_frames, dtype=bool)
    
    return {
        "episode_id": episode_id,
        "data_type": data_type,
        "images": images,
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "intervention_mask": intervention_mask,
    }


def prepare_recap_dataset(
    output_dir: str,
    num_demo_episodes: int = 10,
    num_auto_episodes: int = 5,
    num_intervention_episodes: int = 5,
):
    """
    Prepare a RECAP dataset with mixed data types
    
    Args:
        output_dir: Where to save the dataset
        num_demo_episodes: Number of demonstration episodes
        num_auto_episodes: Number of autonomous episodes
        num_intervention_episodes: Number of intervention episodes
    """
    print("Preparing RECAP dataset...")
    print(f"  Demonstrations: {num_demo_episodes}")
    print(f"  Autonomous: {num_auto_episodes}")
    print(f"  Interventions: {num_intervention_episodes}")
    
    episodes = []
    episode_id = 0
    
    # Create demonstration episodes
    print("\nCreating demonstration episodes...")
    for i in range(num_demo_episodes):
        ep = create_episode_data(episode_id, "demo")
        episodes.append(ep)
        episode_id += 1
    
    # Create autonomous episodes
    print("Creating autonomous episodes...")
    for i in range(num_auto_episodes):
        ep = create_episode_data(episode_id, "auto")
        episodes.append(ep)
        episode_id += 1
    
    # Create intervention episodes
    print("Creating intervention episodes...")
    for i in range(num_intervention_episodes):
        ep = create_episode_data(episode_id, "intervention")
        episodes.append(ep)
        episode_id += 1
    
    # Convert to LeRobot format
    print("\nConverting to LeRobot format...")
    
    # This is a simplified example - actual conversion would require proper handling
    # For a real dataset, use lerobot.common.datasets.push_dataset_to_hub
    
    print(f"Dataset preparation complete!")
    print(f"Total episodes: {len(episodes)}")
    
    # Save summary
    summary = {
        "total_episodes": len(episodes),
        "demo_count": num_demo_episodes,
        "auto_count": num_auto_episodes,
        "intervention_count": num_intervention_episodes,
        "recap_weights": {
            "demo": 1.0,
            "auto": 1.0,
            "intervention": 2.0,
        }
    }
    
    print("\nDataset Summary:")
    print(f"  Total: {summary['total_episodes']}")
    print(f"  Demo: {summary['demo_count']} (weight: {summary['recap_weights']['demo']})")
    print(f"  Auto: {summary['auto_count']} (weight: {summary['recap_weights']['auto']})")
    print(f"  Intervention: {summary['intervention_count']} (weight: {summary['recap_weights']['intervention']})")
    
    return episodes, summary


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare RECAP dataset")
    parser.add_argument("--output_dir", type=str, default="./recap_dataset",
                       help="Output directory for dataset")
    parser.add_argument("--num_demo", type=int, default=10,
                       help="Number of demonstration episodes")
    parser.add_argument("--num_auto", type=int, default=5,
                       help="Number of autonomous episodes")
    parser.add_argument("--num_intervention", type=int, default=5,
                       help="Number of intervention episodes")
    
    args = parser.parse_args()
    
    episodes, summary = prepare_recap_dataset(
        output_dir=args.output_dir,
        num_demo_episodes=args.num_demo,
        num_auto_episodes=args.num_auto,
        num_intervention_episodes=args.num_intervention,
    )
    
    print("\n" + "="*60)
    print("RECAP Dataset Preparation Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Convert to LeRobot format using push_dataset_to_hub")
    print("2. Upload to HuggingFace Hub")
    print("3. Train with: python train_pi_star_recap.py --dataset.repo_id=your_dataset")


if __name__ == "__main__":
    main()

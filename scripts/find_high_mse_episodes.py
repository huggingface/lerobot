#!/usr/bin/env python
"""
Script to find episodes with highest MSE between observation.state and action pairs.

This script:
1. Downloads a LeRobot dataset (if needed, skipping videos)
2. Computes MSE between observation.state and action for each frame
3. Aggregates MSE per episode
4. Returns the top 1% episodes with highest total MSE
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_episode_mse(
    dataset: LeRobotDataset,
    state_key: str = "observation.state",
    action_key: str = "action",
) -> dict[int, float]:
    """
    Compute total MSE between state and action for each episode.
    
    Args:
        dataset: LeRobotDataset to analyze
        state_key: Key for the observation state in the dataset
        action_key: Key for the action in the dataset
        
    Returns:
        Dictionary mapping episode_index to total MSE for that episode
    """
    episode_mse = {}
    
    # Get all unique episode indices
    hf_dataset = dataset.hf_dataset
    
    # Group frames by episode for efficient processing
    logging.info("Computing MSE for each episode...")
    
    # Process all frames and accumulate MSE per episode
    for idx in tqdm(range(len(hf_dataset)), desc="Processing frames"):
        item = hf_dataset[idx]
        
        ep_idx = item["episode_index"]
        if isinstance(ep_idx, torch.Tensor):
            ep_idx = ep_idx.item()
        
        state = item[state_key]
        action = item[action_key]
        
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        
        # Compute MSE for this frame (sum of squared differences across all dimensions)
        mse = np.mean((state - action) ** 2)
        
        if ep_idx not in episode_mse:
            episode_mse[ep_idx] = 0.0
        episode_mse[ep_idx] += mse
    
    return episode_mse


def get_top_mse_episodes(
    episode_mse: dict[int, float],
    top_percent: float = 1.0,
) -> list[int]:
    """
    Get the top X% of episodes with highest total MSE.
    
    Args:
        episode_mse: Dictionary mapping episode_index to total MSE
        top_percent: Percentage of episodes to return (default: 1%)
        
    Returns:
        List of episode indices sorted by MSE (highest first)
    """
    # Sort episodes by MSE in descending order
    sorted_episodes = sorted(episode_mse.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate number of episodes to return
    num_episodes = len(sorted_episodes)
    num_top = max(1, int(np.ceil(num_episodes * top_percent / 100)))
    
    # Extract top episode indices
    top_episodes = [ep_idx for ep_idx, _ in sorted_episodes[:num_top]]
    
    return top_episodes


def find_high_mse_episodes(
    repo_id: str,
    root: str | Path | None = None,
    state_key: str = "observation.state",
    action_key: str = "action",
    top_percent: float = 1.0,
    force_download: bool = False,
) -> tuple[list[int], dict[int, float]]:
    """
    Find episodes with highest MSE between observation.state and action.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        root: Local directory for dataset storage (default: ~/.cache/huggingface/lerobot)
        state_key: Key for the observation state in the dataset
        action_key: Key for the action in the dataset
        top_percent: Percentage of episodes to return (default: 1%)
        force_download: Force re-download of the dataset
        
    Returns:
        Tuple of (list of top episode indices, dict of all episode MSEs)
    """
    logging.info(f"Loading dataset: {repo_id}")
    
    # Load the dataset (skip video download since we only need state/action data)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        download_videos=False,
        force_cache_sync=force_download,
    )
    
    # Verify the dataset has the required features
    if state_key not in dataset.features:
        raise ValueError(f"Dataset does not contain '{state_key}' feature. "
                        f"Available features: {list(dataset.features.keys())}")
    if action_key not in dataset.features:
        raise ValueError(f"Dataset does not contain '{action_key}' feature. "
                        f"Available features: {list(dataset.features.keys())}")
    
    # Check that state and action have the same shape
    state_shape = tuple(dataset.features[state_key]["shape"])
    action_shape = tuple(dataset.features[action_key]["shape"])
    if state_shape != action_shape:
        raise ValueError(f"State shape {state_shape} does not match action shape {action_shape}")
    
    logging.info(f"Dataset loaded successfully:")
    logging.info(f"  - Total episodes: {dataset.meta.total_episodes}")
    logging.info(f"  - Total frames: {dataset.meta.total_frames}")
    logging.info(f"  - State shape: {state_shape}")
    logging.info(f"  - Action shape: {action_shape}")
    logging.info(f"  - Feature names: {dataset.features[state_key].get('names', 'N/A')}")
    
    # Compute MSE for each episode
    episode_mse = compute_episode_mse(dataset, state_key, action_key)
    
    # Get top episodes
    top_episodes = get_top_mse_episodes(episode_mse, top_percent)
    
    return top_episodes, episode_mse


def main():
    parser = argparse.ArgumentParser(
        description="Find episodes with highest MSE between observation.state and action"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace dataset repository ID (e.g., 'lerobot/aloha_sim_insertion_human')",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local directory for dataset storage (default: ~/.cache/huggingface/lerobot)",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default="observation.state",
        help="Key for observation state feature (default: 'observation.state')",
    )
    parser.add_argument(
        "--action-key",
        type=str,
        default="action",
        help="Key for action feature (default: 'action')",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=1.0,
        help="Percentage of episodes to return (default: 1.0)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of the dataset",
    )
    parser.add_argument(
        "--show-all-mse",
        action="store_true",
        help="Show MSE values for all episodes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results (optional)",
    )
    
    args = parser.parse_args()
    
    # Find high MSE episodes
    top_episodes, all_mse = find_high_mse_episodes(
        repo_id=args.repo_id,
        root=args.root,
        state_key=args.state_key,
        action_key=args.action_key,
        top_percent=args.top_percent,
        force_download=args.force_download,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print(f"TOP {args.top_percent}% EPISODES WITH HIGHEST MSE")
    print("=" * 60)
    
    print(f"\nTotal episodes analyzed: {len(all_mse)}")
    print(f"Number of top episodes (top {args.top_percent}%): {len(top_episodes)}")
    
    print(f"\nTop {len(top_episodes)} episode(s) with highest MSE:")
    print("-" * 40)
    for i, ep_idx in enumerate(top_episodes, 1):
        print(f"  {i:3d}. Episode {ep_idx:5d} - Total MSE: {all_mse[ep_idx]:.6f}")
    
    # Statistics
    all_mse_values = list(all_mse.values())
    print(f"\nMSE Statistics:")
    print(f"  - Mean MSE: {np.mean(all_mse_values):.6f}")
    print(f"  - Std MSE: {np.std(all_mse_values):.6f}")
    print(f"  - Min MSE: {np.min(all_mse_values):.6f}")
    print(f"  - Max MSE: {np.max(all_mse_values):.6f}")
    print(f"  - Median MSE: {np.median(all_mse_values):.6f}")
    
    if args.show_all_mse:
        print(f"\nAll episodes sorted by MSE (descending):")
        print("-" * 40)
        sorted_episodes = sorted(all_mse.items(), key=lambda x: x[1], reverse=True)
        for ep_idx, mse in sorted_episodes:
            print(f"  Episode {ep_idx:5d} - Total MSE: {mse:.6f}")
    
    # Save results if output file specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.write(f"# High MSE Episodes Analysis\n")
            f.write(f"# Dataset: {args.repo_id}\n")
            f.write(f"# State key: {args.state_key}\n")
            f.write(f"# Action key: {args.action_key}\n")
            f.write(f"# Top percent: {args.top_percent}%\n\n")
            
            f.write(f"Top {args.top_percent}% episodes:\n")
            for ep_idx in top_episodes:
                f.write(f"{ep_idx},{all_mse[ep_idx]:.6f}\n")
        
        logging.info(f"Results saved to: {output_path}")
    
    # Return the list for programmatic use
    return top_episodes


if __name__ == "__main__":
    main()


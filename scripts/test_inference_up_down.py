#!/usr/bin/env python3
"""
Test inference script for LeRobot using the NLTuan/up-down dataset.
This script demonstrates how to:
1. Load a pretrained policy (SmolVLA).
2. Load a sample observation from the dataset.
3. Run inference to predict actions.
"""

import sys
import torch
from pathlib import Path

# Add the src directory to the python path
sys.path.append(str(Path(__file__).parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

def main():
    # 1. Configuration
    # You can point this to your trained model checkpoint later
    # e.g., "outputs/train/2025-12-30/15-40-00_smolvla/checkpoints/005000/pretrained_model"
    pretrained_path = "lerobot/smolvla_base"
    dataset_repo_id = "NLTuan/up-down"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading policy from: {pretrained_path}")
    
    # 2. Load Policy
    # We load the config first to ensure we have the correct feature shapes
    cfg = PreTrainedConfig.from_pretrained(pretrained_path)
    cfg.device = str(device)
    
    # Load the dataset to get metadata and a sample
    print(f"Loading dataset: {dataset_repo_id}")
    # 2. Load Dataset
    # We force video_backend="pyav" because torchcodec requires system-level ffmpeg
    dataset = LeRobotDataset(dataset_repo_id, video_backend="pyav")
    
    # Instantiate the policy
    policy = make_policy(cfg, ds_meta=dataset.meta)
    policy.eval()
    policy.to(device)
    
    # Create pre/post processors for normalization
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)
    
    # 3. Prepare Input (Initial State)
    # We'll take the first frame of the first episode as our "initial state"
    print("Preparing initial observation from dataset...")
    item = dataset[0]
    
    # The dataset item contains 'observation.image', 'observation.state', etc.
    # We need to add a batch dimension
    observation = {k: v.unsqueeze(0).to(device) for k, v in item.items() if k.startswith("observation")}
    
    # 4. Run Inference
    print("Running inference...")
    with torch.no_grad():
        # Preprocess the observation (normalization, etc.)
        processed_observation = preprocessor(observation)
        
        # Select action
        # For SmolVLA/ACT, this usually returns a single action or a chunk
        action = policy.select_action(processed_observation)
        
        # Postprocess the action (unnormalization)
        unnormalized_action = postprocessor(action)
    
    print("\nInference complete!")
    print(f"Predicted action shape: {unnormalized_action.shape}")
    print(f"Predicted action (first few dims): {unnormalized_action[0, :5]}")
    
    # Note: In a real loop, you would apply this action to your robot/env,
    # get a new observation, and repeat.

if __name__ == "__main__":
    main()

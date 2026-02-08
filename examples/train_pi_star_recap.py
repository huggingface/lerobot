#!/usr/bin/env python3
"""
Example script for training π*₀.₆ RECAP policy with LeRobot

This demonstrates how to use the pi_star_recap policy with LeRobot's training pipeline.
"""

import sys
from pathlib import Path

# Add LeRobot to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.configs import parse_args
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.train import train


def main():
    """
    Train π*₀.₆ RECAP policy
    
    Usage:
        python train_pi_star_recap.py \
            --dataset.repo_id=your_username/your_dataset \
            --output_dir=outputs/pi_star_recap
    """
    
    # Parse arguments
    cfg = parse_args()
    
    # Validate configuration
    assert cfg.policy.type == "pi_star_recap", \
        f"Expected policy.type='pi_star_recap', got '{cfg.policy.type}'"
    
    # Print configuration
    print("="*60)
    print("Training π*₀.₆ RECAP Policy")
    print("="*60)
    print(f"Policy type: {cfg.policy.type}")
    print(f"VLM model: {cfg.policy.vlm_model_name}")
    print(f"IQL expectile (τ): {cfg.policy.iql_expectile}")
    print(f"IQL temperature (β): {cfg.policy.iql_temperature}")
    print(f"Data weights: demo={cfg.policy.demo_weight}, "
          f"auto={cfg.policy.auto_weight}, "
          f"intervention={cfg.policy.intervention_weight}")
    print(f"Output directory: {cfg.output_dir}")
    print("="*60)
    
    # Run training
    train(cfg)


if __name__ == "__main__":
    main()

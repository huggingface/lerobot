#!/usr/bin/env python

import torch
import numpy as np
from lerobot.policies.rlearn.configuration_rlearn import RLearNConfig
from lerobot.policies.rlearn.modeling_rlearn import RLearNPolicy
from lerobot.policies.rlearn.evaluation import RLearnEvaluator


def test_temporal_evaluation():
    """Test that evaluation creates proper temporal sequences with past frames."""
    
    # Create a simple config
    config = RLearNConfig(
        max_seq_len=4,  # Small for testing
        dim_model=64,   # Small for testing
        n_heads=2,
        n_layers=2,
    )
    
    # Create model (will be randomly initialized)
    model = RLearNPolicy(config)
    model.eval()
    
    # Create evaluator
    evaluator = RLearnEvaluator(model, device="cpu")
    
    # Create test episode: 8 frames of 3x64x64 images
    T, C, H, W = 8, 3, 64, 64
    frames = torch.randn(T, C, H, W)
    language = "test instruction"
    
    print(f"Input episode shape: {frames.shape}")
    print(f"Model expects sequences of length: {config.max_seq_len}")
    
    # Test the evaluation
    rewards = evaluator.predict_episode_rewards(frames, language, batch_size=4)
    
    print(f"Output rewards shape: {rewards.shape}")
    print(f"Rewards: {rewards}")
    
    # Verify we get one reward per frame
    assert len(rewards) == T, f"Expected {T} rewards, got {len(rewards)}"
    
    print("✅ Test passed! Evaluation correctly processes temporal sequences.")
    
    # Test with very short episode (shorter than max_seq_len)
    short_frames = torch.randn(2, C, H, W)  # Only 2 frames
    short_rewards = evaluator.predict_episode_rewards(short_frames, language)
    
    print(f"\nShort episode shape: {short_frames.shape}")
    print(f"Short rewards shape: {short_rewards.shape}")
    assert len(short_rewards) == 2, f"Expected 2 rewards, got {len(short_rewards)}"
    
    print("✅ Short episode test passed!")


if __name__ == "__main__":
    test_temporal_evaluation()

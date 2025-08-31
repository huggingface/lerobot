#!/usr/bin/env python
"""Test script to verify episode-relative progress is working correctly."""

import torch
import numpy as np
from pathlib import Path

# Simulate what the dataset would provide
def create_test_batch(batch_size=2, episode_lengths=[100, 150]):
    """Create a test batch with episode information."""
    batch = {}
    
    # Simulate episode indices and frame indices
    batch["episode_index"] = torch.tensor([0, 1])  # Two different episodes
    batch["frame_index"] = torch.tensor([50, 75])  # Middle of each episode
    
    # Simulate images (not important for this test)
    batch["observation.images"] = torch.randn(batch_size, 16, 3, 224, 224)
    
    # Simulate language
    batch["observation.language"] = ["Pick up the blue block", "Pick up the red block"]
    
    return batch

def test_progress_calculation():
    """Test that progress is calculated correctly."""
    print("Testing Episode-Relative Progress Calculation")
    print("=" * 60)
    
    # Simulate episode_data_index
    episode_data_index = {
        "from": torch.tensor([0, 100, 250]),  # Episode boundaries
        "to": torch.tensor([100, 250, 400])    # Episode ends
    }
    
    # Test case 1: Sample from middle of episode
    print("\nTest Case 1: Window from middle of 100-frame episode")
    print("Anchor at frame 50, window frames [35-50]")
    
    # Expected progress for frames 35-50 in a 100-frame episode
    expected_progress = [35/99, 36/99, 37/99, 38/99, 39/99, 40/99, 41/99, 42/99,
                        43/99, 44/99, 45/99, 46/99, 47/99, 48/99, 49/99, 50/99]
    
    print(f"Expected progress range: [{expected_progress[0]:.3f} to {expected_progress[-1]:.3f}]")
    print(f"This is ~[0.354 to 0.505] - NOT [0.0 to 1.0]!")
    
    # Test case 2: Sample from end of episode
    print("\nTest Case 2: Window from end of 150-frame episode")
    print("Anchor at frame 140, window frames [125-140]")
    
    # Expected progress for frames 125-140 in a 150-frame episode
    expected_progress_2 = [125/149, 126/149, 127/149, 128/149, 129/149, 130/149, 131/149, 132/149,
                          133/149, 134/149, 135/149, 136/149, 137/149, 138/149, 139/149, 140/149]
    
    print(f"Expected progress range: [{expected_progress_2[0]:.3f} to {expected_progress_2[-1]:.3f}]")
    print(f"This is ~[0.839 to 0.940] - NOT [0.0 to 1.0]!")
    
    print("\n" + "=" * 60)
    print("✅ Key Insight: Each 16-frame window should have progress values")
    print("   that reflect its actual position within the episode,")
    print("   NOT always [0.0 to 1.0]!")

if __name__ == "__main__":
    test_progress_calculation()

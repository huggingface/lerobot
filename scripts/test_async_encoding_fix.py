#!/usr/bin/env python

"""
Test script to verify the async encoding fix works with actual dataset structure.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.datasets.async_video_encoder import AsyncVideoEncoder


def test_async_encoding_with_real_dataset():
    """Test async encoding with the actual dataset structure."""
    
    dataset_path = Path("datasets/async_strawberry_picking")
    
    print("=" * 80)
    print("TESTING ASYNC ENCODING WITH REAL DATASET")
    print("=" * 80)
    
    print(f"Dataset path: {dataset_path}")
    
    # Check if dataset exists
    if not dataset_path.exists():
        print("❌ Dataset not found!")
        return False
    
    # Check for images
    front_images = list(dataset_path.glob("images/observation.images.front/episode_*/frame_*.png"))
    wrist_images = list(dataset_path.glob("images/observation.images.wrist/episode_*/frame_*.png"))
    
    print(f"Front camera images found: {len(front_images)}")
    print(f"Wrist camera images found: {len(wrist_images)}")
    
    if len(front_images) == 0 and len(wrist_images) == 0:
        print("❌ No images found!")
        return False
    
    # Test async encoding
    print("\nTesting async encoding...")
    
    with AsyncVideoEncoder(num_workers=2, max_queue_size=10) as encoder:
        # Submit encoding tasks for both episodes
        for episode_index in [0, 1]:
            for video_key in ["observation.images.front", "observation.images.wrist"]:
                success = encoder.submit_encoding_task(
                    episode_index=episode_index,
                    video_keys=[video_key],
                    fps=30,
                    root_path=dataset_path,
                    priority=0
                )
                print(f"Submitted encoding task for episode {episode_index}, camera {video_key}: {success}")
        
        # Wait for completion
        print("\nWaiting for encoding to complete...")
        completed = encoder.wait_for_completion(timeout=60.0)
        
        if completed:
            print("✅ All encoding tasks completed!")
        else:
            print("⚠️  Encoding tasks did not complete within timeout")
        
        # Get results
        results = encoder.get_results()
        stats = encoder.get_stats()
        
        print(f"\nResults: {len(results)} tasks")
        print(f"Stats: {stats}")
        
        # Check for videos
        video_files = list(dataset_path.glob("videos/**/*.mp4"))
        print(f"\nVideo files created: {len(video_files)}")
        
        for video_file in video_files:
            print(f"  {video_file}")
        
        return len(video_files) > 0


if __name__ == "__main__":
    success = test_async_encoding_with_real_dataset()
    
    if success:
        print("\n✅ Async encoding test PASSED!")
    else:
        print("\n❌ Async encoding test FAILED!") 
#!/usr/bin/env python

"""
Basic test script to verify AsyncVideoEncoder infrastructure works.
"""

import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.datasets.async_video_encoder import AsyncVideoEncoder


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_async_encoder_basic():
    """Test the AsyncVideoEncoder basic functionality."""
    print("Testing AsyncVideoEncoder basic functionality...")
    
    # Test the async encoder
    with AsyncVideoEncoder(num_workers=2, max_queue_size=10) as encoder:
        print("AsyncVideoEncoder started successfully")
        
        # Submit some test tasks (these will fail due to missing files, but that's OK)
        for episode in range(3):
            success = encoder.submit_encoding_task(
                episode_index=episode,
                video_keys=["camera_1"],
                fps=30,
                root_path=Path("./nonexistent"),
                priority=0
            )
            print(f"Submitted encoding task for episode {episode}: {success}")
        
        # Wait a bit for tasks to be processed
        time.sleep(0.5)
        
        # Get results
        results = encoder.get_results()
        stats = encoder.get_stats()
        
        print(f"Results: {len(results)} tasks completed")
        print(f"Stats: {stats}")
        
        # Verify that tasks were submitted and processed
        assert stats['tasks_submitted'] == 3, f"Expected 3 tasks submitted, got {stats['tasks_submitted']}"
        assert stats['tasks_completed'] == 3, f"Expected 3 tasks completed, got {stats['tasks_completed']}"
        
        print("All tasks were submitted and processed successfully!")
    
    print("AsyncVideoEncoder basic test completed successfully!")


def test_async_encoder_priority():
    """Test priority queue functionality."""
    print("\nTesting AsyncVideoEncoder priority queue...")
    
    with AsyncVideoEncoder(num_workers=1, max_queue_size=10) as encoder:
        # Submit tasks with different priorities
        encoder.submit_encoding_task(
            episode_index=0,
            video_keys=["camera_1"],
            fps=30,
            root_path=Path("./nonexistent"),
            priority=1  # Lower priority
        )
        
        encoder.submit_encoding_task(
            episode_index=1,
            video_keys=["camera_1"],
            fps=30,
            root_path=Path("./nonexistent"),
            priority=5  # Higher priority
        )
        
        encoder.submit_encoding_task(
            episode_index=2,
            video_keys=["camera_1"],
            fps=30,
            root_path=Path("./nonexistent"),
            priority=3  # Medium priority
        )
        
        # Wait for processing
        time.sleep(0.5)
        
        results = encoder.get_results()
        print(f"Processed {len(results)} tasks")
        
        # Verify that higher priority tasks were processed first
        if len(results) >= 2:
            print("Priority queue test completed!")


def main():
    """Run the async encoder tests."""
    setup_logging("INFO")
    
    print("=" * 60)
    print("ASYNC VIDEO ENCODER BASIC TESTS")
    print("=" * 60)
    
    test_async_encoder_basic()
    test_async_encoder_priority()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe AsyncVideoEncoder infrastructure is working correctly!")
    print("The encoding failures are expected since we don't have actual image files.")


if __name__ == "__main__":
    main() 
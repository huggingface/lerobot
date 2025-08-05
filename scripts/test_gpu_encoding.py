#!/usr/bin/env python

"""
Test script to verify GPU-accelerated encoding works with the async encoder.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.datasets.async_video_encoder import AsyncVideoEncoder
from lerobot.datasets.gpu_video_encoder import create_gpu_encoder_config


def test_gpu_async_encoding():
    """Test GPU-accelerated async encoding."""
    
    print("=" * 80)
    print("GPU-ACCELERATED ASYNC ENCODING TEST")
    print("=" * 80)
    
    dataset_path = Path("datasets/async_strawberry_picking")
    
    if not dataset_path.exists():
        print("❌ Test dataset not found. Please run a recording session first.")
        return False
    
    # Test different GPU configurations
    configs = [
        {
            "name": "NVIDIA NVENC H.264",
            "config": create_gpu_encoder_config(
                encoder_type="nvenc",
                codec="h264",
                preset="fast",
                quality=23
            )
        },
        {
            "name": "NVIDIA NVENC HEVC",
            "config": create_gpu_encoder_config(
                encoder_type="nvenc",
                codec="hevc",
                preset="fast",
                quality=23
            )
        },
        {
            "name": "Auto-select best encoder",
            "config": create_gpu_encoder_config(
                encoder_type="auto",
                codec="h264",
                preset="fast",
                quality=23
            )
        }
    ]
    
    for test_config in configs:
        print(f"\n--- Testing {test_config['name']} ---")
        
        # Create async encoder with GPU support
        async_encoder = AsyncVideoEncoder(
            num_workers=2,
            max_queue_size=10,
            enable_logging=True,
            gpu_encoding=True,
            gpu_encoder_config={
                "encoder_type": test_config["config"].encoder_type,
                "codec": test_config["config"].codec,
                "preset": test_config["config"].preset,
                "quality": test_config["config"].quality,
                "gpu_id": 0
            }
        )
        
        with async_encoder:
            # Submit encoding tasks for episode 0
            for video_key in ["observation.images.front", "observation.images.wrist"]:
                success = async_encoder.submit_encoding_task(
                    episode_index=0,
                    video_keys=[video_key],
                    fps=30,
                    root_path=dataset_path,
                    priority=0
                )
                print(f"  Submitted task for {video_key}: {success}")
            
            # Wait for completion
            print("  Waiting for encoding to complete...")
            start_time = time.time()
            completed = async_encoder.wait_for_completion(timeout=120.0)
            encoding_time = time.time() - start_time
            
            if completed:
                print(f"  ✅ All tasks completed in {encoding_time:.2f}s")
                
                # Get results
                results = async_encoder.get_results()
                stats = async_encoder.get_stats()
                
                print(f"  Results: {len(results)} tasks")
                print(f"  Stats: {stats}")
                
                # Check for video files
                video_files = list(dataset_path.glob("videos/**/*.mp4"))
                print(f"  Video files: {len(video_files)}")
                
                for video_file in video_files:
                    print(f"    {video_file}")
                
                if len(video_files) > 0:
                    print(f"  ✅ {test_config['name']} test PASSED!")
                else:
                    print(f"  ❌ {test_config['name']} test FAILED - no videos created")
                    return False
            else:
                print(f"  ❌ {test_config['name']} test FAILED - timeout")
                return False
    
    print("\n" + "=" * 80)
    print("ALL GPU ENCODING TESTS COMPLETED")
    print("=" * 80)
    return True


def test_cpu_vs_gpu_performance():
    """Compare CPU vs GPU encoding performance."""
    
    print("\n" + "=" * 80)
    print("CPU vs GPU PERFORMANCE COMPARISON")
    print("=" * 80)
    
    dataset_path = Path("datasets/async_strawberry_picking")
    
    if not dataset_path.exists():
        print("❌ Test dataset not found. Please run a recording session first.")
        return False
    
    # Test CPU encoding
    print("\n--- Testing CPU Encoding ---")
    cpu_start = time.time()
    
    cpu_encoder = AsyncVideoEncoder(
        num_workers=2,
        max_queue_size=10,
        enable_logging=True,
        gpu_encoding=False
    )
    
    with cpu_encoder:
        for video_key in ["observation.images.front", "observation.images.wrist"]:
            cpu_encoder.submit_encoding_task(
                episode_index=1,
                video_keys=[video_key],
                fps=30,
                root_path=dataset_path,
                priority=0
            )
        
        cpu_encoder.wait_for_completion(timeout=120.0)
        cpu_stats = cpu_encoder.get_stats()
    
    cpu_time = time.time() - cpu_start
    
    # Test GPU encoding
    print("\n--- Testing GPU Encoding ---")
    gpu_start = time.time()
    
    gpu_encoder = AsyncVideoEncoder(
        num_workers=2,
        max_queue_size=10,
        enable_logging=True,
        gpu_encoding=True,
        gpu_encoder_config={
            "encoder_type": "nvenc",
            "codec": "h264",
            "preset": "fast",
            "quality": 23,
            "gpu_id": 0
        }
    )
    
    with gpu_encoder:
        for video_key in ["observation.images.front", "observation.images.wrist"]:
            gpu_encoder.submit_encoding_task(
                episode_index=1,
                video_keys=[video_key],
                fps=30,
                root_path=dataset_path,
                priority=0
            )
        
        gpu_encoder.wait_for_completion(timeout=120.0)
        gpu_stats = gpu_encoder.get_stats()
    
    gpu_time = time.time() - gpu_start
    
    # Compare results
    print(f"\n--- Performance Comparison ---")
    print(f"CPU encoding time: {cpu_time:.2f}s")
    print(f"GPU encoding time: {gpu_time:.2f}s")
    
    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        improvement = ((cpu_time - gpu_time) / cpu_time) * 100
        print(f"GPU is {speedup:.2f}x faster ({improvement:.1f}% improvement)")
    else:
        slowdown = gpu_time / cpu_time
        print(f"GPU is {slowdown:.2f}x slower (this might be due to overhead for short videos)")
    
    print(f"CPU stats: {cpu_stats}")
    print(f"GPU stats: {gpu_stats}")
    
    return True


if __name__ == "__main__":
    print("Testing GPU-accelerated async encoding...")
    
    # Test basic GPU encoding
    success1 = test_gpu_async_encoding()
    
    # Test performance comparison
    success2 = test_cpu_vs_gpu_performance()
    
    if success1 and success2:
        print("\n🎉 All GPU encoding tests PASSED!")
    else:
        print("\n❌ Some GPU encoding tests FAILED!") 
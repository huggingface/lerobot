#!/usr/bin/env python

"""
Monitor async encoding performance during real recording sessions.

This script can be used to check the status of async video encoding
while recording is in progress.
"""

import time
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.datasets.async_video_encoder import AsyncVideoEncoder


def monitor_async_encoder(dataset_path: str, interval: float = 5.0):
    """
    Monitor async encoding performance for a dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        interval: Monitoring interval in seconds
    """
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Dataset path does not exist: {dataset_path}")
        return
    
    print(f"Monitoring async encoding for dataset: {dataset_path}")
    print(f"Monitoring interval: {interval}s")
    print("-" * 80)
    
    try:
        while True:
            # Check for async encoder stats file
            stats_file = dataset_path / "async_encoder_stats.json"
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                print(f"\n[{time.strftime('%H:%M:%S')}] Async Encoder Status:")
                print(f"  Tasks submitted: {stats.get('tasks_submitted', 0)}")
                print(f"  Tasks completed: {stats.get('tasks_completed', 0)}")
                print(f"  Tasks in queue: {stats.get('tasks_in_queue', 0)}")
                print(f"  Average encoding time: {stats.get('average_encoding_time', 0):.2f}s")
                print(f"  Total encoding time: {stats.get('total_encoding_time', 0):.2f}s")
                
                # Calculate efficiency
                if stats.get('tasks_completed', 0) > 0:
                    efficiency = (stats['tasks_completed'] / stats['tasks_submitted']) * 100
                    print(f"  Efficiency: {efficiency:.1f}%")
                
                # Check for any errors
                if stats.get('failed_tasks', 0) > 0:
                    print(f"  ⚠️  Failed tasks: {stats['failed_tasks']}")
                
            else:
                print(f"\n[{time.strftime('%H:%M:%S')}] No async encoder stats found")
                print("  (Async encoding may not be enabled or recording not started)")
            
            # Check for episode directories
            episode_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('episode_')]
            if episode_dirs:
                print(f"  Episodes found: {len(episode_dirs)}")
                
                # Check for videos in recent episodes
                recent_episodes = sorted(episode_dirs)[-3:]  # Last 3 episodes
                for episode_dir in recent_episodes:
                    video_files = list(episode_dir.glob("*.mp4"))
                    if video_files:
                        print(f"    {episode_dir.name}: {len(video_files)} videos")
                    else:
                        print(f"    {episode_dir.name}: No videos yet (encoding in progress?)")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError during monitoring: {e}")


def main():
    """Main function for monitoring async encoding."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor async encoding performance")
    parser.add_argument(
        "dataset_path",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Monitoring interval in seconds (default: 5.0)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ASYNC ENCODING MONITOR")
    print("=" * 80)
    print("\nThis tool monitors async video encoding performance during recording.")
    print("Press Ctrl+C to stop monitoring.\n")
    
    monitor_async_encoder(args.dataset_path, args.interval)


if __name__ == "__main__":
    main() 
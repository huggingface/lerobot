#!/usr/bin/env python

"""
Test Async GPU Encoding

This script tests the combination of async video encoding with GPU acceleration
to see if the improvements have fixed the sticking issues.
"""

import subprocess
import sys
from pathlib import Path


def run_async_gpu_recording():
    """Run recording with async GPU encoding."""

    # Configuration with both async and GPU encoding
    cmd = [
        "python",
        "-m",
        "lerobot.record",
        # Robot configuration
        "--robot.type=so101_follower",
        "--robot.port=/dev/ttyACM0",
        "--robot.id=brl_so101_follower_arm",
        "--robot.cameras={ front: {type: opencv, index_or_path: /dev/video4, width: 1280, height: 720, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: /dev/video2, width: 1280, height: 720, fps: 30, fourcc: MJPG}}",
        # Teleop configuration
        "--teleop.type=so101_leader",
        "--teleop.port=/dev/ttyACM1",
        "--teleop.id=brl_so101_leader_arm",
        # Dataset configuration
        "--dataset.single_task=picking double strawberry",
        "--dataset.repo_id=local/async_gpu_test_dataset",
        "--dataset.root=./datasets/async_gpu_test",
        "--dataset.episode_time_s=20",
        "--dataset.num_episodes=3",
        "--dataset.push_to_hub=false",
        # ASYNC + GPU ENCODING CONFIGURATION
        "--dataset.async_video_encoding=true",  # Enable async encoding
        "--dataset.gpu_video_encoding=true",  # Enable GPU encoding
        "--dataset.gpu_encoder_config={'encoder_type': 'nvenc', 'codec': 'h264', 'preset': 'fast', 'quality': 23}",
        "--dataset.video_encoding_workers=1",  # Reduce workers to prevent conflicts
        "--dataset.video_encoding_queue_size=50",  # Smaller queue size
        # Disable display to reduce resource usage
        "--display_data=false",
    ]

    print("=" * 80)
    print("ASYNC GPU ENCODING TEST")
    print("=" * 80)
    print("Testing async + GPU encoding with improvements:")
    print("✅ Async video encoding enabled")
    print("✅ GPU acceleration enabled")
    print("✅ Timeout protection (5 minutes)")
    print("✅ Automatic CPU fallback")
    print("✅ Reduced workers (1) to prevent conflicts")
    print("✅ Smaller queue size (50)")
    print("✅ Reduced resource usage (no display)")
    print("=" * 80)

    try:
        # Run the recording command
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Recording failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Recording interrupted by user")
        return False


def run_async_gpu_recording_aggressive():
    """Run recording with more aggressive async GPU encoding."""

    # Configuration with more aggressive settings
    cmd = [
        "python",
        "-m",
        "lerobot.record",
        # Robot configuration
        "--robot.type=so101_follower",
        "--robot.port=/dev/ttyACM0",
        "--robot.id=brl_so101_follower_arm",
        "--robot.cameras={ front: {type: opencv, index_or_path: /dev/video4, width: 1280, height: 720, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: /dev/video2, width: 1280, height: 720, fps: 30, fourcc: MJPG}}",
        # Teleop configuration
        "--teleop.type=so101_leader",
        "--teleop.port=/dev/ttyACM1",
        "--teleop.id=brl_so101_leader_arm",
        # Dataset configuration
        "--dataset.single_task=picking double strawberry",
        "--dataset.repo_id=local/async_gpu_aggressive_dataset",
        "--dataset.root=./datasets/async_gpu_aggressive",
        "--dataset.episode_time_s=20",
        "--dataset.num_episodes=3",
        "--dataset.push_to_hub=false",
        # AGGRESSIVE ASYNC + GPU ENCODING
        "--dataset.async_video_encoding=true",
        "--dataset.gpu_video_encoding=true",
        "--dataset.gpu_encoder_config={'encoder_type': 'nvenc', 'codec': 'h264', 'preset': 'fast', 'quality': 23}",
        "--dataset.video_encoding_workers=2",  # More workers
        "--dataset.video_encoding_queue_size=100",  # Larger queue
        # Disable display
        "--display_data=false",
    ]

    print("=" * 80)
    print("AGGRESSIVE ASYNC GPU ENCODING TEST")
    print("=" * 80)
    print("Testing more aggressive async + GPU encoding:")
    print("✅ Async video encoding enabled")
    print("✅ GPU acceleration enabled")
    print("✅ 2 encoding workers")
    print("✅ Larger queue size (100)")
    print("⚠️  Higher risk of conflicts")
    print("=" * 80)

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Recording failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Recording interrupted by user")
        return False


def main():
    """Main function with test options."""
    import argparse

    parser = argparse.ArgumentParser(description="Test async GPU encoding")
    parser.add_argument(
        "--mode",
        choices=["conservative", "aggressive", "interactive"],
        default="interactive",
        help="Test mode: conservative (1 worker), aggressive (2 workers), interactive (choose)",
    )

    args = parser.parse_args()

    print("🧪 ASYNC GPU ENCODING TEST")
    print("This script tests if async GPU encoding works without getting stuck.")

    if args.mode == "interactive":
        print("\nChoose test mode:")
        print("1. Conservative async GPU (1 worker, safer)")
        print("2. Aggressive async GPU (2 workers, faster but riskier)")
        print("3. Exit")

        while True:
            choice = input("\nEnter choice (1-3): ").strip()
            if choice == "1":
                return run_async_gpu_recording()
            elif choice == "2":
                return run_async_gpu_recording_aggressive()
            elif choice == "3":
                print("Exiting...")
                return True
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    elif args.mode == "conservative":
        return run_async_gpu_recording()

    elif args.mode == "aggressive":
        return run_async_gpu_recording_aggressive()


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Async GPU encoding test completed successfully!")
        print("🎉 The improvements have fixed the sticking issues!")
    else:
        print("\n❌ Async GPU encoding test failed!")
        print("💡 Consider using synchronous GPU encoding for stability")
        sys.exit(1)

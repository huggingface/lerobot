#!/usr/bin/env python

"""
Regenerate Dataset Videos with GPU Encoding

This script regenerates MP4 video files for a LeRobot dataset using GPU-accelerated encoding.
It's useful when video files are missing or corrupted.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.datasets.gpu_video_encoder import GPUVideoEncoder, create_gpu_encoder_config
from lerobot.datasets.video_utils import encode_video_frames


def find_episodes(dataset_path: Path) -> list[int]:
    """Find all episode numbers in the dataset."""
    episodes = []

    # Check data directory
    data_dir = dataset_path / "data"
    if data_dir.exists():
        for chunk_dir in data_dir.iterdir():
            if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
                for parquet_file in chunk_dir.glob("episode_*.parquet"):
                    episode_num = int(parquet_file.stem.split("_")[1])
                    episodes.append(episode_num)

    # Check images directory
    images_dir = dataset_path / "images"
    if images_dir.exists():
        for camera_dir in images_dir.iterdir():
            if camera_dir.is_dir() and camera_dir.name.startswith("observation.images."):
                for episode_dir in camera_dir.iterdir():
                    if episode_dir.is_dir() and episode_dir.name.startswith("episode_"):
                        episode_num = int(episode_dir.name.split("_")[1])
                        if episode_num not in episodes:
                            episodes.append(episode_num)

    return sorted(episodes)


def find_cameras(dataset_path: Path) -> list[str]:
    """Find all camera keys in the dataset."""
    cameras = []
    images_dir = dataset_path / "images"

    if images_dir.exists():
        for camera_dir in images_dir.iterdir():
            if camera_dir.is_dir() and camera_dir.name.startswith("observation.images."):
                camera_key = camera_dir.name
                cameras.append(camera_key)

    return sorted(cameras)


def get_chunk_id(episode_num: int) -> str:
    """Get chunk ID for an episode number."""
    return f"chunk-{episode_num // 1000:03d}"


def count_frames(episode_dir: Path) -> int:
    """Count the number of frames in an episode directory."""
    if not episode_dir.exists():
        return 0

    frame_files = list(episode_dir.glob("frame_*.png"))
    return len(frame_files)


def regenerate_video(
    dataset_path: Path,
    episode_num: int,
    camera_key: str,
    gpu_encoder: GPUVideoEncoder,
    use_gpu: bool = True,
    fps: int = 30,
) -> bool:
    """Regenerate a single video file."""

    # Calculate paths
    chunk_id = get_chunk_id(episode_num)
    episode_str = f"episode_{episode_num:06d}"

    # Input: images directory
    images_dir = dataset_path / "images" / camera_key / episode_str

    # Output: video file
    video_path = dataset_path / "videos" / chunk_id / camera_key / f"{episode_str}.mp4"

    # Check if input exists
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False

    # Count frames
    num_frames = count_frames(images_dir)
    if num_frames == 0:
        print(f"❌ No frames found in: {images_dir}")
        return False

    print(f"📹 Regenerating video: {episode_str} - {camera_key} ({num_frames} frames)")

    # Create output directory
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing video file if it exists
    if video_path.exists():
        video_path.unlink()
        print("   Removed existing video file")

    # Encode video
    start_time = time.time()

    if use_gpu and gpu_encoder:
        print("   Using GPU encoding...")
        success = gpu_encoder.encode_video(
            input_dir=images_dir,
            output_path=video_path,
            fps=fps,
            timeout=600,  # 10 minutes timeout
        )

        if not success:
            print("   ⚠️  GPU encoding failed, falling back to CPU...")
            # Fallback to CPU encoding
            encode_video_frames(imgs_dir=images_dir, video_path=video_path, fps=fps, overwrite=True)
            success = True
    else:
        print("   Using CPU encoding...")
        encode_video_frames(imgs_dir=images_dir, video_path=video_path, fps=fps, overwrite=True)
        success = True

    encoding_time = time.time() - start_time

    if success and video_path.exists():
        video_size = video_path.stat().st_size / (1024 * 1024)  # MB
        print(f"   ✅ Success: {video_size:.1f} MB in {encoding_time:.2f}s")
        return True
    else:
        print(f"   ❌ Failed after {encoding_time:.2f}s")
        return False


def main():
    """Main function to regenerate all videos in a dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate dataset videos with GPU encoding")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset directory")
    parser.add_argument(
        "--use-gpu", action="store_true", default=True, help="Use GPU encoding (default: True)"
    )
    parser.add_argument("--use-cpu", action="store_true", help="Force CPU encoding")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--encoder-type", type=str, default="nvenc", help="GPU encoder type (default: nvenc)")
    parser.add_argument("--codec", type=str, default="h264", help="Video codec (default: h264)")
    parser.add_argument("--preset", type=str, default="fast", help="Encoding preset (default: fast)")
    parser.add_argument("--quality", type=int, default=23, help="Quality setting (default: 23)")
    parser.add_argument(
        "--episodes", type=str, help="Comma-separated list of episode numbers to process (e.g., '0,1,2')"
    )

    args = parser.parse_args()

    # Validate dataset path
    if not args.dataset_path.exists():
        print(f"❌ Dataset path does not exist: {args.dataset_path}")
        return 1

    # Determine encoding method
    use_gpu = args.use_gpu and not args.use_cpu

    print("=" * 80)
    print("DATASET VIDEO REGENERATION")
    print("=" * 80)
    print(f"Dataset: {args.dataset_path}")
    print(f"Encoding: {'GPU' if use_gpu else 'CPU'}")
    print(f"FPS: {args.fps}")

    # Find episodes and cameras
    all_episodes = find_episodes(args.dataset_path)
    cameras = find_cameras(args.dataset_path)

    if not all_episodes:
        print("❌ No episodes found in dataset")
        return 1

    if not cameras:
        print("❌ No cameras found in dataset")
        return 1

    print(f"Found {len(all_episodes)} episodes: {all_episodes}")
    print(f"Found {len(cameras)} cameras: {cameras}")

    # Filter episodes if specified
    if args.episodes:
        try:
            episode_list = [int(x.strip()) for x in args.episodes.split(",")]
            episodes = [ep for ep in all_episodes if ep in episode_list]
            if not episodes:
                print(f"❌ No matching episodes found: {episode_list}")
                return 1
        except ValueError:
            print("❌ Invalid episode list format. Use comma-separated numbers (e.g., '0,1,2')")
            return 1
    else:
        episodes = all_episodes

    # Initialize GPU encoder if using GPU
    gpu_encoder = None
    if use_gpu:
        try:
            gpu_config = create_gpu_encoder_config(
                encoder_type=args.encoder_type, codec=args.codec, preset=args.preset, quality=args.quality
            )
            gpu_encoder = GPUVideoEncoder(gpu_config)
            encoder_info = gpu_encoder.get_encoder_info()
            print(f"GPU Encoder: {encoder_info['selected_encoder']} {encoder_info['selected_codec']}")
        except Exception as e:
            print(f"⚠️  GPU encoder initialization failed: {e}")
            print("   Falling back to CPU encoding")
            use_gpu = False

    # Process each episode and camera
    total_videos = len(episodes) * len(cameras)
    successful_videos = 0
    failed_videos = 0

    print(f"\nProcessing {total_videos} videos...")

    for episode_num in episodes:
        for camera_key in cameras:
            success = regenerate_video(
                dataset_path=args.dataset_path,
                episode_num=episode_num,
                camera_key=camera_key,
                gpu_encoder=gpu_encoder,
                use_gpu=use_gpu,
                fps=args.fps,
            )

            if success:
                successful_videos += 1
            else:
                failed_videos += 1

    # Summary
    print("\n" + "=" * 80)
    print("REGENERATION SUMMARY")
    print("=" * 80)
    print(f"Total videos: {total_videos}")
    print(f"Successful: {successful_videos}")
    print(f"Failed: {failed_videos}")
    print(f"Success rate: {successful_videos / total_videos * 100:.1f}%")

    if failed_videos > 0:
        print(f"\n⚠️  {failed_videos} videos failed to regenerate")
        return 1
    else:
        print("\n✅ All videos regenerated successfully!")
        return 0


if __name__ == "__main__":
    exit(main())

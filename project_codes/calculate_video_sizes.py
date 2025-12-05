from pathlib import Path
import os


def get_max_video_sizes(repo_id, name):
    """Get maximum video file size from a dataset."""
    print(f"\n{'='*60}")
    print(f"{name}: {repo_id}")
    print('='*60)

    cache_path = Path.home() / ".cache/huggingface/lerobot" / \
        repo_id.replace("/", "/")
    videos_path = cache_path / "videos"

    if not videos_path.exists():
        print(f"✗ Videos directory not found: {videos_path}")
        return {}

    max_sizes = {}

    # List all camera directories
    camera_dirs = [d for d in videos_path.iterdir() if d.is_dir()]

    for camera_dir in camera_dirs:
        camera_name = camera_dir.name
        max_size = 0

        # Find all video files
        video_files = list(camera_dir.glob("**/*.mp4"))

        for vf in video_files:
            size_mb = vf.stat().st_size / (1024 * 1024)
            if size_mb > max_size:
                max_size = size_mb

        max_sizes[camera_name] = max_size
        print(f"  {camera_name}: max {max_size:.1f} MB")

    overall_max = max(max_sizes.values()) if max_sizes else 0
    print(f"\n  Overall maximum: {overall_max:.1f} MB")

    return max_sizes


def main():
    print("Analyzing video file sizes from your datasets...\n")

    right_sizes = get_max_video_sizes(
        "YieumYoon/bimanual-center-basket-right-rblock",
        "RIGHT Dataset"
    )

    left_sizes = get_max_video_sizes(
        "YieumYoon/bimanual-center-basket-left-rblock",
        "LEFT Dataset"
    )

    print("\n" + "="*60)
    print("RECOMMENDATION FOR VIDEO_FILE_SIZE_IN_MB")
    print("="*60)

    if right_sizes and left_sizes:
        right_max = max(right_sizes.values())
        left_max = max(left_sizes.values())

        # Get max across both datasets
        absolute_max = max(right_max, left_max)

        # When merging, videos can be concatenated, so we need room for 2 videos
        # Plus some safety margin
        recommended_size = (absolute_max * 2) * 1.2  # 20% safety margin

        # Round up to nearest 50 MB
        recommended_size = ((recommended_size // 50) + 1) * 50

        print(f"\nCurrent DEFAULT_VIDEO_FILE_SIZE_IN_MB: 200 MB")
        print(f"\nLargest video file in RIGHT: {right_max:.1f} MB")
        print(f"Largest video file in LEFT: {left_max:.1f} MB")
        print(f"Absolute maximum: {absolute_max:.1f} MB")
        print(f"\nWhen merging, two videos may be concatenated:")
        print(
            f"  {absolute_max:.1f} MB + {absolute_max:.1f} MB = {absolute_max*2:.1f} MB")
        print(f"\nWith 20% safety margin: {recommended_size:.1f} MB")
        print(f"\n" + "="*60)
        print(f"RECOMMENDED VALUE: {int(recommended_size)} MB")
        print(f"="*60)
        print(f"\nTo apply this fix:")
        print(f"1. Edit: src/lerobot/datasets/utils.py")
        print(
            f"2. Change: DEFAULT_VIDEO_FILE_SIZE_IN_MB = {int(recommended_size)}")
        print(f"3. Re-run your merge command")

        # Show exact edit
        print(f"\n" + "="*60)
        print("EXACT CODE CHANGE")
        print("="*60)
        print(f"\nIn src/lerobot/datasets/utils.py, change line 52 to:")
        print(
            f"\nDEFAULT_VIDEO_FILE_SIZE_IN_MB = {int(recommended_size)}  # Increased for merge")


if __name__ == "__main__":
    main()

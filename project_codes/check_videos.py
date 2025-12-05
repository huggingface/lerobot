import os
from pathlib import Path


def check_video_files(repo_id, name):
    """Check actual video file structure."""
    print(f"\n{'='*60}")
    print(f"{name}: {repo_id}")
    print('='*60)

    cache_path = Path.home() / ".cache/huggingface/lerobot" / \
        repo_id.replace("/", "/")
    videos_path = cache_path / "videos"

    if not videos_path.exists():
        print(f"✗ Videos directory not found: {videos_path}")
        return

    print(f"Videos directory: {videos_path}")

    # List all camera directories
    camera_dirs = [d for d in videos_path.iterdir() if d.is_dir()]
    print(f"\nCamera directories: {len(camera_dirs)}")

    for camera_dir in camera_dirs:
        print(f"\n  📁 {camera_dir.name}/")

        # List chunks
        chunk_dirs = sorted([d for d in camera_dir.iterdir() if d.is_dir()])
        print(f"     Chunks: {len(chunk_dirs)}")

        for chunk_dir in chunk_dirs[:3]:  # Show first 3 chunks
            video_files = sorted(chunk_dir.glob("*.mp4"))
            print(f"       📁 {chunk_dir.name}/ - {len(video_files)} videos")

            for vf in video_files[:3]:  # Show first 3 videos
                size_mb = vf.stat().st_size / (1024 * 1024)
                print(f"          {vf.name}: {size_mb:.1f} MB")

                # Get video duration and frame count
                try:
                    import subprocess
                    result = subprocess.run(
                        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                         '-count_frames', '-show_entries', 'stream=nb_read_frames,duration',
                         '-of', 'csv=p=0', str(vf)],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        output = result.stdout.strip().split(',')
                        if len(output) >= 2:
                            duration = output[0]
                            frames = output[1]
                            print(
                                f"             Duration: {duration}s, Frames: {frames}")
                except Exception as e:
                    print(f"             Could not probe: {e}")


def main():
    check_video_files("YieumYoon/bimanual-center-basket-right-rblock", "RIGHT")
    check_video_files("YieumYoon/bimanual-center-basket-left-rblock", "LEFT")
    check_video_files(
        "YieumYoon/bimanual-center-basket-rblock-rlmerged", "MERGED")


if __name__ == "__main__":
    main()

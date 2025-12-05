from lerobot.datasets.lerobot_dataset import LeRobotDataset
import traceback


def analyze_remaining_issues():
    """Analyze the remaining bad episodes."""
    print(f"\n{'='*60}")
    print("ANALYZING REMAINING BAD EPISODES")
    print('='*60)

    ds = LeRobotDataset(
        repo_id="YieumYoon/bimanual-center-basket-rblock-rlmerged-4",
        video_backend="torchcodec"
    )

    bad_episodes = [59, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
                    103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                    115, 116, 117, 118, 119]

    print(f"\nTotal bad episodes: {len(bad_episodes)}")
    print(f"Bad episode range: {min(bad_episodes)} to {max(bad_episodes)}")

    # Analyze pattern
    print(f"\nPattern analysis:")
    print(f"  Episode 59: From RIGHT dataset (0-89)")
    print(f"  Episodes 90-119: From LEFT dataset (90-179)")
    print(f"  Episodes 120-179: Good (from LEFT dataset)")

    ds._ensure_hf_dataset_loaded()

    # Check a few specific bad episodes
    print(f"\n{'='*60}")
    print("CHECKING SPECIFIC BAD EPISODES")
    print('='*60)

    for ep_idx in [59, 90, 91, 119]:
        print(f"\n--- Episode {ep_idx} ---")

        # Find samples for this episode
        ep_samples = [i for i in range(len(ds.hf_dataset))
                      if ds.hf_dataset[i]['episode_index'] == ep_idx]

        if not ep_samples:
            print(f"  No samples found")
            continue

        print(
            f"  Sample range: {ep_samples[0]} to {ep_samples[-1]} ({len(ep_samples)} frames)")

        # Try to load first sample and get error details
        try:
            sample = ds[ep_samples[0]]
            print(f"  ✓ First sample OK (this shouldn't happen!)")
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Error: {error_msg[:150]}")

            # Extract frame indices from error
            if "Invalid frame index=" in error_msg:
                parts = error_msg.split("Invalid frame index=")[1]
                requested = parts.split(" ")[0]
                max_frames = parts.split("must be less than ")[1].split()[
                    0] if "must be less than" in parts else "?"
                print(f"  Requested frame: {requested}")
                print(f"  Max available: {max_frames}")
                print(
                    f"  Difference: {int(requested) - int(max_frames) if max_frames.isdigit() else '?'}")


def check_video_file_structure():
    """Check the video file structure of the merged dataset."""
    from pathlib import Path

    print(f"\n{'='*60}")
    print("VIDEO FILE STRUCTURE ANALYSIS")
    print('='*60)

    cache_path = Path.home() / \
        ".cache/huggingface/lerobot/YieumYoon/bimanual-center-basket-rblock-rlmerged-4/videos"

    if not cache_path.exists():
        print(f"Videos directory not found: {cache_path}")
        return

    for camera_dir in sorted(cache_path.iterdir()):
        if not camera_dir.is_dir():
            continue

        print(f"\n📁 {camera_dir.name}/")

        video_files = sorted(camera_dir.glob("**/*.mp4"))
        print(f"   Total video files: {len(video_files)}")

        for vf in video_files:
            size_mb = vf.stat().st_size / (1024 * 1024)
            rel_path = vf.relative_to(cache_path)
            print(f"   {rel_path}: {size_mb:.1f} MB")

            # Get video info using ffprobe
            try:
                import subprocess
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-count_frames', '-show_entries', 'stream=nb_read_frames,duration',
                     '-of', 'csv=p=0', str(vf)],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    output = result.stdout.strip().split(',')
                    if len(output) >= 2:
                        duration = output[0]
                        frames = output[1]
                        print(f"      Duration: {duration}s, Frames: {frames}")
            except Exception as e:
                print(f"      Could not probe: {e}")


def main():
    analyze_remaining_issues()
    check_video_file_structure()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print("\nProgress:")
    print("  Before fix: 68 bad episodes (38% failure)")
    print("  After fix:  31 bad episodes (17% failure)")
    print("  Improvement: 37 episodes fixed! ✓")
    print("\nRemaining issue:")
    print("  30 consecutive episodes from LEFT dataset (90-119)")
    print("  + 1 episode from RIGHT dataset (59)")
    print("\nThis suggests the video files are STILL too large for proper concatenation.")
    print("The 500 MB limit helped, but may not be enough for ALL videos.")


if __name__ == "__main__":
    main()

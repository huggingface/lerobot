from lerobot.datasets.lerobot_dataset import LeRobotDataset
import json


def check_dataset_videos(repo_id, name):
    """Check video configuration and availability."""
    print(f"\n{'='*60}")
    print(f"Checking: {name}")
    print(f"Repo: {repo_id}")
    print('='*60)

    try:
        ds = LeRobotDataset(
            repo_id=repo_id,
            video_backend="torchcodec",
        )

        print(f"✓ Dataset loaded")
        print(f"  Episodes: {ds.num_episodes}, Samples: {len(ds)}")

        # Check info.json
        if hasattr(ds, 'info'):
            print(f"\n  Dataset Info:")
            if 'camera_keys' in ds.info:
                print(f"    camera_keys: {ds.info['camera_keys']}")
            if 'video' in ds.info:
                print(f"    video: {ds.info['video']}")
            if 'fps' in ds.info:
                print(f"    fps: {ds.info['fps']}")

        # Check the actual features
        ds._ensure_hf_dataset_loaded()
        print(f"\n  HF Dataset features: {ds.hf_dataset.features.keys()}")

        # Check if video is enabled
        print(f"\n  Video enabled: {ds.video}")

        # Try to get a sample with videos
        try:
            print(f"\n  Trying to load sample 0...")
            sample = ds[0]
            print(f"    Sample keys: {sample.keys()}")

            # Check for image/video keys
            video_keys = [k for k in sample.keys(
            ) if 'image' in k.lower() or 'video' in k.lower()]
            if video_keys:
                print(f"    Video/Image keys found: {video_keys}")
                for key in video_keys:
                    val = sample[key]
                    print(
                        f"      {key}: shape={val.shape if hasattr(val, 'shape') else type(val)}")
            else:
                print(f"    ✗ No video/image keys found in sample!")

        except Exception as e:
            print(f"    ✗ Error loading sample: {repr(e)[:200]}")

        # Check cache directory for videos
        if hasattr(ds, 'root'):
            import os
            print(f"\n  Cache root: {ds.root}")

            # Look for video directories
            if os.path.exists(ds.root):
                contents = os.listdir(ds.root)
                print(f"    Contents: {contents}")

                # Check for videos directory
                videos_dir = os.path.join(ds.root, "videos")
                if os.path.exists(videos_dir):
                    video_contents = os.listdir(videos_dir)
                    print(
                        f"    videos/ contents (first 5): {video_contents[:5]}")
                else:
                    print(f"    ✗ No videos/ directory found")

        return True

    except Exception as e:
        print(f"✗ Failed: {repr(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Check all datasets
    check_dataset_videos(
        "YieumYoon/bimanual-center-basket-right-rblock",
        "RIGHT (Original)"
    )

    check_dataset_videos(
        "YieumYoon/bimanual-center-basket-left-rblock",
        "LEFT (Original)"
    )

    check_dataset_videos(
        "YieumYoon/bimanual-center-basket-rblock-rlmerged",
        "MERGED"
    )


if __name__ == "__main__":
    main()

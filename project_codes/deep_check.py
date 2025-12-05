from lerobot.datasets.lerobot_dataset import LeRobotDataset


def deep_check_dataset(repo_id, name):
    """Deep check of dataset structure."""
    print(f"\n{'='*60}")
    print(f"{name}: {repo_id}")
    print('='*60)

    # Load without video backend to see raw data
    ds = LeRobotDataset(repo_id=repo_id)

    print(f"Episodes: {ds.num_episodes}, Samples: {len(ds)}")

    # Check cache location
    print(f"\nCache location: {ds.root}")

    # Check what's in the cache
    import os
    if os.path.exists(ds.root):
        for item in os.listdir(ds.root):
            item_path = os.path.join(ds.root, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
                # List first few items in subdirectory
                subitems = os.listdir(item_path)[:5]
                for subitem in subitems:
                    print(f"     - {subitem}")
                if len(os.listdir(item_path)) > 5:
                    print(
                        f"     ... and {len(os.listdir(item_path)) - 5} more")
            else:
                size = os.path.getsize(item_path)
                print(f"  📄 {item} ({size:,} bytes)")

    # Load and check sample
    print(f"\nSample 0:")
    sample = ds[0]
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape} {value.dtype}")
        else:
            print(f"  {key}: {type(value)} = {value}")

    # Check info/meta files
    print(f"\nDataset metadata:")
    if hasattr(ds, 'meta'):
        print(f"  meta: {ds.meta}")
    if hasattr(ds, 'info'):
        import json
        print(f"  info keys: {list(ds.info.keys())}")
        if 'camera_keys' in ds.info:
            print(f"  camera_keys: {ds.info['camera_keys']}")
        if 'video' in ds.info:
            print(f"  video config: {ds.info['video']}")

    # Try with video backend
    print(f"\n--- With torchcodec backend ---")
    try:
        ds_video = LeRobotDataset(repo_id=repo_id, video_backend="torchcodec")
        sample_video = ds_video[0]
        print(f"✓ Loaded with video backend")
        print(f"  Sample 0 keys: {list(sample_video.keys())}")
        video_keys = [k for k in sample_video.keys(
        ) if 'image' in k or 'video' in k]
        if video_keys:
            print(f"  Video keys: {video_keys}")
            for vk in video_keys:
                print(
                    f"    {vk}: {sample_video[vk].shape if hasattr(sample_video[vk], 'shape') else type(sample_video[vk])}")
        else:
            print(f"  ✗ No video keys even with video_backend!")
    except Exception as e:
        print(f"✗ Error with video backend: {repr(e)[:300]}")


def main():
    deep_check_dataset(
        "YieumYoon/bimanual-center-basket-right-rblock", "RIGHT")
    deep_check_dataset("YieumYoon/bimanual-center-basket-left-rblock", "LEFT")
    deep_check_dataset(
        "YieumYoon/bimanual-center-basket-rblock-rlmerged", "MERGED")


if __name__ == "__main__":
    main()

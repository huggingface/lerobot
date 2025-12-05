from lerobot.datasets.lerobot_dataset import LeRobotDataset
import traceback


def find_problematic_episodes(repo_id, name, bad_episodes):
    """Check specific bad episodes to understand the issue."""
    print(f"\n{'='*60}")
    print(f"{name}: {repo_id}")
    print(f"Checking {len(bad_episodes)} bad episodes")
    print('='*60)

    ds = LeRobotDataset(repo_id=repo_id, video_backend="torchcodec")

    ds._ensure_hf_dataset_loaded()

    # Get first and last frame of each bad episode
    error_summary = {}

    for ep_idx in bad_episodes[:10]:  # Check first 10 bad episodes
        print(f"\n--- Episode {ep_idx} ---")

        # Find samples belonging to this episode
        ep_samples = [i for i in range(
            len(ds.hf_dataset)) if ds.hf_dataset[i]['episode_index'] == ep_idx]

        if not ep_samples:
            print(f"  ✗ No samples found for episode {ep_idx}")
            continue

        print(
            f"  Samples: {len(ep_samples)} (indices {ep_samples[0]} to {ep_samples[-1]})")

        # Try first sample
        try:
            sample = ds[ep_samples[0]]
            print(f"  ✓ First sample loads OK")
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)[:200]
            print(f"  ✗ First sample FAILED: {error_type}: {error_msg}")

            # Track error types
            if error_type not in error_summary:
                error_summary[error_type] = []
            error_summary[error_type].append({
                'episode': ep_idx,
                'sample': ep_samples[0],
                'message': error_msg
            })

            # Get more details
            print(f"\n  Full traceback:")
            try:
                _ = ds[ep_samples[0]]
            except Exception:
                traceback.print_exc()

        # Try middle sample
        if len(ep_samples) > 1:
            mid_idx = ep_samples[len(ep_samples) // 2]
            try:
                sample = ds[mid_idx]
                print(f"  ✓ Middle sample loads OK")
            except Exception as e:
                print(f"  ✗ Middle sample FAILED: {type(e).__name__}")

    print(f"\n{'='*60}")
    print(f"ERROR SUMMARY")
    print('='*60)
    for error_type, errors in error_summary.items():
        print(f"\n{error_type}: {len(errors)} occurrences")
        print(
            f"  First occurrence: Episode {errors[0]['episode']}, Sample {errors[0]['sample']}")
        print(f"  Message: {errors[0]['message']}")


def compare_episode_structure():
    """Compare episode structure between source and merged datasets."""
    print(f"\n{'='*60}")
    print("COMPARING EPISODE STRUCTURES")
    print('='*60)

    right_ds = LeRobotDataset("YieumYoon/bimanual-center-basket-right-rblock")
    left_ds = LeRobotDataset("YieumYoon/bimanual-center-basket-left-rblock")
    merged_ds = LeRobotDataset(
        "YieumYoon/bimanual-center-basket-rblock-rlmerged")

    print(
        f"\nRIGHT:  {right_ds.num_episodes} episodes, {len(right_ds)} samples")
    print(f"LEFT:   {left_ds.num_episodes} episodes, {len(left_ds)} samples")
    print(
        f"MERGED: {merged_ds.num_episodes} episodes, {len(merged_ds)} samples")

    # Check if episodes are consecutive
    right_ds._ensure_hf_dataset_loaded()
    left_ds._ensure_hf_dataset_loaded()
    merged_ds._ensure_hf_dataset_loaded()

    # Get episode ranges
    right_eps = set(right_ds.hf_dataset['episode_index'])
    left_eps = set(left_ds.hf_dataset['episode_index'])
    merged_eps = set(merged_ds.hf_dataset['episode_index'])

    print(f"\nRIGHT episode range: {min(right_eps)} to {max(right_eps)}")
    print(f"LEFT episode range:  {min(left_eps)} to {max(left_eps)}")
    print(f"MERGED episode range: {min(merged_eps)} to {max(merged_eps)}")

    # Map bad episodes to their source
    bad_episodes = [23, 24, 25, 58, 59, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

    print(f"\nBAD EPISODES ANALYSIS:")
    print(f"  Episodes 0-89 should be from RIGHT dataset")
    print(f"  Episodes 90-179 should be from LEFT dataset")

    bad_from_right = [ep for ep in bad_episodes if ep < 90]
    bad_from_left = [ep for ep in bad_episodes if ep >= 90]

    print(f"\n  Bad episodes from RIGHT range (0-89): {len(bad_from_right)}")
    print(f"    {bad_from_right}")
    print(f"\n  Bad episodes from LEFT range (90-179): {len(bad_from_left)}")
    print(
        f"    {bad_from_left[:10]}{'...' if len(bad_from_left) > 10 else ''}")


def main():
    # All bad episodes from your output
    bad_episodes = [23, 24, 25, 58, 59, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
                    124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
                    136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
                    148, 149]

    compare_episode_structure()
    find_problematic_episodes(
        "YieumYoon/bimanual-center-basket-rblock-rlmerged",
        "MERGED",
        bad_episodes
    )


if __name__ == "__main__":
    main()

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import json


def check_dataset(repo_id, name):
    """Check a dataset and return basic info and bad episodes."""
    print(f"\n{'='*60}")
    print(f"Checking: {name}")
    print(f"Repo: {repo_id}")
    print('='*60)

    try:
        ds = LeRobotDataset(
            repo_id=repo_id,
            video_backend="torchcodec",
        )

        print(f"✓ Dataset loaded successfully")
        print(f"  Total length: {len(ds)}")
        print(f"  Total episodes: {ds.num_episodes}")

        # Check for bad samples
        bad_sample_indices = []
        bad_errors = {}

        for i in range(min(len(ds), 10000)):  # Check first 10k samples or all if less
            try:
                _ = ds[i]
            except Exception as e:
                error_msg = repr(e)
                print(f"  [ERROR] Sample {i} failed: {error_msg[:100]}")
                bad_sample_indices.append(i)
                # Categorize errors
                error_type = type(e).__name__
                bad_errors[error_type] = bad_errors.get(error_type, 0) + 1

        # Map to episodes
        bad_episode_indices = set()
        if bad_sample_indices:
            ds._ensure_hf_dataset_loaded()
            for i in bad_sample_indices:
                ep_idx = int(ds.hf_dataset[i]["episode_index"])
                bad_episode_indices.add(ep_idx)

        print(f"\n  Bad samples: {len(bad_sample_indices)}")
        print(f"  Bad episodes: {len(bad_episode_indices)}")
        if bad_errors:
            print(f"  Error types: {bad_errors}")
        if bad_episode_indices:
            print(
                f"  Bad episode indices: {sorted(bad_episode_indices)[:10]}{'...' if len(bad_episode_indices) > 10 else ''}")

        # Check metadata
        print(f"\n  Checking metadata...")
        ds._ensure_hf_dataset_loaded()
        sample = ds.hf_dataset[0]
        print(f"  Available keys: {list(sample.keys())}")
        print(
            f"  Video keys: {[k for k in sample.keys() if 'observation.images' in k]}")

        return {
            'name': name,
            'repo_id': repo_id,
            'total_episodes': ds.num_episodes,
            'total_samples': len(ds),
            'bad_episodes': sorted(bad_episode_indices),
            'bad_samples': len(bad_sample_indices),
            'error_types': bad_errors,
        }

    except Exception as e:
        print(f"✗ Failed to load dataset: {repr(e)}")
        return None


def main():
    # Check the two original datasets
    right_info = check_dataset(
        "YieumYoon/bimanual-center-basket-right-rblock",
        "RIGHT (Original)"
    )

    left_info = check_dataset(
        "YieumYoon/bimanual-center-basket-left-rblock",
        "LEFT (Original)"
    )

    # Check the merged dataset
    merged_info = check_dataset(
        "YieumYoon/bimanual-center-basket-rblock-rlmerged",
        "MERGED (First attempt)"
    )

    merged2_info = check_dataset(
        "YieumYoon/bimanual-center-basket-rblock-rlmerged-2",
        "MERGED-2 (Second attempt)"
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    for info in [right_info, left_info, merged_info, merged2_info]:
        if info:
            print(f"\n{info['name']}:")
            print(f"  Episodes: {info['total_episodes']}")
            print(f"  Samples: {info['total_samples']}")
            print(f"  Bad episodes: {len(info['bad_episodes'])}")
            print(f"  Bad samples: {info['bad_samples']}")
            if info['error_types']:
                print(f"  Errors: {info['error_types']}")

    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print('='*60)

    if right_info and left_info and merged_info:
        expected_episodes = right_info['total_episodes'] + \
            left_info['total_episodes']
        actual_episodes = merged_info['total_episodes']
        print(f"\nExpected merged episodes: {expected_episodes}")
        print(f"Actual merged episodes: {actual_episodes}")
        print(f"Match: {'✓' if expected_episodes == actual_episodes else '✗'}")

        # Check if bad episodes align with one of the source datasets
        if merged_info['bad_episodes']:
            right_ep_range = range(0, right_info['total_episodes'])
            left_ep_range = range(
                right_info['total_episodes'], right_info['total_episodes'] + left_info['total_episodes'])

            bad_from_right = [
                ep for ep in merged_info['bad_episodes'] if ep in right_ep_range]
            bad_from_left = [
                ep for ep in merged_info['bad_episodes'] if ep in left_ep_range]

            print(
                f"\nBad episodes from RIGHT dataset range: {len(bad_from_right)}")
            print(
                f"Bad episodes from LEFT dataset range: {len(bad_from_left)}")

            if bad_from_right:
                print(f"  First few from right: {bad_from_right[:5]}")
            if bad_from_left:
                print(f"  First few from left: {bad_from_left[:5]}")


if __name__ == "__main__":
    main()

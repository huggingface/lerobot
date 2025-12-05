from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    # --- 1. Instantiate the dataset directly (no full TrainPipelineConfig needed) ---
    ds = LeRobotDataset(
        # repo_id="pepijn223/bimanual-so100-handover-cube",
        repo_id="YieumYoon/bimanual-crlbasket-rblock-merged-00",
        # Use the backend that was failing in your run; change to "decord" if you want to test that instead.
        video_backend="torchcodec",
        # video_backend="pyav",
    )
    print("Dataset length:", len(ds))

    bad_sample_indices: list[int] = []

    # --- 2. Scan all samples, catching decode errors ---
    for i in range(len(ds)):
        try:
            _ = ds[i]
        except Exception as e:
            print(f"[ERROR] Sample index {i} failed with: {repr(e)}")
            bad_sample_indices.append(i)

    # --- 3. Map bad samples to episode indices (without decoding videos) ---
    bad_episode_indices: set[int] = set()
    if bad_sample_indices:
        # Ensure underlying HF dataset is loaded
        ds._ensure_hf_dataset_loaded()
        for i in bad_sample_indices:
            ep_idx = int(ds.hf_dataset[i]["episode_index"])
            bad_episode_indices.add(ep_idx)

    print("\n--- Scan finished ---")
    print("Total bad samples:", len(bad_sample_indices))
    print("Bad sample indices:", bad_sample_indices)

    if bad_episode_indices:
        bad_eps_sorted = sorted(bad_episode_indices)
        print("Bad episode_index values:", bad_eps_sorted)

        # Also print a suggested list of good episodes you can pass via --dataset.episodes
        num_eps = ds.num_episodes
        good_eps = [ep for ep in range(
            num_eps) if ep not in bad_episode_indices]
        print(f"\nTotal episodes: {num_eps}")
        print("Suggested good episode list for --dataset.episodes:")
        print(good_eps)


if __name__ == "__main__":
    main()

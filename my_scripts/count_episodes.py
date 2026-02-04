import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    # Set the environment variable as specified in the original command
    if "HF_LEROBOT_HOME" not in os.environ:
        os.environ["HF_LEROBOT_HOME"] = str(Path.home() / ".cache/huggingface/lerobot")

    # Hardcoded dataset path as requested
    dataset_repo_id = "local/lerobot_pick_and_place"

    try:
        ds = LeRobotDataset(dataset_repo_id)
        print(f"Episodes: {ds.meta.total_episodes}")
    except Exception as e:
        print(f"Error loading dataset {dataset_repo_id}: {e}")


if __name__ == "__main__":
    main()

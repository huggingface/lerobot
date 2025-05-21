#!/usr/bin/env python
import argparse
from lerobot.common.datasets.utils import append_jsonlines
import json
from datasets import load_dataset, Dataset
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    write_episode,
    write_episode_stats,
    write_info,
)
from lerobot.common.datasets.compute_stats import compute_episode_stats

# === CONFIGURATION ===
GOAL_SQUARE = "D4"
PIECE_TYPE = "rook"
IS_WHITE = True

def patch_episode_task(dataset: LeRobotDataset, episode_index: int, task_description: str):
    # Ensure task is registered
    if dataset.meta.get_task_index(task_description) is None:
        dataset.meta.add_task(task_description)
    task_index = dataset.meta.get_task_index(task_description)

    # Load parquet
    file_path = dataset.meta.root / dataset.meta.get_data_file_path(episode_index)
    ep_data = load_dataset("parquet", data_files=str(file_path), split="train").to_dict()

    # Patch task_index field
    ep_data["task_index"] = [task_index] * len(ep_data["index"])
    Dataset.from_dict(ep_data).to_parquet(file_path)

    print(f"✅ Patched episode {episode_index} with task_index {task_index}")

    # Update episode metadata
    dataset.meta.episodes[episode_index]["tasks"] = [task_description]
    write_episode(dataset.meta.episodes[episode_index], dataset.meta.root)

    # Recompute stats
    ep_array_data = {
        k: np.array(v) for k, v in ep_data.items()
        if k in dataset.meta.features  # filter out e.g. "task"
    }
    stats = compute_episode_stats(ep_array_data, dataset.meta.features)
    dataset.meta.episodes_stats[episode_index] = stats
    write_episode_stats(episode_index, stats, dataset.meta.root)

def add_metadata_to_dataset(repo_id: str):
    dataset = LeRobotDataset(repo_id=repo_id, force_cache_sync=True)
    print(f"📦 Loaded dataset from Hugging Face: {dataset}")

    for episode_index in dataset.meta.episodes.keys():
        print(f"\n🔁 Episode {episode_index}")

        start_square = input(f"Enter start square for episode {episode_index} (e.g., 'A4'): ").strip().upper()

        task_description = json.dumps({
            "piece": PIECE_TYPE,
            "color": "white" if IS_WHITE else "black",
            "start_square": start_square,
            "goal_square": GOAL_SQUARE
        })

        patch_episode_task(dataset, episode_index, task_description)

    # Update info.json (total_tasks etc.)
    write_info(dataset.meta.info, dataset.meta.root)

    # Cleanly rewrite episodes.jsonl (to remove duplicate entries)
    episodes_path = dataset.meta.root / "meta/episodes.jsonl"
    episodes_path.write_text("")  # clear file
    for episode in dataset.meta.episodes.values():
        append_jsonlines(episode, episodes_path)

    print("\n📤 Pushing updated dataset to Hugging Face...")
    dataset.push_to_hub(tags=["structured-tasks"])
    print("✅ All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amend LeRobotDataset episodes with structured task metadata.")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face dataset repo ID.")
    args = parser.parse_args()

    add_metadata_to_dataset(repo_id=args.repo_id)

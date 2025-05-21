#!/usr/bin/env python
import argparse
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset, Dataset

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    write_episode,
    write_episode_stats,
    write_info,
    append_jsonlines,
)
from lerobot.common.datasets.compute_stats import compute_episode_stats

# === CONFIGURATION ===
GOAL_SQUARE = "D4"
PIECE_TYPE = "rook"
IS_WHITE = True

def update_episode_tasks(dataset):
    print("🔁 Updating episode task strings in memory...")
    for episode_index in dataset.meta.episodes:
        start_square = input(f"Enter start square for episode {episode_index} (e.g., 'A4'): ").strip().upper()
        task_description = json.dumps({
            "piece": PIECE_TYPE,
            "color": "white" if IS_WHITE else "black",
            "start_square": start_square,
            "goal_square": GOAL_SQUARE
        })
        dataset.meta.episodes[episode_index]["tasks"] = [task_description]

def rebuild_task_index_from_episodes(dataset):
    dataset.meta.task_to_task_index.clear()
    dataset.meta.tasks.clear()
    dataset.meta.info["total_tasks"] = 0

    task_set = set()
    for ep in dataset.meta.episodes.values():
        for task in ep["tasks"]:
            task_set.add(task)

    sorted_tasks = sorted(task_set)
    for i, task in enumerate(sorted_tasks):
        dataset.meta.task_to_task_index[task] = i
        dataset.meta.tasks[i] = task
        dataset.meta.info["total_tasks"] += 1

def patch_episode_parquet(dataset: LeRobotDataset, episode_index: int):
    task_str = dataset.meta.episodes[episode_index]["tasks"][0]
    task_index = dataset.meta.task_to_task_index[task_str]

    file_path = dataset.meta.root / dataset.meta.get_data_file_path(episode_index)
    ep_data = load_dataset("parquet", data_files=str(file_path), split="train").to_dict()
    ep_data["task_index"] = [task_index] * len(ep_data["index"])
    Dataset.from_dict(ep_data).to_parquet(file_path)

    # Update stats
    ep_array_data = {
        k: np.array(v) for k, v in ep_data.items()
        if k in dataset.meta.features
    }
    stats = compute_episode_stats(ep_array_data, dataset.meta.features)
    dataset.meta.episodes_stats[episode_index] = stats
    write_episode_stats(episode_index, stats, dataset.meta.root)

    print(f"✅ Episode {episode_index} set to task_index {task_index}")

def add_metadata_to_dataset(repo_id: str):
    dataset = LeRobotDataset(repo_id=repo_id, force_cache_sync=True)
    print(f"📦 Loaded dataset from Hugging Face: {dataset}")

    # Step 1: Prompt user and update task strings in memory
    update_episode_tasks(dataset)

    # Step 2: Rebuild task-to-index mapping
    rebuild_task_index_from_episodes(dataset)

    # Step 3: Patch each episode's parquet file
    for episode_index in dataset.meta.episodes:
        patch_episode_parquet(dataset, episode_index)

    # Step 4: Overwrite metadata files
    write_info(dataset.meta.info, dataset.meta.root)

    episodes_path = dataset.meta.root / "meta/episodes.jsonl"
    episodes_path.write_text("")
    for episode in dataset.meta.episodes.values():
        append_jsonlines(episode, episodes_path)

    tasks_path = dataset.meta.root / "meta/tasks.jsonl"
    tasks_path.write_text("")
    for task, index in sorted(dataset.meta.task_to_task_index.items(), key=lambda kv: kv[1]):
        append_jsonlines({"task_index": index, "task": task}, tasks_path)

    print("\n📤 Pushing updated dataset to Hugging Face...")
    dataset.push_to_hub(tags=["structured-tasks"])
    print("✅ All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch LeRobotDataset episodes with structured task metadata.")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face dataset repo ID.")
    args = parser.parse_args()

    add_metadata_to_dataset(repo_id=args.repo_id)

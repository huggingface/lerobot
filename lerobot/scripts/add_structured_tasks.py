#!/usr/bin/env python

import argparse
import json

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import append_jsonlines, write_info

# === TASK CONFIGURATION ===
GOAL_SQUARE = "D4"
PIECE_TYPE = "rook"
COLOR = "white"


def prompt_and_build_task(start_square: str):
    """Construct a structured task string (as JSON) from user input."""
    return json.dumps(
        {
            "piece": PIECE_TYPE,
            "color": COLOR,
            "start_square": start_square.upper(),
            "goal_square": GOAL_SQUARE,
        }
    )


def update_task_mappings(dataset: LeRobotDataset):
    """Rebuilds task-to-index mappings from episode metadata."""
    dataset.meta.task_to_task_index.clear()
    dataset.meta.tasks.clear()
    dataset.meta.info["total_tasks"] = 0

    unique_tasks = set()
    for ep in dataset.meta.episodes.values():
        for task in ep["tasks"]:
            unique_tasks.add(task)

    sorted_tasks = sorted(unique_tasks)
    for i, task in enumerate(sorted_tasks):
        dataset.meta.task_to_task_index[task] = i
        dataset.meta.tasks[i] = task
        dataset.meta.info["total_tasks"] += 1


def patch_parquet_task_index(dataset: LeRobotDataset, episode_index: int):
    """Updates only the 'task_index' column in the .parquet file for one episode."""
    task_str = dataset.meta.episodes[episode_index]["tasks"][0]
    task_index = dataset.meta.task_to_task_index[task_str]
    file_path = dataset.meta.root / dataset.meta.get_data_file_path(episode_index)

    # Load the original data preserving all columns
    table = pq.read_table(file_path)
    df = table.to_pandas()

    # Add or replace the task_index column
    df["task_index"] = task_index

    # Write back to the same file safely
    new_table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(new_table, file_path)


def save_metadata(dataset: LeRobotDataset):
    """Writes updated episodes.jsonl and tasks.jsonl to disk."""
    episodes_path = dataset.meta.root / "meta/episodes.jsonl"
    episodes_path.write_text("")
    for episode in dataset.meta.episodes.values():
        append_jsonlines(episode, episodes_path)

    tasks_path = dataset.meta.root / "meta/tasks.jsonl"
    tasks_path.write_text("")
    for task, index in sorted(dataset.meta.task_to_task_index.items(), key=lambda kv: kv[1]):
        append_jsonlines({"task_index": index, "task": task}, tasks_path)

    write_info(dataset.meta.info, dataset.meta.root)


def main(repo_id: str):
    dataset = LeRobotDataset(repo_id=repo_id, force_cache_sync=True)
    print(f"📦 Loaded dataset: {repo_id}")
    print("Found episodes:", list(dataset.meta.episodes.keys()))

    # Step 1: Prompt for task metadata
    for ep_idx in dataset.meta.episodes:
        start_square = input(f"Enter start square for episode {ep_idx} (e.g. A7, B7): ").strip().upper()
        task_str = prompt_and_build_task(start_square)
        dataset.meta.episodes[ep_idx]["tasks"] = [task_str]

    # Step 2: Rebuild task index mappings
    update_task_mappings(dataset)

    # Step 3: Patch parquet files with new task_index
    for ep_idx in dataset.meta.episodes:
        patch_parquet_task_index(dataset, ep_idx)

    # Step 4: Save updated metadata
    save_metadata(dataset)

    # Step 5: Push back to Hugging Face
    print("\n🚀 Pushing updated dataset to Hugging Face...")
    dataset.push_to_hub(tags=["structured-tasks"])
    print("✅ Done pushing!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add structured tasks and task_index to an existing LeRobot dataset."
    )
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face dataset repo ID.")
    args = parser.parse_args()
    main(args.repo_id)

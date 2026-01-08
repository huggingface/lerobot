#!/usr/bin/env python
"""Unify all tasks in a dataset to a single task."""

import argparse
import json
from pathlib import Path

import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_tasks


def unify_tasks(repo_id: str, new_task: str):
    """Set all episodes to use a single task."""
    print(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id)
    root = dataset.root
    
    print(f"Current tasks: {list(dataset.meta.tasks['task']) if dataset.meta.tasks is not None else []}")
    
    # 1. Update tasks.parquet to have only one task
    tasks_df = pd.DataFrame({"task": [new_task]})
    write_tasks(tasks_df, root)
    print(f"Set single task: '{new_task}'")
    
    # 2. Update all data parquet files to set task_index=0
    data_dir = root / "data"
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        df["task_index"] = 0
        df.to_parquet(parquet_path)
        print(f"Updated: {parquet_path.relative_to(root)}")
    
    # 3. Update info.json
    info_path = root / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["total_tasks"] = 1
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"\nDone! All {dataset.meta.total_episodes} episodes now use task: '{new_task}'")
    print(f"\nTo push: huggingface-cli upload {repo_id} {root} --repo-type dataset")


def main():
    parser = argparse.ArgumentParser(description="Unify all tasks in a dataset to a single task")
    parser.add_argument("--repo_id", type=str, required=True, help="Dataset repo_id")
    parser.add_argument("--task", type=str, required=True, help="New task description")
    args = parser.parse_args()
    
    unify_tasks(args.repo_id, args.task)


if __name__ == "__main__":
    main()


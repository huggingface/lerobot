"""Aggregate multiple task-specific LeRobot datasets into a single combined dataset."""

import argparse
import os
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple task-specific datasets into a single LeRobot dataset"
    )
    parser.add_argument(
        "--task-datasets-dir",
        type=str,
        required=True,
        help="Directory containing individual task datasets (e.g., /path/to/behavior1k/)",
    )
    parser.add_argument(
        "--aggregated-root",
        type=str,
        required=True,
        help="Path where the aggregated dataset will be written",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=50,
        help="Number of tasks to aggregate (default: 50)",
    )
    parser.add_argument(
        "--task-start-idx",
        type=int,
        default=0,
        help="Starting task index (default: 0)",
    )
    parser.add_argument(
        "--hf-user",
        type=str,
        default=None,
        help="HuggingFace username for repo IDs (defaults to HF_USER env var or 'lerobot')",
    )
    parser.add_argument(
        "--aggregated-repo-id",
        type=str,
        default=None,
        help="Repository ID for the aggregated dataset (defaults to {hf_user}/behavior1k)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the aggregated dataset to the Hugging Face Hub",
    )

    args = parser.parse_args()

    # Determine HF user
    hf_user = args.hf_user or os.environ.get("HF_USER", "lerobot")

    # Set default aggregated repo ID if not provided
    aggregated_repo_id = args.aggregated_repo_id or f"{hf_user}/behavior1k"

    # Generate task indices
    task_indices = range(args.task_start_idx, args.task_start_idx + args.num_tasks)

    # Generate repo IDs for individual tasks
    repo_ids = [f"{hf_user}/behavior1k-task{i:04d}" for i in task_indices]

    # Generate local paths for individual task datasets
    task_datasets_dir = Path(args.task_datasets_dir)
    roots = [task_datasets_dir / f"behavior1k-task{i:04d}" for i in task_indices]

    # Aggregated dataset path
    aggregated_root = Path(args.aggregated_root)

    print(f"🔹 Aggregating {args.num_tasks} task datasets")
    print(f"Task datasets directory: {task_datasets_dir}")
    print(f"Aggregated output: {aggregated_root}")
    print(f"Aggregated repo ID: {aggregated_repo_id}")

    aggregate_datasets(
        repo_ids=repo_ids,
        roots=roots,
        aggr_repo_id=aggregated_repo_id,
        aggr_root=aggregated_root,
    )

    print("✅ Aggregation complete")

    if args.push_to_hub:
        print(f"📤 Pushing aggregated dataset to {aggregated_repo_id}")
        ds = LeRobotDataset(repo_id=aggregated_repo_id, root=aggregated_root)
        ds.push_to_hub()
        print("✅ Successfully pushed to hub")


if __name__ == "__main__":
    main()

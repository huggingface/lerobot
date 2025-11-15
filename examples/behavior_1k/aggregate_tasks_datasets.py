from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    """Aggregate all tasks datasets into a single LeRobotDataset and push it to the hub."""
    task_indices = range(50)

    repo_ids = [f"fracapuano/behavior1k-task{i:04d}" for i in task_indices]

    roots = [Path(f"/fsx/francesco_capuano/behavior1k/behavior1k-task{i:04d}") for i in task_indices]

    aggregated_root = Path("/fsx/francesco_capuano/behavior1k/behavior1k")
    aggregated_repo_id = "fracapuano/behavior1k"

    aggregate_datasets(
        repo_ids=repo_ids,
        roots=roots,
        aggr_repo_id=aggregated_repo_id,
        aggr_root=aggregated_root,
    )

    ds = LeRobotDataset(repo_id=aggregated_repo_id, root=aggregated_root)
    ds.push_to_hub()


if __name__ == "__main__":
    main()

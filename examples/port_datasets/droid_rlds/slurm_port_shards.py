import argparse
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

from examples.port_datasets.droid_rlds.port_droid import DROID_SHARDS
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata


def validate_shard(repo_id):
    """Sanity check that ensure meta data can be loaded and all files are present."""
    meta = LeRobotDatasetMetadata(repo_id)

    if meta.total_episodes == 0:
        raise ValueError("Number of episodes is 0.")

    for ep_idx in range(meta.total_episodes):
        data_path = meta.root / meta.get_data_file_path(ep_idx)

        if not data_path.exists():
            raise ValueError(f"Parquet file is missing in: {data_path}")

        for vid_key in meta.video_keys:
            vid_path = meta.root / meta.get_video_file_path(ep_idx, vid_key)
            if not vid_path.exists():
                raise ValueError(f"Video file is missing in: {vid_path}")


class PortDroidShards(PipelineStep):
    def __init__(
        self,
        raw_dir: Path | str,
        repo_id: str = None,
    ):
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.repo_id = repo_id

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        from datasets.utils.tqdm import disable_progress_bars

        from examples.port_datasets.droid_rlds.port_droid import port_droid
        from lerobot.common.utils.utils import init_logging

        init_logging()
        disable_progress_bars()

        shard_repo_id = f"{self.repo_id}_world_{world_size}_rank_{rank}"

        port_droid(
            self.raw_dir,
            shard_repo_id,
            push_to_hub=False,
            num_shards=world_size,
            shard_index=rank,
        )

        validate_shard(shard_repo_id)


def make_port_executor(
    raw_dir, repo_id, job_name, logs_dir, workers, partition, cpus_per_task, mem_per_cpu, slurm=True
):
    kwargs = {
        "pipeline": [
            PortDroidShards(raw_dir, repo_id),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": DROID_SHARDS,
                "workers": workers,
                "time": "08:00:00",
                "partition": partition,
                "cpus_per_task": cpus_per_task,
                "sbatch_args": {"mem-per-cpu": mem_per_cpu},
            }
        )
        executor = SlurmPipelineExecutor(**kwargs)
    else:
        kwargs.update(
            {
                "tasks": 1,
                "workers": 1,
            }
        )
        executor = LocalPipelineExecutor(**kwargs)

    return executor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing input raw datasets (e.g. `path/to/dataset` or `path/to/dataset/version).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="Repositery identifier on Hugging Face: a community or a user name `/` the name of the dataset, required when push-to-hub is True.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        help="Path to logs directory for `datatrove`.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="port_droid",
        help="Job name used in slurm, and name of the directory created inside the provided logs directory.",
    )
    parser.add_argument(
        "--slurm",
        type=int,
        default=1,
        help="Launch over slurm. Use `--slurm 0` to launch sequentially (useful to debug).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2048,
        help="Number of slurm workers. It should be less than the maximum number of shards.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="Slurm partition. Ideally a CPU partition. No need for GPU partition.",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=8,
        help="Number of cpus that each slurm worker will use.",
    )
    parser.add_argument(
        "--mem-per-cpu",
        type=str,
        default="1950M",
        help="Memory per cpu that each worker will use.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    kwargs["slurm"] = kwargs.pop("slurm") == 1
    port_executor = make_port_executor(**kwargs)
    port_executor.run()


if __name__ == "__main__":
    main()

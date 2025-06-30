import argparse
import logging
from pathlib import Path
import tarfile

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

from examples.port_datasets.agibot_hdf5.download import RAW_REPO_ID, download_meta_data, get_observations_files



class PortAgiBotShards(PipelineStep):
    def __init__(
        self,
        raw_dir: Path | str,
        repo_id: str = None,
    ):
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.repo_id = repo_id

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import shutil
        import logging
        import tarfile
        from datasets.utils.tqdm import disable_progress_bars

        from lerobot.common.constants import HF_LEROBOT_HOME
        from examples.port_datasets.agibot_hdf5.port_agibot import port_agibot
        from examples.port_datasets.agibot_hdf5.download import get_observations_files, download, no_depth, RAW_REPO_ID
        from examples.port_datasets.droid_rlds.port_droid import validate_dataset
        from lerobot.common.utils.utils import init_logging

        init_logging()
        disable_progress_bars()

        shard_repo_id = f"{self.repo_id}_world_{world_size}_rank_{rank}"

        dataset_dir = HF_LEROBOT_HOME / shard_repo_id
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        obs_files, _ = get_observations_files(self.raw_dir, RAW_REPO_ID)
        obs_file = obs_files[rank]

        # Download subset
        download(self.raw_dir, allow_patterns=obs_file)

        tar_path = self.raw_dir / obs_file
        with tarfile.open(tar_path, 'r') as tar:
            extracted_files = tar.getnames()

        task_index = int(tar_path.parent.name)
        episode_names = [int(p) for p in extracted_files if '/' not in p]

        # Untar if needed
        if not all([(tar_path.parent / f"{ep_name}").exists() for ep_name in episode_names]):
            logging.info(f"Untar-ing {tar_path}...")
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=tar_path.parent, filter=no_depth)

        port_agibot(self.raw_dir, shard_repo_id, task_index, episode_names, push_to_hub=False)

        for ep_name in episode_names:
            shutil.rmtree(str(tar_path.parent / f"{ep_name}"))

        tar_path.unlink()

        validate_dataset(shard_repo_id)


def make_port_executor(
    raw_dir, repo_id, job_name, logs_dir, workers, partition, cpus_per_task, mem_per_cpu, slurm=True
):
    download_meta_data(raw_dir)
    obs_files, _ = get_observations_files(raw_dir, RAW_REPO_ID)
    num_shards = len(obs_files)

    kwargs = {
        "pipeline": [
            PortAgiBotShards(raw_dir, repo_id),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": num_shards,
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
                "tasks": num_shards,
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

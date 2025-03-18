import argparse
import logging
import os
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from huggingface_hub import HfApi
from huggingface_hub.constants import REPOCARD_NAME

from examples.port_datasets.droid_rlds.port_droid import DROID_SHARDS
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import create_lerobot_dataset_card


class UploadDataset(PipelineStep):
    def __init__(
        self,
        repo_id: str,
        branch: str | None = None,
        revision: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        private: bool = False,
        **card_kwargs,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.branch = branch
        self.tags = tags
        self.license = license
        self.private = private
        self.card_kwargs = card_kwargs
        self.revision = revision if revision else CODEBASE_VERSION

        if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") != "1":
            logging.warning(
                'HF_HUB_ENABLE_HF_TRANSFER is not set to "1". Install hf_transfer and set the env '
                "variable for faster uploads:\npip install hf-transfer\nexport HF_HUB_ENABLE_HF_TRANSFER=1"
            )

        self.create_repo()

    def create_repo(self):
        hub_api = HfApi()

        meta = LeRobotDatasetMetadata(self.repo_id)
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=self.private,
            repo_type="dataset",
            exist_ok=True,
        )
        if self.branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=self.branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        if not hub_api.file_exists(self.repo_id, REPOCARD_NAME, repo_type="dataset", revision=self.branch):
            card = create_lerobot_dataset_card(
                tags=self.tags, dataset_info=meta.info, license=self.license, **self.card_kwargs
            )
            card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=self.branch)

        def list_files_recursively(directory):
            base_path = Path(directory)
            return [str(file.relative_to(base_path)) for file in base_path.rglob("*") if file.is_file()]

        meta = LeRobotDatasetMetadata(self.repo_id)
        self.file_paths = list_files_recursively(meta.root)
        self.file_paths = sorted(self.file_paths)

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import logging
        import random
        import time
        from itertools import islice

        from huggingface_hub import CommitOperationAdd, create_commit, preupload_lfs_files
        from huggingface_hub.utils import HfHubHTTPError

        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.common.utils.utils import init_logging

        BASE_DELAY = 1.0  # noqa: N806
        MAX_RETRIES = 24  # noqa: N806

        init_logging()

        def chunked(lst, n):
            it = iter(lst)
            return [list(islice(it, size)) for size in [len(lst) // n + (i < len(lst) % n) for i in range(n)]]

        chunks = chunked(self.file_paths, world_size)
        file_paths = chunks[rank]

        if len(file_paths) == 0:
            raise ValueError(file_paths)

        meta = LeRobotDatasetMetadata(self.repo_id)
        additions = [
            CommitOperationAdd(path_in_repo=path, path_or_fileobj=meta.root / path) for path in file_paths
        ]
        logging.info(f"Uploading {','.join(file_paths)} to the hub...")
        preupload_lfs_files(
            repo_id=self.repo_id, repo_type="dataset", additions=additions, revision=self.branch
        )
        logging.info(f"Upload of {','.join(file_paths)} to the hub complete!")

        retries = 0
        while True:
            try:
                create_commit(
                    self.repo_id,
                    repo_type="dataset",
                    operations=additions,
                    commit_message=f"DataTrove upload ({len(additions)} files)",
                    revision=self.branch,
                )
                break
            except HfHubHTTPError as e:
                if "A commit has happened since" in e.server_message:
                    if retries >= MAX_RETRIES:
                        logging.error(f"Failed to create commit after {MAX_RETRIES=}. Giving up.")
                        raise e
                    logging.info("Commit creation race condition issue. Waiting...")
                    time.sleep(BASE_DELAY * 2**retries + random.uniform(0, 2))
                    retries += 1
                else:
                    raise e


def make_upload_executor(
    repo_id, job_name, logs_dir, workers, partition, cpus_per_task, mem_per_cpu, slurm=True
):
    kwargs = {
        "pipeline": [
            UploadDataset(repo_id),
        ],
        "logging_dir": str(logs_dir),
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
                "tasks": DROID_SHARDS,
                "workers": 1,
            }
        )
        executor = LocalPipelineExecutor(**kwargs)

    return executor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        help="Repositery identifier on Hugging Face: a community or a user name `/` the name of the dataset, required when push-to-hub is True.",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
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
    upload_executor = make_upload_executor(**kwargs)
    upload_executor.run()


if __name__ == "__main__":
    main()

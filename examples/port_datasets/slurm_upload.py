#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from huggingface_hub import HfApi
from huggingface_hub.constants import REPOCARD_NAME
from port_droid import DROID_SHARDS

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.utils import create_lerobot_dataset_card
from lerobot.utils.utils import init_logging


class UploadDataset(PipelineStep):
    def __init__(
        self,
        repo_id: str,
        branch: str | None = None,
        revision: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        private: bool = False,
        distant_repo_id: str | None = None,
        **card_kwargs,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.distant_repo_id = self.repo_id if distant_repo_id is None else distant_repo_id
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
        logging.info(f"Loading meta data from {self.repo_id}...")
        meta = LeRobotDatasetMetadata(self.repo_id)

        logging.info(f"Creating repo {self.distant_repo_id}...")
        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.distant_repo_id,
            private=self.private,
            repo_type="dataset",
            exist_ok=True,
        )
        if self.branch:
            hub_api.create_branch(
                repo_id=self.distant_repo_id,
                branch=self.branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        if not hub_api.file_exists(
            self.distant_repo_id, REPOCARD_NAME, repo_type="dataset", revision=self.branch
        ):
            card = create_lerobot_dataset_card(
                tags=self.tags, dataset_info=meta.info, license=self.license, **self.card_kwargs
            )
            card.push_to_hub(repo_id=self.distant_repo_id, repo_type="dataset", revision=self.branch)

            hub_api.create_tag(self.distant_repo_id, tag=CODEBASE_VERSION, repo_type="dataset")

        def list_files_recursively(directory):
            base_path = Path(directory)
            return [str(file.relative_to(base_path)) for file in base_path.rglob("*") if file.is_file()]

        logging.info(f"Listing all local files from {self.repo_id}...")
        self.file_paths = list_files_recursively(meta.root)
        self.file_paths = sorted(self.file_paths)

    def create_chunks(self, lst, n):
        from itertools import islice

        it = iter(lst)
        return [list(islice(it, size)) for size in [len(lst) // n + (i < len(lst) % n) for i in range(n)]]

    def create_commits(self, additions):
        import logging
        import math
        import random
        import time

        from huggingface_hub import create_commit
        from huggingface_hub.utils import HfHubHTTPError

        FILES_BETWEEN_COMMITS = 10  # noqa: N806
        BASE_DELAY = 0.1  # noqa: N806
        MAX_RETRIES = 12  # noqa: N806

        # Split the files into smaller chunks for faster commit
        # and avoiding "A commit has happened since" error
        num_chunks = math.ceil(len(additions) / FILES_BETWEEN_COMMITS)
        chunks = self.create_chunks(additions, num_chunks)

        for chunk in chunks:
            retries = 0
            while True:
                try:
                    create_commit(
                        self.distant_repo_id,
                        repo_type="dataset",
                        operations=chunk,
                        commit_message=f"DataTrove upload ({len(chunk)} files)",
                        revision=self.branch,
                    )
                    # TODO: every 100 chunks super_squach_commits()
                    logging.info("create_commit completed!")
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

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import logging

        from datasets.utils.tqdm import disable_progress_bars
        from huggingface_hub import CommitOperationAdd, preupload_lfs_files

        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.utils.utils import init_logging

        init_logging()
        disable_progress_bars()

        chunks = self.create_chunks(self.file_paths, world_size)
        file_paths = chunks[rank]

        if len(file_paths) == 0:
            raise ValueError(file_paths)

        logging.info("Pre-uploading LFS files...")
        for i, path in enumerate(file_paths):
            logging.info(f"{i}: {path}")

        meta = LeRobotDatasetMetadata(self.repo_id)
        additions = [
            CommitOperationAdd(path_in_repo=path, path_or_fileobj=meta.root / path) for path in file_paths
        ]
        preupload_lfs_files(
            repo_id=self.distant_repo_id, repo_type="dataset", additions=additions, revision=self.branch
        )

        logging.info("Creating commits...")
        self.create_commits(additions)
        logging.info("Done!")


def make_upload_executor(
    repo_id, job_name, logs_dir, workers, partition, cpus_per_task, mem_per_cpu, private=False, slurm=True
):
    kwargs = {
        "pipeline": [
            UploadDataset(repo_id, private=private),
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
        type=Path,
        help="Path to logs directory for `datatrove`.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="upload_droid",
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
        default=50,
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
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Whether to create a private repository.",
    )

    init_logging()

    args = parser.parse_args()
    kwargs = vars(args)
    kwargs["slurm"] = kwargs.pop("slurm") == 1
    upload_executor = make_upload_executor(**kwargs)
    upload_executor.run()


if __name__ == "__main__":
    main()

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
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from port_datasets.droid_rlds.port_droid import DROID_SHARDS


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
        from port_datasets.droid_rlds.port_droid import port_droid, validate_dataset

        from lerobot.utils.utils import init_logging

        init_logging()
        disable_progress_bars()

        shard_repo_id = f"{self.repo_id}_world_{world_size}_rank_{rank}"

        try:
            validate_dataset(shard_repo_id)
            return
        except Exception:
            pass  # nosec B110 - Dataset doesn't exist yet, continue with porting

        port_droid(
            self.raw_dir,
            shard_repo_id,
            push_to_hub=False,
            num_shards=world_size,
            shard_index=rank,
        )

        validate_dataset(shard_repo_id)


class PortRLDSShards(PipelineStep):
    def __init__(
        self,
        raw_dir: Path | str,
        repo_id: str = None,
        num_shards: int = None,
    ):
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.repo_id = repo_id
        self.num_shards = num_shards

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        from datasets.utils.tqdm import disable_progress_bars
        from port_datasets.port_rlds import port_rlds, validate_dataset

        from lerobot.utils.utils import init_logging

        init_logging()
        disable_progress_bars()

        shard_repo_id = f"{self.repo_id}_world_{world_size}_rank_{rank}"

        try:
            validate_dataset(shard_repo_id)
            return
        except Exception:
            pass  # nosec B110 - Dataset doesn't exist yet, continue with porting

        port_rlds(
            self.raw_dir,
            shard_repo_id,
            push_to_hub=False,
            num_shards=world_size,
            shard_index=rank,
        )

        validate_dataset(shard_repo_id)


def make_port_executor(
    raw_dir,
    repo_id,
    job_name,
    logs_dir,
    workers,
    partition,
    cpus_per_task,
    mem_per_cpu,
    slurm=True,
    dataset_type="droid",
    num_shards=None,
):
    # Select appropriate pipeline step based on dataset type
    if dataset_type.lower() == "droid":
        pipeline_step = PortDroidShards(raw_dir, repo_id)
        default_shards = DROID_SHARDS
    elif dataset_type.lower() == "rlds":
        pipeline_step = PortRLDSShards(raw_dir, repo_id, num_shards)
        default_shards = num_shards or workers  # Use num_shards or fallback to workers
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    kwargs = {
        "pipeline": [pipeline_step],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": default_shards,
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
        "--dataset-type",
        type=str,
        choices=["droid", "rlds"],
        default="droid",
        help="Type of dataset to process: 'droid' for DROID datasets or 'rlds' for RLDS/OpenX datasets.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Job name used in slurm, and name of the directory created inside the provided logs directory. Defaults to 'port_{dataset_type}'.",
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
        default=None,
        help="Number of slurm workers. Defaults: 2048 for DROID, 64 for RLDS datasets.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of shards to split the dataset into. For DROID datasets, this is fixed at 2048. For RLDS datasets, defaults to number of workers.",
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

    # Set defaults based on dataset type
    if args.job_name is None:
        args.job_name = f"port_{args.dataset_type}"

    if args.workers is None:
        if args.dataset_type == "droid":
            args.workers = 2048
        else:  # rlds
            args.workers = 64

    # Convert args to kwargs and process
    kwargs = vars(args)
    kwargs["slurm"] = kwargs.pop("slurm") == 1

    port_executor = make_port_executor(**kwargs)
    port_executor.run()


if __name__ == "__main__":
    main()

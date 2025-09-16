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
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from port_datasets.droid_rlds.port_droid import DROID_SHARDS

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.utils.utils import init_logging


class AggregateDatasets(PipelineStep):
    def __init__(
        self,
        repo_ids: list[str],
        aggregated_repo_id: str,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.aggr_repo_id = aggregated_repo_id

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        init_logging()

        # Since aggregate_datasets already handles parallel processing internally,
        # we only need one worker to run the entire aggregation
        if rank == 0:
            logging.info(f"Starting aggregation of {len(self.repo_ids)} datasets into {self.aggr_repo_id}")
            aggregate_datasets(self.repo_ids, self.aggr_repo_id)
            logging.info("Aggregation complete!")
        else:
            logging.info(f"Worker {rank} skipping - only worker 0 performs aggregation")


def make_aggregate_executor(
    repo_ids, repo_id, job_name, logs_dir, workers, partition, cpus_per_task, mem_per_cpu, slurm=True
):
    kwargs = {
        "pipeline": [
            AggregateDatasets(repo_ids, repo_id),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        # For aggregation, we only need 1 task since aggregate_datasets handles everything
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": 1,  # Only need 1 task for aggregation
                "workers": 1,  # Only need 1 worker
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
        "--repo-id",
        type=str,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset, required when push-to-hub is True.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        help="Path to logs directory for `datatrove`.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="aggr_droid",
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
        default=1,  # Changed default to 1 since aggregation doesn't need multiple workers
        help="Number of slurm workers. For aggregation, this should be 1.",
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

    repo_ids = [f"{args.repo_id}_world_{DROID_SHARDS}_rank_{rank}" for rank in range(DROID_SHARDS)]
    aggregate_executor = make_aggregate_executor(repo_ids, **kwargs)
    aggregate_executor.run()


if __name__ == "__main__":
    main()

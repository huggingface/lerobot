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

"""
Aggregate video dataset shards into a single dataset.

After parallel conversion using slurm_convert_to_video.py, this script merges
all the shard datasets into one final dataset.

Example usage:
    python slurm_aggregate_video_shards.py \
        --shards-dir /fsx/jade_choghari/libero_video \
        --output-dir /fsx/jade_choghari/libero_video_final \
        --output-repo-id lerobot_video \
        --num-workers 100 \
        --partition cpu_partition \
        --cpus-per-task 16
"""

import argparse
import logging
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep


class AggregateVideoShards(PipelineStep):
    """Pipeline step that aggregates video dataset shards."""

    def __init__(
        self,
        shards_dir: str | Path,
        output_dir: str | Path,
        output_repo_id: str,
        num_shards: int,
    ):
        super().__init__()
        self.shards_dir = Path(shards_dir)
        self.output_dir = Path(output_dir)
        self.output_repo_id = output_repo_id
        self.num_shards = num_shards

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        """Aggregate all shards into a single dataset."""
        from lerobot.datasets.dataset_tools import merge_datasets
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.utils import init_logging

        init_logging()

        # Only worker 0 performs aggregation
        if rank != 0:
            logging.info(f"Worker {rank} skipping - only worker 0 performs aggregation")
            return

        logging.info(f"Starting aggregation of {self.num_shards} shards")

        # Collect all shard datasets
        shard_datasets = []
        for shard_idx in range(self.num_shards):
            shard_dir = self.shards_dir / f"shard_{shard_idx:04d}"
            if not shard_dir.exists():
                logging.warning(f"Shard directory not found: {shard_dir}")
                continue

            # Find the repo_id for this shard
            shard_repo_id = f"{self.output_repo_id}_shard_{shard_idx:04d}"
            try:
                shard_dataset = LeRobotDataset(shard_repo_id, root=shard_dir)
                shard_datasets.append(shard_dataset)
                logging.info(
                    f"Loaded shard {shard_idx}: {shard_dataset.meta.total_episodes} episodes, "
                    f"{shard_dataset.meta.total_frames} frames"
                )
            except Exception as e:
                logging.error(f"Failed to load shard {shard_idx}: {e}")
                continue

        if len(shard_datasets) == 0:
            raise ValueError(f"No valid shards found in {self.shards_dir}")

        logging.info(f"Successfully loaded {len(shard_datasets)} shards, starting merge")

        # Merge all shards
        self.output_dir.mkdir(parents=True, exist_ok=True)
        merged_dataset = merge_datasets(
            shard_datasets,
            output_repo_id=self.output_repo_id,
            output_dir=self.output_dir,
        )

        logging.info("✓ Aggregation complete!")
        logging.info(f"Merged dataset saved to: {self.output_dir}")
        logging.info(f"Total episodes: {merged_dataset.meta.total_episodes}")
        logging.info(f"Total frames: {merged_dataset.meta.total_frames}")


def make_aggregate_executor(
    shards_dir,
    output_dir,
    output_repo_id,
    num_shards,
    job_name,
    logs_dir,
    partition,
    cpus_per_task,
    mem_per_cpu,
    time_limit,
    slurm=True,
):
    """Create executor for shard aggregation."""
    kwargs = {
        "pipeline": [
            AggregateVideoShards(
                shards_dir=shards_dir,
                output_dir=output_dir,
                output_repo_id=output_repo_id,
                num_shards=num_shards,
            ),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        # Only need 1 worker for aggregation
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": 1,
                "workers": 1,
                "time": time_limit,
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
    parser = argparse.ArgumentParser(
        description="Aggregate video dataset shards into a single dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--shards-dir",
        type=Path,
        required=True,
        help="Directory containing shard_XXXX subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the aggregated dataset",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        required=True,
        help="Repository ID for the aggregated dataset",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Number of shards to aggregate (should match --workers from conversion)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        required=True,
        help="Path to logs directory for datatrove",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="aggregate_video_shards",
        help="Job name for SLURM",
    )
    parser.add_argument(
        "--slurm",
        type=int,
        default=1,
        help="Launch over SLURM (1) or locally (0)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        required=True,
        help="SLURM partition (use CPU partition)",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=16,
        help="Number of CPUs per task (aggregation can use more CPUs)",
    )
    parser.add_argument(
        "--mem-per-cpu",
        type=str,
        default="8G",
        help="Memory per CPU",
    )
    parser.add_argument(
        "--time-limit",
        type=str,
        default="08:00:00",
        help="Time limit for SLURM job",
    )

    args = parser.parse_args()

    # Convert slurm flag to boolean
    slurm = args.slurm == 1

    # Create and run executor
    executor = make_aggregate_executor(
        shards_dir=args.shards_dir,
        output_dir=args.output_dir,
        output_repo_id=args.output_repo_id,
        num_shards=args.num_shards,
        job_name=args.job_name,
        logs_dir=args.logs_dir,
        partition=args.partition,
        cpus_per_task=args.cpus_per_task,
        mem_per_cpu=args.mem_per_cpu,
        time_limit=args.time_limit,
        slurm=slurm,
    )

    logging.info("Starting shard aggregation")
    executor.run()
    logging.info("Aggregation job submitted/completed")


if __name__ == "__main__":
    main()



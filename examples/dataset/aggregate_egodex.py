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
Aggregate EgoDex shards into a single dataset.

After distributed processing creates multiple shards, this script combines
them into a single unified dataset.

Reference: https://arxiv.org/abs/2505.11709, https://github.com/apple/ml-egodex
"""

import argparse
import logging
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep


class AggregateEgoDexDatasets(PipelineStep):
    """Datatrove pipeline step for aggregating EgoDex shards."""

    def __init__(
        self,
        repo_ids: list[str],
        aggregated_repo_id: str,
        local_dir: Path | str | None = None,
        push_to_hub: bool = False,
        hf_repo_id: str | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.aggr_repo_id = aggregated_repo_id
        self.local_dir = Path(local_dir) if local_dir else None
        self.push_to_hub = push_to_hub
        self.hf_repo_id = hf_repo_id if hf_repo_id else aggregated_repo_id

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import logging

        from lerobot.datasets.aggregate import aggregate_datasets
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.utils import init_logging

        init_logging()

        # Only worker 0 performs aggregation (aggregate_datasets handles parallelism internally)
        if rank == 0:
            logging.info(f"Starting aggregation of {len(self.repo_ids)} shards into {self.aggr_repo_id}")

            # Build roots list if local_dir is specified
            roots = None
            if self.local_dir:
                roots = [self.local_dir / repo_id for repo_id in self.repo_ids]
                # Filter to only existing directories
                existing_roots = [r for r in roots if r.exists()]
                if len(existing_roots) != len(self.repo_ids):
                    logging.warning(
                        f"Only {len(existing_roots)} of {len(self.repo_ids)} shard directories found. "
                        "Missing shards will be skipped."
                    )
                # Update repo_ids to match existing roots
                existing_repo_ids = [
                    repo_id for repo_id, r in zip(self.repo_ids, roots, strict=False) if r.exists()
                ]
                roots = existing_roots
                self.repo_ids = existing_repo_ids

            if len(self.repo_ids) == 0:
                logging.error("No shard directories found. Nothing to aggregate.")
                return

            aggr_root = self.local_dir / self.aggr_repo_id if self.local_dir else None

            aggregate_datasets(
                repo_ids=self.repo_ids,
                aggr_repo_id=self.aggr_repo_id,
                roots=roots,
                aggr_root=aggr_root,
            )
            logging.info("Aggregation complete!")

            # Push to Hugging Face Hub if requested
            if self.push_to_hub:
                logging.info(f"Pushing to Hugging Face Hub as {self.hf_repo_id}...")
                dataset = LeRobotDataset(
                    repo_id=self.aggr_repo_id,
                    root=aggr_root,
                )
                # Update repo_id for pushing to different HF account if specified
                dataset.repo_id = self.hf_repo_id
                dataset.push_to_hub(
                    tags=["egodex", "hand", "dexterous", "lerobot"],
                    license="cc-by-nc-nd-4.0",
                )
                logging.info("Push to hub complete!")
        else:
            logging.info(f"Worker {rank} skipping - only worker 0 performs aggregation")


def make_aggregate_executor(
    repo_id,
    num_shards,
    job_name,
    logs_dir,
    partition,
    cpus_per_task,
    mem_per_cpu,
    local_dir,
    push_to_hub,
    hf_repo_id,
    slurm=True,
):
    """Create executor for aggregating EgoDex shards."""
    # Generate repo IDs for all shards
    repo_ids = [f"{repo_id}_world_{num_shards}_rank_{rank}" for rank in range(num_shards)]

    kwargs = {
        "pipeline": [
            AggregateEgoDexDatasets(repo_ids, repo_id, local_dir, push_to_hub, hf_repo_id),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": 1,  # Only need 1 task for aggregation
                "workers": 1,  # Only need 1 worker
                "time": "24:00:00",  # 24 hours for aggregation
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
        description="Aggregate EgoDex dataset shards into a single unified dataset."
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier (base name without shard suffix, e.g., pepijn/egodex-test)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Number of shards to aggregate (must match --workers from slurm_port_egodex.py)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Path to logs directory for datatrove",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="aggr_egodex",
        help="Job name used in SLURM",
    )
    parser.add_argument(
        "--slurm",
        type=int,
        default=1,
        help="Launch over SLURM. Use --slurm 0 to launch locally (for debugging)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="SLURM partition (ideally CPU partition)",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=16,
        help="Number of CPUs for aggregation task",
    )
    parser.add_argument(
        "--mem-per-cpu",
        type=str,
        default="8G",
        help="Memory per CPU for aggregation",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Local directory where shards are stored. If not specified, uses default HF cache.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push aggregated dataset to Hugging Face Hub after aggregation.",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Hugging Face repo ID for upload (e.g., username/dataset-name). Defaults to --repo-id.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    kwargs["slurm"] = kwargs.pop("slurm") == 1

    aggregate_executor = make_aggregate_executor(**kwargs)
    aggregate_executor.run()


if __name__ == "__main__":
    main()


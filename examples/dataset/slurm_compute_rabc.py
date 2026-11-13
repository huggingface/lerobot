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
SLURM-distributed SARM scoring for RA-BC.

Scores every frame of a dataset with SARM across SLURM workers, then merges the
workers' scores into the dataset score `sarm`, which RA-BC reads with
sample_weighting.score_name=sarm and sample_weighting.signal_name=reward.sarm.sparse_progress.

Two subcommands, each a separate SLURM submission:

  compute    – N workers, each scores a subset of episodes as the score `sarm-rank<N>`
  aggregate  – 1 worker, merges the worker scores into `sarm`, optionally pushes it

Both subcommands read and write the same dataset root, so it must be a writable
directory shared by all workers, with the dataset already downloaded.

Usage:
    python slurm_compute_rabc.py compute \\
        --repo-id user/dataset --dataset-root /shared/user/dataset \\
        --reward-model-path user/sarm_model \\
        --stride 10 --device cpu --workers 50 --partition cpu

    python slurm_compute_rabc.py aggregate \\
        --repo-id user/dataset --dataset-root /shared/user/dataset \\
        --partition cpu --push-to-hub
"""

import argparse
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep


class ComputeWorkerScores(PipelineStep):
    """Each worker scores its assigned episodes as its own dataset score."""

    def __init__(
        self, repo_id, dataset_root, reward_model_path, score_name, stride=1, head_mode="sparse", device="cpu"
    ):
        super().__init__()
        self.repo_id = repo_id
        self.dataset_root = dataset_root
        self.reward_model_path = reward_model_path
        self.score_name = score_name
        self.stride = stride
        self.head_mode = head_mode
        self.device = device

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import logging

        from lerobot.configs.rewards import RewardModelConfig
        from lerobot.datasets import LeRobotDataset
        from lerobot.rewards import make_frame_scorer, make_reward_model, score_dataset
        from lerobot.utils.utils import init_logging

        init_logging()

        dataset = LeRobotDataset(self.repo_id, root=self.dataset_root)
        my_episodes = list(range(dataset.num_episodes))[rank::world_size]
        if not my_episodes:
            logging.info(f"Rank {rank}: no episodes assigned")
            return
        logging.info(f"Rank {rank}: {len(my_episodes)} / {dataset.num_episodes} episodes")

        config = RewardModelConfig.from_pretrained(
            self.reward_model_path, cli_overrides=[f"--device={self.device}"]
        )
        scorer = make_frame_scorer(make_reward_model(config), head_mode=self.head_mode, stride=self.stride)
        score_dataset(dataset, scorer, name=f"{self.score_name}-rank{rank:05d}", episodes=my_episodes)


class MergeWorkerScores(PipelineStep):
    """Merge the workers' scores into one dataset score."""

    def __init__(self, repo_id, dataset_root, score_name, push_to_hub=False):
        super().__init__()
        self.repo_id = repo_id
        self.dataset_root = dataset_root
        self.score_name = score_name
        self.push_to_hub = push_to_hub

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import logging

        from lerobot.datasets import LeRobotDataset
        from lerobot.datasets.scores import (
            list_scores,
            merge_scores,
            push_scores_to_hub,
        )
        from lerobot.utils.utils import init_logging

        init_logging()

        if rank != 0:
            return

        dataset = LeRobotDataset(self.repo_id, root=self.dataset_root)
        worker_scores = [name for name in list_scores(dataset) if name.startswith(f"{self.score_name}-rank")]
        if not worker_scores:
            raise ValueError(f"No worker scores named {self.score_name}-rank* in {dataset.root}")
        logging.info(f"Merging {len(worker_scores)} worker scores into {self.score_name!r}")
        merge_scores(dataset, worker_scores, name=self.score_name)

        if self.push_to_hub:
            push_scores_to_hub(dataset, self.score_name)


def make_compute_executor(
    repo_id,
    dataset_root,
    reward_model_path,
    score_name,
    stride,
    head_mode,
    device,
    logs_dir,
    job_name,
    slurm,
    workers,
    partition,
    cpus_per_task,
    mem_per_cpu,
):
    kwargs = {
        "pipeline": [
            ComputeWorkerScores(
                repo_id, str(dataset_root), reward_model_path, score_name, stride, head_mode, device
            ),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": workers,
                "workers": workers,
                "time": "24:00:00",
                "partition": partition,
                "cpus_per_task": cpus_per_task,
                "sbatch_args": {"mem-per-cpu": mem_per_cpu},
            }
        )
        return SlurmPipelineExecutor(**kwargs)

    kwargs.update({"tasks": workers, "workers": 1})
    return LocalPipelineExecutor(**kwargs)


def make_aggregate_executor(
    repo_id,
    dataset_root,
    score_name,
    logs_dir,
    job_name,
    slurm,
    partition,
    cpus_per_task,
    mem_per_cpu,
    push_to_hub,
):
    kwargs = {
        "pipeline": [
            MergeWorkerScores(repo_id, str(dataset_root), score_name, push_to_hub),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": 1,
                "workers": 1,
                "time": "02:00:00",
                "partition": partition,
                "cpus_per_task": cpus_per_task,
                "sbatch_args": {"mem-per-cpu": mem_per_cpu},
            }
        )
        return SlurmPipelineExecutor(**kwargs)

    kwargs.update({"tasks": 1, "workers": 1})
    return LocalPipelineExecutor(**kwargs)


def _add_shared_args(p):
    p.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository identifier, e.g. 'user/dataset'.",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Writable dataset root shared by all workers, with the dataset already downloaded.",
    )
    p.add_argument(
        "--score-name",
        type=str,
        default="sarm",
        help="Name of the merged dataset score; workers write <score-name>-rank<N>.",
    )
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for datatrove logs.",
    )
    p.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="SLURM job name (defaults to rabc_<subcommand>).",
    )
    p.add_argument(
        "--slurm",
        type=int,
        default=1,
        help="1 = submit via SLURM; 0 = run locally (useful for debugging).",
    )
    p.add_argument(
        "--partition",
        type=str,
        default=None,
        help="SLURM partition to submit to.",
    )
    p.add_argument(
        "--cpus-per-task",
        type=int,
        default=4,
        help="Number of CPUs per SLURM task.",
    )
    p.add_argument(
        "--mem-per-cpu",
        type=str,
        default="4G",
        help="Memory per CPU, e.g. '4G' or '1950M'.",
    )


def main():
    parser = argparse.ArgumentParser(
        description="SLURM-distributed SARM scoring for RA-BC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compute subcommand
    cp = sub.add_parser(
        "compute",
        help="Distribute scoring across SLURM workers.",
    )
    _add_shared_args(cp)
    cp.add_argument(
        "--reward-model-path",
        type=str,
        required=True,
        help="Path or HF repo id of the SARM reward model.",
    )
    cp.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Compute every Nth frame; intermediate frames are interpolated (must be >= 1).",
    )
    cp.add_argument(
        "--head-mode",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "both"],
        help="Which reward head(s) to compute.",
    )
    cp.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for reward model inference, e.g. 'cpu' or 'cuda'.",
    )
    cp.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Number of parallel SLURM tasks (one worker score each).",
    )

    # aggregate subcommand
    ap = sub.add_parser(
        "aggregate",
        help="Merge the worker scores into one dataset score.",
    )
    _add_shared_args(ap)
    ap.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the merged score to the Hugging Face Hub.",
    )

    args = parser.parse_args()
    job_name = args.job_name or f"rabc_{args.command}"
    kwargs = vars(args)
    kwargs["slurm"] = kwargs.pop("slurm") == 1
    kwargs["job_name"] = job_name
    command = kwargs.pop("command")

    executor = make_compute_executor(**kwargs) if command == "compute" else make_aggregate_executor(**kwargs)

    executor.run()


if __name__ == "__main__":
    main()

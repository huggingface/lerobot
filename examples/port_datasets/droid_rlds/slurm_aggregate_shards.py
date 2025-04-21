#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import tqdm
from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

from examples.port_datasets.droid_rlds.port_droid import DROID_SHARDS
from lerobot.common.datasets.aggregate import validate_all_metadata
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    legacy_write_episode_stats,
    legacy_write_task,
    write_episode,
    write_info,
)
from lerobot.common.utils.utils import init_logging


class AggregateDatasets(PipelineStep):
    def __init__(
        self,
        repo_ids: list[str],
        aggregated_repo_id: str,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.aggr_repo_id = aggregated_repo_id

        self.create_aggr_dataset()

    def create_aggr_dataset(self):
        init_logging()

        logging.info("Start aggregate_datasets")

        all_metadata = [LeRobotDatasetMetadata(repo_id) for repo_id in self.repo_ids]

        fps, robot_type, features = validate_all_metadata(all_metadata)

        # Create resulting dataset folder
        aggr_meta = LeRobotDatasetMetadata.create(
            repo_id=self.aggr_repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
        )

        logging.info("Find all tasks")
        # find all tasks, deduplicate them, create new task indices for each dataset
        # indexed by dataset index
        datasets_task_index_to_aggr_task_index = {}
        aggr_task_index = 0
        for dataset_index, meta in enumerate(tqdm.tqdm(all_metadata, desc="Find all tasks")):
            task_index_to_aggr_task_index = {}

            for task_index, task in meta.tasks.items():
                if task not in aggr_meta.task_to_task_index:
                    # add the task to aggr tasks mappings
                    aggr_meta.tasks[aggr_task_index] = task
                    aggr_meta.task_to_task_index[task] = aggr_task_index
                    aggr_task_index += 1

                # add task_index anyway
                task_index_to_aggr_task_index[task_index] = aggr_meta.task_to_task_index[task]

            datasets_task_index_to_aggr_task_index[dataset_index] = task_index_to_aggr_task_index

        logging.info("Prepare copy data and videos")
        datasets_ep_idx_to_aggr_ep_idx = {}
        datasets_aggr_episode_index_shift = {}
        aggr_episode_index_shift = 0
        for dataset_index, meta in enumerate(tqdm.tqdm(all_metadata, desc="Prepare copy data and videos")):
            ep_idx_to_aggr_ep_idx = {}

            for episode_index in range(meta.total_episodes):
                aggr_episode_index = episode_index + aggr_episode_index_shift
                ep_idx_to_aggr_ep_idx[episode_index] = aggr_episode_index

            datasets_ep_idx_to_aggr_ep_idx[dataset_index] = ep_idx_to_aggr_ep_idx
            datasets_aggr_episode_index_shift[dataset_index] = aggr_episode_index_shift

            # populate episodes
            for episode_index, episode_dict in meta.episodes.items():
                aggr_episode_index = episode_index + aggr_episode_index_shift
                episode_dict["episode_index"] = aggr_episode_index
                aggr_meta.episodes[aggr_episode_index] = episode_dict

            # populate episodes_stats
            for episode_index, episode_stats in meta.episodes_stats.items():
                aggr_episode_index = episode_index + aggr_episode_index_shift
                aggr_meta.episodes_stats[aggr_episode_index] = episode_stats

            # populate info
            aggr_meta.info["total_episodes"] += meta.total_episodes
            aggr_meta.info["total_frames"] += meta.total_frames
            aggr_meta.info["total_videos"] += len(aggr_meta.video_keys) * meta.total_episodes

            aggr_episode_index_shift += meta.total_episodes

        logging.info("Write meta data")
        aggr_meta.info["total_tasks"] = len(aggr_meta.tasks)
        aggr_meta.info["total_chunks"] = aggr_meta.get_episode_chunk(aggr_episode_index_shift - 1)
        aggr_meta.info["splits"] = {"train": f"0:{aggr_meta.info['total_episodes']}"}

        # create a new episodes jsonl with updated episode_index using write_episode
        for episode_dict in tqdm.tqdm(aggr_meta.episodes.values(), desc="Write episodes"):
            write_episode(episode_dict, aggr_meta.root)

        # create a new episode_stats jsonl with updated episode_index using write_episode_stats
        for episode_index, episode_stats in tqdm.tqdm(
            aggr_meta.episodes_stats.items(), desc="Write episodes stats"
        ):
            legacy_write_episode_stats(episode_index, episode_stats, aggr_meta.root)

        # create a new task jsonl with updated episode_index using write_task
        for task_index, task in tqdm.tqdm(aggr_meta.tasks.items(), desc="Write tasks"):
            legacy_write_task(task_index, task, aggr_meta.root)

        write_info(aggr_meta.info, aggr_meta.root)

        self.datasets_task_index_to_aggr_task_index = datasets_task_index_to_aggr_task_index
        self.datasets_ep_idx_to_aggr_ep_idx = datasets_ep_idx_to_aggr_ep_idx
        self.datasets_aggr_episode_index_shift = datasets_aggr_episode_index_shift

        logging.info("Meta data done writing!")

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import logging
        import shutil

        import pandas as pd

        from lerobot.common.datasets.aggregate import get_update_episode_and_task_func
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.common.utils.utils import init_logging

        init_logging()

        aggr_meta = LeRobotDatasetMetadata(self.aggr_repo_id)
        all_metadata = [LeRobotDatasetMetadata(repo_id) for repo_id in self.repo_ids]

        if world_size != len(all_metadata):
            raise ValueError()

        dataset_index = rank
        meta = all_metadata[dataset_index]
        aggr_episode_index_shift = self.datasets_aggr_episode_index_shift[dataset_index]

        logging.info("Copy data")
        for episode_index in range(meta.total_episodes):
            aggr_episode_index = self.datasets_ep_idx_to_aggr_ep_idx[dataset_index][episode_index]
            data_path = meta.root / meta.get_data_file_path(episode_index)
            aggr_data_path = aggr_meta.root / aggr_meta.get_data_file_path(aggr_episode_index)

            # update episode_index and task_index
            df = pd.read_parquet(data_path)
            update_row_func = get_update_episode_and_task_func(
                aggr_episode_index_shift, self.datasets_task_index_to_aggr_task_index[dataset_index]
            )
            df = df.apply(update_row_func, axis=1)

            aggr_data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(aggr_data_path)

        logging.info("Copy videos")
        for episode_index in range(meta.total_episodes):
            aggr_episode_index = episode_index + aggr_episode_index_shift
            for vid_key in meta.video_keys:
                video_path = meta.root / meta.get_video_file_path(episode_index, vid_key)
                aggr_video_path = aggr_meta.root / aggr_meta.get_video_file_path(aggr_episode_index, vid_key)
                aggr_video_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(video_path, aggr_video_path)

                # copy_command = f"cp {video_path} {aggr_video_path} &"
                # subprocess.Popen(copy_command, shell=True)

        logging.info("Done!")


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

    repo_ids = [f"{args.repo_id}_world_{DROID_SHARDS}_rank_{rank}" for rank in range(DROID_SHARDS)]
    aggregate_executor = make_aggregate_executor(repo_ids, **kwargs)
    aggregate_executor.run()


if __name__ == "__main__":
    main()

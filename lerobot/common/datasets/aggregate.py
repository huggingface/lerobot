import logging
import shutil

import pandas as pd
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import write_episode, write_episode_stats, write_info, write_task
from lerobot.common.utils.utils import init_logging


def validate_all_metadata(all_metadata: list[LeRobotDatasetMetadata]):
    # validate same fps, robot_type, features

    fps = all_metadata[0].fps
    robot_type = all_metadata[0].robot_type
    features = all_metadata[0].features

    for meta in tqdm.tqdm(all_metadata, desc="Validate all meta data"):
        if fps != meta.fps:
            raise ValueError(f"Same fps is expected, but got fps={meta.fps} instead of {fps}.")
        if robot_type != meta.robot_type:
            raise ValueError(
                f"Same robot_type is expected, but got robot_type={meta.robot_type} instead of {robot_type}."
            )
        if features != meta.features:
            raise ValueError(
                f"Same features is expected, but got features={meta.features} instead of {features}."
            )

    return fps, robot_type, features


def get_update_episode_and_task_func(episode_index_to_add, task_index_to_global_task_index):
    def _update(row):
        row["episode_index"] = row["episode_index"] + episode_index_to_add
        row["task_index"] = task_index_to_global_task_index[row["task_index"]]
        return row

    return _update


def aggregate_datasets(repo_ids: list[str], aggr_repo_id: str, aggr_root=None):
    logging.info("Start aggregate_datasets")

    all_metadata = [LeRobotDatasetMetadata(repo_id) for repo_id in repo_ids]

    fps, robot_type, features = validate_all_metadata(all_metadata)

    # Create resulting dataset folder
    aggr_meta = LeRobotDatasetMetadata.create(
        repo_id=aggr_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=aggr_root,
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

    logging.info("Copy data and videos")
    aggr_episode_index_shift = 0
    for dataset_index, meta in enumerate(tqdm.tqdm(all_metadata, desc="Copy data and videos")):
        # cp data
        for episode_index in range(meta.total_episodes):
            aggr_episode_index = episode_index + aggr_episode_index_shift
            data_path = meta.root / meta.get_data_file_path(episode_index)
            aggr_data_path = aggr_meta.root / aggr_meta.get_data_file_path(aggr_episode_index)

            # update episode_index and task_index
            df = pd.read_parquet(data_path)
            update_row_func = get_update_episode_and_task_func(
                aggr_episode_index_shift, datasets_task_index_to_aggr_task_index[dataset_index]
            )
            df = df.apply(update_row_func, axis=1)

            aggr_data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(aggr_data_path)

        # cp videos
        for episode_index in range(meta.total_episodes):
            aggr_episode_index = episode_index + aggr_episode_index_shift
            for vid_key in meta.video_keys:
                video_path = meta.root / meta.get_video_file_path(episode_index, vid_key)
                aggr_video_path = aggr_meta.root / aggr_meta.get_video_file_path(aggr_episode_index, vid_key)
                aggr_video_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(video_path, aggr_video_path)

                # copy_command = f"cp {video_path} {aggr_video_path} &"
                # subprocess.Popen(copy_command, shell=True)

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

    logging.info("write meta data")

    aggr_meta.info["total_chunks"] = aggr_meta.get_episode_chunk(aggr_episode_index_shift - 1)
    aggr_meta.info["splits"] = {"train": f"0:{aggr_meta.info['total_episodes']}"}

    # create a new episodes jsonl with updated episode_index using write_episode
    for episode_dict in aggr_meta.episodes.values():
        write_episode(episode_dict, aggr_meta.root)

    # create a new episode_stats jsonl with updated episode_index using write_episode_stats
    for episode_index, episode_stats in aggr_meta.episodes_stats.items():
        write_episode_stats(episode_index, episode_stats, aggr_meta.root)

    # create a new task jsonl with updated episode_index using write_task
    for task_index, task in aggr_meta.tasks.items():
        write_task(task_index, task, aggr_meta.root)

    write_info(aggr_meta.info, aggr_meta.root)


if __name__ == "__main__":
    init_logging()
    repo_id = "cadene/droid"
    aggr_repo_id = "cadene/droid"
    datetime = "2025-02-22_11-23-54"

    # root = Path(f"/tmp/{repo_id}")
    # if root.exists():
    #     shutil.rmtree(root)
    root = None

    # all_metadata = [LeRobotDatasetMetadata(f"{repo_id}_{datetime}_world_2048_rank_{rank}") for rank in range(2048)]

    # aggregate_datasets(
    #     all_metadata,
    #     aggr_repo_id,
    #     root=root,
    # )

    aggr_dataset = LeRobotDataset(
        repo_id=aggr_repo_id,
        root=root,
    )
    aggr_dataset.push_to_hub(tags=["openx"])

    # for meta in all_metadata:
    #     dataset = LeRobotDataset(repo_id=meta.repo_id, root=meta.root)
    #     dataset.push_to_hub(tags=["openx"])

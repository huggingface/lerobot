import logging
import shutil
from pathlib import Path

import pandas as pd
import tqdm

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_PATH,
    write_info,
    write_stats,
    write_tasks,
)
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


def get_update_episode_and_task_func(episode_index_to_add, old_tasks, new_tasks):
    def _update(row):
        row["episode_index"] = row["episode_index"] + episode_index_to_add
        task = old_tasks.iloc[row["task_index"]].name
        row["task_index"] = new_tasks.loc[task].task_index.item()
        return row

    return _update


def get_update_meta_func(
    meta_chunk_index_to_add,
    meta_file_index_to_add,
    data_chunk_index_to_add,
    data_file_index_to_add,
    videos_chunk_index_to_add,
    videos_file_index_to_add,
    frame_index_to_add,
):
    def _update(row):
        row["meta/episodes/chunk_index"] = row["meta/episodes/chunk_index"] + meta_chunk_index_to_add
        row["meta/episodes/file_index"] = row["meta/episodes/file_index"] + meta_file_index_to_add
        row["data/chunk_index"] = row["data/chunk_index"] + data_chunk_index_to_add
        row["data/file_index"] = row["data/file_index"] + data_file_index_to_add
        for key in videos_chunk_index_to_add:
            row[f"videos/{key}/chunk_index"] = (
                row[f"videos/{key}/chunk_index"] + videos_chunk_index_to_add[key]
            )
            row[f"videos/{key}/file_index"] = row[f"videos/{key}/file_index"] + videos_file_index_to_add[key]
        row["dataset_from_index"] = row["dataset_from_index"] + frame_index_to_add
        row["dataset_to_index"] = row["dataset_to_index"] + frame_index_to_add
        return row

    return _update


def aggregate_datasets(repo_ids: list[str], aggr_repo_id: str, roots: list[Path] = None, aggr_root=None):
    logging.info("Start aggregate_datasets")

    if roots is None:
        all_metadata = [LeRobotDatasetMetadata(repo_id) for repo_id in repo_ids]
    else:
        all_metadata = [
            LeRobotDatasetMetadata(repo_id, root=root) for repo_id, root in zip(repo_ids, roots, strict=False)
        ]

    fps, robot_type, features = validate_all_metadata(all_metadata)
    video_keys = [key for key in features if features[key]["dtype"] == "video"]

    # Create resulting dataset folder
    aggr_meta = LeRobotDatasetMetadata.create(
        repo_id=aggr_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=aggr_root,
    )
    aggr_root = aggr_meta.root

    logging.info("Find all tasks")
    unique_tasks = pd.concat([meta.tasks for meta in all_metadata]).index.unique()
    aggr_meta.tasks = pd.DataFrame({"task_index": range(len(unique_tasks))}, index=unique_tasks)

    num_episodes = 0
    num_frames = 0

    aggr_meta_chunk_idx = 0
    aggr_meta_file_idx = 0

    aggr_data_chunk_idx = 0
    aggr_data_file_idx = 0

    aggr_videos_chunk_idx = dict.fromkeys(video_keys, 0)
    aggr_videos_file_idx = dict.fromkeys(video_keys, 0)

    for meta in tqdm.tqdm(all_metadata, desc="Copy data and videos"):
        meta_chunk_file_ids = {
            (c, f)
            for c, f in zip(
                meta.episodes["meta/episodes/chunk_index"],
                meta.episodes["meta/episodes/file_index"],
                strict=False,
            )
        }
        for chunk_idx, file_idx in meta_chunk_file_ids:
            path = meta.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
            df = pd.read_parquet(path)
            update_meta_func = get_update_meta_func(
                aggr_meta_chunk_idx,
                aggr_meta_file_idx,
                aggr_data_chunk_idx,
                aggr_data_file_idx,
                aggr_videos_chunk_idx,
                aggr_videos_file_idx,
                num_frames,
            )
            df = df.apply(update_meta_func, axis=1)

            aggr_path = aggr_root / DEFAULT_EPISODES_PATH.format(
                chunk_index=aggr_meta_chunk_idx, file_index=aggr_meta_file_idx
            )
            aggr_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(aggr_path)

            aggr_meta_file_idx += 1
            if aggr_meta_file_idx >= DEFAULT_CHUNK_SIZE:
                aggr_meta_file_idx = 0
                aggr_meta_chunk_idx += 1

        # cp videos
        for key in video_keys:
            video_chunk_file_ids = {
                (c, f)
                for c, f in zip(
                    meta.episodes[f"videos/{key}/chunk_index"],
                    meta.episodes[f"videos/{key}/file_index"],
                    strict=False,
                )
            }
            for chunk_idx, file_idx in video_chunk_file_ids:
                path = meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=key, chunk_index=chunk_idx, file_index=file_idx
                )
                aggr_path = aggr_root / DEFAULT_VIDEO_PATH.format(
                    video_key=key,
                    chunk_index=aggr_videos_chunk_idx[key],
                    file_index=aggr_videos_file_idx[key],
                )
                aggr_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(path), str(aggr_path))

                # copy_command = f"cp {video_path} {aggr_video_path} &"
                # subprocess.Popen(copy_command, shell=True)

                aggr_videos_file_idx[key] += 1
                if aggr_videos_file_idx[key] >= DEFAULT_CHUNK_SIZE:
                    aggr_videos_file_idx[key] = 0
                    aggr_videos_chunk_idx[key] += 1

        data_chunk_file_ids = {
            (c, f)
            for c, f in zip(meta.episodes["data/chunk_index"], meta.episodes["data/file_index"], strict=False)
        }
        for chunk_idx, file_idx in data_chunk_file_ids:
            path = meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
            df = pd.read_parquet(path)
            # TODO(rcadene): update frame index
            update_data_func = get_update_episode_and_task_func(num_episodes, meta.tasks, aggr_meta.tasks)
            df = df.apply(update_data_func, axis=1)

            aggr_path = aggr_root / DEFAULT_DATA_PATH.format(
                chunk_index=aggr_data_chunk_idx, file_index=aggr_data_file_idx
            )
            aggr_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(aggr_path)

            aggr_data_file_idx += 1
            if aggr_data_file_idx >= DEFAULT_CHUNK_SIZE:
                aggr_data_file_idx = 0
                aggr_data_chunk_idx += 1

        num_episodes += meta.total_episodes
        num_frames += meta.total_frames

    logging.info("write tasks")
    write_tasks(aggr_meta.tasks, aggr_meta.root)

    logging.info("write info")
    aggr_meta.info["total_episodes"] = sum([meta.total_episodes for meta in all_metadata])
    aggr_meta.info["total_frames"] = sum([meta.total_frames for meta in all_metadata])
    aggr_meta.info["splits"] = {"train": f"0:{aggr_meta.total_episodes}"}
    write_info(aggr_meta.info, aggr_meta.root)

    logging.info("write stats")
    aggr_meta.stats = aggregate_stats([meta.stats for meta in all_metadata])
    write_stats(aggr_meta.stats, aggr_meta.root)


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

import logging
import shutil
from pathlib import Path

import pandas as pd
import tqdm

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    concat_video_files,
    get_parquet_file_size_in_mb,
    get_video_size_in_mb,
    update_chunk_file_indices,
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


def update_data_df(df, src_meta, dst_meta):
    def _update(row):
        row["episode_index"] = row["episode_index"] + dst_meta["total_episodes"]
        row["index"] = row["index"] + dst_meta["total_frames"]
        task = src_meta.tasks.iloc[row["task_index"]].name
        row["task_index"] = dst_meta.tasks.loc[task].task_index.item()
        return row

    return df.apply(_update, axis=1)


def update_meta_data(
    df,
    dst_meta,
    meta_idx,
    data_idx,
    videos_idx,
):
    def _update(row):
        row["meta/episodes/chunk_index"] = row["meta/episodes/chunk_index"] + meta_idx["chunk_index"]
        row["meta/episodes/file_index"] = row["meta/episodes/file_index"] + meta_idx["file_index"]
        row["data/chunk_index"] = row["data/chunk_index"] + data_idx["chunk_index"]
        row["data/file_index"] = row["data/file_index"] + data_idx["file_index"]
        for key, video_idx in videos_idx.items():
            row[f"videos/{key}/chunk_index"] = row[f"videos/{key}/chunk_index"] + video_idx["chunk_index"]
            row[f"videos/{key}/file_index"] = row[f"videos/{key}/file_index"] + video_idx["file_index"]
            row[f"videos/{key}/from_timestamp"] = (
                row[f"videos/{key}/from_timestamp"] + video_idx["latest_duration"]
            )
            row[f"videos/{key}/to_timestamp"] = (
                row[f"videos/{key}/to_timestamp"] + video_idx["latest_duration"]
            )
        row["dataset_from_index"] = row["dataset_from_index"] + dst_meta.info["total_frames"]
        row["dataset_to_index"] = row["dataset_to_index"] + dst_meta.info["total_frames"]
        row["episode_index"] = row["episode_index"] + dst_meta.info["total_episodes"]
        return row

    return df.apply(_update, axis=1)


def aggregate_datasets(repo_ids: list[str], aggr_repo_id: str, roots: list[Path] = None, aggr_root=None):
    logging.info("Start aggregate_datasets")

    # Load metadata
    all_metadata = (
        [LeRobotDatasetMetadata(repo_id) for repo_id in repo_ids]
        if roots is None
        else [
            LeRobotDatasetMetadata(repo_id, root=root) for repo_id, root in zip(repo_ids, roots, strict=False)
        ]
    )
    fps, robot_type, features = validate_all_metadata(all_metadata)
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]

    # Initialize output dataset metadata
    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=aggr_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=aggr_root,
    )

    # Aggregate task info
    logging.info("Find all tasks")
    unique_tasks = pd.concat([m.tasks for m in all_metadata]).index.unique()
    dst_meta.tasks = pd.DataFrame({"task_index": range(len(unique_tasks))}, index=unique_tasks)

    # Track counters and indices
    meta_idx = {"chunk": 0, "file": 0}
    data_idx = {"chunk": 0, "file": 0}
    videos_idx = {
        key: {"chunk": 0, "file": 0, "latest_duration": 0, "episode_duration": 0} for key in video_keys
    }

    # Process each dataset
    for src_meta in tqdm.tqdm(all_metadata, desc="Copy data and videos"):
        videos_idx = aggregate_videos(src_meta, dst_meta, videos_idx)
        data_idx = aggregate_data(src_meta, dst_meta, data_idx)
        meta_idx = aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx, video_keys)

        dst_meta.info["total_episodes"] += src_meta.total_episodes
        dst_meta.info["total_frames"] += src_meta.total_frames

    finalize_aggregation(dst_meta, all_metadata)
    logging.info("Aggregation complete.")


# -------------------------------
# Helper Functions
# -------------------------------


def aggregate_videos(src_meta, dst_meta, videos_idx):
    """
    Aggregates video chunks from a dataset into the aggregated dataset folder.
    """
    for key, video_idx in videos_idx.items():
        # Get unique (chunk, file) combinations
        unique_chunk_file_pairs = {
            (chunk, file)
            for chunk, file in zip(
                src_meta.episodes[f"videos/{key}/chunk_index"],
                src_meta.episodes[f"videos/{key}/file_index"],
                strict=False,
            )
        }

        # Current target chunk/file index
        chunk_idx = video_idx["chunk_idx"]
        file_idx = video_idx["file_idx"]

        for src_chunk_idx, src_file_idx in unique_chunk_file_pairs:
            src_path = src_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=src_chunk_idx,
                file_index=src_file_idx,
            )

            dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )

            if not dst_path.exists():
                # First write to this destination file
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
                continue

            # Check file sizes before appending
            src_size = get_video_size_in_mb(src_path)
            dst_size = get_video_size_in_mb(dst_path)

            if dst_size + src_size >= DEFAULT_VIDEO_FILE_SIZE_IN_MB:
                # Rotate to a new chunk/file
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
                dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=key,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
            else:
                # Append to existing video file
                concat_video_files(
                    [dst_path, src_path],
                    dst_meta.root,
                    key,
                    chunk_idx,
                    file_idx,
                )

        # Update the video index tracking
        video_idx["chunk_idx"] = chunk_idx
        video_idx["file_idx"] = file_idx

        return videos_idx


def aggregate_data(src_meta, dst_meta, data_idx):
    unique_chunk_file_ids = {
        (c, f)
        for c, f in zip(
            src_meta.episodes["data/chunk_index"], src_meta.episodes["data/file_index"], strict=False
        )
    }
    for src_chunk_idx, src_file_idx in unique_chunk_file_ids:
        src_path = src_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=src_chunk_idx, file_index=src_file_idx
        )
        df = pd.read_parquet(src_path)
        df = update_data_df(df, src_meta, dst_meta)

        dst_path = aggr_root / DEFAULT_DATA_PATH.format(
            chunk_index=data_idx["chunk"], file_index=data_idx["file"]
        )
        data_idx = write_parquet_safely(
            df,
            src_path,
            dst_path,
            data_idx,
            DEFAULT_DATA_FILE_SIZE_IN_MB,
            DEFAULT_CHUNK_SIZE,
            DEFAULT_DATA_PATH,
        )

    return data_idx


def aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx):
    chunk_file_ids = {
        (c, f)
        for c, f in zip(
            src_meta.episodes["meta/episodes/chunk_index"],
            src_meta.episodes["meta/episodes/file_index"],
            strict=False,
        )
    }

    for chunk_idx, file_idx in chunk_file_ids:
        src_path = src_meta.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        df = pd.read_parquet(src_path)
        df = update_meta_data(
            df,
            dst_meta,
            meta_idx,
            data_idx,
            videos_idx,
        )

        # for k in video_keys:
        #     video_idx[k]["latest_duration"] += video_idx[k]["episode_duration"]

        dst_path = dst_meta.root / DEFAULT_EPISODES_PATH.format(
            chunk_index=meta_idx["chunk"], file_index=meta_idx["file"]
        )
        write_parquet_safely(
            df,
            src_path,
            dst_path,
            meta_idx,
            DEFAULT_DATA_FILE_SIZE_IN_MB,
            DEFAULT_CHUNK_SIZE,
            DEFAULT_EPISODES_PATH,
        )

    return meta_idx


def write_parquet_safely(
    df: pd.DataFrame,
    src_path: Path,
    dst_path: Path,
    idx: dict[str, int],
    max_mb: float,
    chunk_size: int,
    default_path: str,
):
    """
    Safely appends or creates a Parquet file at dst_path based on size constraints.

    Parameters:
        df (pd.DataFrame): Data to write.
        src_path (Path): Path to source file (used to get size).
        dst_path (Path): Target path for writing.
        idx (dict): Dictionary containing 'chunk' and 'file' indices.
        max_mb (float): Maximum allowed file size in MB.
        chunk_size (int): Maximum number of files per chunk.
        default_path (str): Format string for generating a new file path.

    Returns:
        dict: Updated index dictionary.
    """

    # If destination file doesn't exist, just write the new one
    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_path)
        return idx

    # Otherwise, check if we exceed the size limit
    src_size = get_parquet_file_size_in_mb(src_path)
    dst_size = get_parquet_file_size_in_mb(dst_path)

    if dst_size + src_size >= max_mb:
        # File is too large, move to a new one
        idx["chunk"], idx["file"] = update_chunk_file_indices(idx["chunk"], idx["file"], chunk_size)
        new_path = dst_path.parent / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])
        new_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(new_path)
    else:
        # Append to existing file
        existing_df = pd.read_parquet(dst_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_parquet(dst_path)

    return idx


def finalize_aggregation(aggr_meta, all_metadata):
    logging.info("write tasks")
    write_tasks(aggr_meta.tasks, aggr_meta.root)

    logging.info("write info")
    aggr_meta.info.update(
        {
            "total_tasks": len(aggr_meta.tasks),
            "total_episodes": sum(m.total_episodes for m in all_metadata),
            "total_frames": sum(m.total_frames for m in all_metadata),
            "splits": {"train": f"0:{sum(m.total_episodes for m in all_metadata)}"},
        }
    )
    write_info(aggr_meta.info, aggr_meta.root)

    logging.info("write stats")
    aggr_meta.stats = aggregate_stats([m.stats for m in all_metadata])
    write_stats(aggr_meta.stats, aggr_meta.root)


if __name__ == "__main__":
    init_logging()

    num_shards = 2048
    repo_id = "cadene/droid_1.0.1_v30"
    aggr_repo_id = f"{repo_id}_compact_6"
    tags = ["openx"]

    # num_shards = 210
    # repo_id = "cadene/agibot_alpha_v30"
    # aggr_repo_id = f"{repo_id}"
    # tags = None

    # aggr_root = Path(f"/tmp/{aggr_repo_id}")
    aggr_root = HF_LEROBOT_HOME / aggr_repo_id
    if aggr_root.exists():
        shutil.rmtree(aggr_root)

    repo_ids = []
    roots = []
    for rank in range(num_shards):
        shard_repo_id = f"{repo_id}_world_{num_shards}_rank_{rank}"
        shard_root = HF_LEROBOT_HOME / shard_repo_id
        try:
            meta = LeRobotDatasetMetadata(shard_repo_id, root=shard_root)
            if len(meta.video_keys) == 0:
                continue
            repo_ids.append(shard_repo_id)
            roots.append(shard_root)
        except:
            pass

        if rank == 1:
            break

    aggregate_datasets(
        repo_ids,
        aggr_repo_id,
        roots=roots,
        aggr_root=aggr_root,
    )

    aggr_dataset = LeRobotDataset(repo_id=aggr_repo_id, root=aggr_root)
    # for i in tqdm.tqdm(range(len(aggr_dataset))):
    #     aggr_dataset[i]
    #     pass
    aggr_dataset.push_to_hub(tags=tags, upload_large_folder=True)

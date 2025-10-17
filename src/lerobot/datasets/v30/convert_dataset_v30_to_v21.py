"""Utilities to convert a LeRobot dataset from codebase version v3.0 back to v2.1.

The script mirrors :mod:`lerobot.datasets.v21.convert_dataset_v21_to_v30` but applies the reverse
transformations so an existing dataset created with the new consolidated file
layout can be ported back to the legacy per-episode structure.

Usage examples
--------------

Convert a dataset that already exists locally::

    python src/lerobot/datasets/v30/convert_dataset_v30_to_v21.py \
        --repo-id=lerobot/pusht \
        --root=/path/to/datasets

"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import jsonlines
import numpy as np
import pyarrow.parquet as pq
import tqdm
from huggingface_hub import snapshot_download

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_PATH,
    EPISODES_DIR,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    load_info,
    load_tasks,
    serialize_dict,
    unflatten_dict,
    write_info,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging

V21 = "v2.1"
V30 = "v3.0"

LEGACY_DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH_TEMPLATE = "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4"
MIN_VIDEO_DURATION = 1e-6

def _to_serializable(value: Any) -> Any:
    """Convert numpy/pyarrow values into standard Python types for JSON dumps."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    return value


def validate_local_dataset_version(local_path: Path) -> None:
    info = load_info(local_path)
    dataset_version = info.get("codebase_version", "unknown")
    if dataset_version != V30:
        raise ValueError(
            f"Local dataset has codebase version '{dataset_version}', expected '{V30}'. "
            f"This script converts datasets from v3.0 back to v2.1."
        )


def load_episode_records(root: Path) -> list[dict[str, Any]]:
    """Load the consolidated metadata rows stored in ``meta/episodes``."""

    episodes_dir = root / EPISODES_DIR
    pq_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not pq_paths:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}.")

    records: list[dict[str, Any]] = []
    for pq_path in pq_paths:
        table = pq.read_table(pq_path)
        records.extend(table.to_pylist())

    records.sort(key=lambda rec: int(rec["episode_index"]))
    return records


def convert_tasks(root: Path, new_root: Path) -> None:
    logging.info("Converting tasks parquet to legacy JSONL")
    tasks = load_tasks(root)
    tasks = tasks.sort_values("task_index")

    out_path = new_root / LEGACY_TASKS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(out_path, mode="w") as writer:
        for task, row in tasks.iterrows():
            writer.write({
                "task_index": int(row["task_index"]),
                "task": _to_serializable(task),
            })


def convert_info(
    root: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
) -> None:
    info = load_info(root)
    logging.info("Converting info.json metadata to v2.1 schema")

    total_episodes = info.get("total_episodes") or len(episode_records)
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)

    info["codebase_version"] = V21

    # Restore legacy layout templates.
    info["data_path"] = LEGACY_DATA_PATH_TEMPLATE
    if info.get("video_path") is not None and len(video_keys) > 0:
        info["video_path"] = LEGACY_VIDEO_PATH_TEMPLATE
    else:
        info["video_path"] = None

    # Remove v3-specific sizing hints which do not exist in v2.1.
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)

    # Restore per-feature metadata: camera entries already contain their own fps.
    for key, ft in info["features"].items():
        if ft.get("dtype") != "video":
            ft.pop("fps", None)

    info["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0
    info["total_videos"] = total_episodes * len(video_keys)

    write_info(info, new_root)


def _group_episodes_by_data_file(
    episode_records: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in episode_records:
        key = (
            int(record["data/chunk_index"]),
            int(record["data/file_index"]),
        )
        grouped[key].append(record)
    return grouped


def convert_data(root: Path, new_root: Path, episode_records: list[dict[str, Any]]) -> None:
    logging.info("Converting consolidated parquet files back to per-episode files")
    grouped = _group_episodes_by_data_file(episode_records)

    for (chunk_idx, file_idx), records in tqdm.tqdm(grouped.items(), desc="convert data files"):
        source_path = root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        if not source_path.exists():
            raise FileNotFoundError(f"Expected source parquet file not found: {source_path}")

        table = pq.read_table(source_path)
        records = sorted(records, key=lambda rec: int(rec["dataset_from_index"]))
        file_offset = int(records[0]["dataset_from_index"])

        for record in records:
            episode_index = int(record["episode_index"])
            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            length = stop - start

            if length <= 0:
                raise ValueError(
                    "Invalid episode length computed during data conversion: "
                    f"episode_index={episode_index}, length={length}"
                )

            episode_table = table.slice(start, length)

            dest_chunk = episode_index // DEFAULT_CHUNK_SIZE
            dest_path = new_root / LEGACY_DATA_PATH_TEMPLATE.format(
                chunk_index=dest_chunk,
                episode_index=episode_index,
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(episode_table, dest_path)


def _group_episodes_by_video_file(
    episode_records: Iterable[dict[str, Any]],
    video_key: str,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    chunk_column = f"videos/{video_key}/chunk_index"
    file_column = f"videos/{video_key}/file_index"

    for record in episode_records:
        if chunk_column not in record or file_column not in record:
            continue
        chunk_idx = record.get(chunk_column)
        file_idx = record.get(file_column)
        if chunk_idx is None or file_idx is None:
            continue
        grouped[(int(chunk_idx), int(file_idx))].append(record)
    return grouped


def _validate_video_paths(src: Path, dst: Path) -> None:
    """Validate source and destination paths to prevent security issues."""
    
    # Convert to Path objects if they aren't already
    src = Path(src)
    dst = Path(dst)
    
    # Resolve paths to handle symlinks and normalize them
    try:
        src_resolved = src.resolve()
        dst_resolved = dst.resolve()
    except OSError as exc:
        raise ValueError(f"Invalid path provided: {exc}") from exc
    
    # Check that source file exists and is a regular file
    if not src_resolved.exists():
        raise FileNotFoundError(f"Source video file does not exist: {src_resolved}")
    
    if not src_resolved.is_file():
        raise ValueError(f"Source path is not a regular file: {src_resolved}")
    
    # Validate file extensions for video files
    valid_video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    if src_resolved.suffix.lower() not in valid_video_extensions:
        raise ValueError(f"Source file does not have a valid video extension: {src_resolved}")
    
    if dst_resolved.suffix.lower() not in valid_video_extensions:
        raise ValueError(f"Destination file does not have a valid video extension: {dst_resolved}")
    
    # Check for path traversal attempts in the original paths
    src_str = str(src)
    dst_str = str(dst)
    
    # Ensure paths don't contain null bytes or other control characters
    for path_str, name in [(src_str, "source"), (dst_str, "destination")]:
        if "\0" in path_str:
            raise ValueError(f"Path contains null bytes: {name} path")
        if any(ord(c) < 32 and c not in ["\t", "\n", "\r"] for c in path_str):
            raise ValueError(f"Path contains invalid control characters: {name} path")
    
    # Additional check: ensure resolved paths don't point to system directories
    system_dirs = {"/etc", "/sys", "/proc", "/dev", "/boot", "/root"}
    for resolved_path, name in [(src_resolved, "source"), (dst_resolved, "destination")]:
        path_str = str(resolved_path)
        for sys_dir in system_dirs:
            if path_str.startswith(sys_dir + "/") or path_str == sys_dir:
                raise ValueError(f"Path points to system directory: {name} path {resolved_path}")
    
    # Ensure the destination directory can be created safely
    try:
        dst_parent = dst_resolved.parent
        if not dst_parent.exists():
            # Check if we can create the parent directory structure
            dst_parent.resolve()
    except OSError as exc:
        raise ValueError(f"Cannot create destination directory: {exc}") from exc


def _extract_video_segment(
    src: Path,
    dst: Path,
    start: float,
    end: float,
) -> None:
    # Validate paths to prevent security issues
    _validate_video_paths(src, dst)
    
    # Validate numeric parameters to prevent injection
    if not (0 <= start <= 86400):  # 24 hours max
        raise ValueError(f"Invalid start time: {start}")
    if not (0 <= end <= 86400):  # 24 hours max
        raise ValueError(f"Invalid end time: {end}")
    if start >= end:
        raise ValueError(f"Start time {start} must be less than end time {end}")
    
    duration = max(end - start, MIN_VIDEO_DURATION)
    
    # Validate duration is reasonable
    if duration > 3600:  # 1 hour max
        raise ValueError(f"Video segment duration too long: {duration} seconds")
    
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Build command with validated parameters
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        "-y",
        str(dst),
    ]

    try:
        # Use more secure subprocess call with explicit timeout
        result = subprocess.run(
            cmd, 
            check=True, 
            timeout=300,  # 5 minute timeout
            capture_output=True, 
            text=True
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out while processing video '{src}' -> '{dst}'") from exc
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable not found; it is required for video conversion") from exc
    except subprocess.CalledProcessError as exc:
        error_msg = f"ffmpeg failed while splitting video '{src}' into '{dst}'"
        if exc.stderr:
            error_msg += f". Error: {exc.stderr.strip()}"
        raise RuntimeError(error_msg) from exc


def convert_videos(root: Path, new_root: Path, episode_records: list[dict[str, Any]], video_keys: list[str]) -> None:
    if len(video_keys) == 0:
        logging.info("No video features detected; skipping video conversion")
        return

    logging.info("Converting concatenated MP4 files back to per-episode videos")

    for video_key in video_keys:
        grouped = _group_episodes_by_video_file(episode_records, video_key)
        if len(grouped) == 0:
            logging.info("No video metadata found for key '%s'; skipping", video_key)
            continue

        for (chunk_idx, file_idx), records in tqdm.tqdm(
            grouped.items(), desc=f"convert videos ({video_key})"
        ):
            src_path = root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            if not src_path.exists():
                raise FileNotFoundError(f"Expected MP4 file not found: {src_path}")

            records = sorted(records, key=lambda rec: float(rec[f"videos/{video_key}/from_timestamp"]))

            for record in records:
                episode_index = int(record["episode_index"])
                start = float(record[f"videos/{video_key}/from_timestamp"])
                end = float(record[f"videos/{video_key}/to_timestamp"])

                dest_chunk = episode_index // DEFAULT_CHUNK_SIZE
                dest_path = new_root / LEGACY_VIDEO_PATH_TEMPLATE.format(
                    chunk_index=dest_chunk,
                    video_key=video_key,
                    episode_index=episode_index,
                )

                _extract_video_segment(src_path, dest_path, start=start, end=end)


def convert_episodes_metadata(new_root: Path, episode_records: list[dict[str, Any]]) -> None:
    logging.info("Reconstructing legacy episodes and episodes_stats JSONL files")

    episodes_path = new_root / LEGACY_EPISODES_PATH
    stats_path = new_root / LEGACY_EPISODES_STATS_PATH
    episodes_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(episodes_path, mode="w") as episodes_writer, jsonlines.open(
        stats_path, mode="w"
    ) as stats_writer:
        for record in sorted(episode_records, key=lambda rec: int(rec["episode_index"])):
            legacy_episode = {
                key: value
                for key, value in record.items()
                if not key.startswith("data/")
                and not key.startswith("videos/")
                and not key.startswith("stats/")
                and not key.startswith("meta/")
                and key not in {"dataset_from_index", "dataset_to_index"}
            }

            serializable_episode = {key: _to_serializable(value) for key, value in legacy_episode.items()}
            episodes_writer.write(serializable_episode)

            stats_flat = {key: record[key] for key in record if key.startswith("stats/")}
            stats_nested = unflatten_dict(stats_flat).get("stats", {})
            stats_serialized = serialize_dict(stats_nested)
            stats_writer.write(
                {
                    "episode_index": int(record["episode_index"]),
                    "stats": stats_serialized,
                }
            )


def copy_global_stats(root: Path, new_root: Path) -> None:
    source_stats = root / "meta" / "stats.json"
    if source_stats.exists():
        target_stats = new_root / "meta" / "stats.json"
        target_stats.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_stats, target_stats)


def copy_ancillary_directories(root: Path, new_root: Path) -> None:
    for subdir in ["images"]:
        source = root / subdir
        if source.exists():
            shutil.copytree(source, new_root / subdir, dirs_exist_ok=True)


def convert_dataset(
    repo_id: str,
    root: str | Path | None = None,
    force_conversion: bool = False,
) -> None:
    root = HF_LEROBOT_HOME / repo_id if root is None else Path(root) / repo_id

    if root.exists() and force_conversion:
        logging.info("--force-conversion enabled: removing existing snapshot at %s", root)
        shutil.rmtree(root)

    if root.exists():
        validate_local_dataset_version(root)
        logging.info("Using existing local dataset at %s", root)
    else:
        logging.info("Downloading dataset snapshot from the Hub")
        snapshot_download(repo_id, repo_type="dataset", local_dir=root)

    episode_records = load_episode_records(root)
    video_keys = [
        key
        for key, ft in load_info(root)["features"].items()
        if ft.get("dtype") == "video"
    ]

    backup_root = root.parent / f"{root.name}_{V30}"
    new_root = root.parent / f"{root.name}_{V21}"

    if backup_root.is_dir():
        shutil.rmtree(backup_root)
    if new_root.is_dir():
        shutil.rmtree(new_root)

    new_root.mkdir(parents=True, exist_ok=True)

    convert_info(root, new_root, episode_records, video_keys)
    copy_global_stats(root, new_root)
    convert_tasks(root, new_root)
    convert_data(root, new_root, episode_records)
    convert_videos(root, new_root, episode_records, video_keys)
    convert_episodes_metadata(new_root, episode_records)
    copy_ancillary_directories(root, new_root)

    shutil.move(str(root), str(backup_root))
    shutil.move(str(new_root), str(root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local directory under which the dataset should be stored.",
    )
    parser.add_argument(
        "--force-conversion",
        action="store_true",
        help="Ignore any existing local snapshot and re-download it from the Hub.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    init_logging()
    args = parse_args()
    convert_dataset(**vars(args))

#!/usr/bin/env python3
# Copyright 2024-2025 The HuggingFace Inc. team and contributors.
# Licensed under the Apache-2.0 license.

"""
merge_dataset.py – merge two lerobot datasets while keeping episode
indices unique, concatenating all meta files, and updating
indices within Parquet files.

Example
-------
python merge_dataset.py \
    --dataset1=/data/robot/run_A \
    --dataset2=/data/robot/run_B \
    --output_dir=/data/robot/merged
"""

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Union

import draccus
import pandas as pd


# ─────────────────────────────────── Config ────────────────────────────────── #
@dataclass
class MergeConfig:
    # required
    dataset1: Path
    dataset2: Path
    output_dir: Path

    # optional
    chunk_name: str = "chunk-000"
    copy_parquet: bool = True  # If True, Parquet files are copied and modified.
    copy_videos: bool = True


# ─────────────────────────────────── Helpers ───────────────────────────────── #
NUM_KEYS = [
    "total_episodes",
    "total_frames",
    "total_videos",
]


_NUM_RE = re.compile(r"(\d+)(?=\.parquet$|\.mp4$)")


def _extract_idx(path: Path) -> int:
    m = _NUM_RE.search(path.name)
    if not m:
        raise ValueError(f"Impossible de trouver un index numérique dans {path}")
    return int(m.group(1))


def _natural_sort(paths: Iterable[Path]) -> List[Path]:
    return sorted(paths, key=_extract_idx)


JsonVal = Union[int, float, str, bool, None, List["JsonVal"], Dict[str, "JsonVal"]]


def _shift_any(obj: JsonVal, offset: int) -> JsonVal:
    if isinstance(obj, int):
        return obj + offset
    if isinstance(obj, list):
        return [_shift_any(x, offset) for x in obj]
    if isinstance(obj, dict):
        return {k: _shift_any(v, offset) for k, v in obj.items()}
    return obj


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_parquet_and_update_indices(
    src_root: Path,
    dst_data_dir: Path,
    chunk_name: str,
    episode_idx_offset: int,
    frame_idx_offset: int,
) -> int:
    """
    Copies Parquet files from source to destination, updating internal indices.
    - Renames episode files based on episode_idx_offset.
    - Updates 'episode_index' column in Parquet to the new global episode index.
    - Shifts 'index' and 'frame_index' columns by frame_idx_offset.
    """
    src_chunk_dir = src_root / "data" / chunk_name
    if not src_chunk_dir.exists():
        print(f"Source chunk directory not found: {src_chunk_dir}")
        return 0

    src_files = _natural_sort(src_chunk_dir.glob("episode_*.parquet"))
    if not src_files:
        print(f"No Parquet files found in {src_chunk_dir}")
        return 0

    count_processed = 0
    for src_file_path in src_files:
        original_episode_idx_in_filename = _extract_idx(src_file_path)
        new_episode_global_idx = original_episode_idx_in_filename + episode_idx_offset
        dst_file_path = dst_data_dir / f"episode_{new_episode_global_idx:06d}.parquet"

        try:
            df = pd.read_parquet(src_file_path)
        except Exception as e:
            print(f"Error reading Parquet file {src_file_path}: {e}. Skipping this file.")
            continue

        # 1. Update 'episode_index' column
        if "episode_index" in df.columns:
            df["episode_index"] = new_episode_global_idx
        else:
            print(f"Warning: 'episode_index' column not found in {src_file_path}. Cannot update it.")

        # 2. Update 'index' column (global step/frame counter)
        if "index" in df.columns:
            df["index"] = df["index"] + frame_idx_offset
        elif frame_idx_offset > 0:  # Only warn if an offset was meant to be applied
            print(f"Warning: 'index' column not found in {src_file_path}. Cannot apply frame offset.")

        # 3. Update 'frame_index' column
        if "frame_index" in df.columns:
            df["frame_index"] = df["frame_index"] + frame_idx_offset
        elif frame_idx_offset > 0:  # Only warn if an offset was meant to be applied
            print(f"Warning: 'frame_index' column not found in {src_file_path}. Cannot apply frame offset.")

        try:
            df.to_parquet(dst_file_path)
            count_processed += 1
        except Exception as e:
            print(f"Error writing Parquet file {dst_file_path}: {e}. Skipping this file.")
            if dst_file_path.exists():
                try:
                    dst_file_path.unlink()  # Attempt to remove partially written file
                except OSError as oe:
                    print(f"Error deleting partial file {dst_file_path}: {oe}")
            continue

    return count_processed


def read_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(objs: Iterable[Dict], path: Path) -> None:
    with path.open("w") as f:
        for o in objs:
            f.write(json.dumps(o, separators=(",", ":")) + "\n")


# ────────────────────────────── Meta-file helpers ─────────────────────────── #
def _shift_episode_indices(recs: List[Dict], offset: int, *, shift_index: bool = True) -> None:
    for r in recs:
        r["episode_index"] += offset
        if shift_index and "index" in r:
            idx = r["index"]
            r["index"] = [x + offset for x in idx] if isinstance(idx, list) else idx + offset


def merge_episodes_stats(p1: Path, p2: Path, out: Path, offset: int) -> None:
    a, b = read_jsonl(p1), read_jsonl(p2)
    # Shift relevant indices in records from the second dataset (b)
    _shift_episode_indices(b, offset, shift_index=True)
    # Special handling for 'stats' if it contains indices needing shift
    for r_b in b:
        if "stats" in r_b and isinstance(r_b["stats"], dict) and "episode_index" in r_b["stats"]:
            r_b["stats"]["episode_index"] = _shift_any(r_b["stats"]["episode_index"], offset)
        elif "stats" in r_b:
            r_b["stats"] = _shift_any(r_b["stats"], offset)
    write_jsonl(a + b, out)


def merge_episodes(p1: Path, p2: Path, out: Path, offset: int) -> None:
    a, b = read_jsonl(p1), read_jsonl(p2)
    _shift_episode_indices(b, offset, shift_index=True)
    write_jsonl(a + b, out)


def merge_tasks(t1: Path, t2: Path, out: Path) -> None:
    a, b = read_jsonl(t1), read_jsonl(t2)
    existing = {r["task"]: r["task_index"] for r in a}
    next_idx = max(existing.values()) + 1 if existing else 0

    merged: List[Dict] = a.copy()
    for r in b:
        if r["task"] in existing:
            continue
        merged.append({"task": r["task"], "task_index": next_idx})
        next_idx += 1

    write_jsonl(sorted(merged, key=lambda x: x["task_index"]), out)


def merge_info(i1: Path, i2: Path, out: Path) -> None:
    d1, d2 = json.loads(i1.read_text()), json.loads(i2.read_text())
    merged = d1.copy()
    for k in NUM_KEYS:
        if isinstance(d1.get(k), (int, float)) and isinstance(d2.get(k), (int, float)):
            merged[k] = d1[k] + d2[k]
    # Make split inclusive 0 to N-1
    merged.setdefault("splits", {})["train"] = (
        f"0:{merged['total_episodes'] - 1 if merged['total_episodes'] > 0 else 0}"
    )
    out.write_text(json.dumps(merged, indent=2))


# ────────────────────────────── Video handling ────────────────────────────── #
def copy_videos(src_root: Path, dst_chunk_root: Path, *, start_idx: int = 0, chunk: str) -> int:
    src_video_root = src_root / "videos" / chunk
    if not src_video_root.exists():
        return 0

    episode_count = None
    camera_dirs = sorted(p for p in src_video_root.iterdir() if p.is_dir())
    if not camera_dirs:  # No camera subdirectories
        return 0  # No videos to copy if structure is videos/chunk/camera/episode_*.mp4

    processed_video_episodes = 0
    first_camera_processed = False

    for cam in camera_dirs:
        dst_cam = dst_chunk_root / cam.name
        safe_mkdir(dst_cam)

        vids = _natural_sort(cam.glob("episode_*.mp4"))

        if not first_camera_processed:  # Determine episode_count from the first camera that has videos
            if vids:
                episode_count = len(vids)
                processed_video_episodes = episode_count  # Count episodes only once
                first_camera_processed = True
        elif (
            episode_count is not None and len(vids) != episode_count and vids
        ):  # Only raise error if vids exist and count mismatch
            # If a subsequent camera has a different number of videos, it might indicate an issue
            # or simply that not all cameras recorded all episodes.
            print(
                f"Warning: Camera {cam} has {len(vids)} videos, expected {episode_count} based on earlier camera(s)."
            )

        for src in vids:
            dst_idx = _extract_idx(src) + start_idx
            shutil.copy2(src, dst_cam / f"episode_{dst_idx:06d}.mp4")

    return processed_video_episodes if first_camera_processed else 0


# ────────────────────────────────── Core run ──────────────────────────────── #
def run(cfg: MergeConfig) -> None:
    chunk = cfg.chunk_name
    data_dst_dir = cfg.output_dir / "data" / chunk
    meta_dst_dir = cfg.output_dir / "meta"
    safe_mkdir(data_dst_dir)
    safe_mkdir(meta_dst_dir)

    processed_episodes_d1 = 0
    processed_episodes_d2 = 0
    total_frames_d1_for_offset = 0

    if cfg.copy_parquet:
        info1_path = cfg.dataset1 / "meta" / "info.json"
        if info1_path.exists():
            try:
                info1_data = json.loads(info1_path.read_text())
                total_frames_d1_for_offset = info1_data.get("total_frames", 0)
                if not isinstance(total_frames_d1_for_offset, int):
                    print(
                        f"Warning: 'total_frames' in {info1_path} is not an integer. Using 0 as frame offset for dataset2's Parquet files."
                    )
                    total_frames_d1_for_offset = 0
            except json.JSONDecodeError as e:
                print(
                    f"Warning: Could not parse {info1_path} to get 'total_frames'. Using 0 as frame offset. Error: {e}"
                )
                total_frames_d1_for_offset = 0
        else:
            print(
                f"Warning: {info1_path} not found. Using 0 as frame offset for dataset2's Parquet files. 'index' and 'frame_index' columns in dataset2 Parquets might not be globally unique if dataset1 had frames."
            )
            total_frames_d1_for_offset = 0

    # 1 – Parquet processing
    if cfg.copy_parquet:
        print(f"Processing Parquet files from dataset1: {cfg.dataset1}")
        processed_episodes_d1 = copy_parquet_and_update_indices(
            src_root=cfg.dataset1,
            dst_data_dir=data_dst_dir,
            chunk_name=chunk,
            episode_idx_offset=0,
            frame_idx_offset=0,
        )
        print(f"Finished processing {processed_episodes_d1} Parquet episode files from dataset1.")

        print(f"Processing Parquet files from dataset2: {cfg.dataset2}")
        processed_episodes_d2 = copy_parquet_and_update_indices(
            src_root=cfg.dataset2,
            dst_data_dir=data_dst_dir,
            chunk_name=chunk,
            episode_idx_offset=processed_episodes_d1,
            frame_idx_offset=total_frames_d1_for_offset,
        )
        print(f"Finished processing {processed_episodes_d2} Parquet episode files from dataset2.")
    else:
        print("Skipping Parquet file copying and modification as per configuration (copy_parquet=False).")
        print(
            "Metadata from dataset2 will be merged with an episode offset of 0 if dataset1 Parquets are not processed by this script."
        )

    metadata_episode_offset = processed_episodes_d1

    # Ensure metadata source files exist before attempting to merge
    d1_meta_path = cfg.dataset1 / "meta"
    d2_meta_path = cfg.dataset2 / "meta"

    if not (d1_meta_path.exists() and d2_meta_path.exists()):
        print(
            f"Warning: Metadata directory for dataset1 ({d1_meta_path}) or dataset2 ({d2_meta_path}) not found. Skipping metadata merging."
        )
    else:
        # Episodes Stats
        ep_stats1 = d1_meta_path / "episodes_stats.jsonl"
        ep_stats2 = d2_meta_path / "episodes_stats.jsonl"
        if ep_stats1.exists() and ep_stats2.exists():
            merge_episodes_stats(
                ep_stats1, ep_stats2, meta_dst_dir / "episodes_stats.jsonl", offset=metadata_episode_offset
            )
        else:
            print(
                f"Skipping episodes_stats.jsonl merge due to missing source files ({ep_stats1} or {ep_stats2})."
            )

        # Episodes
        ep1 = d1_meta_path / "episodes.jsonl"
        ep2 = d2_meta_path / "episodes.jsonl"
        if ep1.exists() and ep2.exists():
            merge_episodes(ep1, ep2, meta_dst_dir / "episodes.jsonl", offset=metadata_episode_offset)
        else:
            print(f"Skipping episodes.jsonl merge due to missing source files ({ep1} or {ep2}).")

        # Tasks
        tasks1 = d1_meta_path / "tasks.jsonl"
        tasks2 = d2_meta_path / "tasks.jsonl"
        if tasks1.exists() and tasks2.exists():
            merge_tasks(tasks1, tasks2, meta_dst_dir / "tasks.jsonl")
        else:
            print(f"Skipping tasks.jsonl merge due to missing source files ({tasks1} or {tasks2}).")

        # Info
        info1 = d1_meta_path / "info.json"
        info2 = d2_meta_path / "info.json"
        if info1.exists() and info2.exists():
            merge_info(info1, info2, meta_dst_dir / "info.json")
        else:
            print(f"Skipping info.json merge due to missing source files ({info1} or {info2}).")

    # 5 – Videos
    processed_videos_d1 = 0
    if cfg.copy_videos:
        vid_root = cfg.output_dir / "videos" / chunk
        # Videos for dataset1
        processed_videos_d1 = copy_videos(cfg.dataset1, vid_root, start_idx=0, chunk=chunk)
        # Videos for dataset2, offset by the number of video episodes from dataset1 for this chunk.
        # The `metadata_episode_offset` (which is `processed_episodes_d1` if parquets were copied, or 0 otherwise)
        # should align with video episode numbering.
        _ = copy_videos(cfg.dataset2, vid_root, start_idx=metadata_episode_offset, chunk=chunk)

    total_parquets_processed = processed_episodes_d1 + processed_episodes_d2
    print(
        "\n✅ Merge finished!\n"
        f"  • Parquet files processed/copied : {processed_episodes_d1} (from dataset1) + {processed_episodes_d2} (from dataset2) = {total_parquets_processed}\n"
        f"  • Meta directory       : {meta_dst_dir}\n"
        + (
            f"  • Video episodes copied (d1 count): {processed_videos_d1} (from dataset1, d2 videos also renumbered if copied)\n"
            if cfg.copy_videos
            else ""
        )
    )


# ───────────────────────────── CLI entry-point ───────────────────────────── #
@draccus.wrap()
def main(cfg: MergeConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()

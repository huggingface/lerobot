#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc.
# Licensed under the Apache License, Version 2.0

"""
Convert a LeRobot dataset on the Hub from codebase version 2.1 to 3.0.

Baseline behavior (unchanged):
- Reads the v2.1 layout from the Hub.
- Generates per-episode stats (parquet), checks consistency with legacy stats.
- Removes deprecated `stats.json`.
- Writes updated `meta/info.json` with v3.0 layout metadata.
- Writes v3.0 layout (`data/`, `videos/`, `meta/`).
- By default, pushes to the Hub and tags with "v3.0" (unless --no-push).

New capabilities:
- --no-push: skip Hub deletion/tag/push (for local runs / benchmarking).
- Single-machine parallelism: --max-workers (threads) for faster writes.
- Distributed orchestration (single command UX):
  * --orchestrate: plan + run N workers + stream writer concurrently.
  * --episodes-per-batch: number of episodes per leased batch (workers).
  * --num-workers: parallel worker processes in orchestrator mode.
  * --work-dir: working directory for manifest/temp/state (defaults under ~/.cache/huggingface/lerobot/<repo_id>/_work).
  The orchestrator runs:
    1) Planner: creates a manifest with batches like:
       {"part_id": 0, "episode_start": 0, "episode_end": 9, "status": "pending"}, ...
    2) Worker pool: processes batches, emits temp parquet shards (one per batch).
    3) Streaming Writer: concurrently packs shards into final v3 files using the configured size thresholds.
    4) Finalizer: writes meta/episodes parquet + aggregated stats, swaps _old<->v30, optional push.

NOTE: Workers write *temp shards* only. The single writer owns all final v3 file paths
(no file-level contention, safe concurrent read/write, and consistent with “keep workers’ work separate”).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonlines
import pandas as pd
import pyarrow as pa
import tqdm
from datasets import Dataset, Features, Image
from huggingface_hub import HfApi, snapshot_download
from requests import HTTPError

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    cast_stats_to_numpy,
    flatten_dict,
    get_parquet_file_size_in_mb,
    get_parquet_num_frames,
    get_video_size_in_mb,
    load_info,
    update_chunk_file_indices,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s
from lerobot.utils.constants import HF_LEROBOT_HOME

V21 = "v2.1"

# --------------------------------------------------------------------------------
# Legacy helpers (unaltered behavior; reused by all modes)
# --------------------------------------------------------------------------------

def load_jsonlines(fpath: Path) -> List[Any]:
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)

def legacy_load_episodes(local_dir: Path) -> Dict:
    episodes = load_jsonlines(local_dir / LEGACY_EPISODES_PATH)
    return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}

def legacy_load_episodes_stats(local_dir: Path) -> Dict:
    episodes_stats = load_jsonlines(local_dir / LEGACY_EPISODES_STATS_PATH)
    return {
        item["episode_index"]: cast_stats_to_numpy(item["stats"])
        for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
    }

def legacy_load_tasks(local_dir: Path) -> tuple[Dict, Dict]:
    tasks = load_jsonlines(local_dir / LEGACY_TASKS_PATH)
    tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    task_to_task_index = {task: task_index for task, task_index in tasks.items()}
    return tasks, task_to_task_index

def convert_tasks(root: Path, new_root: Path):
    tasks, _ = legacy_load_tasks(root)
    task_indices = list(tasks.keys())
    task_strings = list(tasks.values())
    df_tasks = pd.DataFrame({"task_index": task_indices}, index=task_strings)
    write_tasks(df_tasks, new_root)


def get_video_keys(root: Path) -> List[str]:
    info = load_info(root)
    features = info["features"]
    return sorted([key for key, ft in features.items() if ft["dtype"] == "video"])

def get_image_keys(root: Path) -> List[str]:
    info = load_info(root)
    features = info["features"]
    return [key for key, ft in features.items() if ft["dtype"] == "image"]

def concat_data_files(paths_to_cat: List[Path], new_root: Path, chunk_idx: int, file_idx: int, image_keys: List[str]):
    # Keep pandas-based behavior for image schema compatibility, as in original code.
    dfs = [pd.read_parquet(p) for p in paths_to_cat]
    df = pd.concat(dfs, ignore_index=True)
    out = new_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = None
    if image_keys:
        schema = Features.from_arrow_schema(pa.Schema.from_pandas(df))
        for k in image_keys:
            schema[k] = Image()
        schema = schema.arrow_schema
    df.to_parquet(out, index=False, schema=schema)

def convert_videos_of_camera(root: Path, new_root: Path, video_key: str, max_mb: int) -> List[Dict]:
    vids_dir = root / "videos"
    ep_paths = sorted(vids_dir.glob(f"*/{video_key}/*.mp4"))
    ep_idx = chunk_idx = file_idx = 0
    size_mb = 0.0
    t = 0.0
    acc: List[Path] = []
    meta: List[Dict] = []
    for p in tqdm.tqdm(ep_paths, desc=f"convert videos of {video_key}"):
        mb = get_video_size_in_mb(p)
        dur = get_video_duration_in_s(p)
        if size_mb + mb >= max_mb and acc:
            concatenate_video_files(
                acc,
                new_root / DEFAULT_VIDEO_PATH.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx),
            )
            for i, _ in enumerate(acc):
                past = ep_idx - len(acc) + i
                meta[past][f"videos/{video_key}/chunk_index"] = chunk_idx
                meta[past][f"videos/{video_key}/file_index"] = file_idx
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_mb = 0.0
            t = 0.0
            acc = []
        meta.append(
            {
                "episode_index": ep_idx,
                f"videos/{video_key}/chunk_index": chunk_idx,
                f"videos/{video_key}/file_index": file_idx,
                f"videos/{video_key}/from_timestamp": t,
                f"videos/{video_key}/to_timestamp": t + dur,
            }
        )
        acc.append(p)
        size_mb += mb
        t += dur
        ep_idx += 1
    if acc:
        concatenate_video_files(
            acc,
            new_root / DEFAULT_VIDEO_PATH.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx),
        )
        for i, _ in enumerate(acc):
            past = ep_idx - len(acc) + i
            meta[past][f"videos/{video_key}/chunk_index"] = chunk_idx
            meta[past][f"videos/{video_key}/file_index"] = file_idx
    return meta

def generate_episode_metadata_dict(episodes_legacy_metadata, episodes_metadata, episodes_stats, episodes_videos=None):
    n = len(episodes_metadata)
    legacy_vals = list(episodes_legacy_metadata.values())
    stats_vals = list(episodes_stats.values())
    stats_keys = list(episodes_stats.keys())
    for i in range(n):
        ep_legacy = legacy_vals[i]
        ep_meta = episodes_metadata[i]
        ep_stats = stats_vals[i]
        ids = {ep_legacy["episode_index"], ep_meta["episode_index"], stats_keys[i]}
        ep_video = {} if episodes_videos is None else episodes_videos[i]
        if episodes_videos is not None:
            ids.add(ep_video["episode_index"])
        if len(ids) != 1:
            raise ValueError(f"Episode indices mismatch: {ids}.")
        row = {**ep_meta, **ep_video, **ep_legacy, **flatten_dict({"stats": ep_stats})}
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        yield row

def convert_episodes_metadata(root: Path, new_root: Path, episodes_meta: List[Dict], episodes_video_meta=None):
    legacy = legacy_load_episodes(root)
    stats = legacy_load_episodes_stats(root)
    sizes = {len(legacy), len(episodes_meta)}
    if episodes_video_meta is not None:
        sizes.add(len(episodes_video_meta))
    if len(sizes) != 1:
        raise ValueError(f"Number of episodes is not the same ({sizes}).")
    ds = Dataset.from_generator(
        lambda: generate_episode_metadata_dict(legacy, episodes_meta, stats, episodes_video_meta)
    )
    write_episodes(ds, new_root)
    write_stats(aggregate_stats(list(stats.values())), new_root)

def convert_info(root: Path, new_root: Path, data_mb: int, video_mb: int):
    info = load_info(root)
    info["codebase_version"] = "v3.0"
    info.pop("total_chunks", None)
    info.pop("total_videos", None)
    info["data_files_size_in_mb"] = data_mb
    info["video_files_size_in_mb"] = video_mb
    info["data_path"] = DEFAULT_DATA_PATH
    info["video_path"] = DEFAULT_VIDEO_PATH
    info["fps"] = int(info["fps"])
    for k in info["features"]:
        if info["features"][k]["dtype"] != "video":
            info["features"][k]["fps"] = info["fps"]
    write_info(info, new_root)

# --------------------------------------------------------------------------------
# Original sequential implementation (kept as-is)
# --------------------------------------------------------------------------------

def convert_data(root: Path, new_root: Path, max_mb: int) -> List[Dict]:
    data_dir = root / "data"
    ep_paths = sorted(data_dir.glob("*/*.parquet"))
    image_keys = get_image_keys(root)
    ep_idx = chunk_idx = file_idx = 0
    size_mb = 0.0
    frames_total = 0
    acc: List[Path] = []
    meta: List[Dict] = []
    for p in ep_paths:
        mb = get_parquet_file_size_in_mb(p)
        n = get_parquet_num_frames(p)
        row = {
            "episode_index": ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": frames_total,
            "dataset_to_index": frames_total + n,
        }
        size_mb += mb
        frames_total += n
        meta.append(row)
        ep_idx += 1
        if size_mb < max_mb:
            acc.append(p)
            continue
        if acc:
            concat_data_files(acc, new_root, chunk_idx, file_idx, image_keys)
        size_mb = mb
        acc = [p]
        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
    if acc:
        concat_data_files(acc, new_root, chunk_idx, file_idx, image_keys)
    return meta

def convert_videos(root: Path, new_root: Path, max_mb: int):
    cams = get_video_keys(root)
    if not cams:
        return None
    metas = [convert_videos_of_camera(root, new_root, cam, max_mb) for cam in cams]
    counts = [len(m) for m in metas]
    if len(set(counts)) != 1:
        raise ValueError(f"All cams dont have same number of episodes ({counts}).")
    merged = []
    n_eps = counts[0]
    for i in range(n_eps):
        row = {}
        for c_idx, _ in enumerate(cams):
            row.update(metas[c_idx][i])
        merged.append(row)
    return merged

# --------------------------------------------------------------------------------
# Single-machine threaded parallel (kept, optional)
# --------------------------------------------------------------------------------

def _group_by_size(sizes_mb: List[float], max_file_mb: int) -> List[Tuple[int, int, List[int]]]:
    groups = []
    chunk_idx = file_idx = 0
    acc = 0.0
    cur = []
    for i, s in enumerate(sizes_mb):
        if acc + s >= max_file_mb and cur:
            groups.append((chunk_idx, file_idx, cur))
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            acc = 0.0
            cur = []
        cur.append(i)
        acc += s
    if cur:
        groups.append((chunk_idx, file_idx, cur))
    return groups

def _write_data_group(group_paths: List[Path], new_root: Path, chunk_idx: int, file_idx: int, image_keys: List[str]):
    concat_data_files(group_paths, new_root, chunk_idx, file_idx, image_keys)

def _parallel_convert_data(root: Path, new_root: Path, max_mb: int, max_workers: int) -> List[Dict]:
    data_dir = root / "data"
    ep_paths = sorted(data_dir.glob("*/*.parquet"))
    sizes = [get_parquet_file_size_in_mb(p) for p in ep_paths]
    frames = [get_parquet_num_frames(p) for p in ep_paths]
    image_keys = get_image_keys(root)
    groups = _group_by_size(sizes, max_mb)
    cum = [0]
    for n in frames:
        cum.append(cum[-1] + n)
    meta = [
        {
            "episode_index": i,
            "data/chunk_index": None,
            "data/file_index": None,
            "dataset_from_index": cum[i],
            "dataset_to_index": cum[i + 1],
        }
        for i in range(len(ep_paths))
    ]
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for (c, f, idxs) in groups:
            paths = [ep_paths[i] for i in idxs]
            tasks.append(ex.submit(_write_data_group, paths, new_root, c, f, image_keys))
            for i in idxs:
                meta[i]["data/chunk_index"] = c
                meta[i]["data/file_index"] = f
        for _ in tqdm.tqdm(as_completed(tasks), total=len(tasks), desc="write data files"):
            pass
    return meta

def _write_video_group(paths: List[Path], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concatenate_video_files(paths, out_path)

def _parallel_convert_videos(root: Path, new_root: Path, max_mb: int, max_workers: int) -> Optional[List[Dict]]:
    cams = get_video_keys(root)
    if not cams:
        return None
    per_cam = []
    for cam in cams:
        vids_dir = root / "videos"
        ep_paths = sorted(vids_dir.glob(f"*/{cam}/*.mp4"))
        sizes = [get_video_size_in_mb(p) for p in ep_paths]
        durs = [get_video_duration_in_s(p) for p in ep_paths]
        groups = _group_by_size(sizes, max_mb)
        cam_meta = [
            {
                "episode_index": i,
                f"videos/{cam}/chunk_index": None,
                f"videos/{cam}/file_index": None,
                f"videos/{cam}/from_timestamp": 0.0,
                f"videos/{cam}/to_timestamp": 0.0,
            }
            for i in range(len(ep_paths))
        ]
        for (c, f, idxs) in groups:
            t = 0.0
            for i in idxs:
                cam_meta[i][f"videos/{cam}/chunk_index"] = c
                cam_meta[i][f"videos/{cam}/file_index"] = f
                cam_meta[i][f"videos/{cam}/from_timestamp"] = t
                cam_meta[i][f"videos/{cam}/to_timestamp"] = t + durs[i]
                t += durs[i]
        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for (c, f, idxs) in groups:
                out = new_root / DEFAULT_VIDEO_PATH.format(video_key=cam, chunk_index=c, file_index=f)
                paths = [ep_paths[i] for i in idxs]
                tasks.append(ex.submit(_write_video_group, paths, out))
            for _ in tqdm.tqdm(as_completed(tasks), total=len(tasks), desc=f"write video files ({cam})"):
                pass
        per_cam.append(cam_meta)
    counts = [len(m) for m in per_cam]
    if len(set(counts)) != 1:
        raise ValueError(f"All cams dont have same number of episodes ({counts}).")
    merged = []
    n_eps = counts[0]
    for i in range(n_eps):
        row = {}
        for cam_idx, _ in enumerate(cams):
            row.update(per_cam[cam_idx][i])
        merged.append(row)
    return merged

# --------------------------------------------------------------------------------
# Distributed orchestration (manifest + worker pool + streaming writer)
# --------------------------------------------------------------------------------

def _work_root(repo_id: str) -> Path:
    return HF_LEROBOT_HOME / repo_id / "_work"

def _manifest_dir(work_dir: Path) -> Path:
    d = work_dir / "manifest"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _tmp_data_dir(work_dir: Path) -> Path:
    d = work_dir / "tmp" / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _state_dir(work_dir: Path) -> Path:
    d = work_dir / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _episodes_total(root: Path) -> int:
    return len(sorted((root / "data").glob("*/*.parquet")))

def _build_batches(episodes_total: int, episodes_per_batch: int):
    batches = []
    start = 0
    pid = 0
    while start < episodes_total:
        end = min(episodes_total - 1, start + episodes_per_batch - 1)
        batches.append({"part_id": pid, "episode_start": start, "episode_end": end, "status": "pending"})
        start = end + 1
        pid += 1
    return batches

def _atomic_write_json(p: Path, obj: dict):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, p)

def _manifest_create(root: Path, work_dir: Path, episodes_per_batch: int, data_mb: int, video_mb: int):
    man_dir = _manifest_dir(work_dir)
    n = _episodes_total(root)
    cams = get_video_keys(root)
    batches = _build_batches(n, episodes_per_batch)
    header = {
        "repo_id": str(root.name),
        "episodes_total": n,
        "episodes_per_batch": episodes_per_batch,
        "batches_total": len(batches),
        "data_files_mb": data_mb,
        "video_files_mb": video_mb,
        "cameras": cams,
    }
    _atomic_write_json(man_dir / "manifest.json", header)
    with jsonlines.open(man_dir / "batches.jsonl", "w") as w:
        for rec in batches:
            w.write(rec)
    # Create pending markers for atomic leasing
    for i in range(len(batches)):
        (man_dir / f"part-{i}.pending").touch()
    return header

def _load_manifest_header(man_dir: Path) -> dict:
    return json.loads((man_dir / "manifest.json").read_text())

def _load_batch_record(man_dir: Path, part_id: int):
    """Return the JSON record for part_id from batches.jsonl, or None if not present."""
    path = man_dir / "batches.jsonl"
    if not path.exists():
        return None
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("part_id") == int(part_id):
                return rec
    return None

def _lease_next_batch(man_dir: Path, total_batches: int) -> Optional[int]:
    """
    Return the leased part_id or None if nothing to do.
    Leasing is done by renaming part-{i}.pending -> part-{i}.leased.<pid>.
    """
    pid = os.getpid()
    for i in range(total_batches):
        pending = man_dir / f"part-{i}.pending"
        if pending.exists():
            try:
                os.replace(pending, man_dir / f"part-{i}.leased.{pid}")
                return i
            except FileNotFoundError:
                continue  # raced
    return None

def _mark_done(man_dir: Path, part_id: int, pid: int):
    leased = man_dir / f"part-{part_id}.leased.{pid}"
    if leased.exists():
        os.replace(leased, man_dir / f"part-{part_id}.done")
    else:
        # might have been force-cleaned; ensure a done marker exists
        (man_dir / f"part-{part_id}.done").touch()

def _compute_frames_by_episode(root: Path) -> List[int]:
    data_dir = root / "data"
    ep_paths = sorted(data_dir.glob("*/*.parquet"))
    return [get_parquet_num_frames(p) for p in ep_paths]

def _compute_video_meta_inputs(root: Path) -> Dict[str, Dict[str, List]]:
    """Return {camera: {'paths': [mp4...], 'sizes_mb': [...], 'durs_s': [...]}}."""
    cams = get_video_keys(root)
    out = {}
    vids_dir = root / "videos"
    for cam in cams:
        mp4s = sorted(vids_dir.glob(f"*/{cam}/*.mp4"))
        out[cam] = {
            "paths": mp4s,
            "sizes_mb": [get_video_size_in_mb(p) for p in mp4s],
            "durs_s": [get_video_duration_in_s(p) for p in mp4s],
        }
    return out

def _worker_run_batch(repo_id: str, root: Path, work_dir: Path, part_id: int, image_keys: List[str]):
    """Concatenate a batch [episode_start..end] into tmp/data/part-XXXXXX.parquet."""
    man_dir = _manifest_dir(work_dir)
    rec = _load_batch_record(man_dir, part_id)
    data_dir = root / "data"
    ep_files = sorted(data_dir.glob("*/*.parquet"))
    start, end = rec["episode_start"], rec["episode_end"]
    dfs = [pd.read_parquet(ep_files[i]) for i in range(start, end + 1)]
    df = pd.concat(dfs, ignore_index=True)
    schema = None
    if image_keys:
        schema = Features.from_arrow_schema(pa.Schema.from_pandas(df))
        for k in image_keys:
            schema[k] = Image()
        schema = schema.arrow_schema
    tmp_out = _tmp_data_dir(work_dir) / f"part-{part_id:06d}.parquet"
    df.to_parquet(tmp_out, index=False, schema=schema)

def _writer_thread(
    repo_id: str,
    root: Path,
    new_root: Path,
    work_dir: Path,
    stop_event: threading.Event,
    episodes_meta_out: List[Dict],
    videos_meta_out: List[Dict],
):
    """
    Consume done parts in order and emit final v3 data files by size threshold.
    Then pack videos per-camera by size threshold. While writing, build per-episode
    mapping for data files (chunk/file & dataset indices) and videos (chunk/file & timestamps).
    """
    man_dir = _manifest_dir(work_dir)
    header = _load_manifest_header(man_dir)
    data_mb = int(header["data_files_mb"])
    video_mb = int(header["video_files_mb"])
    eps_total = int(header["episodes_total"])
    eps_per_batch = int(header["episodes_per_batch"])
    batches_total = int(header["batches_total"])
    image_keys = get_image_keys(root)

    # Precompute episode frame counts & dataset indices
    frames_by_ep = _compute_frames_by_episode(root)
    cum_frames = [0]
    for n in frames_by_ep:
        cum_frames.append(cum_frames[-1] + n)
    # bootstrap episodes_meta_out with dataset_from/to
    episodes_meta_out[:] = [
        {
            "episode_index": i,
            "data/chunk_index": None,
            "data/file_index": None,
            "dataset_from_index": cum_frames[i],
            "dataset_to_index": cum_frames[i + 1],
        }
        for i in range(eps_total)
    ]

    # (A) DATA: stream temp parts -> final v3 data files
    next_part = 0
    chunk_idx = file_idx = 0
    acc_paths: List[Path] = []
    acc_size_mb = 0.0

    def _flush_data_file():
        nonlocal chunk_idx, file_idx, acc_paths, acc_size_mb
        if not acc_paths:
            return
        # read shards then write a final file with proper schema for images
        dfs = [pd.read_parquet(p) for p in acc_paths]
        df = pd.concat(dfs, ignore_index=True)
        out = new_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        out.parent.mkdir(parents=True, exist_ok=True)
        schema = None
        if image_keys:
            schema = Features.from_arrow_schema(pa.Schema.from_pandas(df))
            for k in image_keys:
                schema[k] = Image()
            schema = schema.arrow_schema
        df.to_parquet(out, index=False, schema=schema)
        # rotate indices
        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
        acc_paths = []
        acc_size_mb = 0.0

    while not stop_event.is_set():
        # Stop if all batches are accounted for (prevents reacting to stale .done files)
        if next_part >= batches_total:
            break

        done_marker = man_dir / f"part-{next_part}.done"
        if done_marker.exists():
            rec = _load_batch_record(man_dir, next_part)
            if rec is None:
                # Stale marker from a previous run; skip this index
                next_part += 1
                continue

            shard = _tmp_data_dir(work_dir) / f"part-{next_part:06d}.parquet"
            if not shard.exists():
                # Marker arrived before the shard is visible; wait a bit
                time.sleep(0.05)
                continue

            # If adding this shard would exceed the threshold, flush the current file first.
            shard_mb = get_parquet_file_size_in_mb(shard)
            if acc_size_mb + shard_mb >= data_mb and acc_paths:
                _flush_data_file()

            # Assign mapping for all episodes in this shard to current (chunk_idx, file_idx)
            for ep in range(rec["episode_start"], rec["episode_end"] + 1):
                episodes_meta_out[ep]["data/chunk_index"] = chunk_idx
                episodes_meta_out[ep]["data/file_index"] = file_idx

            # Accumulate shard and continue
            acc_paths.append(shard)
            acc_size_mb += shard_mb
            next_part += 1
            continue

        # nothing to do yet; wait for workers
        time.sleep(0.05)

    # Final flush if anything pending
    _flush_data_file()

    # (B) VIDEOS: pack per camera by size threshold; build episodes video metadata
    # We compute paths/sizes/durations once and then pack in order.
    vmeta_inputs = _compute_video_meta_inputs(root)
    cams = list(vmeta_inputs.keys())
    if cams:
        # init per-episode dicts for videos
        per_cam_meta: Dict[str, List[Dict]] = {}
        for cam in cams:
            per_cam_meta[cam] = [
                {
                    "episode_index": i,
                    f"videos/{cam}/chunk_index": None,
                    f"videos/{cam}/file_index": None,
                    f"videos/{cam}/from_timestamp": 0.0,
                    f"videos/{cam}/to_timestamp": 0.0,
                }
                for i in range(eps_total)
            ]
            paths = vmeta_inputs[cam]["paths"]
            sizes_mb = vmeta_inputs[cam]["sizes_mb"]
            durs_s = vmeta_inputs[cam]["durs_s"]
            v_chunk = v_file = 0
            acc_paths = []
            acc_size = 0.0
            t = 0.0
            for i, (p, mb, dur) in enumerate(zip(paths, sizes_mb, durs_s)):
                # If the next episode would exceed threshold, flush current
                if acc_size + mb >= video_mb and acc_paths:
                    concatenate_video_files(
                        acc_paths,
                        new_root / DEFAULT_VIDEO_PATH.format(video_key=cam, chunk_index=v_chunk, file_index=v_file),
                    )
                    v_chunk, v_file = update_chunk_file_indices(v_chunk, v_file, DEFAULT_CHUNK_SIZE)
                    acc_paths = []
                    acc_size = 0.0
                    t = 0.0
                # Assign mapping for episode i to current (v_chunk, v_file)
                per_cam_meta[cam][i][f"videos/{cam}/chunk_index"] = v_chunk
                per_cam_meta[cam][i][f"videos/{cam}/file_index"] = v_file
                per_cam_meta[cam][i][f"videos/{cam}/from_timestamp"] = t
                per_cam_meta[cam][i][f"videos/{cam}/to_timestamp"] = t + dur
                # Accumulate
                acc_paths.append(p)
                acc_size += mb
                t += dur
            # Flush remainder
            if acc_paths:
                concatenate_video_files(
                    acc_paths,
                    new_root / DEFAULT_VIDEO_PATH.format(video_key=cam, chunk_index=v_chunk, file_index=v_file),
                )
        # merge per-camera dicts per episode
        for ep in range(eps_total):
            row = {"episode_index": ep}
            for cam in cams:
                row.update(per_cam_meta[cam][ep])
            videos_meta_out.append(row)
    else:
        # no videos
        videos_meta_out[:] = []

def _worker_entry(repo_id: str, root: str, work_dir: str, batches_total: int, image_keys: List[str]):
    """Entry point for worker processes."""
    root_path = Path(root)
    work_dir_path = Path(work_dir)
    man_dir = _manifest_dir(work_dir_path)
    
    while True:
        part_id = _lease_next_batch(man_dir, batches_total)
        if part_id is None:
            return
        _worker_run_batch(repo_id, root_path, work_dir_path, part_id, image_keys)
        _mark_done(man_dir, part_id, os.getpid())

def _orchestrate(
    repo_id: str,
    branch: Optional[str],
    data_mb: int,
    video_mb: int,
    episodes_per_batch: int,
    num_workers: int,
    no_push: bool,
    work_dir_opt: Optional[str],
):
    root = HF_LEROBOT_HOME / repo_id
    old_root = HF_LEROBOT_HOME / f"{repo_id}_old"
    new_root = HF_LEROBOT_HOME / f"{repo_id}_v30"
    work_dir = Path(work_dir_opt) if work_dir_opt else _work_root(repo_id)
    # Clean work dir to avoid stale markers/shards from previous runs
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    man_dir = _manifest_dir(work_dir)  # recreate after clean           
    # reset dirs
    if old_root.is_dir() and root.is_dir():
        shutil.rmtree(str(root))
        shutil.move(str(old_root), str(root))
    if new_root.is_dir():
        shutil.rmtree(new_root)

    # read v2.1 snapshot
    snapshot_download(repo_id, repo_type="dataset", revision=V21, local_dir=root)

    # write info + tasks now (cheap; idempotent)
    convert_info(root, new_root, data_mb, video_mb)
    convert_tasks(root, new_root)

    # plan manifest
    header = _manifest_create(root, work_dir, episodes_per_batch, data_mb, video_mb)
    batches_total = int(header["batches_total"])
    image_keys = get_image_keys(root)

    # start writer
    stop_evt = threading.Event()
    episodes_meta_out: List[Dict] = []
    videos_meta_out: List[Dict] = []
    writer = threading.Thread(
        target=_writer_thread,
        args=(repo_id, root, new_root, work_dir, stop_evt, episodes_meta_out, videos_meta_out),
        daemon=True,
    )
    writer.start()


    # run worker pool
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = [
            ex.submit(
                _worker_entry,
                repo_id,
                str(root),
                str(work_dir),
                batches_total,
                image_keys,
            )
            for _ in range(num_workers)
        ]
        for _ in tqdm.tqdm(as_completed(futs), total=len(futs), desc="workers"):
            pass


    # workers done; stop writer after it flushes
    stop_evt.set()
    writer.join()

    # write meta episodes + stats based on mapping collected by writer
    convert_episodes_metadata(root, new_root, episodes_meta_out, videos_meta_out if videos_meta_out else None)

    # swap dirs
    if root.exists():
        shutil.move(str(root), str(old_root))
    shutil.move(str(new_root), str(root))

    if no_push:
        print("[NO-PUSH] Orchestration done; skipped Hub push.")
        return

    # push to hub
    hub_api = HfApi()
    try:
        hub_api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    except HTTPError as e:
        print(f"tag={CODEBASE_VERSION} probably doesn't exist. Skipping exception ({e})")
    hub_api.delete_files(
        delete_patterns=["data/chunk*/episode_*", "meta/*.jsonl", "videos/chunk*"],
        repo_id=repo_id,
        revision=branch,
        repo_type="dataset",
    )
    hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")
    LeRobotDataset(repo_id).push_to_hub()

# --------------------------------------------------------------------------------
# Orchestrator entry + original entry
# --------------------------------------------------------------------------------

def convert_dataset(
    repo_id: str,
    branch: Optional[str] = None,
    data_file_size_in_mb: Optional[int] = None,
    video_file_size_in_mb: Optional[int] = None,
    max_workers: int = 1,
    no_push: bool = False,
    # Orchestrator flags
    orchestrate: bool = False,
    episodes_per_batch: int = 10,
    num_workers: int = 4,
    work_dir: Optional[str] = None,
):
    # Defaults
    if data_file_size_in_mb is None:
        data_file_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_file_size_in_mb is None:
        video_file_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB

    if orchestrate:
        return _orchestrate(
            repo_id=repo_id,
            branch=branch,
            data_mb=int(data_file_size_in_mb),
            video_mb=int(video_file_size_in_mb),
            episodes_per_batch=int(episodes_per_batch),
            num_workers=int(num_workers),
            no_push=bool(no_push),
            work_dir_opt=work_dir,
        )

    # Original single-machine path (sequential by default; threaded when max_workers>1)
    root = HF_LEROBOT_HOME / repo_id
    old_root = HF_LEROBOT_HOME / f"{repo_id}_old"
    new_root = HF_LEROBOT_HOME / f"{repo_id}_v30"

    if old_root.is_dir() and root.is_dir():
        shutil.rmtree(str(root))
        shutil.move(str(old_root), str(root))
    if new_root.is_dir():
        shutil.rmtree(new_root)

    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=V21,
        local_dir=root,
    )

    convert_info(root, new_root, int(data_file_size_in_mb), int(video_file_size_in_mb))
    convert_tasks(root, new_root)

    if max_workers > 1:
        episodes_meta = _parallel_convert_data(root, new_root, int(data_file_size_in_mb), int(max_workers))
        episodes_videos_meta = _parallel_convert_videos(root, new_root, int(video_file_size_in_mb), int(max_workers))
    else:
        episodes_meta = convert_data(root, new_root, int(data_file_size_in_mb))
        episodes_videos_meta = convert_videos(root, new_root, int(video_file_size_in_mb))

    convert_episodes_metadata(root, new_root, episodes_meta, episodes_videos_meta)

    if root.exists():
        shutil.move(str(root), str(old_root))
    shutil.move(str(new_root), str(root))

    if no_push:
        print("[NO-PUSH] Skipping Hub delete/tag/push (local benchmark).")
        return

    hub_api = HfApi()
    try:
        hub_api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    except HTTPError as e:
        print(f"tag={CODEBASE_VERSION} probably doesn't exist. Skipping exception ({e})")
    hub_api.delete_files(
        delete_patterns=["data/chunk*/episode_*", "meta/*.jsonl", "videos/chunk*"],
        repo_id=repo_id,
        revision=branch,
        repo_type="dataset",
    )
    hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")
    LeRobotDataset(repo_id).push_to_hub()

# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repo ID on Hugging Face (e.g. lerobot/pusht).",
    )
    p.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Hub branch to push to (default: main).",
    )
    p.add_argument(
        "--data-file-size-in-mb",
        type=int,
        default=None,
        help="Max MB per v3 data file (default: 100).",
    )
    p.add_argument(
        "--video-file-size-in-mb",
        type=int,
        default=None,
        help="Max MB per v3 video file (default: 500).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Single-machine threaded mode: number of threads for data/video writes (1 = sequential).",
    )
    p.add_argument(
        "--no-push",
        action="store_true",
        help="Skip Hub delete/tag/push. Useful for local runs and benchmarking.",
    )
    # Orchestrator (distributed-style) flags
    p.add_argument(
        "--orchestrate",
        action="store_true",
        help="Single-command plan + worker-pool + streaming writer + finalize.",
    )
    p.add_argument(
        "--episodes-per-batch",
        type=int,
        default=10,
        help="Episodes per leased batch for orchestrator mode.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Worker processes for orchestrator mode.",
    )
    p.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Working directory for manifest/temp/state. Defaults to ~/.cache/huggingface/lerobot/<repo_id>/_work.",
    )

    args = p.parse_args()
    convert_dataset(**vars(args))

#!/usr/bin/env python3
# Copyright 2024-2025 The HuggingFace Inc. team and contributors.
# Licensed under the Apache-2.0 license.

"""
merge_dataset.py – merge two lerobot datasets while keeping episode
indices unique and concatenating all meta files.

Example
-------
python merge_dataset.py \
    --dataset1=/data/robot/run_A \
    --dataset2=/data/robot/run_B \
    --output_dir=/data/robot/merged
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import draccus  # pip install draccus


# ─────────────────────────────────── Config ────────────────────────────────── #
@dataclass
class MergeConfig:
    # required
    dataset1: Path
    dataset2: Path
    output_dir: Path

    # optional
    chunk_name: str = "chunk-000"
    copy_parquet: bool = True
    copy_videos: bool = True


# ─────────────────────────────────── Helpers ───────────────────────────────── #
NUM_KEYS = [
    "total_episodes",
    "total_frames",
    "total_videos",
    "total_chunks",
    "total_tasks",
]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_parquet(src_root: Path, dst_root: Path, *, start_idx: int = 0, chunk: str) -> int:
    src_files = sorted((src_root / "data" / chunk).glob("*.parquet"))
    for i, src in enumerate(src_files, start=start_idx):
        shutil.copy2(src, dst_root / f"episode_{i:06d}.parquet")
    return len(src_files)


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
    _shift_episode_indices(b, offset, shift_index=True)

    for r in b:
        epi = r.get("stats", {}).get("episode_index")
        if epi is None:
            continue
        if isinstance(epi, list):
            shifted = [x + offset for x in epi]
        elif isinstance(epi, dict):
            shifted = {
                k: ([x + offset for x in v] if isinstance(v, list) else v + offset) for k, v in epi.items()
            }
        else:
            shifted = epi + offset
        r["stats"]["episode_index"] = shifted

    write_jsonl(a + b, out)


def merge_episodes(p1: Path, p2: Path, out: Path, offset: int) -> None:
    a, b = read_jsonl(p1), read_jsonl(p2)
    _shift_episode_indices(b, offset)
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
    merged.setdefault("splits", {})["train"] = f"0:{merged['total_episodes']}"
    out.write_text(json.dumps(merged, indent=2))


# ────────────────────────────── Video handling ────────────────────────────── #
def copy_videos(src_root: Path, dst_chunk_root: Path, *, start_idx: int = 0, chunk: str) -> int:
    src_video_root = src_root / "videos" / chunk
    if not src_video_root.exists():
        return 0

    episode_count = None
    for cam in sorted(p for p in src_video_root.iterdir() if p.is_dir()):
        dst_cam = dst_chunk_root / cam.name
        safe_mkdir(dst_cam)

        vids = sorted(cam.glob("episode_*.mp4"))
        if episode_count is None:
            episode_count = len(vids)
        elif len(vids) != episode_count:
            raise ValueError(f"{cam} has {len(vids)} videos, expected {episode_count}")

        for i, src in enumerate(vids, start=start_idx):
            shutil.copy2(src, dst_cam / f"episode_{i:06d}.mp4")

    return episode_count or 0


# ────────────────────────────────── Core run ──────────────────────────────── #
def run(cfg: MergeConfig) -> None:
    chunk = cfg.chunk_name
    data_dst = cfg.output_dir / "data" / chunk
    meta_dst = cfg.output_dir / "meta"
    safe_mkdir(data_dst)
    safe_mkdir(meta_dst)

    # 1 – Parquet
    n1 = n2 = 0
    if cfg.copy_parquet:
        n1 = copy_parquet(cfg.dataset1, data_dst, start_idx=0, chunk=chunk)
        n2 = copy_parquet(cfg.dataset2, data_dst, start_idx=n1, chunk=chunk)

    # 2-4 Meta
    merge_episodes_stats(
        cfg.dataset1 / "meta" / "episodes_stats.jsonl",
        cfg.dataset2 / "meta" / "episodes_stats.jsonl",
        meta_dst / "episodes_stats.jsonl",
        offset=n1,
    )
    merge_episodes(
        cfg.dataset1 / "meta" / "episodes.jsonl",
        cfg.dataset2 / "meta" / "episodes.jsonl",
        meta_dst / "episodes.jsonl",
        offset=n1,
    )
    merge_tasks(
        cfg.dataset1 / "meta" / "tasks.jsonl", cfg.dataset2 / "meta" / "tasks.jsonl", meta_dst / "tasks.jsonl"
    )
    merge_info(
        cfg.dataset1 / "meta" / "info.json", cfg.dataset2 / "meta" / "info.json", meta_dst / "info.json"
    )

    # 5 – Videos
    if cfg.copy_videos:
        vid_root = cfg.output_dir / "videos" / chunk
        n_vid1 = copy_videos(cfg.dataset1, vid_root, start_idx=0, chunk=chunk)
        _ = copy_videos(cfg.dataset2, vid_root, start_idx=n1, chunk=chunk)

    print(
        "\n✅ Merge finished!\n"
        f"  • Parquet files copied : {n1} + {n2} = {n1 + n2}\n"
        f"  • Meta directory       : {meta_dst}\n"
        + (f"  • Video episodes       : {n_vid1} + … (renumbered)\n" if cfg.copy_videos else "")
    )


# ───────────────────────────── CLI entry-point ───────────────────────────── #
@draccus.wrap()
def main(cfg: MergeConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()

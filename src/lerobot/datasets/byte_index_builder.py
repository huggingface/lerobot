"""Build mmap-able byte-index sidecars for LeRobot streaming video fetch."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from .mp4_episode_slice import (
    HEADER_PROBE_BYTES,
    MAX_HEADER_PROBE_BYTES,
    average_fps_from_index,
    episode_keyframes,
    parse_mp4_file_layout,
    parse_mp4_index,
)

logger = logging.getLogger(__name__)

BYTE_INDEX_DIR = "meta/byte_index"
FILES_NAME = "files.parquet"
EPISODES_NAME = "episodes.parquet"
KEYFRAMES_NAME = "keyframes.parquet"


@dataclass
class IndexedFile:
    file_id: int
    file_path: str
    file_size: int
    moov_offset: int
    moov_length: int
    header_length: int
    faststart: bool
    avg_fps: float
    codec: str


def fetch_header_bytes(path: str, file_size: int) -> bytes:
    fs = fsspec.filesystem("hf") if path.startswith("hf://") else fsspec.filesystem("file")
    probe = HEADER_PROBE_BYTES
    while True:
        with fs.open(path, "rb", block_size=max(probe, 2**20), cache_type="none") as f:
            header = f.read(min(probe, file_size))
        try:
            parse_mp4_file_layout(header, file_size)
            return header
        except ValueError as exc:
            if probe >= min(MAX_HEADER_PROBE_BYTES, file_size) or "mdat box not found" not in str(exc):
                raise
            probe = min(probe * 2, MAX_HEADER_PROBE_BYTES, file_size)


def index_video_file(path: str, *, rel_path: str | None = None) -> tuple[IndexedFile, Any]:
    fs = fsspec.filesystem("hf") if path.startswith("hf://") else fsspec.filesystem("file")
    file_size = fs.info(path)["size"]
    header = fetch_header_bytes(path, file_size)
    layout = parse_mp4_file_layout(header, file_size)
    if not layout.faststart:
        logger.warning("non-faststart MP4 (moov after mdat): %s", path)
    mp4_index = parse_mp4_index(header, file_size)
    indexed = IndexedFile(
        file_id=-1,
        file_path=rel_path or path,
        file_size=file_size,
        moov_offset=layout.moov_offset,
        moov_length=layout.moov_length,
        header_length=layout.header_end,
        faststart=layout.faststart,
        avg_fps=average_fps_from_index(mp4_index),
        codec=layout.codec,
    )
    return indexed, mp4_index


def build_byte_index_tables(
    meta,
    data_root: str,
    *,
    file_paths: list[str] | None = None,
    include_keyframes: bool = True,
    workers: int = 8,
    existing_files: dict[str, int] | None = None,
    max_episodes: int | None = None,
    return_mp4_indices: bool = False,
    complete_files_table: bool = False,
) -> tuple[pa.Table, pa.Table, pa.Table | None] | tuple[pa.Table, pa.Table, pa.Table | None, dict[str, Any]]:
    """Build files/episodes/(optional keyframes) Arrow tables."""
    video_keys = list(meta.video_keys)
    n_cams = len(video_keys)
    cam_to_idx = {cam: i for i, cam in enumerate(video_keys)}
    num_episodes = meta.total_episodes if max_episodes is None else min(max_episodes, meta.total_episodes)

    rel_paths: set[str] = set()
    for ep_idx in range(num_episodes):
        for cam in video_keys:
            rel_paths.add(str(meta.get_video_file_path(ep_idx, cam)))
    path_by_rel = {rel: f"{data_root.rstrip('/')}/{rel}" for rel in sorted(rel_paths)}
    if file_paths is None:
        file_paths = list(path_by_rel.values())
    rel_by_path = {path_by_rel[rel]: rel for rel in path_by_rel}

    existing_files = existing_files or {}
    file_meta_by_rel: dict[str, dict[str, Any]] = {}
    mp4_by_rel: dict[str, Any] = {}
    next_file_id = max(existing_files.values(), default=-1) + 1

    to_index = [rel for rel in sorted(rel_paths) if rel not in existing_files]
    if to_index:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(index_video_file, path_by_rel[rel], rel_path=rel): rel for rel in to_index
            }
            for fut in as_completed(futures):
                rel = futures[fut]
                indexed, mp4_index = fut.result()
                indexed.file_id = next_file_id
                mp4_by_rel[rel] = mp4_index
                file_meta_by_rel[rel] = {
                    "file_id": indexed.file_id,
                    "file_path": rel,
                    "file_size": indexed.file_size,
                    "moov_offset": indexed.moov_offset,
                    "moov_length": indexed.moov_length,
                    "header_length": indexed.header_length,
                    "faststart": indexed.faststart,
                    "avg_fps": indexed.avg_fps,
                    "codec": indexed.codec,
                }
                existing_files[rel] = indexed.file_id
                next_file_id += 1

    missing_rels = {
        str(meta.get_video_file_path(ep, cam))
        for ep in range(num_episodes)
        for cam in video_keys
    } - set(mp4_by_rel.keys())
    if missing_rels:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(index_video_file, path_by_rel[rel], rel_path=rel): rel
                for rel in missing_rels
                if rel not in mp4_by_rel
            }
            for fut in as_completed(futures):
                rel = futures[fut]
                _, mp4_index = fut.result()
                mp4_by_rel[rel] = mp4_index

    episode_rows: list[dict[str, Any]] = []
    keyframe_rows: list[dict[str, Any]] = []
    for ep_idx in range(num_episodes):
        for cam in video_keys:
            rel = str(meta.get_video_file_path(ep_idx, cam))
            path = f"{data_root.rstrip('/')}/{rel}"
            if rel not in existing_files:
                raise KeyError(f"file not indexed: {rel}")
            mp4_index = mp4_by_rel[rel]
            ep = meta.episodes[ep_idx]
            from_ts = float(ep[f"videos/{cam}/from_timestamp"])
            to_ts = float(ep[f"videos/{cam}/to_timestamp"])
            span = mp4_index.episode_byte_span(from_ts, to_ts)
            global_episode_id = ep_idx * n_cams + cam_to_idx[cam]
            mdat_length = span.slice_hi - span.slice_lo + 1
            episode_rows.append(
                {
                    "global_episode_id": global_episode_id,
                    "episode_index": ep_idx,
                    "camera_key": cam,
                    "camera_index": cam_to_idx[cam],
                    "file_id": existing_files[rel],
                    "mdat_offset": span.slice_lo,
                    "mdat_length": mdat_length,
                    "frame_count": max(1, round((to_ts - from_ts) * meta.fps)),
                    "first_pts": from_ts,
                    "last_pts": to_ts,
                }
            )
            if include_keyframes:
                timescale = mp4_index.timescale
                for pts_s, byte_off in episode_keyframes(mp4_index, from_ts, to_ts):
                    keyframe_rows.append(
                        {
                            "global_episode_id": global_episode_id,
                            "pts": int(round(pts_s * timescale)),
                            "byte_offset": byte_off,
                        }
                    )

    referenced_rels = {
        str(meta.get_video_file_path(ep, cam)) for ep in range(num_episodes) for cam in video_keys
    }
    if complete_files_table:
        files_table = pa.Table.from_pylist([file_meta_by_rel[rel] for rel in sorted(referenced_rels)])
    elif to_index:
        files_table = pa.Table.from_pylist([file_meta_by_rel[rel] for rel in sorted(to_index)])
    else:
        files_table = None
    episodes_table = pa.Table.from_pylist(episode_rows)
    keyframes_table = pa.Table.from_pylist(keyframe_rows) if include_keyframes and keyframe_rows else None
    if return_mp4_indices:
        return files_table, episodes_table, keyframes_table, mp4_by_rel
    return files_table, episodes_table, keyframes_table


def build_byte_index_in_memory(
    meta,
    data_root: str,
    *,
    workers: int = 8,
    max_episodes: int | None = None,
    include_frame_mappings_cache: bool = False,
):
    """Build a complete byte index resident in RAM (no parquet write, no dataset push)."""
    from .byte_index import EpisodeByteIndex

    num_episodes = meta.total_episodes if max_episodes is None else min(max_episodes, meta.total_episodes)
    files_tbl, episodes_tbl, _, mp4_by_rel = build_byte_index_tables(
        meta,
        data_root,
        include_keyframes=False,
        workers=workers,
        max_episodes=max_episodes,
        return_mp4_indices=True,
        complete_files_table=True,
    )
    index = EpisodeByteIndex(
        None,
        video_keys=list(meta.video_keys),
        num_episodes=num_episodes,
        files_table=files_tbl,
        episodes_table=episodes_tbl,
        mp4_by_rel=mp4_by_rel,
    )
    if include_frame_mappings_cache:
        for ep_idx in range(num_episodes):
            for cam in meta.video_keys:
                index.custom_frame_mappings(ep_idx, cam)
    return index


def write_byte_index(
    output_dir: Path,
    files_table: pa.Table | None,
    episodes_table: pa.Table,
    keyframes_table: pa.Table | None = None,
    *,
    merge_existing: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files_path = output_dir / FILES_NAME
    episodes_path = output_dir / EPISODES_NAME
    keyframes_path = output_dir / KEYFRAMES_NAME

    if merge_existing and files_path.exists() and files_table is not None:
        prev = pq.read_table(files_path)
        files_table = pa.concat_tables([prev, files_table])

    if files_table is not None:
        pq.write_table(files_table, files_path)

    pq.write_table(episodes_table, episodes_path)
    if keyframes_table is not None:
        if merge_existing and keyframes_path.exists():
            keyframes_table = pa.concat_tables([pq.read_table(keyframes_path), keyframes_table])
        pq.write_table(keyframes_table, keyframes_path)


def load_existing_file_ids(index_dir: Path) -> dict[str, int]:
    files_path = index_dir / FILES_NAME
    if not files_path.exists():
        return {}
    table = pq.read_table(files_path, columns=["file_id", "file_path"])
    return {row["file_path"]: int(row["file_id"]) for row in table.to_pylist()}

#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
Validate the integrity of a LeRobot v3.0 dataset, given only its repo id.

This is a single-file, read-only health check. It loads the dataset metadata
(downloading only ``meta/`` from the Hub when needed) and then runs a series of
independent checks, grouped into clearly delimited sections:

    1. Folder architecture & expected files
    2. info.json aggregate counts, splits & path templates
    3. Feature schema & "missing features" (columns present in data)
    4. Episode-metadata indexing continuity (episodes / frames)
    5. Per-data-file scan: episode membership, frame counts, frame_index /
       timestamp monotonicity & continuity, global index uniqueness
    6. tasks.parquet referential integrity
    7. stats.json validity
    8. Video integrity (presence, fps, resolution, timestamp bounds, contiguity)
    9. End-to-end loadability smoke test (LeRobotDataset[0] / [-1])
   10. Hub metadata: repo presence, codebase-version revision, discoverability
       tags, license & README

Each section returns a list of *failures* (hard inconsistencies) and a list of
*warnings* (suspicious but non-fatal). The script prints a per-section report
and exits with code 1 if any failure was detected, 0 otherwise.

Sections 1-4, 6, 7 only need ``meta/`` (cheap). Sections 5, 8, 9 read the data
parquet / video payloads; for files missing locally they are fetched from the
Hub on demand. Section 10 queries the Hub API for repo metadata. Use the flags
below to skip the expensive parts.

Usage:
    # Full check of a Hub dataset (downloads data/videos as needed):
    python examples/dataset/check_dataset_integrity.py --repo-id lerobot/pusht

    # Local dataset, metadata-only (fast):
    python examples/dataset/check_dataset_integrity.py \
        --repo-id lerobot/pusht --root /path/to/pusht --metadata-only

    # Skip the video, smoke-test and Hub sections:
    python examples/dataset/check_dataset_integrity.py \
        --repo-id lerobot/pusht --no-videos --no-smoke-test --no-hub
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

from lerobot.datasets.dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DATA_DIR,
    EPISODES_DIR,
    INFO_PATH,
    STATS_PATH,
    VIDEO_DIR,
)
from lerobot.utils.constants import DEFAULT_FEATURES

# Bookkeeping columns every data parquet file must carry (stored as scalar
# features alongside the user-defined ones). They double as the columns the
# frame-level checks rely on.
BOOKKEEPING_COLUMNS = set(DEFAULT_FEATURES)  # timestamp, frame_index, episode_index, index, task_index

# Default value types considered "numeric" for NaN/Inf and stats sanity checks.
_FLOAT_DTYPES = {"float16", "float32", "float64"}
_INT_DTYPES = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"}


@dataclass
class SectionResult:
    """Outcome of a single check section."""

    name: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


# ----------------------------------------------------------------------------
# Small shared helpers
# ----------------------------------------------------------------------------
def _episodes_dataframe(meta: LeRobotDatasetMetadata):
    """Return the per-episode metadata as a pandas DataFrame, sorted by episode_index.

    ``meta.episodes`` is a HuggingFace ``Dataset`` with the ``stats/*`` columns
    already dropped, so this is metadata-only and cheap.
    """
    df = meta.episodes.to_pandas()
    return df.sort_values("episode_index").reset_index(drop=True)


def _read_parquet_columns(meta, rel_path, columns, fs):
    """Read selected columns of a (possibly remote) parquet file.

    Local files are read directly; otherwise the columns are fetched from the
    Hub via ``HfFileSystem`` range requests, so only the requested columns are
    transferred (never the bulk image payload). Returns a pyarrow Table, or
    ``None`` if the file exists nowhere.
    """
    local_path = meta.root / rel_path
    if local_path.is_file():
        return pq.read_table(local_path, columns=columns)
    hf_path = f"datasets/{meta.repo_id}/{rel_path}"
    if not fs.exists(hf_path, revision=meta.revision):
        return None
    with fs.open(hf_path, "rb", revision=meta.revision) as f:
        return pq.read_table(f, columns=columns)


def _read_parquet_schema(meta, rel_path, fs):
    """Read just the schema (footer) of a (possibly remote) parquet file."""
    local_path = meta.root / rel_path
    if local_path.is_file():
        return pq.read_schema(local_path)
    hf_path = f"datasets/{meta.repo_id}/{rel_path}"
    if not fs.exists(hf_path, revision=meta.revision):
        return None
    with fs.open(hf_path, "rb", revision=meta.revision) as f:
        return pq.read_schema(f)


def _ensure_local_file(meta, rel_path):
    """Return a local path to a dataset file, downloading from the Hub if absent.

    Used for videos (which must be a real local file to probe with PyAV).
    Returns ``None`` if the file cannot be located or downloaded.
    """
    local_path = meta.root / rel_path
    if local_path.is_file():
        return local_path
    try:
        downloaded = hf_hub_download(
            repo_id=meta.repo_id,
            repo_type="dataset",
            filename=rel_path,
            revision=meta.revision,
        )
        return Path(downloaded)
    except Exception:
        return None


def _feature_height_width(ft):
    """Best-effort (height, width) extraction from an image/video feature spec."""
    shape = tuple(ft["shape"])
    names = ft.get("names")
    if names and len(names) == len(shape):
        idx = {n: i for i, n in enumerate(names)}
        if "height" in idx and "width" in idx:
            return shape[idx["height"]], shape[idx["width"]]
    if len(shape) == 3:
        # Heuristic: channel-first (C, H, W) when first dim is small, else (H, W, C).
        if shape[0] <= 4:
            return shape[1], shape[2]
        return shape[0], shape[1]
    return None


def _parse_splits(splits):
    """Turn a ``{"train": "0:100", ...}`` dict into the set of covered episodes."""
    covered = []
    for spec in splits.values():
        if isinstance(spec, str) and ":" in spec:
            start, end = spec.split(":")
            covered.extend(range(int(start), int(end)))
    return covered


def _group_episodes_by_data_file(df):
    """Bucket episode rows by the (chunk, file) data parquet they live in."""
    buckets = defaultdict(list)
    for _, ep in df.iterrows():
        key = (int(ep["data/chunk_index"]), int(ep["data/file_index"]))
        buckets[key].append(ep)
    return buckets


# ============================================================================
# SECTION 1 - Folder architecture & expected files
# ----------------------------------------------------------------------------
# Verify the canonical v3.0 layout exists: the required meta/ files, and the
# data/ (and videos/ when the dataset has video features) directories with at
# least one chunk/file. Only checks local presence; remote-only datasets that
# were just metadata-downloaded will legitimately have no local data/ yet, so
# missing data/video dirs are warnings rather than failures here (Section 5/8
# resolve them against the Hub).
# ============================================================================
def check_folder_architecture(meta) -> SectionResult:
    res = SectionResult("1. Folder architecture & expected files")
    root = meta.root

    # Required metadata files (always pulled with meta/).
    for rel in (INFO_PATH, STATS_PATH, "meta/tasks.parquet"):
        if not (root / rel).is_file():
            res.failures.append(f"missing required metadata file: {rel}")

    # At least one episode-metadata parquet under meta/episodes/.
    episode_meta_files = list((root / EPISODES_DIR).glob("**/*.parquet"))
    if not episode_meta_files:
        res.failures.append(f"no episode metadata parquet found under {EPISODES_DIR}/")

    # data/ directory: warn (not fail) when absent locally, since metadata-only
    # snapshots are valid and Section 5 resolves data against the Hub.
    data_dir = root / DATA_DIR
    if not data_dir.is_dir():
        res.warnings.append(f"no local '{DATA_DIR}/' directory (will resolve files from the Hub)")
    elif not list(data_dir.glob("chunk-*/file-*.parquet")):
        res.warnings.append(f"'{DATA_DIR}/' present but contains no chunk-*/file-*.parquet locally")

    # videos/ only expected when the dataset declares video features.
    if meta.video_keys:
        video_dir = root / VIDEO_DIR
        if not video_dir.is_dir():
            res.warnings.append(f"no local '{VIDEO_DIR}/' directory but dataset has video keys")
        else:
            for key in meta.video_keys:
                if not (video_dir / key).is_dir():
                    res.warnings.append(f"no local video directory for video key {key!r}")

    return res


# ============================================================================
# SECTION 2 - info.json aggregate counts, splits & path templates
# ----------------------------------------------------------------------------
# Cross-check the global counters every other consumer trusts: total_episodes /
# total_frames / total_tasks against the actual metadata, that the splits cover
# exactly [0, total_episodes), and that the path templates carry the expected
# placeholders. fps/chunk sizes are validated by DatasetInfo on load, so we only
# surface the codebase version here.
# ============================================================================
def check_info_consistency(meta, df) -> SectionResult:
    res = SectionResult("2. info.json counts, splits & templates")

    # Codebase version (load already raised on hard-incompatible versions).
    if meta.info.codebase_version != CODEBASE_VERSION:
        res.warnings.append(
            f"info.codebase_version={meta.info.codebase_version!r} != script target {CODEBASE_VERSION!r}"
        )

    # total_episodes vs number of episode rows and max(episode_index)+1.
    n_rows = len(df)
    if meta.total_episodes != n_rows:
        res.failures.append(f"info.total_episodes={meta.total_episodes} but found {n_rows} episode rows")
    if n_rows > 0:
        max_idx = int(df["episode_index"].max())
        if max_idx + 1 != n_rows:
            res.failures.append(f"episode_index range [0, {max_idx}] inconsistent with {n_rows} episode rows")

    # total_frames vs sum of per-episode lengths.
    sum_len = int(df["length"].sum()) if n_rows > 0 else 0
    if meta.total_frames != sum_len:
        res.failures.append(f"info.total_frames={meta.total_frames} but sum(length)={sum_len}")

    # total_tasks vs tasks.parquet row count.
    n_tasks = len(meta.tasks)
    if meta.total_tasks != n_tasks:
        res.failures.append(f"info.total_tasks={meta.total_tasks} but tasks.parquet has {n_tasks} rows")

    # Splits must cover exactly [0, total_episodes) with no gaps/overlaps.
    if meta.info.splits:
        covered = sorted(_parse_splits(meta.info.splits))
        expected = list(range(meta.total_episodes))
        if covered != expected:
            res.failures.append(
                f"splits {meta.info.splits} do not cover exactly [0, {meta.total_episodes}) "
                f"(covered {len(covered)} unique episode(s))"
            )

    # Path templates must contain the placeholders the readers format against.
    if not ("{chunk_index" in meta.data_path and "{file_index" in meta.data_path):
        res.failures.append(f"data_path template missing chunk/file placeholders: {meta.data_path!r}")
    if meta.video_keys:
        vp = meta.video_path or ""
        if not all(tok in vp for tok in ("{video_key", "{chunk_index", "{file_index")):
            res.failures.append(f"video_path template missing placeholders: {meta.video_path!r}")

    return res


# ============================================================================
# SECTION 3 - Feature schema & "missing features"
# ----------------------------------------------------------------------------
# Validate the features dict structurally (dtype/shape/names coherence) and
# confirm the first data file actually carries a column for every non-video
# feature (image features are embedded in the parquet; video features live in
# mp4 files and are intentionally absent from the parquet columns).
# ============================================================================
def check_feature_schema(meta, df, fs, scan_data) -> SectionResult:
    res = SectionResult("3. Feature schema & missing features")
    video_keys = set(meta.video_keys)

    # 3a. Structural validation of each feature spec.
    for key, ft in meta.features.items():
        if "dtype" not in ft or "shape" not in ft:
            res.failures.append(f"feature {key!r} missing 'dtype' or 'shape'")
            continue
        shape = tuple(ft["shape"])
        names = ft.get("names")
        # Vector features: names length must match the (1-D) shape.
        if names is not None and isinstance(names, list) and len(shape) == 1 and len(names) != shape[0]:
            res.failures.append(f"feature {key!r}: len(names)={len(names)} != shape[0]={shape[0]}")
        # Image/video features should be 3-dimensional.
        if ft["dtype"] in ("image", "video") and len(shape) != 3:
            res.failures.append(f"visual feature {key!r} has non-3D shape {shape}")

    # 3b. Column presence in the data parquet (needs to read one file footer).
    if not scan_data:
        res.warnings.append("data scan disabled: skipped data-column presence check")
        return res

    if len(df) == 0:
        return res

    first = df.iloc[0]
    rel = meta.data_path.format(
        chunk_index=int(first["data/chunk_index"]), file_index=int(first["data/file_index"])
    )
    schema = _read_parquet_schema(meta, rel, fs)
    if schema is None:
        res.warnings.append(f"could not read schema of first data file {rel} to check columns")
        return res

    data_columns = set(schema.names)
    expected = {k for k in meta.features if k not in video_keys}
    missing = expected - data_columns
    if missing:
        res.failures.append(f"data file {rel} missing feature columns: {sorted(missing)}")
    # Columns present in data but neither a feature nor bookkeeping -> warn.
    extra = data_columns - set(meta.features) - BOOKKEEPING_COLUMNS
    if extra:
        res.warnings.append(f"data file {rel} has unexpected columns: {sorted(extra)}")

    return res


# ============================================================================
# SECTION 4 - Episode-metadata indexing continuity
# ----------------------------------------------------------------------------
# Independently of any data file, the per-episode metadata must describe a
# contiguous, gap-free indexing of frames into the global frame index:
#   * episode_index == 0, 1, 2, ... in order;
#   * the first episode starts at dataset_from_index == 0;
#   * dataset_to_index - dataset_from_index == length for each episode;
#   * each dataset_from_index equals the previous dataset_to_index (no gaps);
#   * the last dataset_to_index equals info.total_frames.
# This is the "missing episodes or frames according to metadata" check.
# ============================================================================
def check_episode_continuity(meta, df) -> SectionResult:
    res = SectionResult("4. Episode-metadata indexing continuity")
    prev_to = 0
    for expected_idx, (_, row) in enumerate(df.iterrows()):
        ep_idx = int(row["episode_index"])
        if ep_idx != expected_idx:
            res.failures.append(f"episode_index not contiguous: expected {expected_idx}, found {ep_idx}")

        d_from = int(row["dataset_from_index"])
        d_to = int(row["dataset_to_index"])
        length = int(row["length"])

        if d_from != prev_to:
            ref = f"episode {expected_idx - 1} dataset_to_index" if expected_idx > 0 else "start (0)"
            res.failures.append(
                f"episode {ep_idx}: dataset_from_index={d_from} does not match {ref}={prev_to}"
            )
        if d_to - d_from != length:
            res.failures.append(
                f"episode {ep_idx}: dataset_to_index - dataset_from_index = {d_to - d_from} but length = {length}"
            )
        if length <= 0:
            res.failures.append(f"episode {ep_idx}: non-positive length {length}")

        prev_to = d_to

    if len(df) > 0 and prev_to != meta.total_frames:
        res.failures.append(
            f"last dataset_to_index={prev_to} does not match info.total_frames={meta.total_frames}"
        )
    return res


# ============================================================================
# SECTION 5 - Per-data-file scan
# ----------------------------------------------------------------------------
# For each data parquet file referenced by the metadata, read the bookkeeping
# columns (episode_index, frame_index, timestamp, index) and validate:
#   * "missing data files": the file resolves locally or on the Hub;
#   * episode membership: the set of episode_index values in the file matches
#     the set the metadata assigns to it, and the row count matches sum(length);
#   * frame_index per episode runs exactly 0..length-1 (monotonic + continuous);
#   * timestamp == frame_index / fps within tolerance;
#   * the global "index" column is a contiguous 0..total_frames-1 with no
#     duplicates across files (cross-file uniqueness);
#   * each episode_index appears in exactly one data file.
# ============================================================================
def check_data_files(meta, df, fs, fps_tol_s) -> SectionResult:
    res = SectionResult("5. Per-data-file scan (membership, frames, monotonicity)")
    fps = meta.fps
    buckets = _group_episodes_by_data_file(df)

    seen_global_index = set()
    episode_to_file = {}
    duplicate_index_count = 0

    # Per-episode lengths from metadata for cross-checking.
    meta_len = {int(r["episode_index"]): int(r["length"]) for _, r in df.iterrows()}

    for (chunk_idx, file_idx), eps in sorted(buckets.items()):
        rel = meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        meta_eps = {int(e["episode_index"]) for e in eps}
        meta_frames = sum(int(e["length"]) for e in eps)

        # Cross-file uniqueness of episodes (an episode must live in one file).
        for ep in meta_eps:
            if ep in episode_to_file:
                res.failures.append(
                    f"episode {ep} assigned to multiple data files: {episode_to_file[ep]} and ({chunk_idx},{file_idx})"
                )
            else:
                episode_to_file[ep] = (chunk_idx, file_idx)

        try:
            table = _read_parquet_columns(
                meta, rel, ["episode_index", "frame_index", "timestamp", "index"], fs
            )
        except Exception as exc:
            res.failures.append(f"[chunk={chunk_idx:03d} file={file_idx:03d}] failed to read {rel}: {exc}")
            continue
        if table is None:
            res.failures.append(f"[chunk={chunk_idx:03d} file={file_idx:03d}] missing data file: {rel}")
            continue

        cols = table.to_pydict()
        ep_col = cols["episode_index"]
        frame_col = cols["frame_index"]
        ts_col = cols["timestamp"]
        idx_col = cols["index"]
        data_eps = {int(v) for v in ep_col}

        # Episode membership: metadata set vs data set.
        missing = meta_eps - data_eps
        unexpected = data_eps - meta_eps
        if missing:
            res.failures.append(f"{rel}: episodes in metadata but absent from data: {sorted(missing)}")
        if unexpected:
            res.failures.append(f"{rel}: episodes in data but not in metadata: {sorted(unexpected)}")
        if not missing and not unexpected and len(ep_col) != meta_frames:
            res.failures.append(f"{rel}: data rows={len(ep_col)} vs metadata sum(length)={meta_frames}")

        # Per-episode frame_index/timestamp checks + global index collection.
        per_ep_frames = defaultdict(list)
        for ep_v, fr_v, ts_v, ix_v in zip(ep_col, frame_col, ts_col, idx_col, strict=True):
            per_ep_frames[int(ep_v)].append((int(fr_v), float(ts_v)))
            ix = int(ix_v)
            if ix in seen_global_index:
                duplicate_index_count += 1
            else:
                seen_global_index.add(ix)

        for ep, frames in per_ep_frames.items():
            frames.sort(key=lambda p: p[0])
            expected_len = meta_len.get(ep)
            # frame_index must be exactly 0..len-1.
            frame_indices = [f for f, _ in frames]
            if frame_indices != list(range(len(frames))):
                res.failures.append(f"{rel}: episode {ep} frame_index not contiguous 0..{len(frames) - 1}")
            elif expected_len is not None and len(frames) != expected_len:
                res.failures.append(
                    f"{rel}: episode {ep} has {len(frames)} frames but metadata length={expected_len}"
                )
            # timestamp == frame_index / fps within tolerance.
            for f_i, ts in frames:
                if not math.isfinite(ts) or abs(ts - f_i / fps) > fps_tol_s:
                    res.failures.append(
                        f"{rel}: episode {ep} frame {f_i} timestamp={ts:.6f} != {f_i / fps:.6f} (1/fps grid)"
                    )
                    break

    # Global index sanity across all files.
    if duplicate_index_count:
        res.failures.append(
            f"found {duplicate_index_count} duplicated global 'index' value(s) across data files"
        )
    if seen_global_index:
        expected_index = set(range(meta.total_frames))
        if seen_global_index != expected_index:
            missing_n = len(expected_index - seen_global_index)
            extra_n = len(seen_global_index - expected_index)
            res.failures.append(
                f"global 'index' is not a contiguous 0..{meta.total_frames - 1} "
                f"(missing {missing_n}, unexpected {extra_n})"
            )

    return res


# ============================================================================
# SECTION 6 - tasks.parquet referential integrity
# ----------------------------------------------------------------------------
# tasks.parquet maps a task string to a task_index. Validate that task indices
# are a contiguous 0..total_tasks-1 with no duplicates (indices or strings), and
# that every task referenced by an episode exists. Tasks never referenced by any
# episode are reported as warnings (orphans).
# ============================================================================
def check_tasks(meta, df) -> SectionResult:
    res = SectionResult("6. tasks.parquet referential integrity")
    tasks = meta.tasks  # index = task string, column 'task_index'

    indices = sorted(int(i) for i in tasks["task_index"].tolist())
    if indices != list(range(len(indices))):
        res.failures.append(f"task_index values are not a contiguous 0..{len(indices) - 1}: {indices[:20]}")

    # Duplicate task strings (the index of the tasks frame).
    task_strings = list(tasks.index)
    if len(set(task_strings)) != len(task_strings):
        res.failures.append("duplicate task strings found in tasks.parquet")

    # Referential integrity: every task named by an episode must exist.
    known_tasks = set(task_strings)
    referenced = set()
    if "tasks" in df.columns:
        for _, row in df.iterrows():
            ep_tasks = row["tasks"]
            if ep_tasks is None:
                continue
            for t in list(ep_tasks):
                referenced.add(t)
                if t not in known_tasks:
                    res.failures.append(f"episode {int(row['episode_index'])} references unknown task {t!r}")

    # Orphan tasks (declared but never used) -> warning.
    orphans = known_tasks - referenced
    if orphans and referenced:
        sample = sorted(orphans)[:10]
        res.warnings.append(f"{len(orphans)} task(s) never referenced by any episode, e.g. {sample}")

    return res


# ============================================================================
# SECTION 7 - stats.json validity
# ----------------------------------------------------------------------------
# stats.json holds per-feature min/max/mean/std/count used for normalization.
# Validate that each entry has the expected sub-keys, that min <= mean <= max
# element-wise, std >= 0, no NaN/Inf, and that the stored shapes are consistent.
# Missing stats for a feature is a warning (some auxiliary features carry none);
# a stats key that is not a feature is a failure.
# ============================================================================
def check_stats(meta) -> SectionResult:
    res = SectionResult("7. stats.json validity")
    if meta.stats is None:
        res.skipped = True
        res.skip_reason = "no stats.json present"
        return res

    feature_keys = set(meta.features)
    for key, stat in meta.stats.items():
        if key not in feature_keys:
            res.failures.append(f"stats key {key!r} is not a declared feature")
            continue

        for sub in ("min", "max", "mean", "std", "count"):
            if sub not in stat:
                res.failures.append(f"stats[{key!r}] missing '{sub}'")
        if any(sub not in stat for sub in ("min", "max", "mean", "std")):
            continue

        mn = np.asarray(stat["min"], dtype=np.float64)
        mx = np.asarray(stat["max"], dtype=np.float64)
        mean = np.asarray(stat["mean"], dtype=np.float64)
        std = np.asarray(stat["std"], dtype=np.float64)

        # No NaN / Inf anywhere.
        for sub, arr in (("min", mn), ("max", mx), ("mean", mean), ("std", std)):
            if not np.all(np.isfinite(arr)):
                res.failures.append(f"stats[{key!r}]['{sub}'] contains NaN/Inf")

        # Ordering and non-negative std.
        if np.any(mn > mx + 1e-6):
            res.failures.append(f"stats[{key!r}]: min > max somewhere")
        if np.any(mean < mn - 1e-6) or np.any(mean > mx + 1e-6):
            res.failures.append(f"stats[{key!r}]: mean outside [min, max] somewhere")
        if np.any(std < -1e-6):
            res.failures.append(f"stats[{key!r}]: negative std somewhere")

        # count consistency (warning: image stats can be sub-sampled).
        if "count" in stat:
            count = int(np.asarray(stat["count"]).reshape(-1)[0])
            if count != meta.total_frames:
                res.warnings.append(f"stats[{key!r}]['count']={count} != total_frames={meta.total_frames}")

    # Warn about features lacking any stats entry.
    missing_stats = feature_keys - set(meta.stats)
    if missing_stats:
        res.warnings.append(f"features without stats: {sorted(missing_stats)}")

    return res


# ============================================================================
# SECTION 8 - Video integrity
# ----------------------------------------------------------------------------
# For datasets with video features, verify each referenced mp4 (per video key)
# is present (locally or on the Hub), decodable, and consistent with metadata:
#   * "missing video files": the file resolves;
#   * container fps == info.fps;
#   * width/height match the feature shape;
#   * every episode's [from_timestamp, to_timestamp] lies within the video
#     duration, with to > from;
#   * the per-file episode segments are non-overlapping (timeline contiguity);
#   * (to - from) * fps is close to the episode length.
# ============================================================================
def check_videos(meta, df) -> SectionResult:
    res = SectionResult("8. Video integrity")
    if not meta.video_keys:
        res.skipped = True
        res.skip_reason = "dataset has no video features"
        return res

    from lerobot.datasets.video_utils import get_video_duration_in_s, get_video_info

    fps = meta.fps
    meta_len = {int(r["episode_index"]): int(r["length"]) for _, r in df.iterrows()}

    for vid_key in meta.video_keys:
        ft = meta.features[vid_key]
        hw = _feature_height_width(ft)

        # Bucket episodes by the video file they reference for this key.
        file_to_eps = defaultdict(list)
        for _, row in df.iterrows():
            chunk = int(row[f"videos/{vid_key}/chunk_index"])
            file_ = int(row[f"videos/{vid_key}/file_index"])
            file_to_eps[(chunk, file_)].append(row)

        for (chunk, file_), eps in sorted(file_to_eps.items()):
            rel = meta.video_path.format(video_key=vid_key, chunk_index=chunk, file_index=file_)
            path = _ensure_local_file(meta, rel)
            if path is None:
                res.failures.append(f"missing video file: {rel}")
                continue

            try:
                info = get_video_info(path)
                duration = get_video_duration_in_s(path)
            except Exception as exc:
                res.failures.append(f"{rel}: not decodable ({exc})")
                continue

            # fps consistency.
            vfps = info.get("video.fps")
            if vfps is not None and int(vfps) != int(fps):
                res.failures.append(f"{rel}: video fps={vfps} != info.fps={fps}")

            # Resolution consistency.
            if hw is not None:
                vh, vw = info.get("video.height"), info.get("video.width")
                if vh is not None and vw is not None and (int(vh), int(vw)) != (int(hw[0]), int(hw[1])):
                    res.failures.append(f"{rel}: video resolution {vh}x{vw} != feature {hw[0]}x{hw[1]} (HxW)")

            # Timestamp bounds + contiguity within this video file.
            segments = []
            for row in eps:
                ep = int(row["episode_index"])
                t_from = float(row[f"videos/{vid_key}/from_timestamp"])
                t_to = float(row[f"videos/{vid_key}/to_timestamp"])
                segments.append((t_from, t_to, ep))

                if t_from < -1e-6 or t_to <= t_from:
                    res.failures.append(f"{rel}: episode {ep} invalid timestamps [{t_from}, {t_to}]")
                if t_to > duration + 1.0 / fps:
                    res.failures.append(
                        f"{rel}: episode {ep} to_timestamp={t_to:.3f}s exceeds video duration={duration:.3f}s"
                    )
                # (to - from) * fps should be ~ episode length.
                expected_len = meta_len.get(ep)
                if expected_len is not None:
                    n = round((t_to - t_from) * fps)
                    if abs(n - expected_len) > 1:
                        res.warnings.append(
                            f"{rel}: episode {ep} (to-from)*fps={n} differs from length={expected_len}"
                        )

            # Non-overlapping segments along the timeline.
            segments.sort()
            for (a_from, a_to, a_ep), (b_from, b_to, b_ep) in zip(segments, segments[1:], strict=False):
                if b_from < a_to - 1e-6:
                    res.failures.append(
                        f"{rel}: episode {a_ep} [{a_from:.3f},{a_to:.3f}] overlaps episode {b_ep} "
                        f"[{b_from:.3f},{b_to:.3f}]"
                    )

    return res


# ============================================================================
# SECTION 9 - End-to-end loadability smoke test
# ----------------------------------------------------------------------------
# Final sanity: construct a LeRobotDataset and fetch the first and last frames.
# This exercises the full read path (parquet + video decoding + delta-timestamp
# querying) and confirms the returned items expose every declared feature key
# with the expected shape. Failures here usually mean the lower-level checks
# missed something or a payload is corrupt.
# ============================================================================
def check_smoke_test(meta, root) -> SectionResult:
    res = SectionResult("9. End-to-end loadability smoke test")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(meta.repo_id, root=root, revision=meta.revision)
    except Exception as exc:
        res.failures.append(f"LeRobotDataset failed to construct: {exc}")
        return res

    # Length must match total_frames.
    if len(ds) != meta.total_frames:
        res.failures.append(f"len(dataset)={len(ds)} != info.total_frames={meta.total_frames}")
    if len(ds) == 0:
        return res

    expected_keys = set(meta.features)
    for idx in {0, len(ds) - 1}:
        try:
            item = ds[idx]
        except Exception as exc:
            res.failures.append(f"dataset[{idx}] raised: {exc}")
            continue
        missing = expected_keys - set(item)
        if missing:
            res.failures.append(f"dataset[{idx}] missing keys: {sorted(missing)}")

    return res


# ============================================================================
# SECTION 10 - Hugging Face Hub metadata, version tag & discoverability tags
# ----------------------------------------------------------------------------
# Independently of the payload, verify the dataset is properly published on the
# Hub and discoverable:
#   * the repo exists on the Hub;
#   * a version branch/tag matching the codebase version (e.g. ``v3.0``) exists,
#     so consumers can pin the revision they load;
#   * the auto-generated / declared discoverability tags are present
#     (task_categories:robotics, the custom ``LeRobot`` tag, modality:tabular /
#     timeseries / video, format:parquet, size_categories:*);
#   * a license is declared;
#   * a README.md (dataset card) is present.
# Missing repo / version are failures; missing tags / license / README are
# warnings (they hurt discoverability but not loadability).
# ============================================================================
def check_hub_metadata(meta) -> SectionResult:
    res = SectionResult("10. Hugging Face Hub metadata & tags")

    import packaging.version
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    from lerobot.datasets.utils import get_repo_versions

    api = HfApi()
    repo_id = meta.repo_id

    # Is the dataset published on the Hub?
    try:
        info = api.dataset_info(repo_id)
    except RepositoryNotFoundError:
        res.failures.append(f"dataset {repo_id!r} not found on the Hugging Face Hub")
        return res
    except Exception as exc:
        res.warnings.append(f"could not query the Hub for {repo_id!r}: {exc}")
        return res

    # A version branch/tag matching the codebase version must exist.
    try:
        target = packaging.version.parse(CODEBASE_VERSION)
        versions = get_repo_versions(repo_id)
        if target not in versions:
            found = sorted(f"v{v}" for v in versions) or "none"
            res.failures.append(f"no {CODEBASE_VERSION} version branch/tag on the Hub (found: {found})")
    except Exception as exc:
        res.warnings.append(f"could not list repo version refs: {exc}")

    # Discoverability tags (auto-generated by the Hub + declared on the card).
    tags = set(info.tags or [])
    expected_tags = ["task_categories:robotics", "LeRobot", "format:parquet", "modality:tabular", "modality:timeseries"]
    if meta.video_keys:
        expected_tags.append("modality:video")
    for tag in expected_tags:
        if tag not in tags:
            res.warnings.append(f"missing expected Hub tag: {tag}")
    if not any(t.startswith("size_categories:") for t in tags):
        res.warnings.append("missing size category tag (size_categories:*)")

    # License (declared on the card or surfaced as a license:* tag).
    card_data = info.card_data
    has_license = bool(getattr(card_data, "license", None)) or any(t.startswith("license:") for t in tags)
    if not has_license:
        res.warnings.append("no license declared on the Hub")

    # README / dataset card.
    siblings = {s.rfilename for s in (info.siblings or [])}
    if "README.md" not in siblings:
        res.warnings.append("no README.md (dataset card) on the Hub")

    return res


# ----------------------------------------------------------------------------
# Orchestration & reporting
# ----------------------------------------------------------------------------
def run_all_checks(
    repo_id,
    root=None,
    revision=None,
    scan_data=True,
    check_video=True,
    smoke_test=True,
    check_hub=True,
    fps_tol_s=1e-3,
):
    """Load metadata and run every section, returning the list of SectionResults."""
    meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    print(
        f"Loaded metadata for {repo_id!r}: {meta.total_episodes} episodes, "
        f"{meta.total_frames} frames, {meta.total_tasks} tasks, "
        f"{len(meta.video_keys)} video key(s).\n"
    )

    df = _episodes_dataframe(meta)
    fs = HfFileSystem()

    results: list[SectionResult] = []

    # Metadata-only sections (cheap).
    results.append(check_folder_architecture(meta))
    results.append(check_info_consistency(meta, df))
    results.append(check_feature_schema(meta, df, fs, scan_data))
    results.append(check_episode_continuity(meta, df))

    # Data-payload section.
    if scan_data:
        results.append(check_data_files(meta, df, fs, fps_tol_s))
    else:
        results.append(
            SectionResult("5. Per-data-file scan", skipped=True, skip_reason="--metadata-only / --no-data")
        )

    results.append(check_tasks(meta, df))
    results.append(check_stats(meta))

    # Video section.
    if check_video:
        results.append(check_videos(meta, df))
    else:
        results.append(SectionResult("8. Video integrity", skipped=True, skip_reason="--no-videos"))

    # Smoke test.
    if smoke_test:
        results.append(check_smoke_test(meta, root))
    else:
        results.append(
            SectionResult("9. End-to-end loadability smoke test", skipped=True, skip_reason="--no-smoke-test")
        )

    # Hub metadata section (needs network).
    if check_hub:
        results.append(check_hub_metadata(meta))
    else:
        results.append(
            SectionResult("10. Hugging Face Hub metadata & tags", skipped=True, skip_reason="--no-hub")
        )

    return results


def print_report(results) -> int:
    """Print a per-section report and return the total number of failures."""
    total_failures = 0
    total_warnings = 0

    for res in results:
        print("=" * 78)
        if res.skipped:
            print(f"{res.name}: SKIPPED ({res.skip_reason})")
            continue

        status = "OK" if not res.failures else f"FAILED ({len(res.failures)})"
        print(f"{res.name}: {status}")
        for f in res.failures:
            print(f"  [FAIL] {f}")
        for w in res.warnings:
            print(f"  [warn] {w}")

        total_failures += len(res.failures)
        total_warnings += len(res.warnings)

    print("=" * 78)
    if total_failures:
        print(f"RESULT: FAILED - {total_failures} failure(s), {total_warnings} warning(s).")
    else:
        print(f"RESULT: OK - 0 failures, {total_warnings} warning(s).")
    return total_failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-id", required=True, help="Hugging Face dataset repo id (e.g. 'lerobot/pusht')."
    )
    parser.add_argument("--root", default=None, help="Optional local dataset root.")
    parser.add_argument("--revision", default=None, help="Optional git revision (branch, tag, or commit).")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only run metadata sections (skip data scan, videos, and smoke test).",
    )
    parser.add_argument("--no-data", action="store_true", help="Skip the per-data-file scan (Section 5).")
    parser.add_argument(
        "--no-videos", action="store_true", help="Skip the video integrity section (Section 8)."
    )
    parser.add_argument(
        "--no-smoke-test", action="store_true", help="Skip the end-to-end loadability smoke test (Section 9)."
    )
    parser.add_argument(
        "--no-hub", action="store_true", help="Skip the Hugging Face Hub metadata & tags section (Section 10)."
    )
    parser.add_argument(
        "--timestamp-tol",
        type=float,
        default=1e-3,
        help="Tolerance (seconds) for the timestamp == frame_index / fps check.",
    )
    args = parser.parse_args()

    scan_data = not (args.metadata_only or args.no_data)
    check_video = not (args.metadata_only or args.no_videos)
    smoke_test = not (args.metadata_only or args.no_smoke_test)
    check_hub = not args.no_hub

    results = run_all_checks(
        repo_id=args.repo_id,
        root=args.root,
        revision=args.revision,
        scan_data=scan_data,
        check_video=check_video,
        smoke_test=smoke_test,
        check_hub=check_hub,
        fps_tol_s=args.timestamp_tol,
    )
    failures = print_report(results)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

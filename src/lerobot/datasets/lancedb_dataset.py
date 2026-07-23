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
"""Read-only loader for LeRobot datasets stored as LanceDB tables.

Layout of a Lance-backed dataset (video layout):

```
<root>/
  meta/         # standard LeRobot v3.0 metadata (info.json, stats.json, tasks, episodes)
  frames.lance  # one row per frame: tabular features, no pixels
  videos.lance  # one row per source video file: encoded bytes in a blob column
```

Feature keys contain dots (``observation.state``) which Lance column names do not
allow, so columns are stored with dots replaced by underscores and mapped back on
read. Metadata is byte-identical to the standard format and is loaded with
:class:`~lerobot.datasets.dataset_metadata.LeRobotDatasetMetadata`, so stats,
tasks and episode boundaries behave exactly as they do for parquet-backed
datasets.

Reading from the Hub streams the tables directly over ``hf://`` (only ``meta/``
is downloaded locally). Set ``HF_TOKEN`` in the environment for private repos.
"""

from __future__ import annotations

import io
import os
import re
import shutil
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import require_package

from .dataset_metadata import LeRobotDatasetMetadata
from .feature_utils import check_delta_timestamps, get_delta_indices
from .video_utils import FrameTimestampError

FRAMES_TABLE = "frames"
VIDEOS_TABLE = "videos"
VIDEO_BLOB_COLUMN = "video_bytes"
# Byte-index columns on the videos table, written at conversion time by
# ``build_video_byte_index``. They let the remote reader translate a frame
# window into the exact byte ranges its decode needs, fetched for the whole
# batch in one parallel ``fetch_blob_ranges`` call.
#
# Scope of validity:
# - ``kf_indices``/``kf_positions`` (frame index and byte offset of each
#   keyframe) are container/codec agnostic: any video a demuxer can walk has
#   them. Mapping timestamps to frame indices assumes constant frame rate,
#   the same assumption the upstream torchcodec reader makes.
# - ``moov_offset``/``moov_size`` are mp4-specific (the sample-table box a
#   decoder must read before anything else; LeRobot videos are always mp4).
#   For a container without them a converter can store 0/0.
#
# None of this affects correctness: these columns and the constants below
# only decide what gets PREFETCHED. Any byte the decoder needs that was not
# prefetched is served by a fallback ``read_range`` on the blob handle —
# slower (one round trip), never wrong. A dataset with missing or stale
# index data decodes correctly at reduced speed.
VIDEO_INDEX_COLUMNS = ("file_size", "moov_offset", "moov_size", "kf_indices", "kf_positions")
# Prefetch paddings for ffmpeg's I/O pattern. These are not tunables and not
# derivable at runtime (ffmpeg does not expose its read plan); they bound two
# stable, documented behaviors, with the fallback read as the safety net:
# - on open, ffmpeg probes the container head (avio buffer reads over the
#   first ~128 KB on mp4; we prefetch 2x that),
# - after the last requested packet it reads ahead by up to one avio buffer
#   (32 KiB default; we pad each range by 2x that).
# If an ffmpeg upgrade ever reads differently, decodes stay correct and the
# per-source ``fallback_bytes`` counter makes the added round trips visible.
_HEAD_BYTES = 256 * 1024
_RANGE_SLACK = 64 * 1024


def _find_moov(read_at, file_size: int) -> tuple[int, int]:
    """Locate the mp4 ``moov`` box by walking top-level box headers."""
    offset = 0
    while offset < file_size:
        header = read_at(offset, 16)
        box_size = int.from_bytes(header[:4], "big")
        box_type = header[4:8]
        if box_size == 1:
            box_size = int.from_bytes(header[8:16], "big")
        elif box_size == 0:
            box_size = file_size - offset
        if box_type == b"moov":
            return offset, box_size
        offset += box_size
    raise ValueError("no moov box found")


def build_video_byte_index(path: str | Path) -> dict:
    """Compute the byte-index columns for one video file.

    Converters store the returned dict alongside the video's blob row; the
    remote reader uses it to translate frame windows into byte ranges (from
    the keyframe preceding the window to the next keyframe after it).

    Works for any container/codec pyav can demux (keyframe flags and packet
    byte offsets are universal). Frame indices are derived as
    ``round(pts * average_rate)``, i.e. constant frame rate is assumed —
    matching the upstream reader's timestamp-to-index conversion. The moov
    fields are meaningful for mp4 only; see ``VIDEO_INDEX_COLUMNS``.
    """
    import av

    path = Path(path)
    file_size = path.stat().st_size
    kf_entries = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        for packet in container.demux(stream):
            if packet.pts is None or not packet.is_keyframe or packet.pos is None:
                continue
            kf_entries.append((round(float(packet.pts * packet.time_base) * fps), packet.pos))
    kf_entries.sort()
    with open(path, "rb") as f:

        def read_at(offset: int, length: int) -> bytes:
            f.seek(offset)
            return f.read(length)

        moov_offset, moov_size = _find_moov(read_at, file_size)
    return {
        "file_size": file_size,
        "moov_offset": moov_offset,
        "moov_size": moov_size,
        "kf_indices": [index for index, _ in kf_entries],
        "kf_positions": [position for _, position in kf_entries],
    }


class _SparseBlobSource(io.RawIOBase):
    """File-like view over a remote blob backed by prefetched byte ranges.

    Reads inside a prefetched range are served from memory; reads outside
    fall back to one ``read_range`` on the lazy blob handle. The fallback is
    what makes the prefetch heuristics (head bytes, range slack) safe: a
    miss costs one network round trip, never incorrect data.
    ``fallback_bytes`` counts miss traffic — near zero on healthy datasets,
    and the number to check first when remote throughput looks off.
    """

    def __init__(self, size: int, fallback):
        super().__init__()
        self._size = size
        self._fallback = fallback
        self._starts: list[int] = []
        self._chunks: list[bytes] = []
        self._pos = 0
        self.buffered = 0
        self.fallback_bytes = 0

    def add(self, offset: int, data: bytes) -> None:
        import bisect

        index = bisect.bisect_left(self._starts, offset)
        self._starts.insert(index, offset)
        self._chunks.insert(index, data)
        self.buffered += len(data)

    def covers(self, start: int, end: int) -> bool:
        import bisect

        index = bisect.bisect_right(self._starts, start) - 1
        return index >= 0 and self._starts[index] + len(self._chunks[index]) >= end

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self._size + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, buffer) -> int:
        import bisect

        if self._pos >= self._size:
            return 0
        want = min(len(buffer), self._size - self._pos)
        index = bisect.bisect_right(self._starts, self._pos) - 1
        if index >= 0:
            start, chunk = self._starts[index], self._chunks[index]
            inside = self._pos - start
            if inside < len(chunk):
                data = chunk[inside : inside + want]
                buffer[: len(data)] = data
                self._pos += len(data)
                return len(data)
        # Cap the miss at the next buffered range so we never re-fetch bytes
        # we already hold.
        next_index = bisect.bisect_right(self._starts, self._pos)
        if next_index < len(self._starts):
            want = min(want, self._starts[next_index] - self._pos)
        data = self._fallback.read_range(self._pos, want)
        self.fallback_bytes += len(data)
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)


class _VideoDecoderLRU:
    """Per-worker LRU of torchcodec decoders keyed by (video_key, chunk, file).

    Local decoders hold cheap seekable ``BlobFile`` handles (``nbytes=0``).
    Remote decoders hold the materialized video bytes, so eviction is also
    bounded by ``byte_budget`` — with large video files a pure entry-count
    cap would let per-worker RAM grow to ``capacity x file_size``.
    """

    def __init__(self, capacity: int, byte_budget: int | None = None):
        self.capacity = capacity
        self.byte_budget = byte_budget
        self._items: OrderedDict[tuple, tuple[object, int]] = OrderedDict()
        self._total_bytes = 0

    def __contains__(self, key: tuple) -> bool:
        return key in self._items

    def get(self, key: tuple):
        self._items.move_to_end(key)
        return self._items[key][0]

    # Sparse (handle-backed) decoders hold a parsed frame index, not file
    # bytes; they are cheap enough to keep in the hundreds. Only materialized
    # entries (nbytes > 0) count against `capacity` and `byte_budget`.
    SPARSE_CAPACITY = 512

    def put(self, key: tuple, decoder, nbytes: int = 0) -> None:
        if key in self._items:
            self._total_bytes -= self._items[key][1]
        self._items[key] = (decoder, nbytes)
        self._items.move_to_end(key)
        self._total_bytes += nbytes

        def _over_limit() -> bool:
            materialized = sum(1 for _, entry_bytes in self._items.values() if entry_bytes > 0)
            return (
                materialized > self.capacity
                or (self.byte_budget is not None and self._total_bytes > self.byte_budget)
                or len(self._items) > max(self.capacity, self.SPARSE_CAPACITY)
            )

        while len(self._items) > 1 and _over_limit():
            _, (_, evicted_bytes) = self._items.popitem(last=False)
            self._total_bytes -= evicted_bytes


def to_lance_column(key: str) -> str:
    """Map a LeRobot feature key to its Lance column name."""
    return key.replace(".", "_")


def _is_remote_uri(path) -> bool:
    return "://" in str(path)


def _mirror_remote_meta(db_uri: str, local_root: Path) -> None:
    """Copy ``meta/`` from an object-store dataset root to a local cache, once.

    Downloads into a temp directory and renames it into place so an
    interrupted mirror can never be mistaken for a complete one.
    """
    meta_dir = local_root / "meta"
    if meta_dir.exists():
        return
    from pyarrow import fs as pa_fs

    tmp_dir = local_root / f"meta.tmp-{os.getpid()}"
    filesystem, base = pa_fs.FileSystem.from_uri(f"{db_uri}/meta")
    try:
        for info in filesystem.get_file_info(pa_fs.FileSelector(base, recursive=True)):
            if info.type != pa_fs.FileType.File:
                continue
            dst = tmp_dir / info.path[len(base) :].lstrip("/")
            dst.parent.mkdir(parents=True, exist_ok=True)
            with filesystem.open_input_stream(info.path) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
        try:
            tmp_dir.rename(meta_dir)
        except OSError:
            if not meta_dir.exists():  # lost a race to another process is fine
                raise
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def lance_mp_context() -> str:
    """Start method for DataLoader workers reading a Lance-backed dataset.

    ``fork`` is unsafe with Lance's async runtime. ``forkserver`` is preferred
    (workers fork from a clean helper process that never touched the runtime,
    and start faster than with ``spawn``); ``spawn`` is the fallback on
    platforms without it.
    """
    import multiprocessing

    return "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"


@lru_cache(maxsize=32)
def is_lance_dataset(
    repo_id: str | None = None, root: str | Path | None = None, revision: str | None = None
) -> bool:
    """Detect whether a dataset is stored in the Lance layout.

    Checks for a ``frames.lance`` table locally first (under ``root`` or the
    default cache location), then on the Hub with a single targeted API call.
    Does not require lancedb to be installed.
    """
    if root is None and repo_id is None:
        return False
    if root is not None and _is_remote_uri(root):
        # Object-store roots imply the Lance layout; the parquet loader
        # cannot read them at all.
        return True
    local_root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    if (local_root / f"{FRAMES_TABLE}.lance").exists():
        return True
    if repo_id is None:
        return False
    import huggingface_hub

    try:
        paths = huggingface_hub.HfApi().get_paths_info(
            repo_id, [f"{FRAMES_TABLE}.lance"], repo_type="dataset", revision=revision
        )
    except Exception:
        # Unknown repo, no network, auth failure: fall back to the parquet loader,
        # which owns the error reporting for those cases.
        return False
    return len(paths) > 0


class LanceDBDataset(torch.utils.data.Dataset):
    """Map-style dataset over a Lance-backed LeRobot dataset.

    Returns the same item dict as :class:`LeRobotDataset` and satisfies the same
    duck-typed contract (``meta``, ``episodes``, ``absolute_to_relative_idx``,
    ``num_frames``), so it can be consumed by the training pipeline unchanged.

    Implements batched ``__getitems__``: the PyTorch ``DataLoader`` fetcher uses
    it automatically, and all rows a batch needs (including delta-timestamp
    windows) are fetched from the frames table in a single deduplicated read.

    Args:
        repo_id: Hub dataset repository (e.g. ``'lerobot/pusht_lance'``). Tables
            are streamed over ``hf://``; only ``meta/`` is downloaded.
        root: Local directory containing ``meta/`` and the ``.lance`` tables,
            or an object-store URI (``s3://bucket/path``) holding the same
            layout. When both ``repo_id`` and ``root`` are given, local tables
            win if present.
        episodes: Optional episode indices to select. ``None`` means all.
        image_transforms: Optional torchvision v2 transform applied to camera
            frames.
        delta_timestamps: Optional mapping of feature key to relative timestamp
            offsets (seconds) for temporal context windows, as in
            :class:`LeRobotDataset`.
        tolerance_s: Timestamp synchronization tolerance in seconds.
        revision: Hub revision for the ``meta/`` download.
        return_uint8: If True, return RGB frames as raw uint8 tensors instead
            of normalized float32.
        storage_options: Extra options forwarded to ``lancedb.connect`` (e.g.
            object-store credentials).
        video_decoder_cache_size: Max torchcodec decoders kept per worker.
            Defaults to 100 for local tables (decoders hold cheap blob
            handles, matching the upstream decoder cache) and 16 for remote
            tables (each decoder holds the materialized video bytes). A cache
            smaller than the dataset's video file count causes re-fetches;
            for remote tables that means re-downloading whole files.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        return_uint8: bool = False,
        storage_options: dict | None = None,
        video_decoder_cache_size: int | None = None,
    ):
        super().__init__()
        require_package("lancedb", extra="lance")
        if repo_id is None and root is None:
            raise ValueError("Provide `repo_id`, `root`, or both.")

        self.repo_id = repo_id
        self.tolerance_s = tolerance_s
        self._storage_options = storage_options

        if root is not None and _is_remote_uri(root):
            # Object-store root (s3://, gs://, ...): tables are read in place,
            # meta/ is mirrored to a local cache directory once.
            self._db_uri = str(root).rstrip("/")
            self.root = HF_LEROBOT_HOME / "remote" / re.sub(r"[^A-Za-z0-9._-]+", "_", self._db_uri)
            _mirror_remote_meta(self._db_uri, self.root)
        else:
            self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
            if (self.root / f"{FRAMES_TABLE}.lance").exists():
                self._db_uri = str(self.root)
            elif repo_id is not None:
                self._db_uri = f"hf://datasets/{repo_id}"
            else:
                raise FileNotFoundError(f"No '{FRAMES_TABLE}.lance' table under {self.root}.")
        self._is_local = not _is_remote_uri(self._db_uri)

        self.meta = LeRobotDatasetMetadata(
            repo_id if repo_id is not None else str(self.root), root=self.root, revision=revision
        )

        if self.meta.depth_keys:
            raise NotImplementedError("Depth features are not supported by LanceDBDataset yet.")
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self.image_transforms = image_transforms
        self.return_uint8 = return_uint8

        self.episodes = sorted(episodes) if episodes is not None else None
        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.meta.fps)

        # Episode boundaries (absolute frame index space), used to clamp delta
        # windows and compute padding masks without a per-row lookup.
        self._ep_from = np.asarray(self.meta.episodes["dataset_from_index"], dtype=np.int64)
        self._ep_to = np.asarray(self.meta.episodes["dataset_to_index"], dtype=np.int64)

        # Row position in the frames table == absolute frame index. When a
        # subset of episodes is selected, __getitem__ indices are relative and
        # mapped through _rel_to_abs (mirrors DatasetReader's index mapping).
        if self.episodes is not None:
            self._rel_to_abs = np.concatenate(
                [np.arange(self._ep_from[ep], self._ep_to[ep]) for ep in self.episodes]
            )
            self._absolute_to_relative_idx = {
                int(abs_idx): rel_idx for rel_idx, abs_idx in enumerate(self._rel_to_abs)
            }
        else:
            self._rel_to_abs = None
            self._absolute_to_relative_idx = None

        # Tabular features live in the frames table; pixels do not.
        self._tabular_keys = [
            key
            for key in self.meta.features
            if key not in self.meta.video_keys and key not in self.meta.image_keys
        ]
        self._fetch_columns = [to_lance_column(key) for key in self._tabular_keys]
        self._feature_shapes = {
            key: tuple(self.meta.features[key].get("shape") or ()) for key in self._tabular_keys
        }
        # Language/string features pass through as python strings, like the
        # upstream reader; lerobot_collate_fn handles them at batch time.
        self._string_keys = {
            key for key in self._tabular_keys if self.meta.features[key].get("dtype") == "string"
        }

        # Per-episode video file location: which (chunk, file) mp4 holds each
        # episode and where the episode starts inside it (episodes share files
        # in v3.0, so frame timestamps must be shifted by from_timestamp).
        self._video_locator = {
            key: (
                np.asarray(self.meta.episodes[f"videos/{key}/chunk_index"], dtype=np.int64),
                np.asarray(self.meta.episodes[f"videos/{key}/file_index"], dtype=np.int64),
                np.asarray(self.meta.episodes[f"videos/{key}/from_timestamp"], dtype=np.float64),
            )
            for key in self.meta.video_keys
        }

        # Lazily opened per process; see __getstate__.
        self._frames_perm = None
        self._videos_table = None
        self._video_row_ids: dict[tuple, int] | None = None
        self._file_meta: OrderedDict[tuple, dict] = OrderedDict()
        if video_decoder_cache_size is None:
            video_decoder_cache_size = 100 if self._is_local else 16
        # Remote decoders hold materialized video bytes: cap them at 2 GiB per
        # worker so large-file datasets can't multiply into an OOM.
        byte_budget = None if self._is_local else 2 << 30
        self._decoder_cache = _VideoDecoderLRU(video_decoder_cache_size, byte_budget=byte_budget)

    # ── connection management ──────────────────────────────────────────────

    def _ensure_open(self) -> None:
        """Open the frames table handle for this process if needed.

        Handles are opened lazily and dropped on pickling so that each
        DataLoader worker builds its own. Lance's runtime is not fork-safe:
        use ``multiprocessing_context='spawn'`` with ``num_workers > 0``.
        """
        if self._frames_perm is not None:
            return
        import lancedb
        from lancedb.permutation import Permutation

        connect_kwargs = {"storage_options": self._storage_options} if self._storage_options else {}
        db = lancedb.connect(self._db_uri, **connect_kwargs)
        table = db.open_table(FRAMES_TABLE)
        self._frames_perm = (
            Permutation.identity(table).select_columns(self._fetch_columns).with_format("arrow")
        )
        if self.meta.video_keys:
            self._videos_table = db.open_table(VIDEOS_TABLE)
            index = (
                self._videos_table.search()
                .select(["video_key", "chunk_index", "file_index"])
                .with_row_id(True)
                .to_arrow()
            )
            self._video_row_ids = {
                (row["video_key"], row["chunk_index"], row["file_index"]): row["_rowid"]
                for row in index.to_pylist()
            }

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_frames_perm"] = None
        state["_videos_table"] = None
        state["_video_row_ids"] = None
        state["_file_meta"] = OrderedDict()
        state["_decoder_cache"] = _VideoDecoderLRU(
            self._decoder_cache.capacity, byte_budget=self._decoder_cache.byte_budget
        )
        return state

    # ── dataset protocol ───────────────────────────────────────────────────

    @property
    def num_frames(self) -> int:
        return len(self._rel_to_abs) if self._rel_to_abs is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        return self._absolute_to_relative_idx

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def fps(self) -> int:
        return self.meta.fps

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """Batched fetch: one deduplicated frames-table read and one blob fetch per batch."""
        self._ensure_open()
        plans = self._plan_batch(indices)
        rows = sorted({row for plan in plans for row in plan["rows"]})
        row_pos = {row: pos for pos, row in enumerate(rows)}
        columns = self._fetch_rows(rows)
        items = [self._build_item(plan, columns, row_pos) for plan in plans]

        if self.meta.video_keys:
            for item, frames in zip(items, self._decode_videos(plans, columns, row_pos), strict=True):
                item.update(frames)
        if self.image_transforms is not None:
            for item in items:
                for cam_key in self.meta.camera_keys:
                    item[cam_key] = self.image_transforms(item[cam_key])
        return items

    # ── internals ──────────────────────────────────────────────────────────

    def _plan_batch(self, indices: list[int]) -> list[dict]:
        """Resolve each sample to the absolute rows it needs and its padding masks."""
        plans = []
        for idx in indices:
            abs_idx = int(self._rel_to_abs[idx]) if self._rel_to_abs is not None else int(idx)
            ep_idx = int(np.searchsorted(self._ep_from, abs_idx, side="right") - 1)
            start, end = int(self._ep_from[ep_idx]), int(self._ep_to[ep_idx])
            plan = {"abs_idx": abs_idx, "ep_idx": ep_idx, "rows": {abs_idx}, "windows": {}, "padding": {}}
            if self.delta_indices is not None:
                for key, deltas in self.delta_indices.items():
                    window = [min(max(abs_idx + delta, start), end - 1) for delta in deltas]
                    plan["windows"][key] = window
                    plan["rows"].update(window)
                    plan["padding"][f"{key}_is_pad"] = torch.BoolTensor(
                        [not (start <= abs_idx + delta < end) for delta in deltas]
                    )
            plans.append(plan)
        return plans

    def _fetch_rows(self, rows: list[int]) -> dict[str, np.ndarray]:
        """Fetch rows from the frames table and decode columns to numpy.

        Vector features come back as 2-D arrays ``(n_rows, dim)``; scalars as 1-D.
        """
        batch = self._frames_perm.__getitems__(rows)
        columns = {}
        for key, lance_name in zip(self._tabular_keys, self._fetch_columns, strict=True):
            array = batch.column(lance_name)
            if hasattr(array, "combine_chunks"):
                array = array.combine_chunks()
            if hasattr(array, "flatten") and hasattr(array.type, "value_type"):
                values = array.flatten().to_numpy(zero_copy_only=False)
                columns[key] = values.reshape(len(array), -1)
            else:
                columns[key] = array.to_numpy(zero_copy_only=False)
        return columns

    def _decode_videos(
        self, plans: list[dict], columns: dict[str, np.ndarray], row_pos: dict[int, int]
    ) -> list[dict[str, torch.Tensor]]:
        """Decode all camera frames a batch needs, one blob fetch and one decode pass per video file.

        Requested frames are grouped by the mp4 file that contains them; files
        missing from the decoder cache are fetched with a single
        ``fetch_blob_files`` call, and each decoder serves every frame the batch
        needs from its file in one ``get_frames_at`` call.
        """
        timestamps = columns["timestamp"]
        requests: dict[tuple, list[tuple[int, list[float]]]] = defaultdict(list)
        for sample_idx, plan in enumerate(plans):
            ep_idx = plan["ep_idx"]
            for key in self.meta.video_keys:
                window = plan["windows"].get(key, [plan["abs_idx"]])
                chunk_arr, file_arr, from_ts_arr = self._video_locator[key]
                shifted_ts = [float(from_ts_arr[ep_idx]) + float(timestamps[row_pos[row]]) for row in window]
                requests[(key, int(chunk_arr[ep_idx]), int(file_arr[ep_idx]))].append(
                    (sample_idx, shifted_ts)
                )

        if self._is_local:
            decoders = self._ensure_decoders_local(requests)
        else:
            decoders = self._ensure_decoders_remote(requests)

        results: list[dict[str, torch.Tensor]] = [{} for _ in plans]

        def _decode_file(file_key: tuple, file_requests: list[tuple[int, list[float]]]) -> None:
            key, chunk_idx, file_idx = file_key
            decoder = decoders[file_key]
            fps = decoder.metadata.average_fps
            # One decode call per sample window: window frames are consecutive,
            # so each call is a single seek plus a sequential decode. This
            # measures faster than merging all windows into one indices list.
            for sample_idx, shifted_ts in file_requests:
                indices = [round(ts * fps) for ts in shifted_ts]
                batch = decoder.get_frames_at(indices=indices)
                # float64: default float32 quantizes ~1e-3 s at t~10^4 s in
                # long aggregated video files, tripping tolerance_s spuriously.
                distance = (torch.tensor(shifted_ts, dtype=torch.float64) - batch.pts_seconds).abs()
                if (distance >= self.tolerance_s).any():
                    raise FrameTimestampError(
                        f"Query timestamps violate tolerance_s={self.tolerance_s} for video "
                        f"'{key}' (chunk {chunk_idx}, file {file_idx}): queried {shifted_ts}, "
                        f"loaded {batch.pts_seconds.tolist()}."
                    )
                frames = batch.data
                if not self.return_uint8:
                    frames = (frames / 255.0).type(torch.float32)
                results[sample_idx][key] = frames.squeeze(0)

        # Decode files in parallel (decoding releases the GIL), mirroring the
        # per-camera thread pool in DatasetReader._query_videos.
        if len(requests) <= 1:
            for file_key, file_requests in requests.items():
                _decode_file(file_key, file_requests)
        else:
            with ThreadPoolExecutor(max_workers=len(requests)) as pool:
                futures = [pool.submit(_decode_file, k, r) for k, r in requests.items()]
                for future in futures:
                    future.result()
        return results

    def _ensure_decoders_local(self, requests: dict) -> dict:
        """Local tables: decoders read straight off seekable blob handles."""
        from torchcodec.decoders import VideoDecoder

        decoders = {}
        for key in requests:
            if key in self._decoder_cache:
                decoders[key], _ = self._decoder_cache.get(key)
        missing = [key for key in requests if key not in decoders]
        if missing:
            blob_files = self._videos_table.fetch_blob_files(
                VIDEO_BLOB_COLUMN, [self._video_row_ids[key] for key in missing]
            )
            for key, blob_file in zip(missing, blob_files, strict=True):
                # approximate seek mode skips the full-file scan that exact
                # mode performs on decoder creation.
                decoder = VideoDecoder(blob_file, seek_mode="approximate")
                decoders[key] = decoder
                self._decoder_cache.put(key, (decoder, None))
        return decoders

    def _ensure_decoders_remote(self, requests: dict) -> dict:
        """Remote tables: fetch every byte the batch needs in one parallel call.

        Uses the byte-index columns to translate each frame window into a
        keyframe-aligned byte range, batches all ranges (plus container
        head/moov for new files) into a single ``fetch_blob_ranges`` request,
        and feeds decoders through sparse in-memory sources.
        """
        from torchcodec.decoders import VideoDecoder

        self._load_file_meta([key for key in requests if key not in self._file_meta])

        decoders: dict[tuple, object] = {}
        sources: dict[tuple, _SparseBlobSource] = {}
        new_files = []
        for key in requests:
            if key in self._decoder_cache:
                decoders[key], sources[key] = self._decoder_cache.get(key)
            else:
                new_files.append(key)

        range_requests: list[tuple[int, int, int]] = []
        range_targets: list[tuple[tuple, int]] = []  # (file_key, offset)

        def request_range(file_key: tuple, start: int, end: int) -> None:
            source = sources.get(file_key)
            if source is not None and source.covers(start, end):
                return
            range_requests.append((self._video_row_ids[file_key], start, end - start))
            range_targets.append((file_key, start))

        for key in new_files:
            meta = self._file_meta[key]
            request_range(key, 0, min(_HEAD_BYTES, meta["file_size"]))
            request_range(key, meta["moov_offset"], meta["moov_offset"] + meta["moov_size"])
        fps = float(self.meta.fps)
        for key, file_requests in requests.items():
            meta = self._file_meta[key]
            for _, shifted_ts in file_requests:
                first = round(shifted_ts[0] * fps)
                last = round(shifted_ts[-1] * fps)
                start, end = self._window_byte_range(meta, min(first, last), max(first, last))
                request_range(key, start, end)

        if range_requests:
            payloads = self._videos_table.fetch_blob_ranges(VIDEO_BLOB_COLUMN, range_requests)
            fallback_handles: dict[tuple, object] = {}
            if new_files:
                handles = self._videos_table.fetch_blob_files(
                    VIDEO_BLOB_COLUMN, [self._video_row_ids[key] for key in new_files]
                )
                fallback_handles = dict(zip(new_files, handles, strict=True))
            for (file_key, offset), payload in zip(range_targets, payloads, strict=True):
                if file_key not in sources:
                    sources[file_key] = _SparseBlobSource(
                        self._file_meta[file_key]["file_size"], fallback_handles[file_key]
                    )
                sources[file_key].add(offset, payload.as_py())

        for key in new_files:
            decoder = VideoDecoder(sources[key], seek_mode="approximate")
            decoders[key] = decoder
        for key in requests:
            source = sources[key]
            # A cached entry costs its buffered bytes plus the decoder's
            # ffmpeg context, which scales with resolution. Re-accounted every
            # batch as sources grow, so the cache's byte budget alone bounds
            # memory: an oversized entry is simply evicted and rebuilt from
            # two extra prefetch ranges on its next touch.
            height = decoders[key].metadata.height or 0
            width = decoders[key].metadata.width or 0
            context_cost = (8 << 20) + 8 * height * width
            self._decoder_cache.put(key, (decoders[key], source), nbytes=source.buffered + context_cost)
        return decoders

    def _load_file_meta(self, missing: list[tuple]) -> None:
        """Fetch byte-index columns for files not yet in the per-worker cache."""
        if not missing:
            return
        clauses = " OR ".join(
            f"(video_key = '{key}' AND chunk_index = {chunk} AND file_index = {file})"
            for key, chunk, file in missing
        )
        rows = (
            self._videos_table.search()
            .where(clauses)
            .select(["video_key", "chunk_index", "file_index", *VIDEO_INDEX_COLUMNS])
            .to_arrow()
            .to_pylist()
        )
        if len(rows) < len(missing):
            raise ValueError(
                "videos table is missing byte-index columns or rows "
                f"({len(rows)} matches for {len(missing)} files). Re-convert the dataset "
                f"with a converter that writes {VIDEO_INDEX_COLUMNS}."
            )
        for row in rows:
            file_key = (row["video_key"], row["chunk_index"], row["file_index"])
            self._file_meta[file_key] = {
                "file_size": row["file_size"],
                "moov_offset": row["moov_offset"],
                "moov_size": row["moov_size"],
                "kf_indices": np.asarray(row["kf_indices"], dtype=np.int64),
                "kf_positions": np.asarray(row["kf_positions"], dtype=np.int64),
            }
            self._file_meta.move_to_end(file_key)
        while len(self._file_meta) > 512:
            self._file_meta.popitem(last=False)

    @staticmethod
    def _window_byte_range(meta: dict, first_frame: int, last_frame: int) -> tuple[int, int]:
        """Byte range covering frames [first, last]: preceding keyframe to next keyframe."""
        kf_indices, kf_positions = meta["kf_indices"], meta["kf_positions"]
        start_idx = max(int(np.searchsorted(kf_indices, first_frame, side="right")) - 1, 0)
        end_idx = int(np.searchsorted(kf_indices, last_frame, side="right"))
        end = int(kf_positions[end_idx]) if end_idx < len(kf_positions) else meta["file_size"]
        return int(kf_positions[start_idx]), min(end + _RANGE_SLACK, meta["file_size"])

    def _build_item(self, plan: dict, columns: dict[str, np.ndarray], row_pos: dict[int, int]) -> dict:
        base = row_pos[plan["abs_idx"]]
        item = {}
        for key in self._tabular_keys:
            data = columns[key]
            shape = self._feature_shapes[key]
            if key in self._string_keys:
                value = data[base]
                item[key] = value if isinstance(value, str) or value is None else str(value)
            elif key in plan["windows"]:
                window = data[[row_pos[row] for row in plan["windows"][key]]]
                if len(shape) > 1:
                    window = window.reshape(len(window), *shape)
                item[key] = torch.from_numpy(np.ascontiguousarray(window))
            elif data.ndim > 1:
                value = data[base]
                if len(shape) > 1:
                    value = value.reshape(shape)
                item[key] = torch.from_numpy(value.copy())
            else:
                item[key] = torch.tensor(data[base])
        item.update(plan["padding"])
        item["task"] = self.meta.tasks.iloc[int(item["task_index"].item())].name
        return item

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  repo_id={self.repo_id},\n"
            f"  uri={self._db_uri},\n"
            f"  episodes={self.num_episodes} selected / {self.meta.total_episodes} total,\n"
            f"  frames={self.num_frames},\n"
            f")"
        )

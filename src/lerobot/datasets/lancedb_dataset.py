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
  meta/             # standard LeRobot v3.0 metadata (info.json, stats.json, tasks, episodes)
  frames.lance      # one row per frame: tabular features, no pixels
  videos.lance      # one row per source video file: encoded bytes in a blob v2
                    # column + byte-index columns (see VIDEO_INDEX_COLUMNS)
  meta.lance        # one row per meta/ file (path, bytes): the transport for
                    # object-store roots, materialized to a local meta/ cache
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

import bisect
import io
import json
import multiprocessing
import os
import re
import shutil
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from lerobot.configs.video import DEFAULT_DEPTH_UNIT, DepthEncoderConfig
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import _lancedb_available, require_package

if TYPE_CHECKING or _lancedb_available:
    import lancedb
    from lancedb.permutation import Permutation

from .dataset_metadata import LeRobotDatasetMetadata
from .depth_utils import dequantize_depth
from .feature_utils import check_delta_timestamps, get_delta_indices
from .video_utils import FrameTimestampError, decode_video_frames_pyav

FRAMES_TABLE = "frames"
VIDEOS_TABLE = "videos"
META_TABLE = "meta"
VIDEO_BLOB_COLUMN = "video_bytes"
# Byte-index columns on the videos table, written at conversion time by
# ``build_video_byte_index`` (defined below; converters import it as the
# schema contract). They map a frame window to the byte ranges its decode
# needs, so a whole batch's video bytes travel in one ``fetch_blob_ranges``
# call. The keyframe columns work for any container/codec and assume
# constant frame rate (upstream's reader assumes the same); the moov columns
# are mp4-specific (store 0/0 for other containers).
VIDEO_INDEX_COLUMNS = ("file_size", "moov_offset", "moov_size", "kf_indices", "kf_positions")
# Why padding exists: ffmpeg reads more bytes than the frames requested.
# At open it probes the container head; after parsing the moov it reads the
# first media packets; past the last requested packet it reads ahead one
# avio buffer. Any of those bytes missing from the prefetched ranges is
# served by a fallback ``read_range`` on the blob handle — always correct,
# but each miss is a blocking round trip on object storage (one 16-byte
# miss per file once cost a third of droid's whole batch time). So each
# prefetched range is padded to cover ffmpeg's known reads, at 2-4x the
# measured sizes since they are stable behaviors, not tunables:
#   _HEAD_BYTES      open-time head probe (~128 KB seen; keep 2x)
#   _OPEN_READAHEAD  first-packet priming read (~64 KB seen; keep 4x)
#   _RANGE_SLACK     per-range readahead (one 32 KB avio buffer; keep 2x)
# If an ffmpeg upgrade reads differently, decodes stay correct and the
# per-source ``fallback_bytes`` counter makes the extra round trips visible.
_HEAD_BYTES = 256 * 1024
_OPEN_READAHEAD = 256 * 1024
_RANGE_SLACK = 64 * 1024


def _merge_spans(spans: list[tuple[int, int]], gap: int = _RANGE_SLACK) -> list[tuple[int, int]]:
    """Coalesce overlapping or nearby byte ranges into fewer, larger requests.
    Batch plans overlap constantly: a faststart file's head/moov/first-packet
    prefetch trio collapses to one request
    """
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


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
    """The adapter between lance's batched fetches and the decoders' file API.

    lance serves video bytes as BATCHED parallel range fetches (one
    ``fetch_blob_ranges`` wave per DataLoader batch — the fast path);
    torchcodec and pyav consume a SEEKABLE FILE they read incrementally in
    whatever order ffmpeg pleases. This class is the bridge: it buffers the
    wave's ranges (kept disjoint by merge-on-add) and presents them as a
    file. Without it, decoders would sit directly on the lazy blob handle,
    turning every ffmpeg read into an individual round trip through the
    Python sync bridge — the serialized pre-fetch_blob_ranges architecture
    (droid S3: 10-14 samples/s vs 130+ through this class).

    It is also the safety boundary that makes every prefetch heuristic in
    this module (head bytes, range slack, window hints) an OPTIMIZATION
    rather than a correctness requirement: a read outside the buffered
    ranges falls back to one ``read_range`` on the handle — one round trip,
    never incorrect data. ``fallback_bytes`` counts that miss traffic; it
    is ~0 on healthy datasets and the first number to check when remote
    throughput looks off.

    If lancedb ever ships a natively prefetchable BlobFile (buffered
    ``prefetch(ranges)`` with the same fallback semantics), this class
    collapses into it and gets deleted.
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
        """Buffer a fetched range, merging any chunks it overlaps or touches.

        Ranges from different fetch waves overlap routinely (a later wave's
        merged span can straddle an earlier chunk). Keeping chunks disjoint
        is what lets ``covers`` and ``readinto`` check only the single chunk
        preceding a position; with overlapping chunks that lookup lands on
        the shorter chunk and misreports covered bytes as missing, costing a
        spurious fallback round trip per touch.
        """
        end = offset + len(data)
        lo = bisect.bisect_left(self._starts, offset)
        if lo > 0 and self._starts[lo - 1] + len(self._chunks[lo - 1]) >= offset:
            lo -= 1
        hi = bisect.bisect_right(self._starts, end)
        if lo == hi:
            self._starts.insert(lo, offset)
            self._chunks.insert(lo, data)
        else:
            merged_start = min(offset, self._starts[lo])
            merged_end = max(end, max(self._starts[i] + len(self._chunks[i]) for i in range(lo, hi)))
            merged = bytearray(merged_end - merged_start)
            for i in range(lo, hi):
                at = self._starts[i] - merged_start
                merged[at : at + len(self._chunks[i])] = self._chunks[i]
            merged[offset - merged_start : offset - merged_start + len(data)] = data
            del self._starts[lo:hi]
            del self._chunks[lo:hi]
            self._starts.insert(lo, merged_start)
            self._chunks.insert(lo, bytes(merged))
        self.buffered = sum(len(chunk) for chunk in self._chunks)

    def covers(self, start: int, end: int) -> bool:
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

    Local decoders hold cheap seekable ``BlobFile`` handles and cost ~nothing
    (``nbytes=0``). Remote decoders hold prefetched byte ranges plus an ffmpeg
    context, so eviction is also bounded by ``byte_budget`` — a pure
    entry-count cap would let per-worker RAM scale with resolution and
    window accumulation.
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

    def put(self, key: tuple, decoder, nbytes: int = 0) -> None:
        if key in self._items:
            self._total_bytes -= self._items[key][1]
        self._items[key] = (decoder, nbytes)
        self._items.move_to_end(key)
        self._total_bytes += nbytes
        while len(self._items) > 1 and (
            len(self._items) > self.capacity
            or (self.byte_budget is not None and self._total_bytes > self.byte_budget)
        ):
            _, (_, evicted_bytes) = self._items.popitem(last=False)
            self._total_bytes -= evicted_bytes


def to_lance_column(key: str) -> str:
    """Map a LeRobot feature key to its Lance column name."""
    return key.replace(".", "_")


def _is_remote_uri(path) -> bool:
    return "://" in str(path)


def _connect(db_uri: str, storage_options: dict | None):
    options = dict(storage_options or {})
    if _is_remote_uri(db_uri):
        os.environ.setdefault("LANCE_IO_THREADS", "256")
    if db_uri.startswith("hf://") and "token" not in options:
        from huggingface_hub import get_token

        token = get_token()
        if token:
            options["token"] = token
    return lancedb.connect(db_uri, **({"storage_options": options} if options else {}))


def _materialize_meta(db, local_root: Path) -> None:
    """Write ``meta/`` from the dataset's meta table to a local cache, once.

    The meta table is the transport, not the source of truth for consumers:
    files are materialized byte-identical so ``LeRobotDatasetMetadata`` (and
    any other tool expecting the standard layout) reads them unchanged.
    Writes into a temp directory and renames it into place so an interrupted
    materialization can never be mistaken for a complete one.
    """
    meta_dir = local_root / "meta"
    if meta_dir.exists():
        return
    try:
        table = db.open_table(META_TABLE)
    except Exception as error:
        raise FileNotFoundError(
            f"Dataset has no '{META_TABLE}' table. Re-convert it with a converter that "
            "ingests meta/, or create the table in place from an existing meta/ copy."
        ) from error

    tmp_dir = local_root / f"meta.tmp-{os.getpid()}"
    try:
        # Stream row groups: meta/ is usually small, but per-episode stats
        # reach hundreds of MB at droid scale (76k episodes = 566 MB).
        for batch in table.search().select(["path", "data"]).to_batches():
            paths = batch.column("path").to_pylist()
            for i, rel_path in enumerate(paths):
                dst = tmp_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(batch.column("data")[i].as_py())
        try:
            tmp_dir.rename(meta_dir)
        except OSError:
            if not meta_dir.exists():  # lost a race to another process is fine
                raise
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def lance_mp_context() -> str:
    return "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"


@lru_cache(maxsize=32)
def is_lance_dataset(
    repo_id: str | None = None, root: str | Path | None = None, revision: str | None = None
) -> bool:
    if root is None and repo_id is None:
        return False
    if root is not None and _is_remote_uri(root):
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
        video_decoder_cache_size: Max torchcodec decoders kept per worker
            (default 16; each holds prefetched byte ranges and an ffmpeg
            context, additionally bounded by a 2 GiB per-worker byte budget).
            An evicted entry re-prefetches its container header ranges
            (~1 MB) on the next touch.
        depth_output_unit: Physical unit depth features are dequantized to
            (``'mm'`` or ``'m'``), as in :class:`LeRobotDataset`. Depth videos
            decode through pyav (their 16-bit planes are not representable in
            torchcodec's RGB output) over the same prefetched byte ranges.
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
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
    ):
        super().__init__()
        require_package("lancedb", extra="lance")
        if repo_id is None and root is None:
            raise ValueError("Provide `repo_id`, `root`, or both.")

        self.repo_id = repo_id
        self.tolerance_s = tolerance_s
        self._storage_options = storage_options

        if root is not None and _is_remote_uri(root):
            # Object-store root (s3://, gs://, ...): tables are read in
            # place. meta/ is materialized ONCE from the meta table into a
            # local cache dir, because LeRobotDatasetMetadata (upstream's
            # class, kept unforked on purpose) reads files from a directory.
            # The table is the transport; the files are its API.
            self._db_uri = str(root).rstrip("/")
            self.root = HF_LEROBOT_HOME / "remote" / re.sub(r"[^A-Za-z0-9._-]+", "_", self._db_uri)
            if not (self.root / "meta").exists():
                _materialize_meta(_connect(self._db_uri, self._storage_options), self.root)
        else:
            self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
            if (self.root / f"{FRAMES_TABLE}.lance").exists():
                self._db_uri = str(self.root)
            elif repo_id is not None:
                self._db_uri = f"hf://datasets/{repo_id}"
            else:
                raise FileNotFoundError(f"No '{FRAMES_TABLE}.lance' table under {self.root}.")

        self.meta = LeRobotDatasetMetadata(
            repo_id if repo_id is not None else str(self.root), root=self.root, revision=revision
        )

        if set(self.meta.depth_keys) & set(self.meta.image_keys):
            raise NotImplementedError(
                "Depth stored as raw images is not supported by LanceDBDataset; "
                "re-encode depth as video (the standard v3.0 recording path)."
            )
        # Depth videos carry 16-bit planes torchcodec cannot emit, so depth
        # keys decode through pyav over the same prefetched sources.
        self._depth_output_unit = depth_output_unit
        self._depth_encoder_configs = {
            key: DepthEncoderConfig.from_video_info(self.meta.features[key].get("info"))
            for key in self.meta.depth_keys
        }
        # The video path fetches keyframe-aligned byte ranges in one batched call
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
        # Integrity: episode ranges must tile [0, total_frames) exactly.
        # Public datasets ship broken meta more often than you'd hope (droid:
        # 44% orphan rows; berkeley: truncated tails; agibot: empty episodes).
        if len(self._ep_from) and (
            int(self._ep_from[0]) != 0
            or int(self._ep_to[-1]) != self.meta.total_frames
            or (self._ep_from[1:] != self._ep_to[:-1]).any()
        ):
            raise ValueError(
                "Episode boundaries in meta do not tile [0, total_frames) contiguously; "
                "the dataset metadata is inconsistent."
            )

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

        self._task_names = list(self.meta.tasks.index)

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
        # String features pass through as python strings and language columns
        # (list<struct> message rows, lerobot#3467) as python lists of dicts,
        # like the upstream reader; lerobot_collate_fn handles both at batch time.
        self._string_keys = {
            key for key in self._tabular_keys if self.meta.features[key].get("dtype") == "string"
        }
        self._language_keys = {
            key for key in self._tabular_keys if self.meta.features[key].get("dtype") == "language"
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
        self._prefetch_pool: ThreadPoolExecutor | None = None
        self._decode_pool: ThreadPoolExecutor | None = None
        if video_decoder_cache_size is None:
            video_decoder_cache_size = 16
        # Decoders hold buffered video bytes: cap them at 2 GiB per worker so
        # large-file datasets can't multiply into an OOM.
        self._decoder_cache = _VideoDecoderLRU(video_decoder_cache_size, byte_budget=2 << 30)

    def _ensure_open(self) -> None:
        """Open the frames table handle for this process if needed.

        Handles are opened lazily and dropped on pickling so that each
        DataLoader worker builds its own. Lance's runtime is not fork-safe:
        pass ``multiprocessing_context=lance_mp_context()`` (forkserver, or
        spawn where unavailable) to the DataLoader when ``num_workers > 0``.
        """
        if self._frames_perm is not None:
            return
        if self.meta.video_keys:
            self._prefetch_pool = ThreadPoolExecutor(max_workers=1)
            # One persistent pool per worker for decoder creation and frame
            # decoding; 16 matches the old per-batch decoder-creation cap.
            self._decode_pool = ThreadPoolExecutor(max_workers=16)
        db = _connect(self._db_uri, self._storage_options)
        table = db.open_table(FRAMES_TABLE)
        # Integrity: row position == absolute frame index requires exactly
        # total_frames rows. Catches truncated or over-appended tables at
        # open instead of as silently wrong samples.
        n_rows = table.count_rows()
        if n_rows != self.meta.total_frames:
            raise ValueError(
                f"frames table has {n_rows} rows but meta declares "
                f"{self.meta.total_frames} frames; the dataset is truncated or corrupt."
            )
        self._frames_perm = (
            Permutation.identity(table).select_columns(self._fetch_columns).with_format("arrow")
        )
        if self.meta.video_keys:
            self._videos_table = db.open_table(VIDEOS_TABLE)
            # Future improvement for datasets with millions
            # of files: resolve row ids lazily per batch instead of upfront.
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
            # fail fast in case of missing episodes
            referenced = {
                (key, int(chunk), int(file))
                for key, (chunks, files, _) in self._video_locator.items()
                for chunk, file in zip(chunks, files, strict=True)
            }
            missing = referenced - self._video_row_ids.keys()
            if missing:
                raise ValueError(
                    f"videos table is missing {len(missing)} file(s) referenced by episode "
                    f"metadata, e.g. {sorted(missing)[:3]}. The dataset is incomplete or "
                    "was converted against different metadata."
                )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_frames_perm"] = None
        state["_videos_table"] = None
        state["_video_row_ids"] = None
        state["_file_meta"] = OrderedDict()
        state["_prefetch_pool"] = None
        state["_decode_pool"] = None
        state["_decoder_cache"] = _VideoDecoderLRU(
            self._decoder_cache.capacity, byte_budget=self._decoder_cache.byte_budget
        )
        return state

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
        rows, row_pos = self._batch_rows(plans)

        # Video prep (byte-index metadata, header ranges, decoder creation,
        # frame-window byte ranges) needs only the batch's file set and frame
        # indices, so it overlaps with the frames-table fetch. Frame positions
        # inside a file follow from episode boundaries under the constant
        # frame rate the byte index already assumes; the timestamp-based
        # coverage check in _ensure_decoders remains the safety net, so any
        # drift costs at most a re-fetch, never a wrong frame.
        prepared_future = None
        if self.meta.video_keys:
            windows = self._plan_file_windows(plans)
            prepared_future = self._prefetch_pool.submit(self._prepare_files, sorted(windows), windows)

        columns = self._fetch_rows(rows)
        items = [self._build_item(plan, columns, row_pos) for plan in plans]

        if self.meta.video_keys:
            decoded = self._decode_videos(plans, columns, row_pos, prepared_future.result())
            for item, frames in zip(items, decoded, strict=True):
                item.update(frames)
        if self.image_transforms is not None:
            for item in items:
                for cam_key in self.meta.camera_keys:
                    if cam_key in self.meta.depth_keys:
                        continue
                    item[cam_key] = self.image_transforms(item[cam_key])
        return items

    # ── internals ──────────────────────────────────────────────────────────

    def _plan_file_windows(self, plans: list[dict]) -> dict[tuple, list[tuple[int, int]]]:
        """Map each batch sample's video windows to (file key -> frame spans).

        Positions come from episode metadata alone (file-local frame =
        absolute frame - episode start + round(from_timestamp * fps)), no
        timestamps needed. stage 2 re-derives ranges from real timestamps and
        fetches anything missed.
        """
        fps = float(self.meta.fps)
        windows: dict[tuple, list[tuple[int, int]]] = {}
        for plan in plans:
            ep_idx = plan["ep_idx"]
            for key in self.meta.video_keys:
                file_key, span = self._planned_file_window(key, ep_idx, plan, fps)
                windows.setdefault(file_key, []).append(span)
        return windows

    def _plan_batch(self, indices: list[int]) -> list[dict]:
        """Resolve each sample to the absolute rows it needs and its padding masks."""
        plans = []
        for idx in indices:
            abs_idx = self._resolve_abs_idx(idx)
            ep_idx = self._episode_index_for_abs_idx(abs_idx)
            start, end = self._episode_bounds(ep_idx)
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

    def _resolve_abs_idx(self, idx: int) -> int:
        return int(self._rel_to_abs[idx]) if self._rel_to_abs is not None else int(idx)

    def _episode_index_for_abs_idx(self, abs_idx: int) -> int:
        return int(np.searchsorted(self._ep_from, abs_idx, side="right") - 1)

    def _episode_bounds(self, ep_idx: int) -> tuple[int, int]:
        return int(self._ep_from[ep_idx]), int(self._ep_to[ep_idx])

    def _batch_rows(self, plans: list[dict]) -> tuple[list[int], dict[int, int]]:
        rows = sorted({row for plan in plans for row in plan["rows"]})
        return rows, {row: pos for pos, row in enumerate(rows)}

    def _planned_file_window(
        self, key: str, ep_idx: int, plan: dict, fps: float
    ) -> tuple[tuple[str, int, int], tuple[int, int]]:
        ep_start, _ = self._episode_bounds(ep_idx)
        _, _, from_ts_arr = self._video_locator[key]
        window = plan["windows"].get(key, [plan["abs_idx"]])
        base = round(float(from_ts_arr[ep_idx]) * fps) - ep_start
        first = base + min(window) - (1 if key in self.meta.depth_keys else 0)
        return self._video_file_key(key, ep_idx), (first, base + max(window))

    def _fetch_rows(self, rows: list[int]) -> dict[str, np.ndarray]:
        """Fetch rows from the frames table and decode columns to numpy.

        Vector features come back as 2-D arrays ``(n_rows, dim)``; scalars as 1-D.
        Language columns stay python (one list of message dicts per row).
        """
        batch = self._frames_perm.__getitems__(rows)
        columns = {}
        for key, lance_name in zip(self._tabular_keys, self._fetch_columns, strict=True):
            array = batch.column(lance_name)
            if hasattr(array, "combine_chunks"):
                array = array.combine_chunks()
            if key in self._language_keys:
                # Match upstream's datasets.Json() feature: tool_calls entries
                # are stored as JSON text but surfaced as python objects.
                rows = array.to_pylist()
                for row in rows:
                    for msg in row or ():
                        if msg.get("tool_calls"):
                            msg["tool_calls"] = [
                                json.loads(call) if isinstance(call, str) else call
                                for call in msg["tool_calls"]
                            ]
                columns[key] = rows
            elif hasattr(array, "flatten") and hasattr(array.type, "value_type"):
                values = array.flatten().to_numpy(zero_copy_only=False)
                columns[key] = values.reshape(len(array), -1)
            else:
                columns[key] = array.to_numpy(zero_copy_only=False)
        return columns

    def _decode_videos(
        self,
        plans: list[dict],
        columns: dict[str, np.ndarray],
        row_pos: dict[int, int],
        prepared: dict[tuple, tuple],
    ) -> list[dict[str, torch.Tensor]]:
        """Decode all camera frames a batch needs, one blob fetch and one decode pass per video file.

        Requested frames are grouped by the mp4 file that contains them; files
        missing from the decoder cache are fetched with a single
        ``fetch_blob_files`` call, and each decoder serves every frame the batch
        needs from its file in one ``get_frames_at`` call.
        """
        requests = self._build_video_requests(plans, columns["timestamp"], row_pos)
        entries = self._ensure_decoders(requests, prepared)

        results: list[dict[str, torch.Tensor]] = [{} for _ in plans]

        def _decode_file(file_key: tuple, file_requests: list[tuple[int, list[float]]]) -> None:
            key, chunk_idx, file_idx = file_key
            decoder, source = entries[file_key]
            if key in self.meta.depth_keys:
                for sample_idx, shifted_ts in file_requests:
                    results[sample_idx][key] = self._decode_depth_window(source, shifted_ts, file_key)
                return
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
        futures = [self._decode_pool.submit(_decode_file, k, r) for k, r in requests.items()]
        for future in futures:
            future.result()
        return results

    def _build_video_requests(
        self,
        plans: list[dict],
        timestamps: np.ndarray,
        row_pos: dict[int, int],
    ) -> dict[tuple, list[tuple[int, list[float]]]]:
        requests: dict[tuple, list[tuple[int, list[float]]]] = defaultdict(list)
        for sample_idx, plan in enumerate(plans):
            ep_idx = plan["ep_idx"]
            for key in self.meta.video_keys:
                window = plan["windows"].get(key, [plan["abs_idx"]])
                requests[self._video_file_key(key, ep_idx)].append(
                    (sample_idx, self._shifted_video_timestamps(key, ep_idx, window, timestamps, row_pos))
                )
        return requests

    def _prepare_files(
        self, file_keys: list[tuple], windows: dict[tuple, list[tuple[int, int]]] | None = None
    ) -> dict[tuple, tuple]:
        """Stage 1 of video decoding: everything that doesn't need timestamps.

        Loads byte-index metadata, fetches container head/moov/first-packet
        ranges for uncached files in one parallel call, creates their
        decoders, and prefetches the batch's frame-window byte ranges
        (``windows`` maps file key to (first, last) frame pairs).
        Timestamp-independent, so ``__getitems__`` runs it in a background
        thread concurrently with the frames-table fetch.
        """
        from torchcodec.decoders import VideoDecoder

        self._load_file_meta([key for key in file_keys if key not in self._file_meta])

        prepared: dict[tuple, tuple] = {}
        new_files = []
        for key in file_keys:
            if key in self._decoder_cache:
                prepared[key] = self._decoder_cache.get(key)
            else:
                new_files.append(key)

        if new_files:
            handles = self._videos_table.fetch_blob_files(
                VIDEO_BLOB_COLUMN, [self._video_row_ids[key] for key in new_files]
            )
            sources = {
                key: _SparseBlobSource(self._file_meta[key]["file_size"], handle)
                for key, handle in zip(new_files, handles, strict=True)
            }
            spans_by_key: dict[tuple, list[tuple[int, int]]] = {}
            for key in new_files:
                meta = self._file_meta[key]
                spans = [
                    (0, min(_HEAD_BYTES, meta["file_size"])),
                    # Slack past the moov covers the next box header (e.g. mdat)
                    # that ffmpeg reads while walking the container — without it,
                    # every decoder open pays one ~16-byte round trip.
                    (
                        meta["moov_offset"],
                        min(meta["moov_offset"] + meta["moov_size"] + _RANGE_SLACK, meta["file_size"]),
                    ),
                ]
                if len(meta["kf_positions"]):
                    first_packet = int(meta["kf_positions"][0])
                    spans.append((first_packet, min(first_packet + _OPEN_READAHEAD, meta["file_size"])))
                spans_by_key[key] = spans
            self._fetch_spans(spans_by_key, sources)

            # Decoder creation parses the moov sample tables (~ms per file):
            # parallelize across the batch's new files. Depth files get no
            # torchcodec decoder (their 16-bit planes decode through pyav);
            # their entry holds only the prefetched source.
            rgb_files = [key for key in new_files if key[0] not in self.meta.depth_keys]
            created = self._decode_pool.map(
                lambda key: VideoDecoder(sources[key], seek_mode="approximate"), rgb_files
            )
            for key, decoder in zip(rgb_files, created, strict=True):
                prepared[key] = (decoder, sources[key])
            for key in new_files:
                if key[0] in self.meta.depth_keys:
                    prepared[key] = (None, sources[key])

        if windows:
            window_spans: dict[tuple, list[tuple[int, int]]] = {}
            for key, frame_windows in windows.items():
                meta = self._file_meta[key]
                source = prepared[key][1]
                spans = [
                    span
                    for first, last in frame_windows
                    for span in [self._window_byte_range(key[0], meta, first, last)]
                    if not source.covers(*span)
                ]
                if spans:
                    window_spans[key] = spans
            self._fetch_spans(window_spans, {key: prepared[key][1] for key in window_spans})
        return prepared

    def _video_file_key(self, key: str, ep_idx: int) -> tuple[str, int, int]:
        chunk_arr, file_arr, _ = self._video_locator[key]
        return key, int(chunk_arr[ep_idx]), int(file_arr[ep_idx])

    def _shifted_video_timestamps(
        self,
        key: str,
        ep_idx: int,
        window: list[int],
        timestamps: np.ndarray,
        row_pos: dict[int, int],
    ) -> list[float]:
        _, _, from_ts_arr = self._video_locator[key]
        base = float(from_ts_arr[ep_idx])
        return [base + float(timestamps[row_pos[row]]) for row in window]

    def _ensure_decoders(self, requests: dict, prepared: dict[tuple, tuple]) -> dict:
        """Stage 2 of video decoding: fetch this batch's frame windows.

        Translates each window into a keyframe-aligned byte range via the
        byte-index columns and fetches all of them in one parallel
        ``fetch_blob_ranges`` call.
        """
        decoders = {key: decoder for key, (decoder, _) in prepared.items()}
        sources = {key: source for key, (_, source) in prepared.items()}

        spans_by_key: dict[tuple, list[tuple[int, int]]] = {}
        fps = float(self.meta.fps)
        for key, file_requests in requests.items():
            meta = self._file_meta[key]
            source = sources[key]
            spans = []
            reach_back = 1 if key[0] in self.meta.depth_keys else 0
            for _, shifted_ts in file_requests:
                first = round(shifted_ts[0] * fps)
                last = round(shifted_ts[-1] * fps)
                start, end = self._window_byte_range(
                    key[0], meta, min(first, last) - reach_back, max(first, last)
                )
                if not source.covers(start, end):
                    spans.append((start, end))
            spans_by_key[key] = spans
        self._fetch_spans(spans_by_key, sources)

        for key in requests:
            source = sources[key]
            # A cached entry costs its buffered bytes plus the decoder's
            # ffmpeg context, which scales with resolution. Re-accounted every
            # batch as sources grow, so the cache's byte budget alone bounds
            # memory: an oversized entry is simply evicted and rebuilt from
            # two extra prefetch ranges on its next touch.
            decoder = decoders[key]
            if decoder is not None:
                height = decoder.metadata.height or 0
                width = decoder.metadata.width or 0
            else:  # depth: no torchcodec decoder; size from the feature shape
                height, width = (tuple(self.meta.features[key[0]].get("shape") or (0, 0)) + (0, 0))[:2]
            context_cost = (8 << 20) + 8 * height * width
            self._decoder_cache.put(key, (decoder, source), nbytes=source.buffered + context_cost)
        return prepared

    def _decode_depth_window(self, source, shifted_ts: list[float], file_key: tuple) -> torch.Tensor:
        """Decode one depth window: upstream's pyav decoder over our sparse source.

        Depth videos hold 16-bit planes torchcodec cannot emit. Since
        video_utils.decode_video_frames_pyav accepts file-like objects, the
        SAME upstream function serves both loaders here; we add only the
        source seek and the dequantize to metric units.
        """
        source.seek(0)
        frames = decode_video_frames_pyav(
            source, shifted_ts, self.tolerance_s, return_uint8=False, is_depth=True
        )
        config = self._depth_encoder_configs[file_key[0]]
        return dequantize_depth(
            frames,
            depth_min=config.depth_min,
            depth_max=config.depth_max,
            shift=config.shift,
            use_log=config.use_log,
            output_unit=self._depth_output_unit,
        ).squeeze(0)

    def _fetch_spans(
        self, spans_by_key: dict[tuple, list[tuple[int, int]]], sources: dict[tuple, _SparseBlobSource]
    ) -> None:
        """Fetch byte spans for many files in one parallel wave and buffer them.

        Spans are coalesced per file (faststart header trios and adjacent
        windows collapse to single requests) and all files' ranges go out in
        a single ``fetch_blob_ranges`` call.
        """
        range_requests: list[tuple[int, int, int]] = []
        range_targets: list[tuple[tuple, int]] = []
        for key, spans in spans_by_key.items():
            for start, end in _merge_spans(spans):
                range_requests.append((self._video_row_ids[key], start, end - start))
                range_targets.append((key, start))
        if not range_requests:
            return
        payloads = self._videos_table.fetch_blob_ranges(VIDEO_BLOB_COLUMN, range_requests)
        for (key, offset), payload in zip(range_targets, payloads, strict=True):
            sources[key].add(offset, payload.as_py())

    def _load_file_meta(self, missing: list[tuple]) -> None:
        """Fetch byte-index columns for files not yet in the per-worker cache."""
        if not missing:
            return
        # Point reads by _rowid: a filter on plain columns would scan (and
        # decode the large keyframe list columns for) the whole table. Keep
        # the keyframe lists in Arrow/numpy — converting them to Python lists
        # costs seconds per batch at droid scale (~50k entries per file).
        row_ids = ",".join(str(self._video_row_ids[file_key]) for file_key in missing)
        batch = (
            self._videos_table.search()
            .where(f"_rowid IN ({row_ids})")
            .select(["video_key", "chunk_index", "file_index", *VIDEO_INDEX_COLUMNS])
            .to_arrow()
        )
        if batch.num_rows != len(missing):
            raise ValueError(
                "videos table is missing byte-index columns or rows "
                f"({batch.num_rows} matches for {len(missing)} files). Re-convert the dataset "
                f"with a converter that writes {VIDEO_INDEX_COLUMNS}."
            )
        scalars = {
            name: batch.column(name).to_pylist()
            for name in ("video_key", "chunk_index", "file_index", "file_size", "moov_offset", "moov_size")
        }
        kf_index_column = batch.column("kf_indices").combine_chunks()
        kf_position_column = batch.column("kf_positions").combine_chunks()
        index_offsets = kf_index_column.offsets.to_numpy(zero_copy_only=False)
        index_values = kf_index_column.values.to_numpy(zero_copy_only=False)
        position_offsets = kf_position_column.offsets.to_numpy(zero_copy_only=False)
        position_values = kf_position_column.values.to_numpy(zero_copy_only=False)
        for i in range(batch.num_rows):
            file_key = (scalars["video_key"][i], scalars["chunk_index"][i], scalars["file_index"][i])
            self._file_meta[file_key] = {
                "file_size": scalars["file_size"][i],
                "moov_offset": scalars["moov_offset"][i],
                "moov_size": scalars["moov_size"][i],
                "kf_indices": index_values[index_offsets[i] : index_offsets[i + 1]],
                "kf_positions": position_values[position_offsets[i] : position_offsets[i + 1]],
            }
            self._file_meta.move_to_end(file_key)
        while len(self._file_meta) > 2048:
            self._file_meta.popitem(last=False)

    def _window_byte_range(self, key: str, meta: dict, first_frame: int, last_frame: int) -> tuple[int, int]:
        """Byte range covering frames [first, last]: preceding keyframe to next keyframe.

        Depth ranges get 4x the readahead slack: 12-bit depth packets run
        ~90 KB (vs ~64 KB for the standard pad), so a window ending on one
        would otherwise leak its readahead to fallback round trips.
        """
        kf_indices, kf_positions = meta["kf_indices"], meta["kf_positions"]
        start_idx = max(int(np.searchsorted(kf_indices, first_frame, side="right")) - 1, 0)
        end_idx = int(np.searchsorted(kf_indices, last_frame, side="right"))
        end = int(kf_positions[end_idx]) if end_idx < len(kf_positions) else meta["file_size"]
        slack = _RANGE_SLACK * 4 if key in self.meta.depth_keys else _RANGE_SLACK
        return int(kf_positions[start_idx]), min(end + slack, meta["file_size"])

    def _build_item(self, plan: dict, columns: dict[str, np.ndarray], row_pos: dict[int, int]) -> dict:
        base = row_pos[plan["abs_idx"]]
        item = {}
        for key in self._tabular_keys:
            data = columns[key]
            shape = self._feature_shapes[key]
            if key in self._language_keys:
                item[key] = data[base]
            elif key in self._string_keys:
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
        item["task"] = self._task_names[int(item["task_index"].item())]
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

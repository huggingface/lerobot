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

"""Helpers for the Lance dataset reader.

Everything the ``"lance"`` storage format needs outside the reader class:
table naming, connection and remote-root resolution, ``meta/`` materialization,
and the mp4 byte-range machinery (byte-index construction, sparse in-memory
sources over prefetched ranges, a bounded decoder cache). The mp4 helpers only
depend on ``av``/``io``/``bisect`` and are reusable outside Lance.
"""

from __future__ import annotations

import bisect
import io
import multiprocessing
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import av
import huggingface_hub

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import _lancedb_available, require_package

if TYPE_CHECKING or _lancedb_available:
    import lancedb

from .storage import is_remote_uri

# Byte-index columns on the videos table: map a frame window to its byte ranges so a
# batch's video fetch can be batched. Assume constant frame rate; mp4-only.
VIDEO_INDEX_COLUMNS = ("file_size", "moov_offset", "moov_size", "kf_indices", "kf_positions")
# ffmpeg reads more bytes than the frames requested. Padding each prefetched range
# to cover those known reads keeps them off the slow fallback path.
_OPEN_PROBE_BYTES = 256 * 1024
_RANGE_SLACK = 64 * 1024


def _merge_spans(spans: list[tuple[int, int]], gap: int = _RANGE_SLACK) -> list[tuple[int, int]]:
    """Coalesce overlapping or nearby byte ranges into fewer, larger requests."""
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
    """Compute the byte-index columns for one video file. mp4-only"""
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
    """Adapter between range fetches and the decoders' file API."""

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
    eviction is bounded by ``byte_budget`` too, not just count.
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


FRAMES_TABLE = "frames"
VIDEOS_TABLE = "videos"
META_TABLE = "meta"
VIDEO_BLOB_COLUMN = "video_bytes"


def to_lance_column(key: str) -> str:
    return key.replace(".", "_")


def _storage_options(
    db_uri: str, storage_options: dict | None, revision: str | None, token: str | bool | None = None
) -> dict:
    options = dict(storage_options or {})
    if db_uri.startswith("hf://"):
        if "token" not in options:
            if isinstance(token, str):
                options["token"] = token
            elif token is not False:
                stored_token = huggingface_hub.get_token()
                if stored_token:
                    options["token"] = stored_token
        if revision and "revision" not in options:
            options["revision"] = revision
    return options


def _connect(
    db_uri: str,
    storage_options: dict | None,
    revision: str | None = None,
    token: str | bool | None = None,
):
    require_package("lancedb", extra="lancedb")  # earliest common site: also reached via localize_root()
    if is_remote_uri(db_uri):
        os.environ.setdefault("LANCE_IO_THREADS", "256")
    options = _storage_options(db_uri, storage_options, revision, token)
    return lancedb.connect(db_uri, **({"storage_options": options} if options else {}))


def _materialize_meta(db, local_root: Path) -> None:
    """Write ``meta/`` from the meta table to a local cache, once."""
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
        tmp_resolved = tmp_dir.resolve()
        for batch in table.search().select(["path", "data"]).to_batches():
            paths = batch.column("path").to_pylist()
            for i, rel_path in enumerate(paths):
                dst = (tmp_dir / rel_path).resolve()
                if not dst.is_relative_to(tmp_resolved):
                    raise ValueError(f"meta table entry escapes the cache directory: {rel_path!r}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(batch.column("data")[i].as_py())
        try:
            tmp_dir.rename(meta_dir)
        except OSError:
            if not meta_dir.exists():
                raise
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def lance_mp_context() -> str:
    return "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"


def localize_root(
    repo_id: str | None,
    root: str | Path,
    revision: str | None = None,
    token: str | bool | None = None,
    force_cache_sync: bool = False,
) -> Path:
    """Materialize ``meta/`` for a remote Lance dataset and return the local dir holding it.

    Hook used by :mod:`lerobot.datasets.storage` for object-store roots; data
    tables are never downloaded.
    """
    _, local_root = resolve_lance_root(
        repo_id, root, revision=revision, token=token, force_cache_sync=force_cache_sync
    )
    return local_root


def resolve_lance_root(
    repo_id: str | None,
    root: str | Path | None,
    storage_options: dict | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    force_cache_sync: bool = False,
) -> tuple[str, Path]:
    """Resolve a Lance dataset to its connect URI and the local root holding ``meta/``"""
    if root is not None and is_remote_uri(root):
        db_uri = str(root).rstrip("/")
        # Key the cache by revision too: an hf:// root at a non-default revision must not
        # reuse (or overwrite) another revision's materialized meta.
        cache_key = f"{db_uri}@{revision}" if revision else db_uri
        local_root = HF_LEROBOT_HOME / "remote" / re.sub(r"[^A-Za-z0-9._-]+", "_", cache_key)
        if force_cache_sync:
            shutil.rmtree(local_root / "meta", ignore_errors=True)
        if not (local_root / "meta").exists():
            _materialize_meta(_connect(db_uri, storage_options, revision, token), local_root)
        return db_uri, local_root
    root_path = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    if (root_path / f"{FRAMES_TABLE}.lance").exists():
        return str(root_path), root_path
    if repo_id is not None:
        return f"hf://datasets/{repo_id}", root_path
    raise FileNotFoundError(f"No '{FRAMES_TABLE}.lance' table under {root_path}.")

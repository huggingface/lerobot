"""Runtime in-memory byte index loaded from precomputed sidecar parquet."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .byte_index_builder import BYTE_INDEX_DIR, EPISODES_NAME, FILES_NAME, KEYFRAMES_NAME
from .mp4_episode_slice import episode_custom_frame_mappings_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeSliceLookup:
    global_episode_id: int
    file_id: int
    mdat_offset: int
    mdat_length: int
    frame_count: int
    first_pts: float
    last_pts: float
    avg_fps: float

    @property
    def fetch_bytes(self) -> int:
        return self.mdat_length


@dataclass(frozen=True)
class FileLookup:
    file_id: int
    file_path: str
    file_size: int
    moov_offset: int
    moov_length: int
    header_length: int
    faststart: bool
    avg_fps: float
    codec: str


class EpisodeByteIndex:
    """Columnar byte-index resident in numpy arrays for O(1) episode lookup."""

    def __init__(
        self,
        index_dir: str | Path | None,
        *,
        video_keys: list[str],
        num_episodes: int,
        mmap: bool = True,
        files_table: pa.Table | None = None,
        episodes_table: pa.Table | None = None,
        mp4_by_rel: dict[str, Any] | None = None,
    ):
        self.index_dir = Path(index_dir) if index_dir is not None else None
        self.video_keys = list(video_keys)
        self.num_episodes = num_episodes
        self.num_cameras = len(video_keys)
        self._cam_to_idx = {cam: i for i, cam in enumerate(self.video_keys)}
        self._mp4_by_rel = mp4_by_rel
        self._frame_mappings_by_gid: dict[int, bytes] = {}

        t0 = time.perf_counter()
        if files_table is not None and episodes_table is not None:
            files_tbl, episodes_tbl = files_table, episodes_table
        else:
            if self.index_dir is None:
                raise ValueError("index_dir or in-memory tables required")
            files_path = self.index_dir / FILES_NAME
            episodes_path = self.index_dir / EPISODES_NAME
            if not files_path.exists() or not episodes_path.exists():
                raise FileNotFoundError(f"byte index missing under {self.index_dir}")
            files_tbl = pq.read_table(files_path, memory_map=mmap)
            episodes_tbl = pq.read_table(episodes_path, memory_map=mmap)

        self._load_tables(files_tbl, episodes_tbl, mmap=mmap)
        self.build_time_s = time.perf_counter() - t0
        self.load_time_s = self.build_time_s

    def _load_tables(self, files_tbl: pa.Table, episodes_tbl: pa.Table, *, mmap: bool) -> None:
        def col(tbl, name: str):
            array = tbl.column(name).combine_chunks()
            if pa.types.is_boolean(array.type):
                return array.to_numpy(zero_copy_only=False)
            return array.to_numpy()

        self.file_id = col(files_tbl, "file_id")
        self.file_path = files_tbl.column("file_path").to_pylist()
        self.file_size = col(files_tbl, "file_size")
        self.moov_offset = col(files_tbl, "moov_offset")
        self.moov_length = col(files_tbl, "moov_length")
        self.header_length = col(files_tbl, "header_length")
        self.faststart = col(files_tbl, "faststart")
        self.file_avg_fps = col(files_tbl, "avg_fps")
        self.codec = files_tbl.column("codec").to_pylist()

        ep = episodes_tbl
        n = len(ep)
        gid = col(ep, "global_episode_id")
        order = np.argsort(gid)
        self._global_episode_id = gid[order]
        self._episode_index = col(ep, "episode_index")[order]
        self._camera_index = col(ep, "camera_index")[order]
        self._file_id = col(ep, "file_id")[order]
        self._mdat_offset = col(ep, "mdat_offset")[order]
        self._mdat_length = col(ep, "mdat_length")[order]
        self._frame_count = col(ep, "frame_count")[order]
        self._first_pts = col(ep, "first_pts")[order]
        self._last_pts = col(ep, "last_pts")[order]

        expected = self.num_episodes * self.num_cameras
        if n != expected:
            raise ValueError(f"byte index episodes rows {n} != expected {expected}")

        if self.index_dir is not None:
            keyframes_path = self.index_dir / KEYFRAMES_NAME
            if keyframes_path.exists():
                kf_tbl = pq.read_table(keyframes_path, memory_map=mmap)
                self._keyframes_rows = len(kf_tbl)
            else:
                self._keyframes_rows = 0
        else:
            self._keyframes_rows = 0

        self.resident_bytes = int(
            self._global_episode_id.nbytes
            + self._file_id.nbytes
            + self._mdat_offset.nbytes
            + self._mdat_length.nbytes
            + self.file_size.nbytes
        )

    @classmethod
    def from_metadata_root(cls, meta_root: Path, *, video_keys: list[str], num_episodes: int) -> EpisodeByteIndex:
        return cls(meta_root / BYTE_INDEX_DIR, video_keys=video_keys, num_episodes=num_episodes)

    @classmethod
    def from_memory_build(
        cls,
        meta,
        data_root: str,
        *,
        workers: int = 8,
        max_episodes: int | None = None,
        include_frame_mappings_cache: bool = True,
    ) -> EpisodeByteIndex:
        """Build a complete byte index in RAM (no parquet write, no dataset push)."""
        from .byte_index_builder import build_byte_index_in_memory

        return build_byte_index_in_memory(
            meta,
            data_root,
            workers=workers,
            max_episodes=max_episodes,
            include_frame_mappings_cache=include_frame_mappings_cache,
        )

    def lookup(self, episode_index: int, camera_key: str) -> EpisodeSliceLookup:
        cam_idx = self._cam_to_idx[camera_key]
        gid = episode_index * self.num_cameras + cam_idx
        row = int(gid)
        if row < 0 or row >= len(self._global_episode_id):
            raise IndexError(f"episode_index={episode_index} camera={camera_key} out of range")
        file_id = int(self._file_id[row])
        return EpisodeSliceLookup(
            global_episode_id=gid,
            file_id=file_id,
            mdat_offset=int(self._mdat_offset[row]),
            mdat_length=int(self._mdat_length[row]),
            frame_count=int(self._frame_count[row]),
            first_pts=float(self._first_pts[row]),
            last_pts=float(self._last_pts[row]),
            avg_fps=float(self.file_avg_fps[file_id]),
        )

    def file_lookup(self, file_id: int) -> FileLookup:
        return FileLookup(
            file_id=file_id,
            file_path=self.file_path[file_id],
            file_size=int(self.file_size[file_id]),
            moov_offset=int(self.moov_offset[file_id]),
            moov_length=int(self.moov_length[file_id]),
            header_length=int(self.header_length[file_id]),
            faststart=bool(self.faststart[file_id]),
            avg_fps=float(self.file_avg_fps[file_id]),
            codec=self.codec[file_id],
        )

    def header_byte_range(self, file_id: int) -> tuple[int, int]:
        length = int(self.header_length[file_id])
        return 0, max(0, length - 1)

    def custom_frame_mappings(self, episode_index: int, camera_key: str) -> bytes | None:
        cam_idx = self._cam_to_idx[camera_key]
        gid = episode_index * self.num_cameras + cam_idx
        cached = self._frame_mappings_by_gid.get(gid)
        if cached is not None:
            return cached
        if self._mp4_by_rel is None:
            return None
        lookup = self.lookup(episode_index, camera_key)
        rel = self.file_path[lookup.file_id]
        mp4_index = self._mp4_by_rel.get(rel)
        if mp4_index is None:
            return None
        payload = episode_custom_frame_mappings_json(mp4_index, lookup.first_pts, lookup.last_pts)
        self._frame_mappings_by_gid[gid] = payload
        return payload

    def stats_dict(self) -> dict[str, float | int]:
        return {
            "load_time_s": self.load_time_s,
            "build_time_s": self.build_time_s,
            "resident_bytes": self.resident_bytes,
            "frame_mappings_cached": len(self._frame_mappings_by_gid),
            "mp4_indices_cached": len(self._mp4_by_rel or {}),
            "num_files": len(self.file_path),
            "num_episode_rows": len(self._global_episode_id),
        }

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.streaming.mp4 import (
    Mp4Index,
    Mp4SampleSlice,
    fetch_mp4_index,
    synthesized_mp4_size,
)
from lerobot.streaming.range_fetch import make_range_fetcher

if TYPE_CHECKING:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.streaming.sidecar import SidecarSpec


@dataclass(frozen=True)
class EpisodeVideoSpan:
    file_id: int
    mdat_offset: int
    mdat_length: int
    first_pts: float
    last_pts: float
    frame_count: int
    sample_lo: int
    sample_hi: int
    source_start_pts: float


@dataclass(frozen=True)
class VideoFileRecord:
    file_path: str
    file_size: int
    mp4: Mp4Index


class EpisodeVideoManifest:
    _FILE_SIDECAR_CACHE: dict[str, tuple[tuple[int, int], dict[str, VideoFileRecord]]] = {}
    _FILE_SIDECAR_CACHE_LOCK = threading.Lock()

    def __init__(
        self,
        *,
        video_keys: list[str],
        files: list[VideoFileRecord],
        spans: dict[str, np.ndarray],
    ):
        self.video_keys = list(video_keys)
        self._camera_to_id = {key: idx for idx, key in enumerate(self.video_keys)}
        self.files = files
        self.spans = spans

    @classmethod
    def build(
        cls,
        meta: LeRobotDatasetMetadata,
        data_root: str | Path,
        *,
        episode_indices: list[int] | range | None = None,
        range_backend: str = "fsspec",
        workers: int = 8,
        header_probe_bytes: int = 4 * 1024 * 1024,
        max_probe_bytes: int = 64 * 1024 * 1024,
        keyframe_pad_s: float = 0.1,
        keyframe_pad_fraction: float = 0.05,
        sidecar_path: str | Path | None = None,
    ) -> EpisodeVideoManifest:
        meta.ensure_readable()
        video_keys = list(meta.video_keys)
        if episode_indices is None:
            episode_indices = range(int(meta.total_episodes))
        rel_paths = sorted(
            {str(meta.get_video_file_path(ep_idx, key)) for ep_idx in episode_indices for key in video_keys}
        )
        path_to_id = {path: idx for idx, path in enumerate(rel_paths)}
        if sidecar_path is None:
            files = cls._build_file_records(
                rel_paths,
                data_root,
                range_backend=range_backend,
                workers=workers,
                header_probe_bytes=header_probe_bytes,
                max_probe_bytes=max_probe_bytes,
            )
        else:
            records = cls.load_file_sidecar(sidecar_path)
            missing = [path for path in rel_paths if path not in records]
            if missing:
                raise ValueError(
                    f"Sidecar {sidecar_path} is missing {len(missing)} files, first: {missing[0]}"
                )
            files = [records[path] for path in rel_paths]

        total = int(meta.total_episodes)
        num_cameras = len(video_keys)
        spans: dict[str, np.ndarray] = {
            "file_id": np.zeros((total, num_cameras), dtype=np.int32),
            "mdat_offset": np.zeros((total, num_cameras), dtype=np.int64),
            "mdat_length": np.zeros((total, num_cameras), dtype=np.int64),
            "first_pts": np.zeros((total, num_cameras), dtype=np.float64),
            "last_pts": np.zeros((total, num_cameras), dtype=np.float64),
            "frame_count": np.zeros((total, num_cameras), dtype=np.int32),
            "sample_lo": np.zeros((total, num_cameras), dtype=np.int32),
            "sample_hi": np.zeros((total, num_cameras), dtype=np.int32),
            "source_start_pts": np.zeros((total, num_cameras), dtype=np.float64),
        }

        for ep_idx in episode_indices:
            ep = meta.episodes[ep_idx]
            for cam_idx, key in enumerate(video_keys):
                rel_path = str(meta.get_video_file_path(ep_idx, key))
                file_id = path_to_id[rel_path]
                mp4 = files[file_id].mp4
                from_ts = float(ep[f"videos/{key}/from_timestamp"])
                to_ts = float(ep[f"videos/{key}/to_timestamp"])
                sample_slice = mp4.sample_slice(
                    from_ts,
                    to_ts,
                    keyframe_pad_s=keyframe_pad_s,
                    keyframe_pad_fraction=keyframe_pad_fraction,
                    file_size=files[file_id].file_size,
                )
                spans["file_id"][ep_idx, cam_idx] = file_id
                spans["mdat_offset"][ep_idx, cam_idx] = sample_slice.byte_offset
                spans["mdat_length"][ep_idx, cam_idx] = sample_slice.byte_length
                spans["first_pts"][ep_idx, cam_idx] = from_ts
                spans["last_pts"][ep_idx, cam_idx] = to_ts
                spans["frame_count"][ep_idx, cam_idx] = sample_slice.sample_hi - sample_slice.sample_lo + 1
                spans["sample_lo"][ep_idx, cam_idx] = sample_slice.sample_lo
                spans["sample_hi"][ep_idx, cam_idx] = sample_slice.sample_hi
                spans["source_start_pts"][ep_idx, cam_idx] = sample_slice.source_start_pts

        return cls(video_keys=video_keys, files=files, spans=spans)

    @staticmethod
    def _build_file_records(
        rel_paths: list[str],
        data_root: str | Path,
        *,
        range_backend: str,
        workers: int,
        header_probe_bytes: int,
        max_probe_bytes: int,
    ) -> list[VideoFileRecord]:
        fetcher = make_range_fetcher(data_root, range_backend=range_backend, workers=workers)

        def build_file(path: str) -> VideoFileRecord:
            file_size = fetcher.info_size(path)
            mp4 = fetch_mp4_index(
                path,
                fetcher.read_range,
                file_size=file_size,
                header_probe_bytes=header_probe_bytes,
                max_probe_bytes=max_probe_bytes,
            )
            return VideoFileRecord(path, file_size, mp4)

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(build_file, path): path for path in rel_paths}
                records = []
                progress_interval = max(1, len(futures) // 20)
                for completed, future in enumerate(as_completed(futures), start=1):
                    records.append(future.result())
                    if completed == len(futures) or completed % progress_interval == 0:
                        logging.info("Indexed %d/%d MP4 files for streaming sidecar", completed, len(futures))
                return sorted(records, key=lambda record: record.file_path)
        finally:
            fetcher.close()

    @classmethod
    def write_file_sidecar(
        cls,
        sidecar_path: str | Path,
        rel_paths: list[str],
        data_root: str | Path,
        *,
        spec: SidecarSpec,
        range_backend: str = "native-http",
        workers: int = 8,
        header_probe_bytes: int = 4 * 1024 * 1024,
        max_probe_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        records = cls._build_file_records(
            sorted(set(rel_paths)),
            data_root,
            range_backend=range_backend,
            workers=workers,
            header_probe_bytes=header_probe_bytes,
            max_probe_bytes=max_probe_bytes,
        )
        cls.save_file_sidecar(sidecar_path, records, spec=spec)

    @staticmethod
    def save_file_sidecar(
        sidecar_path: str | Path,
        records: list[VideoFileRecord],
        *,
        spec: SidecarSpec,
    ) -> None:
        sidecar_path = Path(sidecar_path)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "sidecar": spec.with_source_files(
                tuple((record.file_path, record.file_size) for record in records)
            ).to_dict(),
            "files": [
                {"file_path": record.file_path, "file_size": record.file_size, "mp4": record.mp4.to_dict()}
                for record in records
            ],
        }
        arrays = {}
        for file_idx, record in enumerate(records):
            arrays[f"{file_idx}/sample_pts"] = record.mp4.sample_pts
            arrays[f"{file_idx}/sample_durations"] = record.mp4.sample_durations
            arrays[f"{file_idx}/sample_sizes"] = record.mp4.sample_sizes
            arrays[f"{file_idx}/sample_offsets"] = record.mp4.sample_offsets
            arrays[f"{file_idx}/sync_samples"] = record.mp4.sync_samples
        np.savez_compressed(sidecar_path, manifest_json=json.dumps(payload).encode("utf-8"), **arrays)
        cache_key = str(sidecar_path.expanduser())
        with EpisodeVideoManifest._FILE_SIDECAR_CACHE_LOCK:
            EpisodeVideoManifest._FILE_SIDECAR_CACHE.pop(cache_key, None)

    @staticmethod
    def load_file_sidecar_metadata(sidecar_path: str | Path) -> dict[str, Any]:
        with np.load(sidecar_path, allow_pickle=False) as data:
            payload = json.loads(bytes(data["manifest_json"]).decode("utf-8"))
        if payload.get("version") != 2 or not isinstance(payload.get("sidecar"), dict):
            raise ValueError(f"Unsupported MP4 sidecar schema in {sidecar_path}")
        return payload["sidecar"]

    @staticmethod
    def validate_file_sidecar(sidecar_path: str | Path, spec: SidecarSpec) -> bool:
        try:
            from lerobot.streaming.sidecar import SidecarSpec

            candidate = SidecarSpec.from_dict(EpisodeVideoManifest.load_file_sidecar_metadata(sidecar_path))
            if not spec.matches(candidate):
                return False
            records = EpisodeVideoManifest.load_file_sidecar(sidecar_path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False

        expected = dict(candidate.source_files)
        actual = {path: record.file_size for path, record in records.items()}
        return actual == expected

    @staticmethod
    def load_file_sidecar(sidecar_path: str | Path) -> dict[str, VideoFileRecord]:
        path = Path(sidecar_path).expanduser()
        cache_key = str(path)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        with EpisodeVideoManifest._FILE_SIDECAR_CACHE_LOCK:
            cached = EpisodeVideoManifest._FILE_SIDECAR_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]

        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(bytes(data["manifest_json"]).decode("utf-8"))
            if payload.get("version") != 2:
                raise ValueError(f"Unsupported MP4 sidecar schema in {path}")
            records = {}
            for file_idx, item in enumerate(payload["files"]):
                arrays = {
                    name: data[f"{file_idx}/{name}"]
                    for name in [
                        "sample_pts",
                        "sample_durations",
                        "sample_sizes",
                        "sample_offsets",
                        "sync_samples",
                    ]
                }
                mp4 = Mp4Index.from_dict(item["mp4"], arrays)
                records[item["file_path"]] = VideoFileRecord(item["file_path"], int(item["file_size"]), mp4)
        with EpisodeVideoManifest._FILE_SIDECAR_CACHE_LOCK:
            EpisodeVideoManifest._FILE_SIDECAR_CACHE[cache_key] = (signature, records)
        return records

    def camera_id(self, camera_key: str) -> int:
        return self._camera_to_id[camera_key]

    def lookup(self, episode_index: int, camera_key: str) -> EpisodeVideoSpan:
        cam = self.camera_id(camera_key)
        return EpisodeVideoSpan(
            file_id=int(self.spans["file_id"][episode_index, cam]),
            mdat_offset=int(self.spans["mdat_offset"][episode_index, cam]),
            mdat_length=int(self.spans["mdat_length"][episode_index, cam]),
            first_pts=float(self.spans["first_pts"][episode_index, cam]),
            last_pts=float(self.spans["last_pts"][episode_index, cam]),
            frame_count=int(self.spans["frame_count"][episode_index, cam]),
            sample_lo=int(self.spans["sample_lo"][episode_index, cam]),
            sample_hi=int(self.spans["sample_hi"][episode_index, cam]),
            source_start_pts=float(self.spans["source_start_pts"][episode_index, cam]),
        )

    def file_lookup(self, file_id: int) -> VideoFileRecord:
        return self.files[file_id]

    def mp4_index(self, episode_index: int, camera_key: str) -> Mp4Index:
        return self.files[self.lookup(episode_index, camera_key).file_id].mp4

    def sample_slice(self, episode_index: int, camera_key: str) -> Mp4SampleSlice:
        span = self.lookup(episode_index, camera_key)
        return Mp4SampleSlice(
            sample_lo=span.sample_lo,
            sample_hi=span.sample_hi,
            byte_offset=span.mdat_offset,
            byte_length=span.mdat_length,
            source_start_pts=span.source_start_pts,
        )

    def episode_byte_size(self, episode_index: int) -> int:
        """Exact synthesized video bytes retained while an episode is active."""
        return sum(
            synthesized_mp4_size(
                self.mp4_index(episode_index, camera_key),
                self.sample_slice(episode_index, camera_key),
            )
            for camera_key in self.video_keys
        )

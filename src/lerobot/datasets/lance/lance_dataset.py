#!/usr/bin/env python
"""
Optional Lance dataset reading wrappers:
- Open lance.dataset(path).
- Provide get_episode(ep_id) to read sequence columns.
- Provide LanceFrameDataset for frame-wise iteration (same field set as LeRobotDataset.hf_dataset[idx]).
- Provide take_blobs and convenient PyAV-based interval decoding for video blobs.
- If lance/av are not installed, raise a friendly error (pytest.importorskip can skip).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyarrow as pa
import torch


@dataclass
class EpisodeData:
    episode_index: int
    task_index: int
    fps: int
    length: int
    timestamps: List[float]
    actions: List[List[float]] | None
    obs_state: List[List[float]] | None


class LanceEpisodeDataset:
    """Lightweight Lance reader.

    Note: This class does not convert the table into an HF Dataset. It focuses on reading per-episode sequence columns and decoding video from blob columns.

    storage_options: Optional[Dict[str, str]] allows configuring remote/object storage (e.g., S3/GCS) or custom reader parameters (e.g., storage_options/read_params) when opening the dataset.
    """

    def __init__(self, lance_path: str | Path, storage_options: Optional[Dict[str, str]] = None):
        self.lance_path = str(lance_path)
        try:
            import lance  # type: ignore
        except Exception as e:
            raise ImportError("Missing 'lance' dependency; unable to open Lance dataset. Please `pip install lance`. ") from e
        # Open dataset and cache Arrow schema via official API
        try:
            self._ds = lance.dataset(self.lance_path, storage_options=storage_options)
        except TypeError:
            # Fallback for older Lance versions that do not accept storage_options
            self._ds = lance.dataset(self.lance_path)
        self._schema: pa.Schema = self._ds.schema

    def num_rows(self) -> int:
        # Official API: count_rows() returns the number of rows (episodes)
        return int(self._ds.count_rows())

    def get_episode(self, ep_id: int) -> EpisodeData:
        """Read a single episode row using dataset.take([ep_id]) and extract sequence columns without loading the full table."""
        if ep_id < 0:
            raise IndexError(f"Episode row index out of range: {ep_id}")
        rb = self._ds.take([ep_id])  # pa.RecordBatch with a single row
        # Access columns via schema indices
        def _col(name: str):
            idx = self._schema.get_field_index(name)
            return rb.column(idx) if idx != -1 else None

        def _get(name: str) -> Any:
            c = _col(name)
            return c[0].as_py() if c is not None else None

        def _tolist(name: str) -> Optional[list]:
            c = _col(name)
            return c[0].to_pylist() if c is not None else None

        timestamps = _tolist("timestamps") or []
        actions = _tolist("actions")
        obs_state = _tolist("obs_state")

        return EpisodeData(
            episode_index=int(_get("episode_index") or ep_id),
            task_index=int(_get("task_index") or 0),
            fps=int(_get("fps") or 0),
            length=int(_get("length") or len(timestamps)),
            timestamps=[float(x) for x in timestamps],
            actions=actions,
            obs_state=obs_state,
        )

    def take_video_blob(self, camera_col: str, ep_id: int):
        """Return a Lance BlobFile (file-like) that can be passed to av.open."""
        try:
            import lance  # type: ignore
        except Exception as e:
            raise ImportError("Missing 'lance' dependency; unable to read blob") from e
        ds = lance.dataset(self.lance_path)
        blobs = ds.take_blobs(camera_col, ids=[ep_id])
        return blobs[0]

    def decode_video_interval(self, camera_col: str, ep_id: int, start_ts: float, end_ts: float):
        """Example: decode a given time interval from a BlobFile with PyAV and return a list of frame images (PIL.Image)."""
        try:
            import av  # type: ignore
        except Exception as e:
            raise ImportError("Missing 'av' dependency; unable to decode video") from e
        blob = self.take_video_blob(camera_col, ep_id)
        frames: List[Any] = []
        with av.open(blob, mode="r") as container:
            v = container.streams.video[0]
            # Compute seek start using time_base
            start_pts = int(start_ts / float(v.time_base)) if v.time_base else 0
            container.seek(start_pts, any_frame=False, stream=v)
            for packet in container.demux(v):
                for frame in packet.decode():
                    t = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
                    if t > end_ts:
                        return frames
                    frames.append(frame.to_image())
        return frames


class LanceFrameDataset(torch.utils.data.Dataset):
    """
    Frame-wise dataset wrapper on a per-episode Lance table, returning the same frame-level dict as LeRobotDataset.hf_dataset[idx]:
    { 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index',
      'action' (optional), 'observation.state' (optional), plus single-frame images for each camera_key (optional) }

    - Precompute a global index mapping to (episode_row_idx, frame_local_idx).
    - __len__ returns the total number of frames (sum(length)).
    - __getitem__ returns a frame-level dict; if image_transforms is provided, it decodes a single frame at the corresponding timestamp.
    - Dependencies lance/av are detected via try-import and raise friendly errors if missing.
    - Numeric fields use torch.tensor to match the existing interface (scalars shaped as (1,)).

    storage_options: Optional[Dict[str, str]] can be used to configure remote/object storage (S3/GCS) or custom read parameters when opening the dataset.
    """

    def __init__(
        self,
        lance_path: str | Path,
        image_transforms: Optional[Callable] = None,
        tolerance_s: float = 1e-4,
        storage_options: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.lance_path = str(lance_path)
        self.image_transforms = image_transforms
        self.tolerance_s = tolerance_s

        # Dependency check
        try:
            import lance  # type: ignore
        except Exception as e:
            raise ImportError(
                "Missing 'lance' dependency; unable to read Lance table. Please `pip install lance`."
            ) from e

        # Open Lance dataset via official API and cache schema
        try:
            self._ds = lance.dataset(self.lance_path, storage_options=storage_options)
        except TypeError:
            # Fallback for older Lance versions that do not accept storage_options
            self._ds = lance.dataset(self.lance_path)
        self._schema: pa.Schema = self._ds.schema

        # Detect video columns using the 'video_' prefix; restore original camera_key name (dot-separated)
        self._video_cols: List[str] = [
            name
            for name in self._schema.names
            if name.startswith("video_") and self._schema.field(name).type == pa.large_binary()
        ]
        self._camera_keys: List[str] = [self._denorm_cam_col(c) for c in self._video_cols]

        # Build global frame-index mapping from 'length' column only to avoid full-table load
        builder = self._ds.scanner().columns(["length"])  # ScannerBuilder
        scanner = builder.to_scanner()
        lengths_tbl = scanner.to_table()  # Arrow Table with only 'length'
        lengths: List[int] = lengths_tbl.column(0).to_pylist()
        self._episode_lengths = [int(x) for x in lengths]
        self._prefix: List[int] = []
        total = 0
        for L in self._episode_lengths:
            self._prefix.append(total)
            total += L
        self._total_frames = total
        # Build global index -> (ep_row_idx, frame_local_idx) mapping (simple expansion for __getitem__)
        self._global_map: List[Tuple[int, int]] = []
        for ep_row, L in enumerate(self._episode_lengths):
            for f in range(L):
                self._global_map.append((ep_row, f))

    @staticmethod
    def _denorm_cam_col(col_name: str) -> str:
        """Restore column name 'video_observation_images_cam' to 'observation.images.cam'."""
        assert col_name.startswith("video_")
        rest = col_name[len("video_"):]
        return rest.replace("_", ".")

    def __len__(self) -> int:
        return self._total_frames

    def _get_episode_row(self, row_idx: int) -> Dict[str, Any]:
        # Read a single row via dataset.take([row_idx]) to avoid slicing from a preloaded full table
        rb = self._ds.take([row_idx])  # pa.RecordBatch with a single row
        def _col(name: str):
            idx = self._schema.get_field_index(name)
            return rb.column(idx) if idx != -1 else None
        get = lambda name: (_col(name)[0].as_py() if _col(name) is not None else None)
        tolist = lambda name: (_col(name)[0].to_pylist() if _col(name) is not None else None)
        return {
            "episode_index": int(get("episode_index") or row_idx),
            "task_index": int(get("task_index") or 0),
            "fps": int(get("fps") or 0),
            "length": int(get("length") or 0),
            "timestamps": tolist("timestamps") or [],
            "actions": tolist("actions"),
            "obs_state": tolist("obs_state"),
        }

    def _decode_single_frame(self, camera_col: str, ep_row_idx: int, ts: float) -> Optional[torch.Tensor]:
        """Decode a single frame at the specified timestamp from a camera blob; returns torch.Tensor(C,H,W, float32)."""
        if self.image_transforms is None:
            return None
        try:
            import av  # type: ignore
        except Exception as e:
            raise ImportError("Missing 'av' dependency; unable to decode video; please `pip install av`. ") from e
        blobs = self._ds.take_blobs(camera_col, ids=[ep_row_idx])
        if not blobs:
            return None
        blob = blobs[0]
        with av.open(blob, mode="r") as container:
            v = container.streams.video[0]
            start_pts = int(ts / float(v.time_base)) if v.time_base else 0
            container.seek(start_pts, any_frame=False, stream=v)
            chosen = None
            for packet in container.demux(v):
                for frame in packet.decode():
                    t = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
                    # Choose the first frame (>= ts)
                    if t >= ts - self.tolerance_s:
                        chosen = frame
                        break
                if chosen is not None:
                    break
        if chosen is None:
            return None
        img = chosen.to_image()  # PIL.Image
        # Convert image to torch.Tensor(C,H,W), float32, [0,1] by default
        try:
            # Avoid hard dependency on torchvision; use torch.from_numpy + dimension permutation
            import numpy as np  # noqa: F401
            arr = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 255.0
        except Exception:
            # Fallback: if conversion fails, return None
            return None
        # Apply external image_transforms (if v2 or custom Callable, call directly)
        try:
            arr = self.image_transforms(arr)
        except Exception:
            # If transforms fail, do not halt the flow
            pass
        return arr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self._total_frames:
            raise IndexError(f"Index out of range: {idx} / {self._total_frames}")
        ep_row_idx, f_local = self._global_map[idx]
        ep = self._get_episode_row(ep_row_idx)
        # Basic fields
        ts_val = float(ep["timestamps"][f_local]) if ep["timestamps"] else (float(f_local) / float(max(1, ep["fps"])))
        item: Dict[str, Any] = {
            "timestamp": torch.tensor([ts_val], dtype=torch.float32),
            "frame_index": torch.tensor([f_local], dtype=torch.int64),
            "episode_index": torch.tensor([ep["episode_index"]], dtype=torch.int64),
            "index": torch.tensor([idx], dtype=torch.int64),
            "task_index": torch.tensor([ep["task_index"]], dtype=torch.int64),
        }
        # Optional numeric modalities
        if ep["actions"] is not None:
            act_row = ep["actions"][f_local]
            item["action"] = torch.tensor(act_row, dtype=torch.float32)
        if ep["obs_state"] is not None:
            st_row = ep["obs_state"][f_local]
            item["observation.state"] = torch.tensor(st_row, dtype=torch.float32)
        # Optional visual modality: restore original camera_key names
        if self.image_transforms is not None and len(self._video_cols) > 0:
            for cam_col, cam_key in zip(self._video_cols, self._camera_keys):
                frame = self._decode_single_frame(cam_col, ep_row_idx, ts_val)
                if frame is not None:
                    item[cam_key] = frame
        return item

    def take(self, ids: List[int]) -> List[Dict[str, Any]]:
        """Return items for given frame-level indices, supporting random point queries."""
        return [self.__getitem__(i) for i in ids]

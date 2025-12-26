#!/usr/bin/env python
"""
Dual-table Lance dataset readers for LeRobot:
- LanceEpisodesTable: one row per episode, stores video blobs and episode metadata
- LanceFramesTable: one row per frame, stores lightweight numeric columns and references (episode_index, frame_index, timestamp, task_index, index, action/obs_state, optional embedding)

Design goals:
- Enable efficient random frame access and contiguous window sampling (k frames)
- Support conditional filtering by episode_id and timestamp ranges
- Decode visual frames from episode-level blobs without duplicating video per-frame

Notes:
- Requires 'lance' and 'pyarrow' for table access; 'av' for decoding video frames.
- The frames table must include at least: episode_index, frame_index, timestamp, task_index, index
- The episodes table must include per-camera blob columns and *_from_ts/_to_ts fields
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import torch


@dataclass
class FrameRow:
    episode_index: int
    frame_index: int
    timestamp: float
    index: int
    task_index: int
    action: Optional[List[float]] = None
    obs_state: Optional[List[float]] = None


class LanceEpisodesTable:
    """Reader for the episodes Lance dataset (one row per episode).

    Provides lightweight metadata access and blob decoding helpers.
    """

    def __init__(self, episodes_path: str | Path, storage_options: Optional[Dict[str, str]] = None):
        self.episodes_path = str(episodes_path)
        try:
            import lance  # type: ignore
        except Exception as e:
            raise ImportError("Missing 'lance' dependency; unable to open episodes Lance dataset. Please `pip install lance`. ") from e
        try:
            self._ds = lance.dataset(self.episodes_path, storage_options=storage_options)
        except TypeError:
            self._ds = lance.dataset(self.episodes_path)
        self._schema: pa.Schema = self._ds.schema

        # Detect video columns and from/to timestamp columns
        self._video_cols: List[str] = [
            name
            for name in self._schema.names
            if name.startswith("video_") and self._schema.field(name).type == pa.large_binary()
        ]

    def num_rows(self) -> int:
        return int(self._ds.count_rows())

    def get_episode_metadata(self, ep_row_idx: int) -> Dict[str, Any]:
        rb = self._ds.take([ep_row_idx])
        def _get(name: str):
            i = self._schema.get_field_index(name)
            return rb.column(i)[0].as_py() if i != -1 else None
        def _tolist(name: str):
            i = self._schema.get_field_index(name)
            return rb.column(i)[0].to_pylist() if i != -1 else None
        out = {
            "episode_index": int(_get("episode_index") or ep_row_idx),
            "task_index": int(_get("task_index") or 0),
            "fps": int(_get("fps") or 0),
            "length": int(_get("length") or 0),
            "timestamps": _tolist("timestamps") or [],
        }
        return out

    def take_video_blob(self, camera_col: str, ep_row_idx: int):
        try:
            import lance  # type: ignore
        except Exception as e:
            raise ImportError("Missing 'lance' dependency; unable to read blob") from e
        blobs = self._ds.take_blobs(camera_col, ids=[ep_row_idx])
        return blobs[0] if blobs else None

    def decode_video_frame(self, camera_col: str, ep_row_idx: int, ts_in_clip: float, tolerance_s: float = 1e-4):
        """Decode a single frame from a per-episode blob at the specified timestamp in the episode clip.
        Returns a torch.Tensor (C,H,W) float32 in [0,1] when successful; returns None if cannot decode.
        """
        try:
            import av  # type: ignore
        except Exception as e:
            raise ImportError("Missing 'av' dependency; unable to decode video") from e
        blob = self.take_video_blob(camera_col, ep_row_idx)
        if blob is None:
            return None
        with av.open(blob, mode="r") as container:
            v = container.streams.video[0]
            start_pts = int(ts_in_clip / float(v.time_base)) if v.time_base else 0
            container.seek(start_pts, any_frame=False, stream=v)
            chosen = None
            for packet in container.demux(v):
                for frame in packet.decode():
                    t = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
                    if t >= ts_in_clip - tolerance_s:
                        chosen = frame
                        break
                if chosen is not None:
                    break
        if chosen is None:
            return None
        img = chosen.to_image()
        try:
            import numpy as np  # noqa: F401
            arr = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 255.0
        except Exception:
            return None
        return arr


class LanceFramesTable(torch.utils.data.Dataset):
    """Frame-wise dataset on top of a frames Lance table and an episodes Lance table.

    - frames_ds: one row per frame (episode_index, frame_index, timestamp, index, task_index, optional action/obs_state)
    - episodes_ds: one row per episode with video_* blob columns used for decoding images

    Supports:
    - __len__ total frames
    - __getitem__ single frame dict with optional decoded images per camera key
    - take(ids) random frame sampling
    - sample_window(start_idx, k) contiguous window within the same episode
    - filter_by_episode(episode_idx) -> list[int] of frame ids in that episode
    - filter_by_ts_range(start_ts, end_ts) -> list[int] of frame ids matching timestamp range
    """

    def __init__(
        self,
        frames_path: str | Path,
        episodes_path: str | Path,
        image_transforms: Optional[Any] = None,
        tolerance_s: float = 1e-4,
        storage_options: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.frames_path = str(frames_path)
        self.episodes_path = str(episodes_path)
        self.image_transforms = image_transforms
        self.tolerance_s = tolerance_s

        try:
            import lance  # type: ignore
        except Exception as e:
            raise ImportError("Missing 'lance' dependency; unable to read Lance tables. Please `pip install lance`. ") from e

        try:
            self._frames_ds = lance.dataset(self.frames_path, storage_options=storage_options)
        except TypeError:
            self._frames_ds = lance.dataset(self.frames_path)
        try:
            self._episodes_ds = lance.dataset(self.episodes_path, storage_options=storage_options)
        except TypeError:
            self._episodes_ds = lance.dataset(self.episodes_path)

        self._frames_schema: pa.Schema = self._frames_ds.schema
        self._episodes_schema: pa.Schema = self._episodes_ds.schema

        # Detect camera columns from episodes schema
        self._video_cols: List[str] = [
            name
            for name in self._episodes_schema.names
            if name.startswith("video_") and self._episodes_schema.field(name).type == pa.large_binary()
        ]
        self._camera_keys: List[str] = [self._denorm_cam_col(c) for c in self._video_cols]

        # Build an index from episode_index to (row_id, length)
        # Assumption: episode_index equals row index; fall back to lookup map otherwise
        self._ep_idx_to_row: Dict[int, int] = {}
        self._ep_idx_to_length: Dict[int, int] = {}
        # Read episode_index/length columns via scanner for minimal load
        builder = self._episodes_ds.scanner().columns(["episode_index", "length"])  # type: ignore
        tbl = builder.to_scanner().to_table()
        epi = tbl.column(0).to_pylist()
        lens = tbl.column(1).to_pylist()
        for i, e in enumerate(epi):
            self._ep_idx_to_row[int(e)] = i
        for e, L in zip(epi, lens, strict=False):
            self._ep_idx_to_length[int(e)] = int(L)

        # Count total frames
        self._total_frames = int(self._frames_ds.count_rows())

    @staticmethod
    def _denorm_cam_col(col_name: str) -> str:
        assert col_name.startswith("video_")
        rest = col_name[len("video_"):]
        return rest.replace("_", ".")

    def __len__(self) -> int:
        return self._total_frames

    def _take_frame_row(self, idx: int) -> FrameRow:
        rb = self._frames_ds.take([idx])
        sch = self._frames_schema
        def _get(name: str):
            i = sch.get_field_index(name)
            return rb.column(i)[0].as_py() if i != -1 else None
        def _tolist(name: str):
            i = sch.get_field_index(name)
            return rb.column(i)[0].to_pylist() if i != -1 else None
        return FrameRow(
            episode_index=int(_get("episode_index")),
            frame_index=int(_get("frame_index")),
            timestamp=float(_get("timestamp")),
            index=int(_get("index")),
            task_index=int(_get("task_index")),
            action=_tolist("action"),
            obs_state=_tolist("obs_state"),
        )

    def _decode_single_frame(self, camera_col: str, ep_index: int, ts_in_clip: float) -> Optional[torch.Tensor]:
        # Map episode_index to row id
        ep_row = self._ep_idx_to_row.get(int(ep_index), int(ep_index))
        try:
            import av  # type: ignore
        except Exception as e:
            raise ImportError("Missing 'av' dependency; unable to decode video; please `pip install av`. ") from e
        blobs = self._episodes_ds.take_blobs(camera_col, ids=[ep_row])
        if not blobs:
            return None
        blob = blobs[0]
        with av.open(blob, mode="r") as container:
            v = container.streams.video[0]
            start_pts = int(ts_in_clip / float(v.time_base)) if v.time_base else 0
            container.seek(start_pts, any_frame=False, stream=v)
            chosen = None
            for packet in container.demux(v):
                for frame in packet.decode():
                    t = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
                    if t >= ts_in_clip - self.tolerance_s:
                        chosen = frame
                        break
                if chosen is not None:
                    break
        if chosen is None:
            return None
        img = chosen.to_image()
        try:
            import numpy as np  # noqa: F401
            arr = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 255.0
        except Exception:
            return None
        # Apply optional transforms
        if self.image_transforms is not None:
            try:
                arr = self.image_transforms(arr)
            except Exception:
                pass
        return arr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self._total_frames:
            raise IndexError(f"Index out of range: {idx} / {self._total_frames}")
        fr = self._take_frame_row(idx)
        item: Dict[str, Any] = {
            "timestamp": torch.tensor([fr.timestamp], dtype=torch.float32),
            "frame_index": torch.tensor([fr.frame_index], dtype=torch.int64),
            "episode_index": torch.tensor([fr.episode_index], dtype=torch.int64),
            "index": torch.tensor([fr.index], dtype=torch.int64),
            "task_index": torch.tensor([fr.task_index], dtype=torch.int64),
        }
        if fr.action is not None:
            item["action"] = torch.tensor(fr.action, dtype=torch.float32)
        if fr.obs_state is not None:
            item["observation.state"] = torch.tensor(fr.obs_state, dtype=torch.float32)
        # Optional visual modality
        if self.image_transforms is not None and len(self._video_cols) > 0:
            for cam_col, cam_key in zip(self._video_cols, self._camera_keys):
                frame = self._decode_single_frame(cam_col, fr.episode_index, fr.timestamp)
                if frame is not None:
                    item[cam_key] = frame
        return item

    def take(self, ids: List[int]) -> List[Dict[str, Any]]:
        return [self.__getitem__(i) for i in ids]

    def sample_window(self, start_idx: int, k: int) -> List[Dict[str, Any]]:
        """Return a contiguous window of k frames within the same episode.
        Stops early if reaching the end of the episode.
        """
        if start_idx < 0 or start_idx >= self._total_frames:
            raise IndexError(f"Start idx out of range: {start_idx}")
        start_row = self._take_frame_row(start_idx)
        ep = start_row.episode_index
        # We rely on monotonically increasing global indices grouped by episode. If not guaranteed, a more
        # robust approach would scan and filter the frames table by episode_index and frame_index range.
        out: List[Dict[str, Any]] = []
        for i in range(k):
            idx = start_idx + i
            if idx >= self._total_frames:
                break
            row = self._take_frame_row(idx)
            if row.episode_index != ep:
                break
            out.append(self.__getitem__(idx))
        return out

    def filter_by_episode(self, episode_idx: int, limit: Optional[int] = None) -> List[int]:
        """Return a list of frame ids belonging to the given episode. Best-effort without full scan."""
        # Use a scanner to select just episode_index, index columns then filter in Python to avoid relying on Lance filter API compatibility
        builder = self._frames_ds.scanner().columns(["episode_index", "index"])  # type: ignore
        tbl = builder.to_scanner().to_table()
        epi = tbl.column(0).to_pylist()
        ids = tbl.column(1).to_pylist()
        out = [int(i) for e, i in zip(epi, ids, strict=False) if int(e) == int(episode_idx)]
        if limit is not None:
            out = out[: int(limit)]
        return out

    def filter_by_ts_range(self, start_ts: float, end_ts: float, limit: Optional[int] = None) -> List[int]:
        builder = self._frames_ds.scanner().columns(["timestamp", "index"])  # type: ignore
        tbl = builder.to_scanner().to_table()
        ts_list = tbl.column(0).to_pylist()
        ids = tbl.column(1).to_pylist()
        out = [int(i) for t, i in zip(ts_list, ids, strict=False) if float(start_ts) <= float(t) <= float(end_ts)]
        if limit is not None:
            out = out[: int(limit)]
        return out

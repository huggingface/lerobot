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

from collections import OrderedDict, defaultdict
from collections.abc import Callable
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


class _VideoDecoderLRU:
    """Per-worker LRU of torchcodec decoders keyed by (video_key, chunk, file).

    Each decoder holds a seekable ``BlobFile`` over the video's bytes in the
    Lance table, so evicting the least-recently-used entry also releases its
    blob handle. Capacity bounds worker RAM on datasets with many video files.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._items: OrderedDict[tuple, object] = OrderedDict()

    def __contains__(self, key: tuple) -> bool:
        return key in self._items

    def get(self, key: tuple):
        self._items.move_to_end(key)
        return self._items[key]

    def put(self, key: tuple, decoder) -> None:
        self._items[key] = decoder
        self._items.move_to_end(key)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)


def to_lance_column(key: str) -> str:
    """Map a LeRobot feature key to its Lance column name."""
    return key.replace(".", "_")


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
        root: Local directory containing ``meta/`` and the ``.lance`` tables.
            When both ``repo_id`` and ``root`` are given, local tables win if
            present.
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
        video_decoder_cache_size: int = 8,
    ):
        super().__init__()
        require_package("lancedb", extra="lance")
        if repo_id is None and root is None:
            raise ValueError("Provide `repo_id`, `root`, or both.")

        self.repo_id = repo_id
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        self.tolerance_s = tolerance_s
        self._storage_options = storage_options

        self.meta = LeRobotDatasetMetadata(
            repo_id if repo_id is not None else str(self.root), root=self.root, revision=revision
        )

        if (self.root / f"{FRAMES_TABLE}.lance").exists():
            self._db_uri = str(self.root)
        elif repo_id is not None:
            self._db_uri = f"hf://datasets/{repo_id}"
        else:
            raise FileNotFoundError(f"No '{FRAMES_TABLE}.lance' table under {self.root}.")

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
        self._decoder_cache = _VideoDecoderLRU(video_decoder_cache_size)

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
        state["_decoder_cache"] = _VideoDecoderLRU(self._decoder_cache.capacity)
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

        # Fetch all missing video files in one call. Local references keep
        # this batch's decoders alive even if the LRU evicts them.
        from torchcodec.decoders import VideoDecoder

        decoders = {key: self._decoder_cache.get(key) for key in requests if key in self._decoder_cache}
        missing = [key for key in requests if key not in decoders]
        if missing:
            blob_files = self._videos_table.fetch_blob_files(
                VIDEO_BLOB_COLUMN, [self._video_row_ids[key] for key in missing]
            )
            for key, blob_file in zip(missing, blob_files, strict=True):
                decoder = VideoDecoder(blob_file)
                decoders[key] = decoder
                self._decoder_cache.put(key, decoder)

        results: list[dict[str, torch.Tensor]] = [{} for _ in plans]
        for (key, _chunk, _file), file_requests in requests.items():
            decoder = decoders[(key, _chunk, _file)]
            fps = decoder.metadata.average_fps
            wanted = sorted({round(ts * fps) for _, shifted_ts in file_requests for ts in shifted_ts})
            frame_pos = {frame_idx: pos for pos, frame_idx in enumerate(wanted)}
            batch = decoder.get_frames_at(indices=wanted)
            for sample_idx, shifted_ts in file_requests:
                positions = [frame_pos[round(ts * fps)] for ts in shifted_ts]
                loaded_ts = batch.pts_seconds[positions]
                distance = (torch.tensor(shifted_ts) - loaded_ts).abs()
                if (distance >= self.tolerance_s).any():
                    raise FrameTimestampError(
                        f"Query timestamps violate tolerance_s={self.tolerance_s} for video "
                        f"'{key}' (chunk {_chunk}, file {_file}): queried {shifted_ts}, "
                        f"loaded {loaded_ts.tolist()}."
                    )
                frames = batch.data[positions]
                if not self.return_uint8:
                    frames = (frames / 255.0).type(torch.float32)
                results[sample_idx][key] = frames.squeeze(0)
        return results

    def _build_item(self, plan: dict, columns: dict[str, np.ndarray], row_pos: dict[int, int]) -> dict:
        base = row_pos[plan["abs_idx"]]
        item = {}
        for key in self._tabular_keys:
            data = columns[key]
            shape = self._feature_shapes[key]
            if key in plan["windows"]:
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

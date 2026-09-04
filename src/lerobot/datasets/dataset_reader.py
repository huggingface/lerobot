#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Private reader component for LeRobotDataset. Handles random-access reading (HF dataset, delta indices, video decoding)."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets
import torch

from lerobot.configs import (
    DEFAULT_DEPTH_UNIT,
    DEPTH_METER_UNIT,
    DepthEncoderConfig,
)

from .dataset_metadata import LeRobotDatasetMetadata
from .depth_utils import MM_PER_METRE, dequantize_depth
from .feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
)
from .io_utils import (
    hf_transform_to_torch,
    load_nested_dataset,
)
from .utils import resolve_episode_indices
from .video_utils import decode_video_frames


class BaseDatasetReader(ABC):
    """Read-side data access contract for :class:`LeRobotDataset`.

    A reader owns row fetching and video decoding for one storage format and
    returns fully assembled frame dicts — tabular features, delta-timestamp
    windows, padding masks, decoded video frames — so every format produces
    the same items. ``LeRobotDataset`` delegates ``__getitem__`` and
    ``__getitems__`` to it and keeps everything else (metadata, episode
    selection, the public API). Subclasses define their own constructor
    (their inputs legitimately differ) and must be picklable so
    ``DataLoader`` workers can reopen their own connections.

    Subclasses must set :attr:`episodes` (the selected episode indices, or
    ``None`` for all) during construction.
    """

    episodes: list[int] | None

    @property
    @abstractmethod
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""

    @property
    @abstractmethod
    def num_episodes(self) -> int:
        """Number of episodes selected."""

    @property
    @abstractmethod
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        """Mapping from absolute frame indices to relative row positions.

        Non-None only for episode-filtered datasets where absolute indices
        (from metadata) differ from positions in the filtered view.
        """

    @abstractmethod
    def get_item(self, idx: int) -> dict:
        """Return one fully assembled frame dict for a relative index."""

    def get_items(self, indices: list[int]) -> list[dict]:
        """Return frame dicts for a batch of relative indices.

        Subclasses may override this with a batched implementation.
        """
        return [self.get_item(idx) for idx in indices]

    def __len__(self) -> int:
        return self.num_frames

    def set_image_transforms(self, image_transforms: Callable | None) -> None:
        """Replace the transform applied to visual observations."""
        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self._image_transforms = image_transforms

    def clear_image_transforms(self) -> None:
        """Remove the transform applied to visual observations."""
        self._image_transforms = None


class DatasetReader(BaseDatasetReader):
    """Default reader serving the parquet/mp4 storage format.

    Owns: hf_dataset, _absolute_to_relative_idx, delta_indices.
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        root: Path,
        episodes: list[int] | None,
        tolerance_s: float,
        video_backend: str,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Callable | None,
        return_uint8: bool = False,
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
    ):
        """Initialize the reader with metadata, filtering, and transform config.

        The HF dataset is not loaded here — call :meth:`try_load` or
        :meth:`load_and_activate` afterward.

        Args:
            meta: Dataset metadata instance.
            root: Local dataset root directory.
            episodes: Optional list of episode indices to select. ``None``
                means all episodes.
            tolerance_s: Timestamp synchronization tolerance in seconds.
            video_backend: Video decoding backend identifier.
            delta_timestamps: Optional dict mapping feature keys to lists of
                relative timestamp offsets for temporal context windows.
            image_transforms: Optional torchvision v2 transform applied to
                visual features.
            return_uint8: If True, return RGB video frames as raw uint8 tensors
                instead of normalized float32.
            depth_output_unit: Physical unit depth maps are dequantized to
                (``"m"`` or ``"mm"``). Defaults to ``"mm"``.
        """
        self._meta = meta
        self.root = root
        self.episodes = resolve_episode_indices(episodes, meta.total_episodes)
        self._tolerance_s = tolerance_s
        self._video_backend = video_backend
        self.set_image_transforms(image_transforms)
        self._return_uint8 = return_uint8
        self._depth_output_unit = depth_output_unit

        self.hf_dataset: datasets.Dataset | None = None
        self._absolute_to_relative_idx: dict[int, int] | None = None
        self._column_views: dict[str, datasets.Dataset] = {}
        self._column_views_source: datasets.Dataset | None = None
        self._column_views_transform: Callable | None = None

        # Setup delta_indices (doesn't depend on hf_dataset)
        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)

        self._depth_encoder_configs: dict[str, DepthEncoderConfig] = {
            vid_key: DepthEncoderConfig.from_video_info(self._meta.features[vid_key].get("info"))
            for vid_key in self._meta.depth_keys
        }

        # Get the input unit of each depth feature stored as raw images.
        self._image_depth_units: dict[str, str | None] = {
            key: (self._meta.features[key].get("info") or {}).get("depth_unit")
            for key in self._meta.depth_keys
            if key in self._meta.image_keys
        }

    def try_load(self) -> bool:
        """Attempt to load from local cache. Returns True if data is sufficient."""
        try:
            self.hf_dataset = self._load_hf_dataset()
        except (FileNotFoundError, NotADirectoryError):
            self.hf_dataset = None
            return False
        if not self._check_cached_episodes_sufficient():
            self.hf_dataset = None
            return False
        self._build_index_mapping()
        return True

    def load_and_activate(self) -> None:
        """Load HF dataset from disk and build index mapping. Call after data is on disk."""
        self.hf_dataset = self._load_hf_dataset()
        self._build_index_mapping()

    def _build_index_mapping(self) -> None:
        """Build absolute-to-relative index mapping from loaded hf_dataset."""
        self._absolute_to_relative_idx = None
        if self.episodes is not None and self.hf_dataset is not None:
            indices = self.hf_dataset.data.column("index").to_numpy()
            self._absolute_to_relative_idx = dict(zip(indices.tolist(), range(len(indices)), strict=True))

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self._meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self._meta.total_episodes

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        """Mapping from absolute frame indices to HF dataset row positions."""
        if self.hf_dataset is None:
            self.load_and_activate()
        return self._absolute_to_relative_idx

    def _load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        features = get_hf_features_from_features(self._meta.features)
        self._validate_language_columns_declared(features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _validate_language_columns_declared(self, features: datasets.Features) -> None:
        """Require language columns stored in Parquet to be declared in metadata."""
        # Leave empty datasets to fail through the normal loading path.
        try:
            sample = next((self.root / "data").glob("*/*.parquet"))
        except StopIteration:
            return

        from pyarrow import parquet as _pq  # noqa: PLC0415

        # LeRobot shards are schema-uniform, so one schema represents the dataset.
        schema_names = set(_pq.read_schema(sample).names)
        from .language import LANGUAGE_COLUMNS  # noqa: PLC0415

        missing = sorted(set(LANGUAGE_COLUMNS) & schema_names - set(features))
        if missing:
            raise ValueError(
                f"Dataset Parquet files contain language feature(s) missing from metadata: {missing}. "
                "Metadata must describe the stored data; add the entries returned by "
                "lerobot.datasets.language.language_feature_info() to meta/info.json['features'] "
                "or rerun the annotation pipeline's metadata synchronization."
            )

    def _check_cached_episodes_sufficient(self) -> bool:
        """Check if the cached dataset contains all requested episodes and their video files."""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }

        if self.episodes is None:
            requested_episodes = set(range(self._meta.total_episodes))
        else:
            requested_episodes = set(self.episodes)

        if not requested_episodes.issubset(available_episodes):
            return False

        if len(self._meta.video_keys) > 0:
            for ep_idx in requested_episodes:
                for vid_key in self._meta.video_keys:
                    video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False

        return True

    def get_episodes_file_paths(self) -> list[Path]:
        """Return deduplicated file paths (data + video) for selected episodes.

        Used to build the ``allow_patterns`` list for ``snapshot_download``.
        """
        episodes = self.episodes if self.episodes is not None else list(range(self._meta.total_episodes))
        fpaths = [str(self._meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self._meta.video_keys) > 0:
            video_files = [
                str(self._meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self._meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        # episodes are stored in the same files, so we return unique paths only
        fpaths = list(set(fpaths))
        return fpaths

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """Compute query indices for delta timestamps."""
        ep = self._meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _to_relative(self, indices: list[int]) -> list[int]:
        """Map absolute frame indices to relative row positions in ``hf_dataset``.

        Passthrough when the dataset is not episode-filtered.
        """
        if self._absolute_to_relative_idx is None:
            return indices
        return [self._absolute_to_relative_idx[i] for i in indices]

    def _column_view(self, key: str) -> datasets.Dataset:
        """Return a cached single-column view of ``hf_dataset``.

        ``select_columns`` is a zero-copy schema projection: row queries on the
        view fetch and decode only ``key``. By contrast, ``hf_dataset[indices]``
        (and, since a custom transform disables the lazy-``Column`` fast path in
        ``datasets`` >= 4.4, also ``hf_dataset[key][indices]``) fetches and
        decodes entire rows. On image datasets that decodes every embedded
        camera image of every queried row just to read a low-dimensional column
        like ``action`` (#2895). The view keeps the ``hf_transform_to_torch``
        transform, which is column-wise, so outputs are identical to a plain
        row query.
        """
        transform = self.hf_dataset.format["format_kwargs"].get("transform")
        if self._column_views_source is not self.hf_dataset or self._column_views_transform is not transform:
            # hf_dataset was (re)loaded or its transform changed: drop stale views
            self._column_views = {}
            self._column_views_source = self.hf_dataset
            self._column_views_transform = transform
        if key not in self._column_views:
            self._column_views[key] = self.hf_dataset.select_columns(key)
        return self._column_views[key]

    def _get_query_timestamps(
        self,
        current_ts: list[float],
        query_indices_per_item: list[dict[str, list[int]] | None],
    ) -> list[dict[str, list[float]]]:
        """Timestamps to decode for each requested item, as one ``{video_key: [timestamp, ...]}`` dict.

        Per video key: the referenced rows' timestamps if the item has a delta
        window on it, else the item's own ``current_ts``. Timestamps are read
        through the cached single-column view (see :meth:`_column_view`).
        ``current_ts`` and ``query_indices_per_item`` are batch-aligned (indices
        ABSOLUTE), so every referenced row is read from Arrow in a single shot.
        """
        # Pass 1: per item, collect the relative rows each video key needs a
        # timestamp for.
        rel_per_item: list[dict[str, list[int]]] = []
        needed: set[int] = set()
        for q_idx in query_indices_per_item:
            rel: dict[str, list[int]] = {}
            if q_idx is not None:
                for key in self._meta.video_keys:
                    if key in q_idx:  # this item has a delta window on this video key
                        rel[key] = self._to_relative(q_idx[key])
                        needed.update(rel[key])
            rel_per_item.append(rel)

        # Single Arrow read for every referenced row, keyed by row for lookup below.
        ts_lookup: dict[int, float] = {}
        if needed:
            rel_sorted = sorted(needed)
            column = self._column_view("timestamp")[rel_sorted]["timestamp"]
            ts_lookup = {rel: float(column[j]) for j, rel in enumerate(rel_sorted)}

        # Pass 2: assemble per item; keys without a delta window fall back to current_ts.
        return [
            {
                key: [ts_lookup[r] for r in rel[key]] if key in rel else [current_ts[i]]
                for key in self._meta.video_keys
            }
            for i, rel in enumerate(rel_per_item)
        ]

    def _query_hf_dataset(self, query_indices_per_item: list[dict[str, list[int]]]) -> list[dict]:
        """Tabular columns to gather for each requested item, as one ``{key: stacked tensor}`` dict.

        Per non-video key: the referenced rows stacked into the item's delta window.
        ``query_indices_per_item`` are batch-aligned (indices ABSOLUTE). Each key
        is read through its cached single-column view (see :meth:`_column_view`),
        so only that column is decoded — never the embedded camera images of the
        queried rows. Every row a key needs across the batch is read in a single
        shot, then redistributed (preserving per-item order and duplicates).
        """
        # Pass 1: per item, collect the relative rows each non-video key needs.
        rel_per_item: list[dict[str, list[int]]] = []
        per_key_rows: dict[str, set[int]] = {}
        for query_indices in query_indices_per_item:
            rel = {
                key: self._to_relative(q_idx)
                for key, q_idx in query_indices.items()
                if key not in self._meta.video_keys
            }
            rel_per_item.append(rel)
            for key, q in rel.items():
                per_key_rows.setdefault(key, set()).update(q)

        if not per_key_rows:
            return [{} for _ in query_indices_per_item]

        # Pass 2: one column-pruned Arrow read per key over its row union, keyed by row.
        gathered: dict[str, tuple[list, dict[int, int]]] = {}
        for key, rows in per_key_rows.items():
            rel_sorted = sorted(rows)
            column = self._column_view(key)[rel_sorted][key]
            gathered[key] = (column, {rel: j for j, rel in enumerate(rel_sorted)})

        return [
            {key: torch.stack([gathered[key][0][gathered[key][1][r]] for r in q]) for key, q in rel.items()}
            for rel in rel_per_item
        ]

    def _query_videos(
        self, query_timestamps_per_item: list[dict[str, list[float]]], ep_idxs: list[int]
    ) -> list[dict[str, torch.Tensor]]:
        """Decode video frames for a batch, grouped by physical MP4 file.

        All (file, timestamp) requests across the batch are grouped so each file
        is opened/seeked once (amortizing decode when consecutive samples share
        files. ``query_timestamps`` are within-episode, so the episode's ``from_timestamp``
        offset is applied here.

        Note: When using data workers (e.g. DataLoader with num_workers>0), do not
        call this in the main process. It will result in a Segmentation Fault.
        """
        # Group the queried timestamps by physical MP4 files.
        group_ts: dict[str, list[float]] = {}
        group_meta: dict[str, str] = {}  # path -> vid_key
        segments: list[tuple[int, str, int, int]] = []  # (item, path, start, length)
        for i, (query_ts_per_key, ep_idx) in enumerate(zip(query_timestamps_per_item, ep_idxs, strict=True)):
            ep = self._meta.episodes[ep_idx]
            for vid_key, query_ts in query_ts_per_key.items():
                from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
                path = str(self.root / self._meta.get_video_file_path(ep_idx, vid_key))
                buf = group_ts.setdefault(path, [])
                start = len(buf)
                buf.extend(from_timestamp + ts for ts in query_ts)
                segments.append((i, path, start, len(query_ts)))
                group_meta[path] = vid_key

        def _decode(path: str) -> tuple[str, torch.Tensor]:
            vid_key = group_meta[path]
            frames = decode_video_frames(
                path,
                group_ts[path],
                self._tolerance_s,
                self._video_backend,
                return_uint8=self._return_uint8,
                is_depth=vid_key in self._meta.depth_keys,
            )
            if vid_key in self._meta.depth_keys:
                depth_encoder = self._depth_encoder_configs[vid_key]
                frames = dequantize_depth(
                    frames,
                    depth_min=depth_encoder.depth_min,
                    depth_max=depth_encoder.depth_max,
                    shift=depth_encoder.shift,
                    use_log=depth_encoder.use_log,
                    output_unit=self._depth_output_unit,
                )
            return path, frames

        # Decode each physical MP4 file in parallel.
        paths = list(group_ts)
        if len(paths) <= 1:
            decoded = dict(_decode(p) for p in paths)
        else:
            with ThreadPoolExecutor(max_workers=min(len(paths), 8)) as pool:
                decoded = dict(pool.map(_decode, paths))

        result: list[dict[str, torch.Tensor]] = [{} for _ in query_timestamps_per_item]
        for i, path, start, length in segments:
            result[i][group_meta[path]] = decoded[path][start : start + length].squeeze(0)
        return result

    def get_item(self, idx) -> dict:
        """Return one fully assembled frame dict for a single *relative* index.

        "Relative" is the row position in the (possibly episode-filtered) ``hf_dataset``,
        not the dataset-wide absolute index (see :attr:`absolute_to_relative_idx`).
        Delegates to :meth:`get_items` so single- and batched-access share one
        code path (and identical output).

        Args:
            idx: Relative row position in the loaded ``hf_dataset``.

        Returns:
            The fully assembled frame dict for ``idx``.
        """
        return self.get_items([idx])[0]

    def get_items(self, indices: list[int]) -> list[dict]:
        """Assemble frame dicts for a batch of *relative* indices.

        "Relative" indices are row positions in the (possibly episode-filtered) ``hf_dataset``,
        not the dataset-wide absolute indices (see :attr:`absolute_to_relative_idx`).

        Tabular rows are gathered from the Arrow-backed HF dataset in one shot,
        and all requested video frames are grouped by physical MP4 file so each
        file is opened/seeked once per batch (amortizing decode across the
        batch; effective when consecutive samples share files).

        Args:
            indices: Relative row positions in the loaded ``hf_dataset``.

        Returns:
            One fully assembled frame dict per entry in ``indices``, in order.
        """
        if self.hf_dataset is None:
            # One-shot load after finalize()
            self.load_and_activate()
        if len(indices) == 0:
            return []

        n = len(indices)

        # Batched tabular gather: one Arrow read for all base rows.
        base = self.hf_dataset[indices]
        items: list[dict] = [{key: base[key][i] for key in base} for i in range(n)]
        ep_idxs = [int(items[i]["episode_index"]) for i in range(n)]
        abs_idxs = [int(items[i]["index"]) for i in range(n)]

        # Delta windows: per-item absolute query indices + padding, then one
        # batched tabular gather across the whole batch.
        query_indices_per_item: list[dict[str, list[int]] | None] = [None] * n
        if self.delta_indices is not None:
            for i in range(n):
                query_indices_per_item[i], padding = self._get_query_indices(abs_idxs[i], ep_idxs[i])
                items[i].update(padding)
            for i, tabular in enumerate(self._query_hf_dataset(query_indices_per_item)):
                items[i].update(tabular)

        # Video frames: batched decode, grouped by physical MP4 across the batch.
        if len(self._meta.video_keys) > 0:
            current_ts = [float(items[i]["timestamp"]) for i in range(n)]
            query_timestamps = self._get_query_timestamps(current_ts, query_indices_per_item)
            for i, video in enumerate(self._query_videos(query_timestamps, ep_idxs)):
                items[i].update(video)

        for i in range(n):
            item = items[i]
            # Apply image transforms to RGB cameras.
            if self._image_transforms is not None:
                for cam in self._meta.camera_keys:
                    if cam in self._meta.depth_keys:
                        continue
                    item[cam] = self._image_transforms(item[cam])

            # Convert depth features to the output unit.
            for key, stored_unit in self._image_depth_units.items():
                if key in item and stored_unit is not None and stored_unit != self._depth_output_unit:
                    item[key] = (
                        item[key] * MM_PER_METRE
                        if stored_unit == DEPTH_METER_UNIT
                        else item[key] / MM_PER_METRE
                    )

            item["task"] = self._meta.tasks.iloc[int(item["task_index"])].name

        return items

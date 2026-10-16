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
from functools import partial
from pathlib import Path

import datasets
import torch

from lerobot.configs import DEFAULT_DEPTH_UNIT

from .dataset_metadata import LeRobotDatasetMetadata
from .feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
)
from .io_utils import (
    hf_transform_to_torch,
    load_nested_dataset,
)
from .utils import delta_window, resolve_episode_indices, shift_timestamps, task_name
from .video_utils import (
    apply_rgb_transforms,
    convert_image_depth_units,
    decode_video_frames,
    depth_encoder_configs,
    dequantize_depth_frames,
    image_depth_units,
)


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

        self._depth_encoder_configs = depth_encoder_configs(meta)
        self._image_depth_units = image_depth_units(meta)

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

    def load_and_activate(self) -> datasets.Dataset:
        """Load HF dataset from disk and build index mapping. Call after data is on disk.

        Returns the loaded dataset (also stored in :attr:`hf_dataset`).
        """
        self.hf_dataset = self._load_hf_dataset()
        self._build_index_mapping()
        return self.hf_dataset

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
        hf_dataset.set_transform(partial(hf_transform_to_torch, features=self._meta.features))
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

    def get_episodes_file_paths(self) -> list[str]:
        """Return deduplicated relative file paths (data + video) for selected episodes.

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
        if self.delta_indices is None:
            raise RuntimeError("Query indices require delta_timestamps, but the reader has none.")
        ep = self._meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices: dict[str, list[int]] = {}
        padding: dict[str, torch.Tensor] = {}
        for key, delta_idx in self.delta_indices.items():
            query_indices[key], padding[f"{key}_is_pad"] = delta_window(abs_idx, delta_idx, ep_start, ep_end)
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
        hf_dataset = self.hf_dataset
        if hf_dataset is None:
            raise RuntimeError("hf_dataset is not loaded; call load_and_activate() first.")
        transform = hf_dataset.format["format_kwargs"].get("transform")
        stale = self._column_views_source is not hf_dataset or self._column_views_transform is not transform
        if stale:
            # hf_dataset was (re)loaded or its transform changed: drop stale views
            self._column_views = {}
            self._column_views_source = hf_dataset
            self._column_views_transform = transform
        if key not in self._column_views:
            self._column_views[key] = hf_dataset.select_columns(key)
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

    def _query_hf_dataset(self, query_indices_per_item: list[dict[str, list[int]] | None]) -> list[dict]:
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
                for key, q_idx in (query_indices or {}).items()
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

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault.
        """
        ep = self._meta.episodes[ep_idx]

        def _decode_single(vid_key: str, query_ts: list[float]) -> tuple[str, torch.Tensor]:
            shifted_query_ts = shift_timestamps(query_ts, ep[f"videos/{vid_key}/from_timestamp"])
            video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(
                video_path,
                shifted_query_ts,
                self._tolerance_s,
                self._video_backend,
                return_uint8=self._return_uint8,
                is_depth=vid_key in self._meta.depth_keys,
            )
            if vid_key in self._meta.depth_keys:
                frames = dequantize_depth_frames(
                    frames, self._depth_encoder_configs[vid_key], self._depth_output_unit
                )
            return vid_key, frames.squeeze(0)

        items = list(query_timestamps.items())

        # Single camera: no threading overhead
        if len(items) <= 1:
            return {vid_key: _decode_single(vid_key, query_ts)[1] for vid_key, query_ts in items}

        # Multi-camera: decode in parallel (video decoding releases the GIL)
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            futures = [pool.submit(_decode_single, k, ts) for k, ts in items]
            return dict(f.result() for f in futures)

    def get_item(self, idx: int) -> dict:
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
        while video frames are decoded one item at a time to avoid competing with the multiple workers of the DataLoader.

        Args:
            indices: Relative row positions in the loaded ``hf_dataset``.

        Returns:
            One fully assembled frame dict per entry in ``indices``, in order.
        """
        # One-shot load after finalize()
        hf_dataset = self.hf_dataset if self.hf_dataset is not None else self.load_and_activate()
        if len(indices) == 0:
            return []

        n = len(indices)

        # Batched tabular gather: one Arrow read for all base rows.
        base = hf_dataset[indices]
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

        # Video frames: decoded one item at a time. We do not group decoding by physical
        # MP4 across the batch as it competes with the multiple workers of the DataLoader.
        if len(self._meta.video_keys) > 0:
            current_ts = [float(items[i]["timestamp"]) for i in range(n)]
            query_timestamps = self._get_query_timestamps(current_ts, query_indices_per_item)
            for item, query_ts, ep_idx in zip(items, query_timestamps, ep_idxs, strict=True):
                item.update(self._query_videos(query_ts, ep_idx))

        for item in items:
            apply_rgb_transforms(item, self._image_transforms, self._meta.camera_keys, self._meta.depth_keys)
            convert_image_depth_units(item, self._image_depth_units, self._depth_output_unit)
            item["task"] = task_name(self._meta.tasks, item["task_index"])

        return items

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

"""mcap storage backend for :class:`~lerobot.datasets.lerobot_dataset.LeRobotDataset`.

Serves datasets whose ``meta/info.json`` declares ``"storage_format": "mcap"``: one
``.mcap`` file per episode holds every feature as a per-message channel, next to the
standard ``meta/`` directory. mcap is async and per-message-timestamped — there is no
shared fixed-fps grid the way there is in the default parquet/mp4 or lance formats.

The alignment between mcap's real message timestamps and LeRobot's fixed ``fps`` grid
is resolved once, at conversion time (see :mod:`lerobot.datasets.mcap_convert`), and
stored as extra ``meta/episodes`` columns: a sorted ``log_time_ns`` array per channel
per episode (:func:`lerobot.datasets.mcap_utils.log_time_column`) and one
``episode_start_time_ns`` per episode (the grid's time origin). The reader never
rebuilds this index — it only ever reads it back from already-loaded Arrow columns
(see :meth:`McapDatasetReader.__init__`), mirroring how the lance backend's
``kf_indices``/``kf_positions`` byte-index is built once by the converter and only
ever read by :class:`~lerobot.datasets.lance_backend.LanceDatasetReader`.

Everything here is an implementation detail of the ``"mcap"`` storage format; the
public entry point is ``LeRobotDataset`` (see :mod:`lerobot.datasets.storage`).
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from lerobot.configs.video import DEFAULT_DEPTH_UNIT
from lerobot.utils.import_utils import _mcap_available, require_package

from .dataset_metadata import LeRobotDatasetMetadata
from .dataset_reader import BaseDatasetReader
from .feature_utils import check_delta_timestamps, get_delta_indices
from .io_utils import load_image_as_numpy

# Re-exported: storage.py resolves localize_root on this module.
from .mcap_utils import (  # noqa: F401
    EPISODE_START_TIME_COLUMN,
    MCAP_CHUNK_INDEX_COLUMN,
    MCAP_FILE_INDEX_COLUMN,
    episode_mcap_path,
    load_topic_map,
    localize_root,
    log_time_column,
    topic_for_key,
    unpack_tensor,
)
from .utils import resolve_episode_indices
from .video_utils import FrameTimestampError

if TYPE_CHECKING or _mcap_available:
    from mcap.reader import make_reader

# Administrative columns that are pure functions of (episode metadata, grid index) —
# they're never stored as mcap messages, just computed on read (see _build_item).
_COMPUTED_KEYS = ("timestamp", "frame_index", "episode_index", "index")

# meta/episodes columns every mcap dataset must carry, checked up front with a clear
# error -- mirrors how the lance backend validates its own VIDEO_INDEX_COLUMNS/video
# files instead of letting a missing column surface as a raw pyarrow KeyError.
_REQUIRED_EPISODE_COLUMNS = (
    "dataset_from_index",
    "dataset_to_index",
    EPISODE_START_TIME_COLUMN,
    MCAP_CHUNK_INDEX_COLUMN,
    MCAP_FILE_INDEX_COLUMN,
)


class McapDatasetReader(BaseDatasetReader):
    """Dataset reader serving mcap-formatted LeRobot datasets.

    Instantiated by :class:`LeRobotDataset` through the reader lookup table
    (see :mod:`lerobot.datasets.storage`); returns the same item dicts as the
    default parquet/mp4 pipeline.

    Args:
        meta: Already-loaded dataset metadata (never rebuilt or mutated here).
        root: Local dir with ``meta/`` and per-episode ``.mcap`` files under ``data/``.
        episodes: Episode indices to select. ``None`` means all.
        image_transforms: Optional torchvision v2 transform for camera frames.
        delta_timestamps: Feature key -> relative timestamp offsets (seconds), as
            in :class:`LeRobotDataset`.
        tolerance_s: Timestamp synchronization tolerance in seconds.
        revision: Accepted for constructor-contract uniformity; unused (local-disk only, v1).
        return_uint8: Return RGB frames as raw uint8 instead of normalized float32.
        depth_output_unit: Accepted for constructor-contract uniformity; unused
            (depth features are not supported in v1).
        token: Accepted for constructor-contract uniformity; unused (local-disk only, v1).
        decode_pool_size: Max threads used to fetch/decode messages per batch.
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        return_uint8: bool = False,
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
        token: str | bool | None = None,
        decode_pool_size: int | None = None,
    ):
        require_package("mcap", extra="mcap")
        self.meta = meta
        self.root = Path(root) if root is not None else meta.root
        self.tolerance_s = tolerance_s
        self.return_uint8 = return_uint8
        self.set_image_transforms(image_transforms)
        self._decode_pool_size = decode_pool_size or 16

        if self.meta.image_keys:
            raise NotImplementedError(
                f"Image-backed features are not supported by the mcap backend: {self.meta.image_keys}. "
                "Re-encode them as video (dtype='video')."
            )
        if self.meta.depth_keys:
            raise NotImplementedError(
                f"Depth features are not supported by the mcap backend yet: {self.meta.depth_keys}."
            )

        self.episodes = resolve_episode_indices(episodes, meta.total_episodes)
        if self.episodes is not None and not self.episodes:
            raise ValueError("None of the requested episodes are in the dataset.")

        missing_columns = [c for c in _REQUIRED_EPISODE_COLUMNS if c not in self.meta.episodes.column_names]
        if missing_columns:
            raise ValueError(
                f"meta/episodes is missing column(s) required by the mcap reader: {missing_columns}. "
                "Re-convert the dataset with a converter that writes the mcap sync index (see "
                "lerobot.datasets.mcap_convert.convert_lerobot_dataset_to_mcap / ingest_mcap_episodes), "
                "or confirm storage_format='mcap' is correct for this dataset."
            )

        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.meta.fps)

        self._ep_from = self._episode_numpy("dataset_from_index", np.int64)
        self._ep_to = self._episode_numpy("dataset_to_index", np.int64)
        if len(self._ep_from) and (
            int(self._ep_from[0]) != 0
            or int(self._ep_to[-1]) != self.meta.total_frames
            or (self._ep_from[1:] != self._ep_to[:-1]).any()
        ):
            raise ValueError(
                "Episode boundaries in meta do not tile [0, total_frames) contiguously; "
                "the dataset metadata is inconsistent."
            )
        self._episode_start_ns = self._episode_numpy(EPISODE_START_TIME_COLUMN, np.int64)
        self._mcap_chunk = self._episode_numpy(MCAP_CHUNK_INDEX_COLUMN, np.int64)
        self._mcap_file = self._episode_numpy(MCAP_FILE_INDEX_COLUMN, np.int64)

        # Mirrors the lance backend's "videos table is missing N file(s) referenced by
        # episode metadata" check -- only for the episodes actually selected, so a
        # filtered `episodes=[...]` subset doesn't require every episode's file to exist.
        episodes_to_check = self.episodes if self.episodes is not None else range(self.meta.total_episodes)
        missing_files = [
            (ep_idx, path)
            for ep_idx in episodes_to_check
            if not (path := episode_mcap_path(self.root, int(self._mcap_chunk[ep_idx]), int(self._mcap_file[ep_idx]))).exists()
        ]
        if missing_files:
            ep_idx, path = missing_files[0]
            raise FileNotFoundError(
                f"{len(missing_files)} episode mcap file(s) referenced by episode metadata are missing "
                f"on disk, e.g. episode {ep_idx} -> {path}. The dataset is incomplete or was converted "
                "against a different root."
            )

        if self.episodes is not None:
            self._rel_to_abs = np.concatenate(
                [np.arange(self._ep_from[ep], self._ep_to[ep]) for ep in sorted(self.episodes)]
            )
            self._absolute_to_relative_idx = {
                int(abs_idx): rel_idx for rel_idx, abs_idx in enumerate(self._rel_to_abs)
            }
        else:
            self._rel_to_abs = None
            self._absolute_to_relative_idx = None

        self._task_names = list(self.meta.tasks.index)

        # task_index is usually a real per-message channel (multi-task-per-episode
        # recordings, e.g. from convert_lerobot_dataset_to_mcap). Raw external ingestion
        # (mcap_convert.ingest_mcap_episodes) instead stores one constant task_index per
        # episode -- there's no native "task label" topic in a robot's own recording --
        # as a plain int column, and _build_item treats it like frame_index/timestamp.
        self._task_index_is_channel = log_time_column("task_index") in self.meta.episodes.column_names
        if not self._task_index_is_channel:
            self._task_index_per_episode = self._episode_numpy("task_index", np.int64)

        self._tabular_keys = [
            key
            for key in self.meta.features
            if key not in _COMPUTED_KEYS
            and key not in self.meta.video_keys
            and key not in self.meta.image_keys
            and (key != "task_index" or self._task_index_is_channel)
        ]
        self._stored_keys = self._tabular_keys + list(self.meta.video_keys)
        # A declared shape of (1,) means "scalar-per-frame" throughout this codebase
        # (get_hf_features_from_features maps it to datasets.Value, not a length-1
        # sequence) -- store () so packed/decoded values round-trip as 0-dim tensors,
        # matching what the default/lance readers produce for the same features.
        self._feature_shapes = {
            key: ()
            if tuple(self.meta.features[key].get("shape") or ()) == (1,)
            else tuple(self.meta.features[key].get("shape") or ())
            for key in self._tabular_keys
        }
        self._np_dtypes = {key: np.dtype(self.meta.features[key]["dtype"]) for key in self._tabular_keys}
        topic_map = load_topic_map(self.root)
        self._topics = {key: topic_map.get(key, topic_for_key(key)) for key in self._stored_keys}
        self._key_for_topic = {topic: key for key, topic in self._topics.items()}

        # Mirrors the lance backend's "videos table is missing byte-index columns... Re-convert
        # the dataset with a converter that writes {VIDEO_INDEX_COLUMNS}" check: the sync-index
        # column is this format's equivalent load-bearing, converter-computed data.
        missing_log_time_columns = [
            col
            for key in self._stored_keys
            if (col := log_time_column(key)) not in self.meta.episodes.column_names
        ]
        if missing_log_time_columns:
            raise ValueError(
                f"meta/episodes is missing {len(missing_log_time_columns)} sync-index column(s) required "
                f"by the mcap reader: {missing_log_time_columns}. Re-convert the dataset with a converter "
                "that writes these columns (see lerobot.datasets.mcap_convert)."
            )

        # Per-episode sorted log_time_ns array for each stored key: self._log_time_ns[key][ep_idx].
        # Plain numpy arrays fetched from meta.episodes (already in RAM, no I/O) -- unlike lance's
        # open table handles, these are picklable as-is, so no _ensure_open()/__getstate__ dance
        # is needed for them; only the thread pool needs that treatment (see below).
        self._log_time_ns: dict[str, list[np.ndarray]] = {
            key: [
                np.asarray(ep_times, dtype=np.int64) for ep_times in self.meta.episodes[log_time_column(key)]
            ]
            for key in self._stored_keys
        }

        self._decode_pool: ThreadPoolExecutor | None = None

    def _episode_numpy(self, name: str, dtype: type[np.generic]) -> np.ndarray:
        column = self.meta.episodes.data.column(name).to_numpy(zero_copy_only=False)
        return column.astype(dtype, copy=False)

    def _ensure_open(self) -> None:
        if self._decode_pool is None:
            self._decode_pool = ThreadPoolExecutor(max_workers=self._decode_pool_size)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_decode_pool"] = None
        return state

    def close(self) -> None:
        """Shut down worker threads; reads reopen lazily."""
        if self._decode_pool is not None:
            self._decode_pool.shutdown(wait=False, cancel_futures=True)
            self._decode_pool = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    @property
    def num_frames(self) -> int:
        return len(self._rel_to_abs) if self._rel_to_abs is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        return self._absolute_to_relative_idx

    def get_item(self, idx: int) -> dict:
        return self.get_items([idx])[0]

    def get_items(self, indices: list[int]) -> list[dict]:
        """Batched fetch: one message scan per episode touched, deduplicated across the batch."""
        self._ensure_open()
        plans = self._plan_batch(indices)

        # 1. Union, per episode and key, of the grid positions the whole batch needs.
        grid_needs: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
        for plan in plans:
            ep_idx = plan["ep_idx"]
            for key in self._stored_keys:
                window = plan["windows"].get(key, [plan["abs_idx"]])
                grid_needs[ep_idx][key].update(window)

        # 2. Resolve each grid position to the nearest real message timestamp, enforcing tolerance.
        matched_ns: dict[int, dict[str, dict[int, int]]] = {}
        for ep_idx, per_key in grid_needs.items():
            matched_ns[ep_idx] = {}
            for key, grid_idxs in per_key.items():
                grid_arr = np.array(sorted(grid_idxs), dtype=np.int64)
                matched_ns[ep_idx][key] = self._match_targets(ep_idx, key, grid_arr)

        # 3. Fetch messages, one open file per episode touched (parallel across episodes).
        def _fetch(ep_idx: int) -> tuple[int, dict[str, dict[int, bytes]]]:
            keys_needed = {key: set(m.values()) for key, m in matched_ns[ep_idx].items()}
            return ep_idx, self._fetch_episode_messages(ep_idx, keys_needed)

        futures = [self._decode_pool.submit(_fetch, ep_idx) for ep_idx in grid_needs]
        raw_by_ep = dict(future.result() for future in futures)

        # 4. Decode each (grid position, key) exactly once, even if requested by several samples.
        decoded: dict[tuple[int, str], np.ndarray] = {}
        for ep_idx, per_key in grid_needs.items():
            for key, grid_idxs in per_key.items():
                raw = raw_by_ep[ep_idx][key]
                grid_to_ns = matched_ns[ep_idx][key]
                for grid_idx in grid_idxs:
                    decoded[(grid_idx, key)] = self._decode_message(key, raw[grid_to_ns[grid_idx]])

        items = [self._build_item(plan, decoded) for plan in plans]
        if self._image_transforms is not None:
            for item in items:
                for cam_key in self.meta.camera_keys:
                    item[cam_key] = self._image_transforms(item[cam_key])
        return items

    def _plan_batch(self, indices: list[int]) -> list[dict]:
        plans = []
        for idx in indices:
            abs_idx = self._resolve_abs_idx(idx)
            ep_idx = self._episode_index_for_abs_idx(abs_idx)
            start, end = self._episode_bounds(ep_idx)
            plan = {"abs_idx": abs_idx, "ep_idx": ep_idx, "start": start, "windows": {}, "padding": {}}
            if self.delta_indices is not None:
                for key, deltas in self.delta_indices.items():
                    window = [min(max(abs_idx + delta, start), end - 1) for delta in deltas]
                    plan["windows"][key] = window
                    plan["padding"][f"{key}_is_pad"] = torch.BoolTensor(
                        [not (start <= abs_idx + delta < end) for delta in deltas]
                    )
            plans.append(plan)
        return plans

    def _resolve_abs_idx(self, idx: int) -> int:
        idx = int(idx)
        if idx < 0:
            idx += self.num_frames
        if not 0 <= idx < self.num_frames:
            raise IndexError(f"Index {idx} is out of range for a dataset of {self.num_frames} frames.")
        return int(self._rel_to_abs[idx]) if self._rel_to_abs is not None else idx

    def _episode_index_for_abs_idx(self, abs_idx: int) -> int:
        return int(np.searchsorted(self._ep_from, abs_idx, side="right") - 1)

    def _episode_bounds(self, ep_idx: int) -> tuple[int, int]:
        return int(self._ep_from[ep_idx]), int(self._ep_to[ep_idx])

    def _match_targets(self, ep_idx: int, key: str, grid_idx: np.ndarray) -> dict[int, int]:
        """Nearest-message match: grid position -> real message log_time_ns, within tolerance_s."""
        fps = float(self.meta.fps)
        idx_within_episode = grid_idx - self._ep_from[ep_idx]
        target_ns = self._episode_start_ns[ep_idx] + np.round(idx_within_episode * (1e9 / fps)).astype(
            np.int64
        )

        arr = self._log_time_ns[key][ep_idx]
        if len(arr) == 0:
            raise FrameTimestampError(f"Channel '{key}' has no messages in episode {ep_idx}.")
        pos = np.searchsorted(arr, target_ns)
        right = arr[np.clip(pos, 0, len(arr) - 1)]
        left = arr[np.clip(pos - 1, 0, len(arr) - 1)]
        right_dist = np.abs(right - target_ns)
        left_dist = np.abs(target_ns - left)
        use_right = right_dist <= left_dist
        matched = np.where(use_right, right, left)
        distance_ns = np.where(use_right, right_dist, left_dist)

        tolerance_ns = self.tolerance_s * 1e9
        violations = distance_ns > tolerance_ns
        if violations.any():
            raise FrameTimestampError(
                f"Query timestamps violate tolerance_s={self.tolerance_s} for channel '{key}' "
                f"(episode {ep_idx}): requested grid frames {grid_idx[violations].tolist()}, "
                f"nearest real timestamps were {distance_ns[violations].tolist()} ns away."
            )
        return dict(zip(grid_idx.tolist(), matched.tolist(), strict=True))

    # Below this many wasted messages per needed message, a single combined scan across
    # every key is cheaper than paying per-call overhead for narrower, per-key clusters
    # (measured empirically: ~11 keys x separate calls cost more than scanning through a
    # dense/contiguous batch's small amount of genuine waste).
    _SCAN_WASTE_THRESHOLD = 3.0

    def _fetch_episode_messages(
        self, ep_idx: int, keys_needed: dict[str, set[int]]
    ) -> dict[str, dict[int, bytes]]:
        """Fetch every needed message from one episode's mcap file, one open file handle
        shared across every call this makes.

        Two strategies, chosen per batch from data already in RAM (no file I/O needed to
        decide): a dense/contiguous batch -- the common case -- gets ONE ``iter_messages``
        call across every key's combined range, exactly as cheap as before. A batch
        scattered across a long episode (e.g. ``shuffle=True`` on a dataset with few, long
        episodes) would force that single call to decompress and iterate through mostly
        wasted messages just to discard them, so instead the union of every key's needed
        timestamps is split into contiguous clusters (``_cluster_windows``) and each
        cluster is fetched with its own narrow, still-multi-topic call -- more calls, but
        each mostly used, letting the mcap chunk index skip the untouched middle instead
        of scanning through it.
        """
        path = episode_mcap_path(self.root, int(self._mcap_chunk[ep_idx]), int(self._mcap_file[ep_idx]))
        matched: dict[str, dict[int, bytes]] = {key: {} for key in keys_needed}
        all_ns = [ns for ns_set in keys_needed.values() for ns in ns_set]
        global_min, global_max = min(all_ns), max(all_ns)
        total_needed = sum(len(v) for v in keys_needed.values())
        total_if_combined = sum(
            self._messages_in_range(key, ep_idx, global_min, global_max) for key in keys_needed
        )

        with open(path, "rb") as f:
            reader = make_reader(f)
            if total_if_combined <= total_needed * self._SCAN_WASTE_THRESHOLD:
                topics = [self._topics[key] for key in keys_needed]
                for _schema, channel, message in reader.iter_messages(
                    topics=topics, start_time=global_min, end_time=global_max + 1, log_time_order=True
                ):
                    key = self._key_for_topic[channel.topic]
                    if message.log_time in keys_needed[key]:
                        matched[key][message.log_time] = message.data
            else:
                # Cluster the UNION of every key's needed timestamps together, not each
                # key separately: matched timestamps for different keys at the SAME
                # requested grid position land close in time (they're all near that
                # position's target_ns), so grouping across keys collapses a scattered
                # batch into ~one cluster per distinct grid position -- each serving
                # every key in one combined call -- instead of one cluster per
                # (key, point) pair, which was measured to multiply call count (and its
                # per-call overhead) by the number of keys for no benefit.
                sorted_ns = np.unique(np.array(all_ns, dtype=np.int64))
                min_period_ns = min(self._typical_period_ns(key, ep_idx) for key in keys_needed)
                topics = [self._topics[key] for key in keys_needed]
                for lo, hi in self._cluster_windows(sorted_ns, min_period_ns):
                    for _schema, channel, message in reader.iter_messages(
                        topics=topics, start_time=lo, end_time=hi + 1, log_time_order=True
                    ):
                        key = self._key_for_topic[channel.topic]
                        if message.log_time in keys_needed[key]:
                            matched[key][message.log_time] = message.data
        return matched

    def _messages_in_range(self, key: str, ep_idx: int, lo_ns: int, hi_ns: int) -> int:
        """How many real messages of `key` fall within [lo_ns, hi_ns] -- what a combined
        scan over that range would have to iterate through, computed from the in-memory
        log_time_ns index alone (no file I/O)."""
        arr = self._log_time_ns[key][ep_idx]
        lo = np.searchsorted(arr, lo_ns, side="left")
        hi = np.searchsorted(arr, hi_ns, side="right")
        return int(hi - lo)

    def _typical_period_ns(self, key: str, ep_idx: int) -> float:
        """Median inter-message gap for `key` in the whole episode (not just the
        requested subset -- that would be biased by however scattered the query is)."""
        arr = self._log_time_ns[key][ep_idx]
        if len(arr) < 2:
            return 1.0
        return float(np.median(np.diff(arr)))

    @staticmethod
    def _cluster_windows(
        sorted_ns: np.ndarray, typical_period_ns: float, max_gap_periods: float = 4.0
    ) -> list[tuple[int, int]]:
        """Split sorted timestamps into (start, end) windows, starting a new window
        wherever the gap to the next needed timestamp exceeds `max_gap_periods` times
        the channel's typical message spacing. A dense/contiguous run of requested
        timestamps -- gaps close to the channel's own rate -- always collapses to one
        window spanning all of them, exactly like the single-window scan this replaces.
        """
        if len(sorted_ns) == 0:
            return []
        if len(sorted_ns) == 1:
            return [(int(sorted_ns[0]), int(sorted_ns[0]))]
        threshold = max(typical_period_ns * max_gap_periods, 1.0)
        split_points = np.where(np.diff(sorted_ns) > threshold)[0] + 1
        return [(int(group[0]), int(group[-1])) for group in np.split(sorted_ns, split_points)]

    def _decode_message(self, key: str, data: bytes) -> np.ndarray:
        if key in self.meta.video_keys:
            return load_image_as_numpy(
                data, dtype=np.uint8 if self.return_uint8 else np.float32, is_depth=False
            )
        return unpack_tensor(data, self._np_dtypes[key], self._feature_shapes[key])

    def _build_item(self, plan: dict, decoded: dict[tuple[int, str], np.ndarray]) -> dict:
        abs_idx, ep_idx, start = plan["abs_idx"], plan["ep_idx"], plan["start"]
        fps = self.meta.fps
        idx_within_episode = abs_idx - start

        item: dict = {
            "frame_index": torch.tensor(idx_within_episode),
            "episode_index": torch.tensor(ep_idx),
            "index": torch.tensor(abs_idx),
            "timestamp": torch.tensor(idx_within_episode / fps, dtype=torch.float32),
        }
        if not self._task_index_is_channel:
            item["task_index"] = torch.tensor(int(self._task_index_per_episode[ep_idx]))

        for key in self._tabular_keys:
            if key in plan["windows"]:
                window = plan["windows"][key]
                stacked = np.stack([decoded[(g, key)] for g in window])
                item[key] = torch.from_numpy(np.ascontiguousarray(stacked))
            else:
                # np.ascontiguousarray promotes 0-dim arrays to shape (1,) -- np.array(..)
                # (a copy) preserves rank, matching the default/lance readers' scalar shape
                # for administrative fields declared with shape (1,) (see _feature_shapes).
                item[key] = torch.from_numpy(np.array(decoded[(abs_idx, key)]))

        for key in self.meta.video_keys:
            window = plan["windows"].get(key, [abs_idx])
            frames = torch.stack([torch.from_numpy(decoded[(g, key)]) for g in window])
            # Always squeeze dim 0, matching the default/lance readers: it only
            # collapses when the window has exactly one entry (no delta requested,
            # or a user-specified single-timestamp window) -- otherwise a no-op.
            item[key] = frames.squeeze(0)

        item.update(plan["padding"])
        item["task"] = self._task_names[int(item["task_index"].item())]
        return item

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  repo_id={self.meta.repo_id},\n"
            f"  root={self.root},\n"
            f"  episodes={self.num_episodes} selected / {self.meta.total_episodes} total,\n"
            f"  frames={self.num_frames},\n"
            f")"
        )


# The class lerobot.datasets.storage instantiates for storage_format "mcap".
DATASET_READER = McapDatasetReader

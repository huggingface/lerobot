#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

from lerobot.utils.constants import HF_LEROBOT_HOME

from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .feature_utils import get_delta_indices
from .io_utils import item_to_torch
from .utils import (
    check_version_compatibility,
    find_float_index,
    is_float_in_list,
)
from .video_utils import (
    VideoDecoderCache,
    decode_video_frames_torchcodec,
)

logger = logging.getLogger(__name__)

# Bound the default frame-level shuffle buffer: rows are tabular-only (~KB each), so this is
# roughly a few hundred MB of host RAM per consumer at the cap.
_MAX_DEFAULT_FRAME_BUFFER = 200_000


class StreamingLeRobotDataset(torch.utils.data.IterableDataset):
    """LeRobotDataset with streaming capabilities, built on native HF `datasets` primitives.

    The tabular side is a pure `datasets` pipeline::

        load_dataset(streaming=True)                          # parquet shards from the Hub / a bucket
          -> reshard()                                        # 1 shard == 1 row group == 1 episode
          -> split_dataset_by_node(rank, world_size)          # disjoint shards per rank
          -> batch(by_column="episode_index")                 # whole episodes (one per shard)
          -> shuffle(episode_pool_size, max_buffer_input_shards)  # K random episodes, global perm
          -> map(explode + exact delta windows)               # episode -> frames, windows are exact
          -> shuffle(buffer_size=frame_shuffle_buffer_size)   # frame-level interleave

    and this class is a thin torch ``IterableDataset`` wrapper around it that decodes video
    per emitted sample (decode-on-exit), applies image transforms, and attaches the task
    string. DataLoader workers are split natively by `datasets` (disjoint shards per worker),
    and resume uses the native ``state_dict`` / ``load_state_dict``.

    Random-episode admission (Plan B): the LeRobot writer stores one Parquet row group per
    episode, so ``datasets.IterableDataset.reshard()`` makes one shard == one episode (no new
    files; shards are (file, row_group) pairs). ``shuffle`` then permutes shard order globally and
    fills its buffer from ``max_buffer_input_shards`` shards concurrently, so the episode pool is a
    uniformly-random sample of the corpus regardless of how many episodes are packed per file.
    ``max_buffer_input_shards`` is the number of concurrently-live random episodes; set it
    ``>= batch_size`` for the per-batch distinct-episode fraction to approach 1.

    Requirement: ONE ROW GROUP PER EPISODE. Recorded datasets satisfy this; bulk
    ``df.to_parquet`` / ``push_to_hub`` / aggregate paths collapse row groups and are rejected at
    init (see ``validate_row_groups``). Old collapsed datasets still load fine for the map-style
    path; only this streaming random-episode path requires the invariant.

    Randomness: a batch mixes up to ``episode_pool_size`` distinct episodes; delta windows are
    exact slices of the resident episode with correct padding at episode boundaries.

    Resume: ``state_dict()`` / ``load_state_dict()`` delegate to `datasets`. Samples sitting in
    the shuffle buffers at checkpoint time are skipped on resume (documented `datasets`
    behavior), so resume never repeats data but may drop up to roughly
    ``episode_pool_size x episode_len + frame_shuffle_buffer_size`` frames — negligible at
    training scale. The contract is exact with ``num_workers=0``; with DataLoader workers use
    ``torchdata.stateful_dataloader.StatefulDataLoader``, which checkpoints each worker's
    dataset state through this same protocol.

    Example:
        ```python
        dataset = StreamingLeRobotDataset(
            repo_id="your-dataset-repo-id",
            delta_timestamps={"action": [0.0, 0.1, 0.2]},
            episode_pool_size=1024,
        )
        for sample in dataset:
            ...
        ```
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        streaming: bool = True,
        episode_pool_size: int | None = 1024,
        max_buffer_input_shards: int | None = None,
        frame_shuffle_buffer_size: int | None = None,
        buffer_size: int | None = None,
        max_num_shards: int | None = None,
        seed: int = 42,
        rng: np.random.Generator | None = None,
        shuffle: bool = True,
        return_uint8: bool = False,
        rank: int | None = None,
        world_size: int | None = None,
        video_decoder_cache_size: int | None = None,
        data_files_root: str | None = None,
        validate_row_groups: bool = True,
    ):
        """Initialize a StreamingLeRobotDataset.

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset.
            root (Path | None, optional): Local directory to use for local datasets. When omitted, Hub
                metadata is resolved through a revision-safe snapshot cache under
                ``$HF_LEROBOT_HOME/hub``.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list.
            image_transforms (Callable | None, optional): Transform to apply to image data.
            tolerance_s (float, optional): Tolerance in seconds for timestamp matching.
            revision (str, optional): Git revision id (branch name, tag, or commit hash).
            force_cache_sync (bool, optional): Flag to sync and refresh local files first.
            streaming (bool, optional): Whether to stream the dataset or load it all. Defaults to True.
            episode_pool_size (int, optional): Whole episodes each consumer keeps open to shuffle
                across — the randomness knob. Larger mixes more episodes per batch (closer to
                map-style uniform) at the cost of cold-start latency and frame-buffer RAM.
                Defaults to 1024.
            max_buffer_input_shards (int | None, optional): Number of shards (== episodes, after
                ``reshard()``) the episode-pool ``shuffle`` reads from concurrently — i.e. the count
                of concurrently-live random episodes feeding the pool from a global shard permutation.
                Set ``>= batch_size`` for the per-batch distinct-episode fraction to approach 1.
                Defaults to ``episode_pool_size``.
            frame_shuffle_buffer_size (int | None, optional): Frame-level shuffle buffer after the
                episode pool. Defaults to ``episode_pool_size x average episode length`` (capped),
                which matches the pool's mixing radius.
            buffer_size (int | None, optional): Deprecated; superseded by ``episode_pool_size``.
            max_num_shards (int | None, optional): Deprecated; `datasets` handles shard-to-worker
                assignment natively.
            seed (int, optional): Reproducibility random seed.
            rng (np.random.Generator | None, optional): Deprecated; ignored.
            shuffle (bool, optional): Whether to shuffle. False yields episodes in stream order.
            rank (int | None, optional): This process' rank for distributed training. Each rank streams
                a disjoint set of shards via ``split_dataset_by_node``. When omitted, resolved from
                Accelerate (``process_index``) or the ``RANK`` env var, defaulting to 0.
            world_size (int | None, optional): Total number of distributed processes. When omitted,
                resolved from Accelerate or ``WORLD_SIZE``, defaulting to 1. For an even per-rank split,
                ``num_shards % world_size == 0`` should hold (warned otherwise).
            video_decoder_cache_size (int | None, optional): Max number of open video decoders to retain.
                When omitted, sized to the episode pool's working set, capped at 128.
            data_files_root (str | None, optional): fsspec root holding the bulk ``data/`` and ``videos/``
                trees (e.g. ``hf://buckets/<owner>/<name>``). When set, parquet and video bytes are read
                from there while metadata still loads from ``repo_id`` on the Hub.
            validate_row_groups (bool, optional): When True (default), verify at init that the dataset
                stores one Parquet row group per episode (sampling data-file footers) and that
                ``num_shards`` is divisible by ``world_size`` for distributed runs, raising a clear
                ``ValueError`` otherwise. Set False to skip the checks (e.g. single-process debugging);
                the divisibility check then downgrades to a warning.
        """
        super().__init__()
        self.repo_id = repo_id
        self._requested_root = Path(root) if root else None
        self.root = self._requested_root if self._requested_root is not None else HF_LEROBOT_HOME / repo_id
        self.streaming_from_local = root is not None

        self.image_transforms = image_transforms
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.seed = seed
        if rng is not None:
            logger.warning("StreamingLeRobotDataset: `rng` is deprecated and ignored; use `seed`.")
        if buffer_size is not None:
            logger.warning(
                "StreamingLeRobotDataset: `buffer_size` is deprecated and ignored; "
                "use `episode_pool_size` (whole episodes, not frames)."
            )
        if max_num_shards is not None:
            logger.warning(
                "StreamingLeRobotDataset: `max_num_shards` is deprecated and ignored; "
                "`datasets` assigns shards to DataLoader workers natively."
            )
        self.shuffle = shuffle

        self.streaming = streaming
        self.episode_pool_size = max(1, episode_pool_size) if episode_pool_size else 1024
        self.max_buffer_input_shards = (
            max(1, max_buffer_input_shards) if max_buffer_input_shards else self.episode_pool_size
        )
        self.validate_row_groups = validate_row_groups
        self._return_uint8 = return_uint8

        self.rank, self.world_size = self._resolve_distributed(rank, world_size)
        self.video_decoder_cache_size = video_decoder_cache_size
        self.data_files_root = data_files_root.rstrip("/") if data_files_root else None

        # We cache the video decoders to avoid re-initializing them at each frame (avoiding a ~10x slowdown)
        self.video_decoder_cache = None
        self._epoch = 0
        self._in_flight_epoch = 0

        if self._requested_root is not None:
            self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self._requested_root, self.revision, force_cache_sync=force_cache_sync
        )
        self.root = self.meta.root
        self.revision = self.meta.revision
        # Check version
        check_version_compatibility(self.repo_id, self.meta._version, CODEBASE_VERSION)

        self.delta_timestamps = None
        self.delta_indices = None

        if delta_timestamps is not None:
            self._validate_delta_timestamp_keys(delta_timestamps)  # raises ValueError if invalid
            self.delta_timestamps = delta_timestamps
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        if self.data_files_root is not None:
            # Bulk data lives in an fsspec root (e.g. an HF storage bucket); metadata stays on the Hub.
            self.hf_dataset: datasets.IterableDataset = load_dataset(
                "parquet",
                split="train",
                streaming=self.streaming,
                data_files=f"{self.data_files_root}/data/*/*.parquet",
            )
        else:
            self.hf_dataset = load_dataset(
                self.repo_id if not self.streaming_from_local else str(self.root),
                split="train",
                streaming=self.streaming,
                data_files="data/*/*.parquet",
                revision=self.revision,
            )

        # Drop any parquet columns not declared in the dataset's feature contract. Some revisions / sources
        # (e.g. an unversioned bucket holding `main`) carry extra, possibly variable-length annotation
        # columns such as `language_events`; left in, they leak into the sample and break default DataLoader
        # collation across frames of differing length. On a clean revision this is a no-op.
        known_columns = set(self.meta.features)
        extra_columns = [c for c in (self.hf_dataset.column_names or []) if c not in known_columns]
        if extra_columns:
            self.hf_dataset = self.hf_dataset.remove_columns(extra_columns)

        # Reshard Parquet per row group so 1 shard == 1 row group == 1 episode (the LeRobot writer
        # emits one row group per episode). This lets the episode-pool shuffle admit uniformly-random
        # episodes from a global shard permutation, independent of how many episodes are packed per file.
        if self.streaming:
            self.hf_dataset = self.hf_dataset.reshard()
        self.num_shards = self.hf_dataset.num_shards

        if self.validate_row_groups and self.streaming:
            self._validate_row_groups_per_episode()

        avg_episode_len = max(1, round(self.meta.total_frames / max(1, self.meta.total_episodes)))
        self.frame_shuffle_buffer_size = (
            frame_shuffle_buffer_size
            if frame_shuffle_buffer_size is not None
            else min(self.episode_pool_size * avg_episode_len, _MAX_DEFAULT_FRAME_BUFFER)
        )

        self._pipeline = self._build_pipeline()

    @property
    def num_frames(self):
        return self.meta.total_frames

    @property
    def num_episodes(self):
        return self.meta.total_episodes

    @property
    def fps(self):
        return self.meta.fps

    @staticmethod
    def _resolve_distributed(rank: int | None, world_size: int | None) -> tuple[int, int]:
        """Resolve (rank, world_size) for distributed streaming.

        Explicit arguments win. Otherwise prefer an already-initialized Accelerate state, then the
        ``RANK``/``WORLD_SIZE`` env vars set by launchers, and finally fall back to single-process (0, 1).
        """
        import os

        if rank is not None and world_size is not None:
            return rank, world_size

        try:
            from accelerate.state import PartialState

            if PartialState._shared_state:  # only read it if already initialized; never initialize here
                state = PartialState()
                return state.process_index, state.num_processes
        except Exception:
            logger.debug("Could not resolve distributed state from Accelerate; using env/defaults.")

        env_rank = os.environ.get("RANK")
        env_world = os.environ.get("WORLD_SIZE")
        if env_rank is not None and env_world is not None:
            return int(env_rank), int(env_world)

        return 0, 1

    def _resolve_data_root(self) -> str:
        """fsspec root that holds the bulk ``data/`` parquet tree (revision-qualified for the Hub)."""
        if self.data_files_root is not None:
            return self.data_files_root
        if self.streaming and not self.streaming_from_local:
            return f"hf://datasets/{self.repo_id}@{self.revision}"
        return str(self.root)

    def _episode_files(self) -> dict[tuple[int, int], list[int]]:
        """Map each data file ``(chunk_index, file_index)`` to the episode indices it stores."""
        file_to_eps: dict[tuple[int, int], list[int]] = {}
        for ep in range(self.meta.total_episodes):
            row = self.meta.episodes[ep]
            key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
            file_to_eps.setdefault(key, []).append(ep)
        return file_to_eps

    def _validate_row_groups_per_episode(self, sample_files: int = 32) -> None:
        """Verify the dataset stores ONE ROW GROUP PER EPISODE so each episode is an independently
        addressable shard after ``reshard()``. Cheap (footer-only) and sampled.

        Raises:
            ValueError: if a sampled data file collapses several episodes into fewer row groups, or
                the whole dataset is one row group per file while holding many more episodes than files.
        """
        import fsspec
        import pyarrow.parquet as pq

        file_to_eps = self._episode_files()
        num_data_files = len(file_to_eps)

        # Whole-dataset extreme: reshard() could not split beyond file granularity (one row group per
        # file) yet there are many more episodes than files -> collapsed.
        if self.num_shards <= num_data_files and self.meta.total_episodes > self.num_shards:
            raise ValueError(
                f"{self.repo_id}: after reshard() the stream still has only {self.num_shards} shard(s) "
                f"for {self.meta.total_episodes} episodes across {num_data_files} data file(s) — i.e. one "
                "row group per file. StreamingLeRobotDataset random-episode shuffling requires ONE ROW "
                "GROUP PER EPISODE so each episode is an independently addressable shard after reshard(). "
                "Re-emit through the LeRobot writer (one write_table per episode) or fix the aggregate / "
                "annotate / push_to_hub writer that collapsed the row groups, then re-upload. Recorded "
                "datasets already satisfy this. Pass validate_row_groups=False to bypass (random-episode "
                "quality will degrade)."
            )

        data_root = self._resolve_data_root()
        rng = np.random.default_rng(self.seed)
        keys = list(file_to_eps)
        chosen = rng.choice(len(keys), size=min(sample_files, len(keys)), replace=False)
        for i in chosen:
            chunk_idx, file_idx = keys[int(i)]
            n_ep = len(file_to_eps[(chunk_idx, file_idx)])
            rel = self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
            path = f"{data_root}/{rel}"
            with fsspec.open(path, "rb") as f:
                pf = pq.ParquetFile(f)
                n_rg = pf.num_row_groups
                num_rows = pf.metadata.num_rows
            if n_rg < n_ep:
                raise ValueError(
                    f"{path}: stored as {n_rg} Parquet row group(s) ({num_rows} rows across "
                    f"{n_ep} episodes). StreamingLeRobotDataset random-episode shuffling requires ONE ROW "
                    "GROUP PER EPISODE so each episode becomes an independently addressable shard after "
                    "reshard(). This file was written by a bulk df.to_parquet / push_to_hub / aggregate "
                    "path that collapses row groups. Re-emit through the LeRobot writer (one write_table "
                    "per episode) or fix the aggregate/annotate writer, then re-upload. Recorded datasets "
                    "already satisfy this. Pass validate_row_groups=False to bypass (quality will degrade)."
                )

    def _build_pipeline(self) -> datasets.IterableDataset:
        """Assemble the native tabular pipeline (everything except video decode)."""
        ds = self.hf_dataset
        if self.world_size > 1:
            if ds.num_shards % self.world_size != 0:
                msg = (
                    f"num_shards ({ds.num_shards}) is not divisible by world_size ({self.world_size}). "
                    "After reshard() num_shards == the episode count, and split_dataset_by_node only "
                    "assigns shards evenly when num_shards % world_size == 0; otherwise every rank "
                    "streams (and pays for) the full dataset and keeps only 1/world_size of it. Pin "
                    "world_size to a divisor of the episode count, or drop/pad episodes to a divisible "
                    "count with the dataset tools. Set validate_row_groups=False to downgrade to a warning."
                )
                if self.validate_row_groups:
                    raise ValueError(msg)
                logger.warning(msg)
            ds = split_dataset_by_node(ds, rank=self.rank, world_size=self.world_size)

        ds = ds.batch(by_column="episode_index")
        episode_columns = list(ds.column_names or self.hf_dataset.column_names or [])
        if self.shuffle:
            max_input_shards = max(1, min(self.max_buffer_input_shards, ds.num_shards))
            ds = ds.shuffle(
                seed=self.seed,
                buffer_size=self.episode_pool_size,
                max_buffer_input_shards=max_input_shards,
            )
        # A row-count-changing batched map must drop the input columns explicitly; the exploded
        # frames re-emit them (windowed keys replaced by their delta windows + *_is_pad masks).
        ds = ds.map(self._explode_episodes, batched=True, remove_columns=episode_columns)
        if self.shuffle:
            ds = ds.shuffle(seed=self.seed + 1, buffer_size=max(2, self.frame_shuffle_buffer_size))
        return ds

    def _tabular_window_keys(self) -> list[str]:
        if self.delta_indices is None:
            return []
        return [key for key in self.delta_indices if key not in self.meta.video_keys]

    def _explode_episodes(self, episode_batch: dict[str, list[list]]) -> dict[str, list]:
        """Episode batches -> per-frame rows, with exact tabular delta windows and pad masks.

        Runs inside the `datasets` pipeline (plain Python values, no torch). For each windowed key
        the original per-frame value is replaced by its delta window (list of values, clamped to
        the episode bounds) plus a ``{key}_is_pad`` mask, mirroring the map-style dataset.
        """
        window_keys = set(self._tabular_window_keys())
        out: dict[str, list] = {key: [] for key in episode_batch if key not in window_keys}
        for key in window_keys:
            out[key] = []
            out[f"{key}_is_pad"] = []

        num_episodes = len(episode_batch["episode_index"])
        for e in range(num_episodes):
            length = len(episode_batch["episode_index"][e])
            for key, column in episode_batch.items():
                if key in window_keys:
                    continue
                out[key].extend(column[e])
            for key in window_keys:
                episode_column = episode_batch[key][e]
                deltas = self.delta_indices[key]
                for t in range(length):
                    window = []
                    is_pad = []
                    for delta in deltas:
                        j = t + delta
                        window.append(episode_column[min(max(j, 0), length - 1)])
                        is_pad.append(not 0 <= j < length)
                    out[key].append(window)
                    out[f"{key}_is_pad"].append(is_pad)
        return out

    def _make_video_decoder_cache(self) -> VideoDecoderCache:
        """Size the decoder cache to the pool's working set (pool episodes x cameras), capped at 128."""
        if self.video_decoder_cache_size is not None:
            return VideoDecoderCache(max_size=self.video_decoder_cache_size)
        num_cameras = len(self.meta.video_keys)
        if num_cameras == 0:
            return VideoDecoderCache()
        return VideoDecoderCache(max_size=min((self.episode_pool_size + 1) * num_cameras, 128))

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # `datasets` reshuffles (and re-permutes shard order) per epoch from (seed, epoch);
        # DataLoader workers each advance their own copy's counter in lockstep. The in-flight
        # epoch is tracked separately so a mid-iteration state_dict() records the epoch the
        # stream position actually belongs to. Only advance when shuffling: after reshard() the
        # stream has one shard per episode, and set_epoch(n>0) re-permutes shard order even without
        # a shuffle op, so an unshuffled stream must pin epoch 0 to repeat the same order each pass.
        if self.shuffle:
            self._in_flight_epoch = self._epoch
            self._epoch += 1
        else:
            self._in_flight_epoch = 0
        self._pipeline.set_epoch(self._in_flight_epoch)
        self.video_decoder_cache = self._make_video_decoder_cache()

        iterator = iter(self._pipeline)
        while True:
            try:
                row = next(iterator)
            except StopIteration:
                return
            yield self._finalize_sample(row)

    def _finalize_sample(self, row: dict) -> dict:
        """Torch conversion + video decode (decode-on-exit) + transforms + task for one frame."""
        window_keys = self._tabular_window_keys()
        pad_masks = {f"{key}_is_pad": torch.BoolTensor(row.pop(f"{key}_is_pad")) for key in window_keys}
        item = item_to_torch(row)
        item.update(pad_masks)

        if len(self.meta.video_keys) > 0:
            ep_idx = int(item["episode_index"])
            current_ts = float(item["timestamp"])
            # Per-camera episode-local bounds [0, duration]: out-of-episode deltas pad instead of
            # decoding against a neighbouring episode sharing the same video file.
            episode_boundaries_ts = {
                key: (
                    0.0,
                    self.meta.episodes[ep_idx][f"videos/{key}/to_timestamp"]
                    - self.meta.episodes[ep_idx][f"videos/{key}/from_timestamp"],
                )
                for key in self.meta.video_keys
            }
            original_timestamps = self._make_timestamps_from_indices(current_ts, self.delta_indices)
            query_timestamps = self._get_query_timestamps(
                current_ts, self.delta_indices, episode_boundaries_ts
            )
            video_frames = self._query_videos(query_timestamps, ep_idx)

            if self.image_transforms is not None:
                for cam in self.meta.camera_keys:
                    video_frames[cam] = self.image_transforms(video_frames[cam])

            item.update(video_frames)
            if self.delta_indices is not None:
                item.update(
                    self._get_video_frame_padding_mask(video_frames, query_timestamps, original_timestamps)
                )

        item["task"] = self.meta.tasks.iloc[int(item["task_index"])].name
        return item

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch the next ``__iter__`` will use (reshuffles the native pipeline)."""
        self._epoch = epoch

    def state_dict(self) -> dict:
        """Native `datasets` stream state. Exact contract with ``num_workers=0``; with DataLoader
        workers use ``torchdata.stateful_dataloader.StatefulDataLoader`` (it checkpoints each
        worker's copy through this protocol). Samples in the shuffle buffers are skipped on
        resume (never repeated), bounded by the pool + frame buffer sizes.
        """
        return {"pipeline": self._pipeline.state_dict(), "epoch": self._in_flight_epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        # Resume continues inside the recorded epoch: the next __iter__ replays that epoch's
        # shuffle order from the restored stream position, then advances normally.
        self._epoch = int(state_dict.get("epoch", 0))
        self._pipeline.load_state_dict(state_dict["pipeline"])

    def _make_timestamps_from_indices(
        self, start_ts: float, indices: dict[str, list[int]] | None = None
    ) -> dict[str, list[float]]:
        if indices is not None:
            return {
                key: (start_ts + torch.tensor(indices[key]) / self.fps).tolist()
                for key in self.delta_timestamps
            }
        else:
            return dict.fromkeys(self.meta.video_keys, [start_ts])

    def _make_padding_camera_frame(self, camera_key: str):
        """Variable-shape padding frame for given camera keys, given in (H, W, C)"""
        return torch.zeros(self.meta.info.features[camera_key]["shape"]).permute(-1, 0, 1)

    def _get_video_frame_padding_mask(
        self,
        video_frames: dict[str, torch.Tensor],
        query_timestamps: dict[str, list[float]],
        original_timestamps: dict[str, list[float]],
    ) -> dict[str, torch.BoolTensor]:
        padding_mask = {}

        for video_key, timestamps in original_timestamps.items():
            if video_key not in video_frames:
                continue  # only padding on video keys that are available
            frames = []
            mask = []
            padding_frame = self._make_padding_camera_frame(video_key)
            for ts in timestamps:
                if is_float_in_list(ts, query_timestamps[video_key]):
                    idx = find_float_index(ts, query_timestamps[video_key])
                    frames.append(video_frames[video_key][idx, :])
                    mask.append(False)
                else:
                    frames.append(padding_frame)
                    mask.append(True)

            padding_mask[f"{video_key}_is_pad"] = torch.BoolTensor(mask)

        return padding_mask

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
        episode_boundaries_ts: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        keys_to_timestamps = self._make_timestamps_from_indices(current_ts, query_indices)
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = keys_to_timestamps[key]
                # Clamp out timesteps outside of episode boundaries
                query_timestamps[key] = torch.clamp(
                    torch.tensor(timestamps), *episode_boundaries_ts[key]
                ).tolist()

            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """

        item = {}
        for video_key, query_ts in query_timestamps.items():
            # query_ts is episode-local; shift to the absolute in-file timeline by the episode's offset.
            from_timestamp = self.meta.episodes[ep_idx][f"videos/{video_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            rel_path = str(self.meta.get_video_file_path(ep_idx, video_key))
            if self.data_files_root is not None:
                root = self.data_files_root
            elif self.streaming and not self.streaming_from_local:
                root = self.meta.url_root
            else:
                root = self.root
            video_path = f"{root}/{rel_path}"
            frames = decode_video_frames_torchcodec(
                video_path,
                shifted_query_ts,
                self.tolerance_s,
                decoder_cache=self.video_decoder_cache,
                return_uint8=self._return_uint8,
            )

            item[video_key] = frames.squeeze(0) if len(query_ts) == 1 else frames

        return item

    def _validate_delta_timestamp_keys(self, delta_timestamps: dict[list[float]]) -> None:
        """
        Validate that all keys in delta_timestamps correspond to actual features in the dataset.

        Raises:
            ValueError: If any delta timestamp key doesn't correspond to a dataset feature.
        """
        if delta_timestamps is None:
            return

        # Get all available feature keys from the dataset metadata
        available_features = set(self.meta.features.keys())

        # Get all keys from delta_timestamps
        delta_keys = set(delta_timestamps.keys())

        # Find any keys that don't correspond to features
        invalid_keys = delta_keys - available_features

        if invalid_keys:
            raise ValueError(
                f"The following delta_timestamp keys do not correspond to dataset features: {invalid_keys}. "
                f"Available features are: {sorted(available_features)}"
            )

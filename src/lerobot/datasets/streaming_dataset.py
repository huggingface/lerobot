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
import math
import os
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator
from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

from lerobot.utils.constants import HF_LEROBOT_HOME, LOOKAHEAD_BACKTRACKTABLE, LOOKBACK_BACKTRACKTABLE

from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .feature_utils import get_delta_indices
from .io_utils import item_to_torch
from .utils import (
    check_version_compatibility,
    find_float_index,
    is_float_in_list,
    safe_shard,
)
from .video_utils import (
    VideoDecoderCache,
    decode_video_frames_torchcodec,
)

logger = logging.getLogger(__name__)


class LookBackError(Exception):
    """
    Exception raised when trying to look back in the history of a Backtrackable object.
    """

    pass


class LookAheadError(Exception):
    """
    Exception raised when trying to look ahead in the future of a Backtrackable object.
    """

    pass


class Backtrackable[T]:
    """
    Wrap any iterator/iterable so you can step back up to `history` items
    and look ahead up to `lookahead` items.

    This is useful for streaming datasets where you need to access previous and future items
    but can't load the entire dataset into memory.

    Example:
    -------
    ```python
    ds = load_dataset("c4", "en", streaming=True, split="train")
    rev = Backtrackable(ds, history=3, lookahead=2)

    x0 = next(rev)  # forward
    x1 = next(rev)
    x2 = next(rev)

    # Look ahead
    x3_peek = rev.peek_ahead(1)  # next item without moving cursor
    x4_peek = rev.peek_ahead(2)  # two items ahead

    # Look back
    x1_again = rev.peek_back(1)  # previous item without moving cursor
    x0_again = rev.peek_back(2)  # two items back

    # Move backward
    x1_back = rev.prev()  # back one step
    next(rev)  # returns x2, continues forward from where we were
    ```
    """

    __slots__ = ("_source", "_back_buf", "_ahead_buf", "_cursor", "_history", "_lookahead")

    def __init__(self, iterable: Iterable[T], *, history: int = 1, lookahead: int = 0):
        if history < 1:
            raise ValueError("history must be >= 1")
        if lookahead <= 0:
            raise ValueError("lookahead must be > 0")

        self._source: Iterator[T] = iter(iterable)
        self._back_buf: deque[T] = deque(maxlen=history)
        self._ahead_buf: deque[T] = deque(maxlen=lookahead) if lookahead > 0 else deque()
        self._cursor: int = 0
        self._history = history
        self._lookahead = lookahead

    def __iter__(self) -> "Backtrackable[T]":
        return self

    def __next__(self) -> T:
        # If we've stepped back, consume from back buffer first
        if self._cursor < 0:  # -1 means "last item", etc.
            self._cursor += 1
            return self._back_buf[self._cursor]

        # If we have items in the ahead buffer, use them first
        item = self._ahead_buf.popleft() if self._ahead_buf else next(self._source)

        # Add current item to back buffer and reset cursor
        self._back_buf.append(item)
        self._cursor = 0
        return item

    def prev(self) -> T:
        """
        Step one item back in history and return it.
        Raises IndexError if already at the oldest buffered item.
        """
        if len(self._back_buf) + self._cursor <= 1:
            raise LookBackError("At start of history")

        self._cursor -= 1
        return self._back_buf[self._cursor]

    def peek_back(self, n: int = 1) -> T:
        """
        Look `n` items back (n=1 == previous item) without moving the cursor.
        """
        if n < 0 or n + 1 > len(self._back_buf) + self._cursor:
            raise LookBackError("peek_back distance out of range")

        return self._back_buf[self._cursor - (n + 1)]

    def peek_ahead(self, n: int = 1) -> T:
        """
        Look `n` items ahead (n=1 == next item) without moving the cursor.
        Fills the ahead buffer if necessary.
        """
        if n < 1:
            raise LookAheadError("peek_ahead distance must be 1 or more")
        elif n > self._lookahead:
            raise LookAheadError("peek_ahead distance exceeds lookahead limit")

        # Fill ahead buffer if we don't have enough items
        while len(self._ahead_buf) < n:
            try:
                item = next(self._source)
                self._ahead_buf.append(item)

            except StopIteration as err:
                raise LookAheadError("peek_ahead: not enough items in source") from err

        return self._ahead_buf[n - 1]

    def history(self) -> list[T]:
        """
        Return a copy of the buffered history (most recent last).
        The list length ≤ `history` argument passed at construction.
        """
        if self._cursor == 0:
            return list(self._back_buf)

        # When cursor<0, slice so the order remains chronological
        return list(self._back_buf)[: self._cursor or None]

    def can_peek_back(self, steps: int = 1) -> bool:
        """
        Check if we can go back `steps` items without raising an IndexError.
        """
        return steps <= len(self._back_buf) + self._cursor

    def can_peek_ahead(self, steps: int = 1) -> bool:
        """
        Check if we can peek ahead `steps` items.
        This may involve trying to fill the ahead buffer.
        """
        if self._lookahead > 0 and steps > self._lookahead:
            return False

        # Try to fill ahead buffer to check if we can peek that far
        try:
            while len(self._ahead_buf) < steps:
                if self._lookahead > 0 and len(self._ahead_buf) >= self._lookahead:
                    return False
                item = next(self._source)
                self._ahead_buf.append(item)
            return True
        except StopIteration:
            return False


class StreamingLeRobotDataset(torch.utils.data.IterableDataset):
    """LeRobotDataset with streaming capabilities.

    This class extends LeRobotDataset to add streaming functionality, allowing data to be streamed
    rather than loaded entirely into memory. This is especially useful for large datasets that may
    not fit in memory or when you want to quickly explore a dataset without downloading it completely.

    The key innovation is using a Backtrackable iterator that maintains a bounded buffer of recent
    items, allowing us to access previous frames for delta timestamps without loading the entire
    dataset into memory.

    Example:
        Basic usage:
        ```python
        from lerobot.common.datasets.streaming_dataset import StreamingLeRobotDataset

        # Create a streaming dataset with delta timestamps
        delta_timestamps = {
            "observation.image": [-1.0, -0.5, 0.0],  # 1 sec ago, 0.5 sec ago, current
            "action": [0.0, 0.1, 0.2],  # current, 0.1 sec future, 0.2 sec future
        }

        dataset = StreamingLeRobotDataset(
            repo_id="your-dataset-repo-id",
            delta_timestamps=delta_timestamps,
            streaming=True,
            buffer_size=1000,
        )

        # Iterate over the dataset
        for i, item in enumerate(dataset):
            print(f"Sample {i}: Episode {item['episode_index']} Frame {item['frame_index']}")
            # item will contain stacked frames according to delta_timestamps
            if i >= 10:
                break
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
        buffer_size: int = 1000,
        max_num_shards: int = 16,
        seed: int = 42,
        rng: np.random.Generator | None = None,
        shuffle: bool = True,
        return_uint8: bool = False,
        rank: int | None = None,
        world_size: int | None = None,
        video_decoder_cache_size: int | None = None,
        data_files_root: str | None = None,
        video_decode_device: str = "cpu",
        episode_pool_size: int | None = None,
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
            buffer_size (int, optional): Buffer size for shuffling when streaming. Defaults to 1000.
            max_num_shards (int, optional): Number of shards to re-shard the input dataset into. Defaults to 16.
            seed (int, optional): Reproducibility random seed.
            rng (np.random.Generator | None, optional): Random number generator.
            shuffle (bool, optional): Whether to shuffle the dataset across exhaustions. Defaults to True.
            rank (int | None, optional): This process' rank for distributed (multi-GPU/multi-node) training.
                Each rank streams a disjoint set of shards via ``split_dataset_by_node``. When omitted, it is
                resolved from Accelerate (``process_index``) or the ``RANK`` env var, defaulting to 0.
            world_size (int | None, optional): Total number of distributed processes. When omitted, resolved
                from Accelerate (``num_processes``) or the ``WORLD_SIZE`` env var, defaulting to 1 (no sharding).
                For an even per-rank split, ``num_shards % world_size == 0`` should hold.
            video_decoder_cache_size (int | None, optional): Max number of open video decoders to retain.
                When omitted, it defaults to ``(concurrent active shards + 1) × num_cameras`` so the working
                set of live decoders never thrashes. See :class:`VideoDecoderCache`.
            data_files_root (str | None, optional): fsspec root holding the bulk ``data/`` and ``videos/``
                trees (e.g. an HF storage bucket ``hf://buckets/<owner>/<name>``). When set, parquet and
                video frames are read from there while metadata still loads from ``repo_id`` on the Hub.
                Resolves through fsspec exactly like ``hf://``; use it to benchmark bucket / prewarmed-bucket
                sources without copying the (small) metadata.
            video_decode_device (str, optional): Device for video decoding, passed to the torchcodec
                ``VideoDecoder``. Defaults to ``"cpu"``. Set to ``"cuda"`` to offload H.264/H.265 decode to
                the GPU's dedicated NVDEC engine (independent of the training SMs), which requires a
                CUDA-enabled torchcodec build. Note: ``"cuda"`` decode inside ``DataLoader`` workers needs
                the ``spawn`` start method (CUDA cannot init in forked workers).
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
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.shuffle = shuffle

        self.streaming = streaming
        self.buffer_size = buffer_size
        self.max_num_shards = max_num_shards
        self._return_uint8 = return_uint8

        self.rank, self.world_size = self._resolve_distributed(rank, world_size)
        self.video_decoder_cache_size = video_decoder_cache_size
        self.data_files_root = data_files_root.rstrip("/") if data_files_root else None
        self.video_decode_device = video_decode_device
        # A3 shuffle: when set, iterate by keeping this many full episodes live in memory and sampling
        # frames uniformly across them (mixing radius = episode_pool_size episodes), instead of the
        # default per-shard reservoir. Tabular deltas become exact in-episode index lookups (no
        # Backtrackable). Trades video-decode locality for much stronger shuffle.
        self.episode_pool_size = episode_pool_size

        # We cache the video decoders to avoid re-initializing them at each frame (avoiding a ~10x slowdown)
        self.video_decoder_cache = None
        # Shared [hits, misses, evictions, decode_ns, fetch_ns] tensor so DataLoader workers aggregate
        # decoder-cache stats and component timings into one place the main process can read after
        # iteration (see video_decoder_cache_stats() / timing_stats()).
        self._cache_counters = torch.zeros(5, dtype=torch.int64).share_memory_()
        # Resume state captured by load_state_dict() and consumed at the next __iter__.
        self._resume_state: dict | None = None

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

        self.num_shards = min(self.hf_dataset.num_shards, max_num_shards)

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
    def _iter_random_indices(
        rng: np.random.Generator, buffer_size: int, random_batch_size=100
    ) -> Iterator[int]:
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))

    @staticmethod
    def _infinite_generator_over_elements(rng: np.random.Generator, elements: list[int]) -> Iterator[int]:
        while True:
            yield rng.choice(elements)

    @staticmethod
    def _resolve_distributed(rank: int | None, world_size: int | None) -> tuple[int, int]:
        """Resolve (rank, world_size) for distributed streaming.

        Explicit arguments win. Otherwise prefer an already-initialized Accelerate state, then the
        ``RANK``/``WORLD_SIZE`` env vars set by launchers, and finally fall back to single-process (0, 1).
        """
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

    def _make_video_decoder_cache(self, num_active_shards: int) -> VideoDecoderCache:
        """Size the decoder cache to the working set of live shards so it does not thrash.

        Each shard mid-episode keeps one open decoder per camera; with several shards iterated
        concurrently the working set is ``num_active_shards × num_cameras``. We add one shard worth of
        margin so the round-robin never evicts a still-live decoder.
        """
        if self.video_decoder_cache_size is not None:
            return VideoDecoderCache(
                max_size=self.video_decoder_cache_size,
                counters=self._cache_counters,
                device=self.video_decode_device,
            )
        num_cameras = len(self.meta.video_keys)
        if num_cameras == 0:
            return VideoDecoderCache(counters=self._cache_counters, device=self.video_decode_device)
        return VideoDecoderCache(
            max_size=(num_active_shards + 1) * num_cameras,
            counters=self._cache_counters,
            device=self.video_decode_device,
        )

    # TODO(fracapuano): Implement multi-threaded prefetching to accelerate data loading.
    # The current sequential iteration is a bottleneck. A producer-consumer pattern
    # could be used with a ThreadPoolExecutor to run `make_frame` (especially video decoding)
    # in parallel, feeding a queue from which this iterator will yield processed items.
    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # Distributed correctness: each rank streams a disjoint set of shards (order preserved).
        ds = self.hf_dataset
        if self.world_size > 1:
            ds = split_dataset_by_node(ds, rank=self.rank, world_size=self.world_size)

        num_shards = min(ds.num_shards, self.max_num_shards)
        shard_indices = list(range(num_shards))

        # DataLoader workers within this rank further split the shards so they don't yield duplicates.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard_indices = shard_indices[worker_info.id :: worker_info.num_workers]

        self.video_decoder_cache = self._make_video_decoder_cache(len(shard_indices))

        # keep the same seed across exhaustions if shuffle is False, otherwise shuffle data across exhaustions
        rng = np.random.default_rng(self.seed) if not self.shuffle else self.rng

        # Best-effort resume: restore RNG + exhausted shards and rewind each shard's HF stream. The
        # shuffle buffer is re-warmed rather than restored, so resumption is not bit-exact (acceptable
        # for pretraining); the underlying stream may also skip the few frames Backtrackable read ahead.
        resume = self._resume_state
        self._resume_state = None
        self._exhausted: set[int] = set(resume["exhausted"]) if resume is not None else set()
        if resume is not None:
            rng.bit_generator.state = resume["rng"]

        self._shards: dict[int, datasets.IterableDataset] = {}
        for idx in shard_indices:
            shard = safe_shard(ds, idx, num_shards)
            if resume is not None and str(idx) in resume["shards"]:
                shard.load_state_dict(resume["shards"][str(idx)])
            self._shards[idx] = shard

        # A3 episode-pool shuffle (opt-in): sample frames uniformly across many fully-loaded episodes.
        if self.episode_pool_size:
            shard_iters = {
                idx: iter(self._shards[idx]) for idx in shard_indices if idx not in self._exhausted
            }
            yield from self._iter_episode_pool(shard_iters, rng)
            return

        buffer_indices_generator = self._iter_random_indices(rng, self.buffer_size)

        idx_to_backtrack_dataset = {
            idx: self._make_backtrackable_dataset(shard)
            for idx, shard in self._shards.items()
            if idx not in self._exhausted
        }

        # This buffer is populated while iterating on the dataset's shards
        # the logic is to add 2 levels of randomness:
        # (1) sample one shard at random from the ones available, and
        # (2) sample one frame from the shard sampled at (1)
        # Buffer entries are (partial, video_spec): undecoded tabular rows. Video is decoded by
        # _attach_video only when a sample leaves the buffer, keeping peak memory ~prefetch-bounded.
        frames_buffer = []
        while available_shards := list(idx_to_backtrack_dataset.keys()):
            shard_key = next(self._infinite_generator_over_elements(rng, available_shards))
            backtrack_dataset = idx_to_backtrack_dataset[shard_key]  # selects which shard to iterate on

            try:
                for frame in self.make_frame(backtrack_dataset):
                    if len(frames_buffer) == self.buffer_size:
                        i = next(buffer_indices_generator)  # samples a element from the buffer
                        yield self._attach_video(*frames_buffer[i])  # decode just-in-time on the way out
                        frames_buffer[i] = frame
                    else:
                        frames_buffer.append(frame)
                    break  # random shard sampled, switch shard
            except (
                RuntimeError,
                StopIteration,
            ):  # NOTE: StopIteration inside a generator throws a RuntimeError since python 3.7
                del idx_to_backtrack_dataset[shard_key]  # Remove exhausted shard, onto another shard
                self._exhausted.add(shard_key)

        # Once shards are all exhausted, shuffle the buffer and yield the remaining frames (decoding each).
        rng.shuffle(frames_buffer)
        for partial, video_spec in frames_buffer:
            yield self._attach_video(partial, video_spec)

    def state_dict(self) -> dict:
        """Capture resume state: per-shard HF stream position, exhausted shards, and RNG state.

        Must be called after iteration has started (so the shard streams exist). Restore the returned
        dict with :meth:`load_state_dict` before re-iterating. The shuffle buffer is not captured, so
        resumption is not bit-exact — see :meth:`__iter__`.
        """
        if not hasattr(self, "_shards"):
            raise RuntimeError("state_dict() requires the dataset to have been iterated at least once.")
        return {
            "shards": {str(idx): shard.state_dict() for idx, shard in self._shards.items()},
            "exhausted": sorted(self._exhausted),
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Stage resume state captured by :meth:`state_dict`; applied at the next ``__iter__``."""
        self._resume_state = state_dict

    def video_decoder_cache_stats(self) -> dict[str, int | float]:
        """Decoder-cache reuse aggregated across DataLoader workers via the shared counter tensor.

        Unlike ``self.video_decoder_cache.stats()`` (which only reflects the main process), this sums
        hits/misses/evictions over every worker. Counts are lock-free across processes, so treat them as
        approximate; the ``hit_rate`` ratio is preserved.
        """
        hits, misses, evictions = (int(x) for x in self._cache_counters[:3].tolist())
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "evictions": evictions,
            "hit_rate": round(hits / total, 4) if total else 0.0,
        }

    def timing_stats(self) -> dict[str, float]:
        """Cumulative seconds spent in video decode and parquet/sample fetch, summed across DataLoader
        workers via the shared counter tensor. These overlap in wall-clock (workers run in parallel), so
        compare them to ``num_workers x wallclock`` — not to wallclock directly — to get time fractions.
        """
        decode_ns, fetch_ns = (int(x) for x in self._cache_counters[3:5].tolist())
        return {"decode_s_total": round(decode_ns / 1e9, 2), "fetch_s_total": round(fetch_ns / 1e9, 2)}

    def _get_window_steps(
        self, delta_timestamps: dict[str, list[float]] | None = None, dynamic_bounds: bool = False
    ) -> tuple[int, int]:
        if delta_timestamps is None:
            return 1, 1

        if not dynamic_bounds:
            # Fix the windows
            lookback = LOOKBACK_BACKTRACKTABLE
            lookahead = LOOKAHEAD_BACKTRACKTABLE
        else:
            # Dynamically size the windows to exactly cover the requested delta_timestamps (in frames).
            # This removes the fixed LOOKAHEAD_BACKTRACKTABLE ceiling, which would raise LookAheadError for
            # long horizons (e.g. a SARM window of 8 steps spaced 1s = ~160 frames @ fps20).
            all_timestamps = sum(delta_timestamps.values(), [])
            lookback = math.floor(min(all_timestamps) * self.fps)
            lookahead = math.ceil(max(all_timestamps) * self.fps)

            # When lookback is >=0 it means no negative timesteps have been provided
            lookback = 0 if lookback >= 0 else -lookback

        return lookback, lookahead

    def _make_backtrackable_dataset(self, dataset: datasets.IterableDataset) -> Backtrackable:
        lookback, lookahead = self._get_window_steps(self.delta_timestamps, dynamic_bounds=True)
        # Backtrackable.peek_back(n) needs `history >= n + 1`, so reach a frame `lookback` steps back requires
        # history = lookback + 1. history must be >= 1 and lookahead > 0, so clamp both to at least 1.
        return Backtrackable(dataset, history=max(1, lookback + 1), lookahead=max(1, lookahead))

    def _make_timestamps_from_indices(
        self, start_ts: float, indices: dict[str, list[int]] | None = None
    ) -> dict[str, list[float]]:
        if indices is not None:
            return {
                key: (
                    start_ts + torch.tensor(indices[key]) / self.fps
                ).tolist()  # NOTE: why not delta_timestamps directly?
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

    def make_frame(self, dataset_iterator: Backtrackable) -> Generator:
        """Build a frame's tabular content and defer the video decode.

        Yields a ``(partial, video_spec)`` pair: ``partial`` holds all non-video fields (tabular
        features, tabular delta windows + padding, task); ``video_spec`` carries what
        :meth:`_attach_video` needs to decode the camera frames just-in-time at yield time. Deferring
        the decode keeps the shuffle reservoir holding ~KB tabular rows instead of multi-MB decoded
        images, which collapses peak memory.
        """
        _t0 = time.perf_counter_ns()
        item = next(dataset_iterator)
        self._cache_counters[4] += time.perf_counter_ns() - _t0  # parquet/sample fetch time
        item = item_to_torch(item)

        updates = []  # list of "updates" to apply to the item retrieved from hf_dataset (w/o camera features)

        # Get episode index from the item
        ep_idx = item["episode_index"]

        # `timestamp` is episode-local (restarts at 0 each episode). The absolute in-file timestamp is
        # `from_timestamp + timestamp`, applied per camera at decode time (see `_query_videos`), mirroring
        # the map-style reader. Using `index / fps` here is a dataset-global value that only matches the
        # file timeline when the whole dataset is a single video (e.g. small test fixtures), and otherwise
        # decodes out-of-range frames on multi-file v3 datasets.
        current_ts = float(item["timestamp"])

        # Per-camera episode-local bounds [0, duration]. Query timestamps are clamped into this range so
        # out-of-episode deltas pad rather than decode against a neighbouring episode in the same file.
        episode_boundaries_ts = {
            key: (
                0.0,
                self.meta.episodes[ep_idx][f"videos/{key}/to_timestamp"]
                - self.meta.episodes[ep_idx][f"videos/{key}/from_timestamp"],
            )
            for key in self.meta.video_keys
        }

        # Apply delta querying logic if necessary
        if self.delta_indices is not None:
            query_result, padding = self._get_delta_frames(dataset_iterator, item)
            updates.append(query_result)
            updates.append(padding)

        # Defer the (memory-heavy) video decode: capture only what _attach_video needs to decode the
        # camera frames at yield time, so the shuffle buffer holds ~KB tabular rows, not MB of pixels.
        video_spec = None
        if len(self.meta.video_keys) > 0:
            original_timestamps = self._make_timestamps_from_indices(current_ts, self.delta_indices)
            # Some timestamps might not be available considering the episode's boundaries
            query_timestamps = self._get_query_timestamps(
                current_ts, self.delta_indices, episode_boundaries_ts
            )
            video_spec = (query_timestamps, original_timestamps, ep_idx)

        result = item.copy()
        for update in updates:
            result.update(update)

        result["task"] = self.meta.tasks.iloc[item["task_index"]].name

        yield result, video_spec

    def _attach_video(self, result: dict, video_spec: tuple | None) -> dict:
        """Decode the camera frames for a buffered sample and merge them in (counterpart to make_frame).

        This is where torchcodec decode actually runs — on one sample at a time as it leaves the shuffle
        buffer — so peak memory is bounded by the prefetch queue rather than ``buffer_size`` decoded frames.
        """
        if video_spec is None:
            return result
        query_timestamps, original_timestamps, ep_idx = video_spec
        video_frames = self._query_videos(query_timestamps, ep_idx)
        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                video_frames[cam] = self.image_transforms(video_frames[cam])
        result.update(video_frames)
        if self.delta_indices is not None:
            # We always return the same number of frames. Unavailable frames are padded.
            padding_mask = self._get_video_frame_padding_mask(
                video_frames, query_timestamps, original_timestamps
            )
            result.update(padding_mask)
        return result

    @staticmethod
    def _ep_id(raw_item: dict) -> int:
        """Episode index of a raw (pre-torch) HF stream row, coerced to a plain int."""
        return int(np.asarray(raw_item["episode_index"]).reshape(-1)[0])

    def _read_one_episode(self, sid: int, shard_iters: dict, carry: dict) -> list[dict] | None:
        """Read one full episode (contiguous rows) from a shard iterator, or None if exhausted.

        Episodes are contiguous in the stream, so we read until ``episode_index`` changes and stash the
        first row of the next episode in ``carry`` to start the following read.
        """
        it = shard_iters[sid]
        first = carry[sid]
        carry[sid] = None
        if first is None:
            first = next(it, None)
            if first is None:
                return None
        ep = self._ep_id(first)
        rows = [first]
        for row in it:
            if self._ep_id(row) != ep:
                carry[sid] = row  # belongs to the next episode; start there next time
                break
            rows.append(row)
        return rows

    def _make_frame_from_episode(self, ep_rows: list[dict], p: int) -> tuple[dict, tuple | None]:
        """Build ``(partial, video_spec)`` for frame ``p`` of a fully-loaded episode (A3).

        All temporal neighbors live in ``ep_rows``, so tabular delta windows are exact index lookups
        with correct episode-boundary padding — no Backtrackable, no lookahead pre-read. Video is still
        decoded just-in-time by :meth:`_attach_video`.
        """
        item = ep_rows[p]
        ep_idx = item["episode_index"]
        current_ts = float(item["timestamp"])
        length = len(ep_rows)

        updates = []
        if self.delta_indices is not None:
            query_result, padding = {}, {}
            for key, deltas in self.delta_indices.items():
                if key in self.meta.video_keys:
                    continue  # visual frames are decoded separately
                frames, is_pad = [], []
                for d in deltas:
                    q = p + d
                    clamped = min(max(q, 0), length - 1)  # out-of-episode neighbors pad to the boundary
                    frames.append(ep_rows[clamped][key])
                    is_pad.append(q != clamped)
                query_result[key] = torch.stack(frames)
                padding[f"{key}_is_pad"] = torch.BoolTensor(is_pad)
            updates.append(query_result)
            updates.append(padding)

        video_spec = None
        if len(self.meta.video_keys) > 0:
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
            video_spec = (query_timestamps, original_timestamps, ep_idx)

        result = item.copy()
        for update in updates:
            result.update(update)
        result["task"] = self.meta.tasks.iloc[item["task_index"]].name
        return result, video_spec

    def _iter_episode_pool(self, shard_iters: dict, rng: np.random.Generator) -> Iterator[dict]:
        """A3 shuffle: keep ``episode_pool_size`` full episodes live and sample frames uniformly across
        them. Each episode costs ~one sequential read (IO-cheap); the mixing radius is the pool size.

        ``tickets`` holds one (slot, frame_pos) entry per live, not-yet-emitted frame; swap-remove gives
        O(1) uniform sampling without replacement. When an episode drains it is evicted and a fresh one
        is read in, keeping the pool full.
        """
        carry = {sid: None for sid in shard_iters}
        live = set(shard_iters)
        pool: dict[int, dict] = {}  # slot -> {"rows": [...], "remaining": int}
        tickets: list[tuple[int, int]] = []
        next_slot = 0

        def load_episode() -> bool:
            nonlocal next_slot
            while live:
                sid = int(rng.choice(tuple(live)))
                rows = self._read_one_episode(sid, shard_iters, carry)
                if rows is None:
                    live.discard(sid)
                    continue
                ep_rows = [item_to_torch(r) for r in rows]
                pool[next_slot] = {"rows": ep_rows, "remaining": len(ep_rows)}
                tickets.extend((next_slot, p) for p in range(len(ep_rows)))
                next_slot += 1
                return True
            return False

        while len(pool) < self.episode_pool_size and load_episode():
            pass

        while tickets:
            i = int(rng.integers(len(tickets)))
            slot, p = tickets[i]
            tickets[i] = tickets[-1]  # swap-remove: O(1) sampling without replacement
            tickets.pop()
            partial, video_spec = self._make_frame_from_episode(pool[slot]["rows"], p)
            yield self._attach_video(partial, video_spec)
            pool[slot]["remaining"] -= 1
            if pool[slot]["remaining"] == 0:
                del pool[slot]  # free the episode's frames
                load_episode()  # refill to keep the pool (and mixing radius) full

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
            if self.data_files_root is not None:
                root = self.data_files_root
            elif self.streaming and not self.streaming_from_local:
                root = self.meta.url_root
            else:
                root = self.root
            video_path = f"{root}/{self.meta.get_video_file_path(ep_idx, video_key)}"
            _t0 = time.perf_counter_ns()
            frames = decode_video_frames_torchcodec(
                video_path,
                shifted_query_ts,
                self.tolerance_s,
                decoder_cache=self.video_decoder_cache,
                return_uint8=self._return_uint8,
            )
            self._cache_counters[3] += time.perf_counter_ns() - _t0  # video decode time

            item[video_key] = frames.squeeze(0) if len(query_ts) == 1 else frames

        return item

    def _get_delta_frames(self, dataset_iterator: Backtrackable, current_item: dict):
        # TODO(fracapuano): Modularize this function, refactor the code
        """Get frames with delta offsets using the backtrackable iterator.

        Args:
            current_item (dict): Current item from the iterator.
            ep_idx (int): Episode index.

        Returns:
            tuple: (query_result, padding) - frames at delta offsets and padding info.
        """
        current_episode_idx = current_item["episode_index"]

        # Prepare results
        query_result = {}
        padding = {}

        for key, delta_indices in self.delta_indices.items():
            if key in self.meta.video_keys:
                continue  # visual frames are decoded separately

            target_frames = []
            is_pad = []

            # Create a results dictionary to store frames in processing order, then reconstruct original order for stacking
            delta_results = {}

            # Separate and sort deltas by difficulty (easier operations first)
            negative_deltas = sorted([d for d in delta_indices if d < 0], reverse=True)  # [-1, -2, -3, ...]
            positive_deltas = sorted([d for d in delta_indices if d > 0])  # [1, 2, 3, ...]
            zero_deltas = [d for d in delta_indices if d == 0]

            # Process zero deltas (current frame)
            for delta in zero_deltas:
                delta_results[delta] = (
                    current_item[key],
                    False,
                )

            # Process negative deltas in order of increasing difficulty
            lookback_failed = False

            last_successful_frame = current_item[key]

            for delta in negative_deltas:
                if lookback_failed:
                    delta_results[delta] = (last_successful_frame, True)
                    continue

                try:
                    steps_back = abs(delta)
                    if dataset_iterator.can_peek_back(steps_back):
                        past_item = dataset_iterator.peek_back(steps_back)
                        past_item = item_to_torch(past_item)

                        if past_item["episode_index"] == current_episode_idx:
                            delta_results[delta] = (past_item[key], False)
                            last_successful_frame = past_item[key]

                        else:
                            raise LookBackError("Retrieved frame is from different episode!")
                    else:
                        raise LookBackError("Cannot go back further than the history buffer!")

                except LookBackError:
                    delta_results[delta] = (last_successful_frame, True)
                    lookback_failed = True  # All subsequent negative deltas will also fail

            # Process positive deltas in order of increasing difficulty
            lookahead_failed = False
            last_successful_frame = current_item[key]

            for delta in positive_deltas:
                if lookahead_failed:
                    delta_results[delta] = (last_successful_frame, True)
                    continue

                try:
                    if dataset_iterator.can_peek_ahead(delta):
                        future_item = dataset_iterator.peek_ahead(delta)
                        future_item = item_to_torch(future_item)

                        if future_item["episode_index"] == current_episode_idx:
                            delta_results[delta] = (future_item[key], False)
                            last_successful_frame = future_item[key]

                        else:
                            raise LookAheadError("Retrieved frame is from different episode!")
                    else:
                        raise LookAheadError("Cannot go ahead further than the lookahead buffer!")

                except LookAheadError:
                    delta_results[delta] = (last_successful_frame, True)
                    lookahead_failed = True  # All subsequent positive deltas will also fail

            # Reconstruct original order for stacking
            for delta in delta_indices:
                frame, is_padded = delta_results[delta]

                # add batch dimension for stacking
                target_frames.append(frame)  # frame.unsqueeze(0))
                is_pad.append(is_padded)

            # Stack frames and add to results
            if target_frames:
                query_result[key] = torch.stack(target_frames)
                padding[f"{key}_is_pad"] = torch.BoolTensor(is_pad)

        return query_result, padding

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

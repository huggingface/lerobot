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
import contextlib
import inspect
import logging
import os
import shutil
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import datasets
import fsspec
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
    safe_shard,
)
from .video_utils import (
    VideoDecoderCache,
    decode_video_frames_torchcodec,
)

logger = logging.getLogger(__name__)

# datasets >= 5 groups a stream into whole-episode batches natively (Arrow-side accumulation,
# https://github.com/huggingface/datasets/pull/8172); older versions fall back to a Python row loop.
_HAS_BATCH_BY_COLUMN = "by_column" in inspect.signature(datasets.IterableDataset.batch).parameters

_MASK_64 = (1 << 64) - 1


def _mix64(x: int) -> int:
    """SplitMix64 finalizer (64-bit integer hash) for seed derivation."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK_64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK_64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK_64
    x ^= x >> 31
    return x


@contextlib.contextmanager
def _suppress_hf_worker_split():
    """Hide the torch DataLoader worker context from `datasets` while we drain its streams.

    `datasets` detects torch workers and re-splits its shards across them internally
    (`_iter_pytorch`); this dataset already assigns disjoint shards per worker, so the second
    split silently drops data whenever a per-worker stream has fewer internal shards than there
    are workers — and on datasets 5.0 it also crashes `batch(by_column=...)`. The patch is local
    to this DataLoader worker process and restored on exit.
    """
    original = torch.utils.data.get_worker_info
    torch.utils.data.get_worker_info = lambda: None
    try:
        yield
    finally:
        torch.utils.data.get_worker_info = original


class _PooledEpisode:
    """A fully-loaded episode's tabular rows plus emission bookkeeping."""

    __slots__ = ("episode_index", "rows", "remaining", "video_rel_paths")

    def __init__(self, episode_index: int, rows: list[dict], video_rel_paths: list[str]):
        self.episode_index = episode_index
        self.rows = rows
        self.remaining = list(range(len(rows)))
        self.video_rel_paths = video_rel_paths


class _VideoPrefetcher:
    """Background downloader of episode video files into a local cache (decode-on-exit support).

    Files are refcounted because LeRobot v3 packs several episodes per video file: a file is
    downloaded once when the first pooled episode referencing it is admitted and deleted when
    the last one is evicted. Downloads resolve through fsspec (hf://, s3://, https://, ...).
    """

    def __init__(self, remote_root: str, cache_dir: Path, max_workers: int = 4):
        self._remote_root = remote_root.rstrip("/")
        self._cache_dir = cache_dir
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="video-prefetch")
        self._refcounts: dict[str, int] = {}
        self._futures: dict[str, Future] = {}

    def acquire(self, rel_path: str) -> None:
        self._refcounts[rel_path] = self._refcounts.get(rel_path, 0) + 1
        if rel_path not in self._futures:
            self._futures[rel_path] = self._executor.submit(self._download, rel_path)

    def _download(self, rel_path: str) -> Path:
        local = self._cache_dir / rel_path
        if local.exists():
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + ".tmp")
        with fsspec.open(f"{self._remote_root}/{rel_path}", "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 22)
        tmp.rename(local)
        return local

    def wait_local(self, rel_path: str) -> Path | None:
        """Block until the file is cached; None when not tracked or the download failed."""
        future = self._futures.get(rel_path)
        if future is None:
            return None
        try:
            return future.result()
        except Exception as e:
            logger.warning(f"Video prefetch failed for {rel_path} ({e}); decoding from remote instead.")
            return None

    def release(self, rel_path: str) -> None:
        count = self._refcounts.get(rel_path, 0) - 1
        if count > 0:
            self._refcounts[rel_path] = count
            return
        self._refcounts.pop(rel_path, None)
        future = self._futures.pop(rel_path, None)
        if future is None:
            return
        if not future.cancel():
            try:
                local = future.result()
                local.unlink(missing_ok=True)
            except Exception:
                logger.debug(f"Could not delete cached video {rel_path}.", exc_info=True)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class StreamingLeRobotDataset(torch.utils.data.IterableDataset):
    """LeRobotDataset with streaming capabilities.

    Streams frames from the Hub (or any fsspec source) without downloading the dataset. The
    iteration strategy is an *episode pool*: each consumer keeps ``episode_pool_size`` whole
    episodes' tabular rows in RAM (a few KB per episode) and emits uniformly random frames
    across them, so a batch mixes up to ``batch_size`` distinct episodes. Because a frame's
    whole episode is resident, ``delta_timestamps`` windows are exact array slices with correct
    padding at episode boundaries. Video is decoded only when a sample is emitted
    (decode-on-exit), so pool memory stays tabular-sized; when streaming from a remote source,
    each pooled episode's video files are prefetched to a local cache in the background and
    deleted on eviction.

    Distribution: ranks stream disjoint shards via ``split_dataset_by_node`` and DataLoader
    workers split a rank's shards further, so every frame is consumed exactly once per epoch
    across the whole fleet. Each consumer's order is a pure function of
    ``(seed, epoch, rank, worker)``, which makes resume a deterministic fast-forward (see
    :meth:`load_state_dict`).

    Example:
        ```python
        dataset = StreamingLeRobotDataset(
            repo_id="your-dataset-repo-id",
            delta_timestamps={"action": [0.0, 0.1, 0.2]},
            episode_pool_size=64,
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
        episode_pool_size: int | None = 64,
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
        video_decode_device: str = "cpu",
        prefetch_videos: bool = True,
        video_prefetch_workers: int = 4,
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
            episode_pool_size (int, optional): Number of whole episodes each consumer keeps open to
                shuffle across — the randomness knob. Larger mixes more episodes per batch (closer to
                map-style uniform) at the cost of cold-start latency; RAM stays small because the pool
                holds tabular rows only. Defaults to 64.
            buffer_size (int | None, optional): Deprecated; superseded by ``episode_pool_size``.
            max_num_shards (int | None, optional): Cap on the number of stream shards. None (default)
                uses every underlying parquet shard, which is required to feed many DataLoader workers.
            seed (int, optional): Reproducibility random seed.
            rng (np.random.Generator | None, optional): Deprecated; ignored (the RNG is derived from
                ``(seed, epoch, rank, worker)`` so consumers are decorrelated and runs reproducible).
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
            video_decode_device (str, optional): Device for torchcodec decode. ``"cuda"`` offloads to
                NVDEC (needs a CUDA torchcodec build and ``spawn`` DataLoader workers).
            prefetch_videos (bool, optional): When streaming from a remote source, download each pooled
                episode's video files to a local cache in the background so decode-on-exit reads local
                bytes instead of paying network seek latency. Defaults to True.
            video_prefetch_workers (int, optional): Download threads per consumer. Defaults to 4.
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
        self.shuffle = shuffle

        self.streaming = streaming
        if buffer_size is not None:
            logger.warning(
                "StreamingLeRobotDataset: `buffer_size` is deprecated and ignored; "
                "use `episode_pool_size` (whole episodes, not frames)."
            )
        self.episode_pool_size = max(1, episode_pool_size) if episode_pool_size else 64
        self.max_num_shards = max_num_shards
        self._return_uint8 = return_uint8

        self.rank, self.world_size = self._resolve_distributed(rank, world_size)
        self.video_decoder_cache_size = video_decoder_cache_size
        self.data_files_root = data_files_root.rstrip("/") if data_files_root else None
        self.video_decode_device = video_decode_device
        self.prefetch_videos = prefetch_videos
        self.video_prefetch_workers = video_prefetch_workers

        # We cache the video decoders to avoid re-initializing them at each frame (avoiding a ~10x slowdown)
        self.video_decoder_cache = None
        self._prefetcher: _VideoPrefetcher | None = None
        # Shared [hits, misses, evictions, decode_ns, fetch_ns] tensor so DataLoader workers aggregate
        # decoder-cache stats and component timings into one place the main process can read after
        # iteration (see video_decoder_cache_stats() / timing_stats()).
        self._cache_counters = torch.zeros(5, dtype=torch.int64).share_memory_()
        # Deterministic fast-forward resume (see load_state_dict): per-consumer epoch counter and
        # number of samples still to skip.
        self._epoch = 0
        self._ff_remaining = 0
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

        self.num_shards = (
            self.hf_dataset.num_shards
            if self.max_num_shards is None
            else min(self.hf_dataset.num_shards, self.max_num_shards)
        )

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

    def _consumer_rng(self, epoch: int, worker_id: int) -> np.random.Generator:
        """RNG derived from (seed, epoch, rank, worker): reproducible, decorrelated consumers."""
        state = _mix64(self.seed)
        for salt in (self.rank, worker_id, epoch if self.shuffle else 0):
            state = _mix64(state ^ _mix64(salt))
        return np.random.default_rng(state)

    def _make_video_decoder_cache(self) -> VideoDecoderCache:
        """Size the decoder cache to the pool's working set (pool episodes x cameras), capped at 128."""
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
            max_size=min((self.episode_pool_size + 1) * num_cameras, 128),
            counters=self._cache_counters,
            device=self.video_decode_device,
        )

    def _make_prefetcher(self) -> _VideoPrefetcher | None:
        if not self.prefetch_videos or len(self.meta.video_keys) == 0:
            return None
        if self.data_files_root is not None:
            remote_root = self.data_files_root
        elif self.streaming and not self.streaming_from_local:
            remote_root = self.meta.url_root
        else:
            return None  # video bytes are already local
        return _VideoPrefetcher(
            remote_root,
            cache_dir=self.root / "streaming_video_cache",
            max_workers=self.video_prefetch_workers,
        )

    @staticmethod
    def _iter_shard_episodes(shard: datasets.IterableDataset) -> Iterator[tuple[int, list[dict]]]:
        """Yield (episode_index, rows) for each complete episode of a shard stream.

        On datasets >= 5 the grouping runs natively in Arrow via ``batch(by_column=...)``
        (one accumulation per episode instead of one Python dict per row); older versions
        use the equivalent row loop.
        """
        if _HAS_BATCH_BY_COLUMN:
            for batch in shard.batch(by_column="episode_index"):
                keys = list(batch.keys())
                num_rows = len(batch["episode_index"])
                rows = [{key: batch[key][i] for key in keys} for i in range(num_rows)]
                yield int(batch["episode_index"][0]), rows
            return
        rows: list[dict] = []
        current: int | None = None
        for item in shard:
            ep_idx = int(item["episode_index"])
            if current is None:
                current = ep_idx
            if ep_idx != current:
                yield current, rows
                rows = []
                current = ep_idx
            rows.append(item)
        if rows:
            yield current, rows

    def _admit_episode(self, ep_idx: int, rows: list[dict], prefetcher: _VideoPrefetcher | None):
        video_rel_paths = [str(self.meta.get_video_file_path(ep_idx, key)) for key in self.meta.video_keys]
        if prefetcher is not None:
            for rel in video_rel_paths:
                prefetcher.acquire(rel)
        torch_rows = [item_to_torch(row) for row in rows]
        return _PooledEpisode(ep_idx, torch_rows, video_rel_paths)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        ds = self.hf_dataset
        if self.world_size > 1:
            if ds.num_shards % self.world_size != 0:
                logger.warning(
                    f"num_shards ({ds.num_shards}) is not divisible by world_size ({self.world_size}): "
                    "datasets falls back to example-level splitting where every rank reads (and pays "
                    "for) the full stream. Re-shard the dataset or adjust world size."
                )
            ds = split_dataset_by_node(ds, rank=self.rank, world_size=self.world_size)

        num_shards = ds.num_shards if self.max_num_shards is None else min(ds.num_shards, self.max_num_shards)
        shard_indices = list(range(num_shards))

        # DataLoader workers within this rank further split the shards so they don't yield duplicates.
        worker_info = torch.utils.data.get_worker_info()
        worker_id, num_workers = (worker_info.id, worker_info.num_workers) if worker_info else (0, 1)
        shard_indices = shard_indices[worker_id::num_workers]
        if not shard_indices:
            logger.warning(
                f"Worker {worker_id} owns no shards ({num_shards} shards < {num_workers} workers): "
                "it will yield nothing. Reduce num_workers or re-shard the dataset."
            )
            return

        self.video_decoder_cache = self._make_video_decoder_cache()
        prefetcher = self._make_prefetcher()
        self._prefetcher = prefetcher

        epoch = self._epoch
        self._epoch += 1
        rng = self._consumer_rng(epoch, worker_id)
        # Workers beyond the shard count yield nothing and are stopped by the DataLoader, so the
        # batch round-robin effectively runs over min(num_workers, num_shards) active workers.
        self._consume_resume_state(worker_id, min(num_workers, num_shards))

        # Round-robin episode admission across this consumer's shard streams (deterministic).
        streams = [self._iter_shard_episodes(safe_shard(ds, idx, num_shards)) for idx in shard_indices]
        next_stream = 0

        pool: list[_PooledEpisode] = []
        total_remaining = 0

        def admit() -> int:
            nonlocal next_stream, total_remaining
            admitted = 0
            while len(pool) < self.episode_pool_size and streams:
                stream = streams[next_stream % len(streams)]
                fetch_start = time.perf_counter_ns()
                try:
                    ep_idx, rows = next(stream)
                except StopIteration:
                    streams.remove(stream)
                    continue
                finally:
                    self._cache_counters[4] += time.perf_counter_ns() - fetch_start
                next_stream += 1
                episode = self._admit_episode(ep_idx, rows, prefetcher)
                pool.append(episode)
                total_remaining += len(episode.remaining)
                admitted += 1
            return admitted

        worker_split_guard = (
            _suppress_hf_worker_split() if worker_info is not None else contextlib.nullcontext()
        )
        try:
            with worker_split_guard:
                admit()
                while pool:
                    if self.shuffle:
                        # Uniform draw over every remaining frame in the pool: pick the episode by
                        # cumulative remaining count, then a random remaining position (swap-pop).
                        draw = int(rng.integers(total_remaining))
                        for episode in pool:
                            if draw < len(episode.remaining):
                                break
                            draw -= len(episode.remaining)
                        pick = int(rng.integers(len(episode.remaining)))
                        frame_pos = episode.remaining[pick]
                        episode.remaining[pick] = episode.remaining[-1]
                        episode.remaining.pop()
                    else:
                        episode = pool[0]
                        frame_pos = episode.remaining.pop(0)
                    total_remaining -= 1

                    if self._ff_remaining > 0:
                        self._ff_remaining -= 1
                    else:
                        yield self._make_pool_sample(episode, frame_pos)

                    if not episode.remaining:
                        pool.remove(episode)
                        if prefetcher is not None:
                            for rel in episode.video_rel_paths:
                                prefetcher.release(rel)
                        admit()
        finally:
            if prefetcher is not None:
                prefetcher.shutdown()
            self._prefetcher = None

    def load_state_dict(self, state_dict: dict) -> None:
        """Stage a deterministic fast-forward resume, applied from the next ``__iter__``.

        ``state_dict`` holds ``{"batches_consumed": int, "batch_size": int}`` — what the trainer
        already knows at checkpoint time. Because every consumer's order is a pure function of
        (seed, epoch, rank, worker), resume replays the stream while skipping emission (tabular
        reads only, no video decode) until each worker reaches its own consumed count; the
        DataLoader's round-robin batch assignment makes that count derivable per worker. Exact
        within an epoch; crossing epoch boundaries may drift by < one batch per worker per epoch
        when ``drop_last`` discards partial batches.
        """
        self._resume_state = {
            "batches_consumed": int(state_dict["batches_consumed"]),
            "batch_size": int(state_dict["batch_size"]),
        }

    def _consume_resume_state(self, worker_id: int, active_workers: int) -> None:
        if self._resume_state is None:
            return
        batches = self._resume_state["batches_consumed"]
        batch_size = self._resume_state["batch_size"]
        self._resume_state = None
        if worker_id >= active_workers:
            return  # this worker owns no shards and never delivered a batch
        # The DataLoader assigns batch j to active worker j % active_workers.
        my_batches = batches // active_workers + (1 if batches % active_workers > worker_id else 0)
        self._ff_remaining = my_batches * batch_size
        if self._ff_remaining:
            logger.info(
                f"Streaming resume: worker {worker_id} fast-forwarding {self._ff_remaining} samples "
                "(tabular reads only, no video decode)."
            )

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
        """Cumulative seconds spent in video decode and episode (tabular) fetch, summed across
        DataLoader workers via the shared counter tensor. These overlap in wall-clock (workers run
        in parallel), so compare them to ``num_workers x wallclock`` for time fractions.
        """
        decode_ns, fetch_ns = (int(x) for x in self._cache_counters[3:5].tolist())
        return {"decode_s_total": round(decode_ns / 1e9, 2), "fetch_s_total": round(fetch_ns / 1e9, 2)}

    def _make_pool_sample(self, episode: _PooledEpisode, frame_pos: int) -> dict:
        """Assemble a full training sample for one pooled frame (tabular slices + video decode)."""
        rows = episode.rows
        item = dict(rows[frame_pos])
        ep_idx = episode.episode_index
        num_rows = len(rows)
        current_ts = float(item["timestamp"])

        updates: list[dict] = []
        if self.delta_indices is not None:
            updates.extend(self._pool_delta_frames(rows, frame_pos, num_rows))

        if len(self.meta.video_keys) > 0:
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
            decode_start = time.perf_counter_ns()
            video_frames = self._query_videos(query_timestamps, ep_idx)
            self._cache_counters[3] += time.perf_counter_ns() - decode_start

            if self.image_transforms is not None:
                for cam in self.meta.camera_keys:
                    video_frames[cam] = self.image_transforms(video_frames[cam])

            updates.append(video_frames)
            if self.delta_indices is not None:
                updates.append(
                    self._get_video_frame_padding_mask(video_frames, query_timestamps, original_timestamps)
                )

        result = item
        for update in updates:
            result.update(update)
        result["task"] = self.meta.tasks.iloc[item["task_index"]].name
        return result

    def _pool_delta_frames(self, rows: list[dict], frame_pos: int, num_rows: int) -> list[dict]:
        """Exact delta windows by slicing the resident episode; clamped + padded at boundaries."""
        query_result: dict = {}
        padding: dict = {}
        for key, deltas in self.delta_indices.items():
            if key in self.meta.video_keys:
                continue  # visual frames are decoded separately
            frames = []
            is_pad = []
            for delta in deltas:
                j = frame_pos + delta
                valid = 0 <= j < num_rows
                frames.append(rows[min(max(j, 0), num_rows - 1)][key])
                is_pad.append(not valid)
            query_result[key] = torch.stack(frames)
            padding[f"{key}_is_pad"] = torch.BoolTensor(is_pad)
        return [query_result, padding]

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
            local = self._prefetcher.wait_local(rel_path) if self._prefetcher is not None else None
            if local is not None:
                video_path = str(local)
            else:
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

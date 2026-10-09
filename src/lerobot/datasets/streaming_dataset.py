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
import io
import os
import warnings
from collections import deque
from collections.abc import Callable, Generator, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import datasets
import numpy as np
import torch

from lerobot.configs import DEFAULT_DEPTH_UNIT
from lerobot.streaming.episode_cache import EpisodeByteCache
from lerobot.streaming.episode_parquet import EpisodeParquetReader
from lerobot.streaming.episode_pool import ExactCoveragePool, StreamingSamplingStrategy
from lerobot.streaming.location import StorageLocation
from lerobot.streaming.manifest import EpisodeVideoManifest
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import get_safe_default_video_backend

from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .feature_utils import check_delta_timestamps, get_delta_indices, get_hf_features_from_features
from .io_utils import hf_transform_to_torch
from .language import LANGUAGE_COLUMNS
from .storage import is_bucket_root
from .streaming_sidecar import (
    ensure_dataset_mp4_sidecar,
    streaming_data_root,
)
from .utils import (
    check_version_compatibility,
    delta_window,
    resolve_episode_indices,
    shift_timestamps,
    task_name,
)
from .video_utils import (
    apply_rgb_transforms,
    convert_image_depth_units,
    decode_video_frames_pyav,
    depth_encoder_configs,
    dequantize_depth_frames,
    image_depth_units,
    normalize_rgb_frames,
)


@dataclass(frozen=True)
class _EpisodeData:
    """Keep prepared numeric columns and fallback feature views for one resident episode.

    Numeric columns are NumPy arrays: item assembly indexes them and wraps the copies with
    ``torch.from_numpy``, which keeps the GIL. Small torch ops (indexing, ``clone``, ``item``)
    release the GIL and must take it back, which waits behind the other decode threads.
    """

    dataset: datasets.Dataset
    columns: dict[str, datasets.Dataset]
    numeric: dict[str, np.ndarray]
    other: datasets.Dataset | None
    dataset_from_index: int
    video_from_timestamps: dict[str, float]

    def get_item(self, index: int) -> dict[str, Any]:
        """Return an independently owned row without reformatting prepared numeric columns."""
        item = self.other[index] if self.other is not None else {}
        item.update({key: torch.from_numpy(np.array(values[index])) for key, values in self.numeric.items()})
        return item

    def get_column(self, key: str, indices: list[int]) -> torch.Tensor:
        """Gather a temporal window, preserving repeated indices and sample ownership."""
        if key in self.numeric:
            return torch.from_numpy(self.numeric[key][indices])
        return torch.stack(self.columns[key][indices][key])

    def column_values(self, key: str, indices: list[int]) -> list[Any]:
        """Return a temporal window as Python scalars, without building a tensor."""
        if key in self.numeric:
            return self.numeric[key][indices].tolist()
        return self.get_column(key, indices).tolist()

    def row_int(self, key: str, index: int, item: dict[str, Any]) -> int:
        """Read one integer field of a row, from its NumPy column when the column is numeric."""
        values = self.numeric.get(key)
        return int(values[index]) if values is not None else int(item[key])


def _balanced_episode_shards(
    episode_indices: list[int],
    episode_frame_counts: Mapping[int, int],
    *,
    world_size: int,
) -> list[list[int]]:
    """Assign whole episodes deterministically with greedy frame-count balancing."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    shards: list[list[int]] = [[] for _ in range(world_size)]
    shard_frames = [0] * world_size
    for episode in sorted(episode_indices, key=lambda item: (-episode_frame_counts[item], item)):
        rank = min(range(world_size), key=lambda item: (shard_frames[item], item))
        shards[rank].append(episode)
        shard_frames[rank] += episode_frame_counts[episode]
    return shards


@dataclass(frozen=True)
class _WorkerShare:
    """One DataLoader worker's part of a rank: its episodes and its share of the rank limits."""

    worker_id: int
    num_workers: int
    episodes: list[int]
    pool_size: int
    byte_budget: int


def _batches_per_worker(num_batches: int, batch_limits: list[int | None]) -> tuple[list[int], int]:
    """Replay the DataLoader turns for ``num_batches`` batches of a rank.

    An in-order DataLoader takes batches from its active workers in turn, and a worker leaves the
    turn after its last batch. ``None`` marks a worker that never ends (``repeat=True``).

    Returns:
        `tuple[list[int], int]`: The batches each worker produced, and the worker of the next batch.
    """
    counts = [0] * len(batch_limits)
    if all(limit is None for limit in batch_limits):
        full_turns, extra = divmod(num_batches, len(batch_limits))
        return [full_turns + (1 if worker < extra else 0) for worker in range(len(batch_limits))], extra
    active = [worker for worker, limit in enumerate(batch_limits) if limit is None or limit > 0]
    turn = 0
    for _ in range(num_batches):
        if not active:
            break
        worker = active[turn]
        counts[worker] += 1
        limit = batch_limits[worker]
        if limit is not None and counts[worker] >= limit:
            active.pop(turn)
        else:
            turn += 1
        turn = turn % len(active) if active else 0
    return counts, active[turn] if active else 0


class StreamingLeRobotDataset(torch.utils.data.IterableDataset[dict[str, Any]]):
    """Episode-scoped streaming reader for LeRobot datasets.

    Metadata is cached locally, while each rank reads only the Parquet rows and MP4 byte ranges
    needed for the complete episodes it owns. With N DataLoader workers, each worker owns a disjoint,
    frame-balanced part of the rank's episodes, with its own episode pool and 1/N of the byte budget.
    Each process has its own Python GIL, so N workers decode and assemble samples in parallel.
    Episode ownership is disjoint and every selected frame is yielded exactly once per iteration.
    MP4 sidecars are resolved automatically and built in a revision-keyed local cache when absent.

    Example:
        Basic usage:
        ```python
        from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

        # Create a streaming dataset with delta timestamps
        delta_timestamps = {
            "observation.image": [-1.0, -0.5, 0.0],  # 1 sec ago, 0.5 sec ago, current
            "action": [0.0, 0.1, 0.2],  # current, 0.1 sec future, 0.2 sec future
        }

        dataset = StreamingLeRobotDataset(
            repo_id="your-dataset-repo-id",
            delta_timestamps=delta_timestamps,
        )

        # Iterate over the dataset
        for i, item in enumerate(dataset):
            print(f"Sample {i}: Episode {item['episode_index']} Frame {item['frame_index']}")
            # item will contain stacked frames according to delta_timestamps
            if i >= 10:
                break
        ```
    """

    # Set in __init__; class defaults keep partially built instances (tests) on the one-worker path.
    _worker_pool_size: int | None = None
    _explicit_decoder_cache_size: int | None = None
    # Logical worker offset of DataLoader worker 0 for the current iterator (see `_worker_share`).
    _worker_rotation = 0

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        streaming: bool | None = None,
        buffer_size: int = 1000,
        max_num_shards: int = 16,
        seed: int = 42,
        rng: np.random.Generator | None = None,
        shuffle: bool = True,
        return_uint8: bool = False,
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
        video_backend: str | None = None,
        data_root: str | Path | None = None,
        episode_pool_size: int | None = None,
        prefetch_episodes: int = 8,
        byte_budget_gb: float = 8.0,
        repeat: bool = False,
        *,
        repo_type: Literal["dataset", "bucket"] = "dataset",
        token: str | bool | None = None,
        decode_threads: int = 2,
        decoded_queue_size: int = 8,
        video_decoder_cache_size: int | None = None,
        native_http_connections: int | None = None,
        native_http_subranges: int = 1,
        sampling_strategy: StreamingSamplingStrategy | str = StreamingSamplingStrategy.REMAINING,
    ) -> None:
        """Initialize an episode-scoped streaming reader.

        Args:
            repo_id (`str`):
                Hub dataset or bucket identifier.
            root (`str | Path | None`, *optional*):
                Local dataset directory, or local metadata cache in bucket mode. An
                ``hf://buckets/OWNER/BUCKET`` URI selects bucket mode for that bucket.
            episodes (`list[int] | None`, *optional*):
                Episode indices to select; None selects the complete dataset.
            image_transforms (`Callable[[torch.Tensor], torch.Tensor] | None`, *optional*):
                Transform applied to decoded RGB images in planned sample order.
            delta_timestamps (`dict[str, list[float]] | None`, *optional*):
                Per-feature history or future offsets in seconds, padded at episode boundaries.
            tolerance_s (`float`, *optional*, defaults to `1e-4`):
                Maximum timestamp error allowed when matching decoded frames.
            revision (`str | None`, *optional*):
                Hub revision to resolve to an immutable dataset commit.
            force_cache_sync (`bool`, *optional*, defaults to `False`):
                Refresh locally cached dataset metadata.
            streaming (`bool | None`, *optional*):
                Deprecated and ignored: this class always streams. Use ``dataset.streaming`` in the
                training config to choose between streaming and map-style loading.
            buffer_size (`int`, *optional*, defaults to `1000`):
                Legacy setting used to derive the pool size when episode_pool_size is omitted.
            max_num_shards (`int`, *optional*, defaults to `16`):
                Maximum episode-fetch concurrency of one DataLoader worker (or of the training process
                without workers). ``lerobot-train`` sets it from ``--num_workers``.
            seed (`int`, *optional*, defaults to `42`):
                Seed for deterministic episode admission and anchor sampling.
            rng (`np.random.Generator | None`, *optional*):
                Deprecated and ignored; set seed instead.
            shuffle (`bool`, *optional*, defaults to `True`):
                Advance the seeded sample plan between epochs; False replays the same plan.
            return_uint8 (`bool`, *optional*, defaults to `False`):
                Return RGB pixels as uint8 instead of float32 values in [0, 1].
            depth_output_unit (`str`, *optional*):
                Physical output unit for depth maps: "mm" by default, or "m".
            video_backend (`str | None`, *optional*):
                RGB decoder backend. None uses the platform-safe default; TorchCodec failures
                fall back to PyAV. Depth videos use PyAV.
            data_root (`str | Path | None`, *optional*):
                Payload root override for direct Python use; accepts local paths and fsspec URLs.
            episode_pool_size (`int | None`, *optional*):
                Maximum active episodes of each DataLoader worker, also limited by the compressed-byte
                budget. None mixes ``min(buffer_size, 32)`` episodes for the rank and gives each of the
                N workers 1/N of them. A larger pool mixes more episodes in each batch, but uses more
                memory and fetches more data before the first sample.
            prefetch_episodes (`int`, *optional*, defaults to `8`):
                Pending episodes eligible for speculative prefetch beyond the active pool, for each
                DataLoader worker.
            byte_budget_gb (`float`, *optional*, defaults to `8.0`):
                Per-rank reservation limit in GiB for synthesized video bytes, not total RAM. Each of the
                N DataLoader workers of a rank gets 1/N of it.
            repeat (`bool`, *optional*, defaults to `False`):
                Repeat rank-local coverage epochs, allowing batches to span epoch boundaries.
            repo_type (`Literal["dataset", "bucket"]`, *optional*, defaults to `"dataset"`):
                Whether repo_id identifies a dataset repository or a Storage Bucket.
            token (`str | bool | None`, *optional*):
                Hub authentication retained for worker I/O, never serialized into sidecars.
            decode_threads (`int`, *optional*, defaults to `2`):
                Parallel sample-assembly and video-decode workers.
            decoded_queue_size (`int`, *optional*, defaults to `8`):
                Maximum samples prepared ahead, delivered in planner order.
            video_decoder_cache_size (`int | None`, *optional*):
                Open video decoder cap per rank, divided between the DataLoader workers; None allows one
                per active episode-camera pair of each worker.
            native_http_connections (`int | None`, *optional*):
                Per-rank HTTP connection limit; None derives it from fetch concurrency.
            native_http_subranges (`int`, *optional*, defaults to `1`):
                Maximum concurrent subrequests for one sufficiently large byte range.
            sampling_strategy (`StreamingSamplingStrategy | str`, *optional*, defaults to `"remaining"`):
                Weight episodes by remaining anchors, or draw one anchor per episode each
                shuffled round. Neither strategy is a global uniform shuffle.
        """
        super().__init__()
        if is_bucket_root(repo_id, root):
            # Metadata then uses the regular bucket cache, as with repo_type="bucket".
            root, repo_type = None, "bucket"
        if repo_type not in ("dataset", "bucket"):
            raise ValueError(f"repo_type must be 'dataset' or 'bucket', got {repo_type!r}")
        self.repo_id = repo_id
        self.repo_type = repo_type
        self._requested_root = Path(root) if root else None
        self.root = self._requested_root if self._requested_root is not None else HF_LEROBOT_HOME / repo_id
        self.streaming_from_local = root is not None and repo_type == "dataset"

        self.image_transforms = image_transforms
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision: str | None = revision if revision else CODEBASE_VERSION
        self.seed = seed
        if rng is not None:
            warnings.warn(
                "rng is deprecated and has no effect; use seed for reproducible streaming order.",
                FutureWarning,
                stacklevel=2,
            )
        self.shuffle = shuffle

        if streaming is not None:
            warnings.warn(
                "StreamingLeRobotDataset(streaming=...) is deprecated and has no effect; "
                "choose streaming with the dataset.streaming training option instead.",
                FutureWarning,
                stacklevel=2,
            )
        self.streaming = True
        self.buffer_size = buffer_size
        self.max_num_shards = max_num_shards
        self._return_uint8 = return_uint8
        self._depth_output_unit = depth_output_unit
        self._streaming_io_token = None if self.streaming_from_local else token
        self._video_backend = video_backend if video_backend is not None else get_safe_default_video_backend()
        if self._video_backend == "video_reader":
            self._video_backend = "pyav"
        if self._video_backend not in {"torchcodec", "pyav"}:
            raise ValueError(f"Unsupported video backend: {self._video_backend}")
        self.sampling_strategy = StreamingSamplingStrategy(sampling_strategy)
        self.episode_pool_size = episode_pool_size or min(buffer_size, 32)
        # An explicit pool size is the pool of each DataLoader worker; the default is split between them.
        self._worker_pool_size = episode_pool_size
        self.prefetch_episodes = prefetch_episodes
        self.byte_budget = int(byte_budget_gb * 1024**3)
        self.decode_threads = decode_threads
        # An empty queue could never hold a sample, so at least one is always prepared ahead.
        self.decoded_queue_size = max(1, decoded_queue_size)
        self.native_http_connections = native_http_connections
        self.native_http_subranges = native_http_subranges
        self.repeat = repeat
        self._next_epoch = 0
        self._active_epoch = 0
        self._resume_offset = 0
        self._resume_batch_size = 1
        self._state_offset = 0

        if self._requested_root is not None:
            self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id,
            self._requested_root,
            self.revision,
            force_cache_sync=force_cache_sync,
            repo_type=repo_type,
            token=token,
        )
        self.root = self.meta.root
        self.revision = self.meta.revision
        self.meta.rescale_depth_stats(self._depth_output_unit)
        # Check version
        check_version_compatibility(self.repo_id, self.meta._version, CODEBASE_VERSION)
        self.video_decoder_cache_size = (
            video_decoder_cache_size
            if video_decoder_cache_size is not None
            else max(1, self.episode_pool_size * len(self.meta.video_keys))
        )
        self._explicit_decoder_cache_size = video_decoder_cache_size

        self._depth_encoder_configs = depth_encoder_configs(self.meta)
        self._image_depth_units = image_depth_units(self.meta)

        resolved_episodes = resolve_episode_indices(episodes, self.meta.total_episodes)
        # Each episode is owned once: duplicates would double-count frames in the coverage plan.
        self._selected_episodes = (
            list(range(self.meta.total_episodes))
            if resolved_episodes is None
            else list(dict.fromkeys(resolved_episodes))
        )

        self.delta_timestamps: dict[str, list[float]] | None = None
        self.delta_indices: dict[str, list[int]] | None = None

        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.fps, tolerance_s)
            self.delta_timestamps = delta_timestamps
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        self._data_root = streaming_data_root(
            self.meta,
            requested_root=self._requested_root,
            configured_data_root=str(data_root) if data_root is not None else None,
            token=self._streaming_io_token,
        )
        sidecar_backend = StorageLocation.parse(self._data_root).range_backend
        self._sidecar_path = ensure_dataset_mp4_sidecar(
            self.meta,
            self._data_root,
            workers=max_num_shards,
            range_backend=sidecar_backend,
            token=self._streaming_io_token,
        )
        self._hf_features = get_hf_features_from_features(self.meta.features)
        self._projected_columns = tuple(self._hf_features)
        self.num_shards = min(max_num_shards, max(1, len(self._selected_episodes)))

    @property
    def num_frames(self) -> int:
        """Return the frame count across all selected episodes, before rank sharding."""
        return sum(self._episode_frame_count(episode) for episode in self._selected_episodes)

    @property
    def num_episodes(self) -> int:
        """Return the number of selected episodes across all ranks."""
        return len(self._selected_episodes)

    @property
    def fps(self) -> int:
        """Return the dataset's recording frame rate."""
        return self.meta.fps

    @property
    def depth_output_unit(self) -> str:
        """Physical unit (``"m"`` or ``"mm"``) depth maps are returned in on read."""
        return self._depth_output_unit

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield rank-local samples for one coverage epoch, or repeat when configured."""
        # A new DataLoader iterator starts its worker turns at worker 0; a resume offset sets it again.
        self._worker_rotation = 0
        if self.repeat:
            return self._repeat_iterator()
        return self._iter_once()

    def _repeat_iterator(self) -> Generator[dict[str, Any], None, None]:
        """Repeat nonempty rank-local epochs and close each iterator on early exit."""
        while True:
            with closing(self._iter_once()) as iterator:
                try:
                    first = next(iterator)
                except StopIteration:
                    return
                yield first
                yield from iterator

    def _iter_once(self) -> Generator[dict[str, Any], None, None]:
        """Yield one rank-local coverage plan with bounded prefetch and ordered decoding."""
        epoch = self._next_epoch if self.shuffle else 0
        self._active_epoch = epoch

        rank_episodes, _rank, _world_size = self._rank_episodes()
        share, resume_offset = self._worker_share(rank_episodes, self._resume_offset)
        consumer_episodes = share.episodes
        self._resume_offset = 0
        consumer_frame_count = sum(self._episode_frame_count(episode) for episode in consumer_episodes)
        if not self.repeat and resume_offset >= consumer_frame_count:
            return  # this worker already gave all its frames; without repeat there is no next epoch
        if consumer_frame_count:
            worker_epoch_delta, resume_offset = divmod(resume_offset, consumer_frame_count)
        else:
            worker_epoch_delta = 0
        if self.shuffle:
            epoch += worker_epoch_delta
        self._state_offset = resume_offset
        self._active_epoch = epoch
        if self.shuffle:
            self._next_epoch = epoch + 1

        max_workers = min(self.max_num_shards, max(1, share.pool_size + self.prefetch_episodes))
        with ExitStack() as resources:
            video_cache = self._make_video_cache(consumer_episodes, max_workers, share)
            if video_cache is not None:
                resources.callback(video_cache.close)
            episode_byte_sizes = (
                {episode: video_cache.manifest.episode_byte_size(episode) for episode in consumer_episodes}
                if video_cache is not None
                else None
            )
            planner = ExactCoveragePool(
                [(episode, self._episode_frame_count(episode)) for episode in consumer_episodes],
                pool_size=share.pool_size,
                sampling_strategy=self.sampling_strategy,
                seed=self.seed,
                epoch=epoch,
                episode_byte_sizes=episode_byte_sizes,
                byte_budget=share.byte_budget if episode_byte_sizes is not None else None,
            )
            for _ in range(resume_offset):
                try:
                    next(planner)
                except StopIteration:
                    return
            planner.newly_admitted.clear()
            planner.evicted.clear()

            parquet_reader = EpisodeParquetReader(
                self._data_root,
                columns=self._projected_columns,
                token=self._streaming_io_token,
            )
            parquet_executor = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="lerobot-parquet"
            )
            resources.callback(parquet_executor.shutdown, wait=True, cancel_futures=True)
            decode_executor = ThreadPoolExecutor(
                max_workers=self.decode_threads,
                thread_name_prefix="lerobot-decode",
            )
            resources.callback(decode_executor.shutdown, wait=True, cancel_futures=True)
            episode_futures: dict[int, Future[_EpisodeData]] = {}
            decoded_futures: deque[Future[dict[str, Any]]] = deque()
            scheduled_episodes: set[int] = set()
            retained_video_episodes: set[int] = set()
            if video_cache is not None:
                for episode_index in planner.resident:
                    video_cache.retain_episode(episode_index, wait=True)
                    retained_video_episodes.add(episode_index)

            def submit(episode_index: int) -> Future[_EpisodeData]:
                """Reuse or schedule the table load for an admitted episode."""
                future = episode_futures.get(episode_index)
                if future is None:
                    future = parquet_executor.submit(
                        self._load_episode_dataset, parquet_reader, episode_index
                    )
                    episode_futures[episode_index] = future
                return future

            def schedule_frontier() -> None:
                """Prefetch tables and video ranges from the same bounded admission frontier."""
                frontier = [*planner.resident, *planner.prefetch_candidates(self.prefetch_episodes)]
                for episode_index in frontier:
                    if episode_index in scheduled_episodes:
                        continue
                    submit(episode_index)
                    if video_cache is not None and not video_cache.submit_prefetch(episode_index):
                        continue
                    scheduled_episodes.add(episode_index)

            def decode_item(
                episode_future: Future[_EpisodeData],
                episode_index: int,
                frame_index: int,
            ) -> dict[str, Any]:
                """Assemble a planned sample after its episode table is ready."""
                return self._make_episode_item(
                    episode_future.result(),
                    episode_index,
                    frame_index,
                    video_cache=video_cache,
                )

            def update_frontier() -> None:
                """Release drained episodes and schedule newly admitted or prefetched episodes."""
                for evicted_episode in planner.evicted:
                    episode_futures.pop(evicted_episode, None)
                    if video_cache is not None and evicted_episode in retained_video_episodes:
                        video_cache.release_episode(evicted_episode)
                        retained_video_episodes.remove(evicted_episode)
                if video_cache is not None:
                    for admitted_episode in planner.newly_admitted:
                        if admitted_episode not in retained_video_episodes:
                            # The last anchor may still be decoding after planner eviction.
                            # Wait for its byte lease before admitting the replacement.
                            video_cache.retain_episode(admitted_episode, wait=True)
                            retained_video_episodes.add(admitted_episode)
                planner.evicted.clear()
                planner.newly_admitted.clear()
                schedule_frontier()

            schedule_frontier()
            planner_exhausted = False
            while decoded_futures or not planner_exhausted:
                while not planner_exhausted and len(decoded_futures) < self.decoded_queue_size:
                    try:
                        episode_index, frame_index = next(planner)
                    except StopIteration:
                        planner_exhausted = True
                        break

                    episode_future = submit(episode_index)
                    if video_cache is not None:
                        video_cache.retain_episode(episode_index)
                    try:
                        decoded_future = decode_executor.submit(
                            decode_item,
                            episode_future,
                            episode_index,
                            frame_index,
                        )
                    except Exception:
                        if video_cache is not None:
                            video_cache.release_episode(episode_index)
                        raise
                    if video_cache is not None:

                        def release_video_episode(
                            _future: Future[dict[str, Any]],
                            retained_episode: int = episode_index,
                            retained_cache: EpisodeByteCache = video_cache,
                        ) -> None:
                            """Release this sample's byte lease when its decode future completes."""
                            retained_cache.release_episode(retained_episode)

                        decoded_future.add_done_callback(release_video_episode)
                    decoded_futures.append(decoded_future)
                    update_frontier()

                if not decoded_futures:
                    continue
                item = decoded_futures.popleft().result()
                self._state_offset += 1
                yield item
            self._active_epoch = epoch + 1 if self.shuffle else 0
            self._state_offset = 0

    def _rank_episodes(self) -> tuple[list[int], int, int]:
        """Resolve the distributed rank and its deterministic, frame-balanced episode shard."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid distributed rank/world size: rank={rank}, world_size={world_size}")
        counts = {episode: self._episode_frame_count(episode) for episode in self._selected_episodes}
        shards = _balanced_episode_shards(self._selected_episodes, counts, world_size=world_size)
        return shards[rank], rank, world_size

    def _worker_share(self, rank_episodes: list[int], rank_offset: int = 0) -> tuple[_WorkerShare, int]:
        """Give the current DataLoader worker its episodes, limits and resume offset.

        Each worker owns a frame-balanced part of the rank's episodes. On resume, the rank offset
        counts ``step * batch_size`` samples. The DataLoader took those batches from its workers in
        turn, but a new DataLoader starts its turns at worker 0. So DataLoader worker ``p`` takes
        the part of logical worker ``(p + r) % N``, where ``r`` is the worker of the next batch.

        Returns:
            `tuple[_WorkerShare, int]`: The worker's share, and its own resume offset in samples.
        """
        worker = torch.utils.data.get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker is not None else (0, 1)
        if num_workers > 1 and len(rank_episodes) < num_workers:
            raise ValueError(
                f"This rank owns {len(rank_episodes)} streaming episodes, fewer than its {num_workers} "
                "DataLoader workers. Reduce --num_workers."
            )
        episodes, offset = rank_episodes, rank_offset
        if num_workers > 1:
            counts = {episode: self._episode_frame_count(episode) for episode in rank_episodes}
            shards = _balanced_episode_shards(rank_episodes, counts, world_size=num_workers)
            # Repeat epochs of one iterator keep the rotation that the resume of its first epoch set.
            batches, next_worker = [0] * num_workers, self._worker_rotation
            if rank_offset:
                batch_size = self._resume_batch_size
                if rank_offset % batch_size:
                    raise ValueError(
                        f"A resume offset of {rank_offset} samples is not a whole number of batches of "
                        f"{batch_size}; resuming with {num_workers} DataLoader workers needs whole batches."
                    )
                frames = [sum(counts[episode] for episode in shard) for shard in shards]
                # With repeat, a worker never ends unless it has no frames; without it, after its last batch.
                limits: list[int | None] = [
                    0 if not total else None if self.repeat else -(-total // batch_size) for total in frames
                ]
                batches, next_worker = _batches_per_worker(rank_offset // batch_size, limits)
                batches = [count * batch_size for count in batches]
            self._worker_rotation = next_worker
            worker_id = (worker_id + next_worker) % num_workers
            episodes, offset = shards[worker_id], batches[worker_id]
        if self._worker_pool_size is not None:
            pool_size = self._worker_pool_size
        else:
            pool_size = max(1, round(self.episode_pool_size / num_workers))
        share = _WorkerShare(
            worker_id=worker_id,
            num_workers=num_workers,
            episodes=episodes,
            pool_size=pool_size,
            byte_budget=max(1, self.byte_budget // num_workers),
        )
        return share, offset

    @cached_property
    def _episode_frame_counts(self) -> np.ndarray:
        """Read only episode boundaries once for rank balancing and coverage planning."""
        boundaries = self.meta.episodes.select_columns(
            ["dataset_from_index", "dataset_to_index"]
        ).with_format(None)[:]
        return np.asarray(boundaries["dataset_to_index"], dtype=np.int64) - np.asarray(
            boundaries["dataset_from_index"], dtype=np.int64
        )

    def _episode_frame_count(self, episode_index: int) -> int:
        """Return the complete episode length from its absolute dataset boundaries."""
        return int(self._episode_frame_counts[episode_index])

    def num_frames_for_rank(self, rank: int, world_size: int, num_workers: int) -> int:
        """Return frames owned by one training rank under balanced whole-episode sharding."""
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid distributed rank/world size: rank={rank}, world_size={world_size}")
        counts = {episode: self._episode_frame_count(episode) for episode in self._selected_episodes}
        shards = _balanced_episode_shards(self._selected_episodes, counts, world_size=world_size)
        if num_workers > 1 and len(shards[rank]) < num_workers:
            raise ValueError(
                f"Rank {rank} owns {len(shards[rank])} streaming episodes, fewer than its {num_workers} "
                "DataLoader workers. Reduce --num_workers."
            )
        return sum(counts[episode] for episode in shards[rank])

    def num_episodes_for_rank(self, rank: int, world_size: int) -> int:
        """Return the number of whole episodes one training rank owns."""
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid distributed rank/world size: rank={rank}, world_size={world_size}")
        counts = {episode: self._episode_frame_count(episode) for episode in self._selected_episodes}
        return len(_balanced_episode_shards(self._selected_episodes, counts, world_size=world_size)[rank])

    def _load_episode_dataset(
        self,
        reader: EpisodeParquetReader,
        episode_index: int,
    ) -> _EpisodeData:
        """Load one episode, preparing fixed-shape numeric columns once for temporal queries."""
        table = reader.read_episode(
            self.meta.get_data_file_path(episode_index),
            episode_index=episode_index,
            expected_rows=self._episode_frame_count(episode_index),
        )
        # from_dict applies the declared HF feature encoders (images, nested language/JSON fields)
        # while retaining the episode-sized memory bound.
        dataset = datasets.Dataset.from_dict(table.to_pydict(), features=self._hf_features)
        dataset.set_transform(hf_transform_to_torch)
        numeric: dict[str, np.ndarray] = {}
        for key, feature in self._hf_features.items():
            if key in LANGUAGE_COLUMNS:
                continue
            while isinstance(feature, datasets.List) and feature.length >= 0:
                feature = feature.feature
            if not isinstance(feature, datasets.Value) or feature.dtype not in {
                "bool",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "float16",
                "float32",
                "float64",
            }:
                continue
            values = dataset.select_columns(key).with_format(None)[:][key]
            try:
                # Same values and dtypes as torch.stack(hf_transform_to_torch(...)[key]) (both infer
                # from Python values, not the Arrow dtype), but one tensor per episode column
                # instead of one per row. Nullable columns stay on the hf_transform_to_torch path.
                numeric[key] = torch.tensor(values).numpy()
            except (TypeError, ValueError, RuntimeError):
                continue
        other_keys = [key for key in dataset.column_names if key not in numeric]
        episode = self.meta.episodes[episode_index]
        # Zero-copy views share the episode's lifetime and fixed HF transform. Project
        # before row lookup so action/state windows cannot decode unrelated images.
        column_keys = set(self.delta_indices or ()) - set(self.meta.video_keys)
        if self.meta.video_keys:
            column_keys.add("timestamp")
        return _EpisodeData(
            dataset=dataset,
            columns={key: dataset.select_columns(key) for key in sorted(column_keys) if key not in numeric},
            numeric=numeric,
            other=dataset.select_columns(other_keys) if other_keys else None,
            # Read once per resident episode: a metadata row lookup formats every stats column.
            dataset_from_index=int(episode["dataset_from_index"]),
            video_from_timestamps={
                key: float(episode[f"videos/{key}/from_timestamp"]) for key in self.meta.video_keys
            },
        )

    def _make_video_cache(
        self,
        episode_indices: list[int],
        workers: int,
        share: _WorkerShare,
    ) -> EpisodeByteCache | None:
        """Build the worker-local video manifest and its bounded byte cache."""
        if self._sidecar_path is None or not episode_indices:
            return None
        range_backend = StorageLocation.parse(self._data_root).range_backend
        manifest = EpisodeVideoManifest.build(
            self.meta,
            self._data_root,
            episode_indices=episode_indices,
            range_backend=range_backend,
            workers=workers,
            sidecar_path=self._sidecar_path,
            token=self._streaming_io_token,
        )
        return EpisodeByteCache(
            manifest,
            self._data_root,
            byte_budget=share.byte_budget,
            workers=workers,
            range_backend=range_backend,
            native_http_connections=self.native_http_connections,
            native_http_subranges=self.native_http_subranges,
            max_open_decoders=(
                max(1, self._explicit_decoder_cache_size // share.num_workers)
                if self._explicit_decoder_cache_size is not None
                else max(1, share.pool_size * len(self.meta.video_keys))
            ),
            video_backend=self._video_backend,
            tolerance_s=self.tolerance_s,
            token=self._streaming_io_token,
        )

    def _make_episode_item(
        self,
        episode_data: _EpisodeData,
        episode_index: int,
        frame_index: int,
        *,
        video_cache: EpisodeByteCache | None,
    ) -> dict[str, Any]:
        """Assemble an anchor's temporal windows, padding masks and decoded camera frames."""
        episode_length = len(episode_data.dataset)
        item = episode_data.get_item(frame_index)
        episode_start = episode_data.dataset_from_index

        # Episode-local windows: the whole episode is resident, so rows are 0..length-1.
        windows: dict[str, list[int]] = {}
        for key, deltas in self._delta_arrays.items():
            windows[key], item[f"{key}_is_pad"] = delta_window(frame_index, deltas, 0, episode_length)
            if key not in self.meta.video_keys:
                item[key] = episode_data.get_column(key, windows[key])

        if self.meta.video_keys:
            if video_cache is None:
                raise RuntimeError("Video dataset streaming requires an episode byte cache")
            for video_key in self.meta.video_keys:
                target_indices = windows.get(video_key, [frame_index])
                local_timestamps = [
                    float(timestamp) for timestamp in episode_data.column_values("timestamp", target_indices)
                ]
                query_timestamps = shift_timestamps(
                    local_timestamps, episode_data.video_from_timestamps[video_key]
                )
                if video_key in self.meta.depth_keys:
                    source_start = video_cache.manifest.lookup(episode_index, video_key).source_start_pts
                    frames = decode_video_frames_pyav(
                        io.BytesIO(video_cache.get_bytes(episode_index, video_key)),
                        [timestamp - source_start for timestamp in query_timestamps],
                        self.tolerance_s,
                        return_uint8=False,
                        is_depth=True,
                    )
                    frames = dequantize_depth_frames(
                        frames, self._depth_encoder_configs[video_key], self._depth_output_unit
                    )
                else:
                    frames = normalize_rgb_frames(
                        video_cache.get_frames(episode_index, video_key, query_timestamps), self._return_uint8
                    )
                item[video_key] = frames.squeeze(0)

        # Runs on the decode thread, so augmentation parallelizes with decoding.
        self._apply_image_transforms(item)
        convert_image_depth_units(item, self._image_depth_units, self._depth_output_unit)
        item["task"] = self._task_names[episode_data.row_int("task_index", frame_index, item)]
        if episode_data.row_int("episode_index", frame_index, item) != episode_index:
            raise RuntimeError(f"Episode reader returned episode {item['episode_index']} for {episode_index}")
        if episode_data.row_int("index", frame_index, item) != episode_start + frame_index:
            raise RuntimeError(
                f"Episode {episode_index} frame {frame_index} has unexpected absolute index {item['index']}"
            )
        return item

    @cached_property
    def _delta_arrays(self) -> dict[str, np.ndarray]:
        """Frame offsets of each temporal window, converted once for ``delta_window``."""
        return {key: np.asarray(deltas, dtype=np.int64) for key, deltas in (self.delta_indices or {}).items()}

    @cached_property
    def _task_names(self) -> list[Any]:
        """Task strings by ``task_index``: one list lookup in place of a pandas row lookup per sample."""
        return [task_name(self.meta.tasks, index) for index in range(len(self.meta.tasks))]

    def _apply_image_transforms(self, item: dict[str, Any]) -> None:
        """Transform RGB images in place, leaving physical depth values unchanged."""
        apply_rgb_transforms(item, self.image_transforms, self.meta.camera_keys, self.meta.depth_keys)

    def state_dict(self) -> dict[str, int]:
        """Return the iterator's current rank-local position.

        Returns:
            `dict[str, int]`: Epoch, yielded-sample offset and resume batch size.

        Note:
            With DataLoader prefetch, yielded samples may not yet have been consumed by training.
            The training pipeline restores its position from completed steps instead. With several
            DataLoader workers, each worker process holds only its own position.
        """
        return {
            "epoch": self._active_epoch,
            "offset": self._state_offset,
            "batch_size": self._resume_batch_size,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Restore the next iterator's anchor position without fetching skipped samples.

        Args:
            state (`dict[str, int]`):
                Rank-local epoch, sample offset and batch size. Missing keys default to 0, 0 and 1.

        Raises:
            ValueError: If the epoch or offset is negative, or the batch size is not positive.

        Note:
            Reproducing anchor order requires the same dataset, seed, sampler, pool settings,
            distributed topology and number of DataLoader workers. With several workers, the offset
            is the rank's offset in whole batches of ``batch_size``; each worker derives its own part,
            because the DataLoader takes batches from its workers in turn. Random image transforms
            are not restored by this method.
        """
        epoch = int(state.get("epoch", 0))
        offset = int(state.get("offset", 0))
        batch_size = int(state.get("batch_size", 1))
        if epoch < 0 or offset < 0 or batch_size <= 0:
            raise ValueError(
                "Streaming dataset epoch/offset must be non-negative and batch_size must be positive"
            )
        self._next_epoch = epoch
        self._active_epoch = epoch
        self._resume_offset = offset
        self._resume_batch_size = batch_size
        self._state_offset = offset

    def set_epoch(self, epoch: int) -> None:
        """Select a non-negative epoch and reset the rank-local resume offset.

        Args:
            epoch (`int`):
                Coverage epoch used to seed the next sample plan when shuffle is enabled.

        Raises:
            ValueError: If epoch is negative.
        """
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._next_epoch = epoch
        self._active_epoch = epoch
        self._resume_offset = 0
        self._resume_batch_size = 1
        self._state_offset = 0

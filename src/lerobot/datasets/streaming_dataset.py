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
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import datasets
import numpy as np
import torch

from lerobot.configs import DEFAULT_DEPTH_UNIT, DEPTH_METER_UNIT, DepthEncoderConfig
from lerobot.streaming.episode_video import EpisodeByteCache, EpisodeVideoManifest, ExactCoveragePool
from lerobot.utils.constants import HF_LEROBOT_HOME

from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .depth_utils import MM_PER_METRE, dequantize_depth
from .episode_parquet import EpisodeParquetReader
from .feature_utils import check_delta_timestamps, get_delta_indices, get_hf_features_from_features
from .io_utils import hf_transform_to_torch
from .streaming_sidecar import (
    ensure_dataset_mp4_sidecar,
    range_backend_for_root,
    streaming_data_root,
)
from .utils import check_version_compatibility
from .video_utils import decode_video_frames_pyav


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


class StreamingLeRobotDataset(torch.utils.data.IterableDataset):
    """Episode-scoped streaming reader for LeRobot datasets.

    Metadata is cached locally, while each rank reads only the Parquet rows and MP4 byte ranges
    needed for the complete episodes it owns. One DataLoader worker per rank owns the logical pool;
    its bounded result queue provides decoded-batch prefetch without creating independent samplers.
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

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
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
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
        data_root: str | Path | None = None,
        episode_pool_size: int | None = None,
        prefetch_episodes: int = 8,
        byte_budget_gb: float = 8.0,
        repeat: bool = False,
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
            streaming (bool, optional): Compatibility flag retained by the public API.
            buffer_size (int, optional): Compatibility setting used to derive the default episode pool size.
            max_num_shards (int, optional): Maximum number of concurrent episode loading workers.
            seed (int, optional): Reproducibility random seed.
            rng (np.random.Generator | None, optional): Random number generator.
            shuffle (bool, optional): Whether to shuffle the dataset across exhaustions. Defaults to True.
            depth_output_unit (str, optional): Physical unit depth maps are dequantized to ("m" or "mm").
                Defaults to "mm".
            data_root (str | Path | None, optional): Dataset payload root. Supports local paths, ``hf://``,
                and fsspec URLs.
            episode_pool_size (int | None, optional): Number of complete episodes in the sampling pool.
            prefetch_episodes (int, optional): Episodes prefetched beyond the active pool.
            byte_budget_gb (float, optional): Per-rank upper bound for synthesized episode video bytes.
            repeat (bool, optional): Repeat rank-local exact-coverage epochs without yielding a
                short final training batch. The training factory enables this; direct iteration is
                finite by default.
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
        self._depth_output_unit = depth_output_unit
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if max_num_shards <= 0:
            raise ValueError("max_num_shards must be positive")
        if episode_pool_size is not None and episode_pool_size <= 0:
            raise ValueError("episode_pool_size must be positive")
        if prefetch_episodes < 0:
            raise ValueError("prefetch_episodes must be non-negative")
        if byte_budget_gb <= 0:
            raise ValueError("byte_budget_gb must be positive")
        self.episode_pool_size = episode_pool_size or min(buffer_size, 32)
        self.prefetch_episodes = prefetch_episodes
        self.byte_budget = int(byte_budget_gb * 1024**3)
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
            self.repo_id, self._requested_root, self.revision, force_cache_sync=force_cache_sync
        )
        self.root = self.meta.root
        self.revision = self.meta.revision
        self.meta.rescale_depth_stats(self._depth_output_unit)
        # Check version
        check_version_compatibility(self.repo_id, self.meta._version, CODEBASE_VERSION)

        self._depth_encoder_configs: dict[str, DepthEncoderConfig] = {
            vid_key: DepthEncoderConfig.from_video_info(self.meta.features[vid_key].get("info"))
            for vid_key in self.meta.depth_keys
        }

        # Input unit of each depth feature stored as raw images (dequantized separately from videos).
        self._image_depth_units: dict[str, str | None] = {
            key: (self.meta.features[key].get("info") or {}).get("depth_unit")
            for key in self.meta.depth_keys
            if key in self.meta.image_keys
        }

        selected_episodes = list(range(self.meta.total_episodes)) if episodes is None else list(episodes)
        if len(set(selected_episodes)) != len(selected_episodes):
            raise ValueError("episodes must not contain duplicates")
        invalid_episodes = [
            episode for episode in selected_episodes if episode < 0 or episode >= self.meta.total_episodes
        ]
        if invalid_episodes:
            raise ValueError(
                f"Episode indices out of range for dataset with {self.meta.total_episodes} episodes: "
                f"{invalid_episodes}"
            )
        self._selected_episodes = selected_episodes

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
        )
        sidecar_backend = range_backend_for_root(self._data_root)
        self._sidecar_path = ensure_dataset_mp4_sidecar(
            self.meta,
            self._data_root,
            workers=max_num_shards,
            range_backend=sidecar_backend,
        )
        self._hf_features = get_hf_features_from_features(self.meta.features)
        self._projected_columns = tuple(self._hf_features)
        self.num_shards = min(max_num_shards, max(1, len(self._selected_episodes)))

    @property
    def num_frames(self) -> int:
        return sum(self._episode_frame_count(episode) for episode in self._selected_episodes)

    @property
    def num_episodes(self) -> int:
        return len(self._selected_episodes)

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def depth_output_unit(self) -> str:
        """Physical unit (``"m"`` or ``"mm"``) depth maps are returned in on read."""
        return self._depth_output_unit

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.repeat:
            return self._repeat_iterator()
        return self._iter_once()

    def _repeat_iterator(self) -> Iterator[dict[str, torch.Tensor]]:
        while True:
            iterator = self._iter_once()
            try:
                first = next(iterator)
            except StopIteration:
                return
            yield first
            yield from iterator

    def _iter_once(self) -> Iterator[dict[str, torch.Tensor]]:
        epoch = self._next_epoch if self.shuffle else 0
        self._active_epoch = epoch

        worker = torch.utils.data.get_worker_info()
        if worker is not None and worker.num_workers > 1:
            raise RuntimeError(
                "StreamingLeRobotDataset uses one rank-level sampling pool and supports at most "
                "one DataLoader worker per rank"
            )
        resume_offset = self._resume_offset
        self._resume_offset = 0
        consumer_episodes, _rank, _world_size = self._rank_episodes()
        consumer_frame_count = sum(self._episode_frame_count(episode) for episode in consumer_episodes)
        if consumer_frame_count:
            worker_epoch_delta, resume_offset = divmod(resume_offset, consumer_frame_count)
        else:
            worker_epoch_delta = 0
        epoch += worker_epoch_delta
        self._active_epoch = epoch
        if self.shuffle:
            self._next_epoch = epoch + 1

        max_workers = min(self.max_num_shards, max(1, self.episode_pool_size + self.prefetch_episodes))
        video_cache = self._make_video_cache(consumer_episodes, max_workers)
        episode_byte_sizes = (
            {episode: video_cache.manifest.episode_byte_size(episode) for episode in consumer_episodes}
            if video_cache is not None
            else None
        )
        planner = ExactCoveragePool(
            [(episode, self._episode_frame_count(episode)) for episode in consumer_episodes],
            pool_size=self.episode_pool_size,
            seed=self.seed,
            epoch=epoch,
            episode_byte_sizes=episode_byte_sizes,
            byte_budget=self.byte_budget if episode_byte_sizes is not None else None,
        )
        for _ in range(resume_offset):
            try:
                next(planner)
            except StopIteration:
                return
        planner.newly_admitted.clear()
        planner.evicted.clear()

        parquet_reader = EpisodeParquetReader(self._data_root, columns=self._projected_columns)
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lerobot-parquet")
        episode_futures: dict[int, Future[datasets.Dataset]] = {}
        scheduled_episodes: set[int] = set()
        retained_video_episodes: set[int] = set()
        if video_cache is not None:
            for episode_index in planner.resident:
                video_cache.retain_episode(episode_index)
                retained_video_episodes.add(episode_index)

        def submit(episode_index: int) -> Future[datasets.Dataset]:
            future = episode_futures.get(episode_index)
            if future is None:
                future = executor.submit(self._load_episode_dataset, parquet_reader, episode_index)
                episode_futures[episode_index] = future
            return future

        def schedule_frontier() -> None:
            frontier = [*planner.resident, *planner.prefetch_candidates(self.prefetch_episodes)]
            for episode_index in frontier:
                if episode_index in scheduled_episodes:
                    continue
                submit(episode_index)
                if video_cache is not None:
                    video_cache.submit_prefetch(episode_index)
                scheduled_episodes.add(episode_index)

        schedule_frontier()
        try:
            while True:
                try:
                    episode_index, frame_index = next(planner)
                except StopIteration:
                    self._active_epoch = epoch + 1 if self.shuffle else 0
                    self._state_offset = 0
                    break

                episode_dataset = submit(episode_index).result()
                item = self._make_episode_item(
                    episode_dataset,
                    episode_index,
                    frame_index,
                    video_cache=video_cache,
                )

                for evicted_episode in planner.evicted:
                    episode_futures.pop(evicted_episode, None)
                    if video_cache is not None and evicted_episode in retained_video_episodes:
                        video_cache.release_episode(evicted_episode)
                        retained_video_episodes.remove(evicted_episode)
                if video_cache is not None:
                    for admitted_episode in planner.newly_admitted:
                        if admitted_episode not in retained_video_episodes:
                            video_cache.retain_episode(admitted_episode)
                            retained_video_episodes.add(admitted_episode)
                planner.evicted.clear()
                planner.newly_admitted.clear()
                schedule_frontier()

                self._state_offset += 1
                yield item
        finally:
            for future in episode_futures.values():
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            if video_cache is not None:
                video_cache.close()

    def _rank_episodes(self) -> tuple[list[int], int, int]:
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

    def _episode_frame_count(self, episode_index: int) -> int:
        episode = self.meta.episodes[episode_index]
        return int(episode["dataset_to_index"] - episode["dataset_from_index"])

    def num_frames_for_rank(self, rank: int, world_size: int, num_workers: int) -> int:
        """Return frames owned by one training rank under balanced whole-episode sharding."""
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid distributed rank/world size: rank={rank}, world_size={world_size}")
        if num_workers > 1:
            raise ValueError("Rank-level streaming supports at most one DataLoader worker per rank")
        counts = {episode: self._episode_frame_count(episode) for episode in self._selected_episodes}
        shards = _balanced_episode_shards(self._selected_episodes, counts, world_size=world_size)
        return sum(counts[episode] for episode in shards[rank])

    def _load_episode_dataset(
        self,
        reader: EpisodeParquetReader,
        episode_index: int,
    ) -> datasets.Dataset:
        table = reader.read_episode(
            self.meta.get_data_file_path(episode_index),
            episode_index=episode_index,
            expected_rows=self._episode_frame_count(episode_index),
        )
        # from_dict applies the declared HF feature encoders (images, nested language/JSON fields)
        # while retaining the episode-sized memory bound.
        dataset = datasets.Dataset.from_dict(table.to_pydict(), features=self._hf_features)
        dataset.set_transform(hf_transform_to_torch)
        return dataset

    def _make_video_cache(
        self,
        episode_indices: list[int],
        workers: int,
    ) -> EpisodeByteCache | None:
        if self._sidecar_path is None or not episode_indices:
            return None
        range_backend = range_backend_for_root(self._data_root)
        manifest = EpisodeVideoManifest.build(
            self.meta,
            self._data_root,
            episode_indices=episode_indices,
            range_backend=range_backend,
            workers=workers,
            sidecar_path=self._sidecar_path,
        )
        decoder_limit = max(1, min(64, self.episode_pool_size * max(1, len(self.meta.video_keys))))
        return EpisodeByteCache(
            manifest,
            self._data_root,
            byte_budget=self.byte_budget,
            workers=workers,
            range_backend=range_backend,
            max_open_decoders=decoder_limit,
        )

    def _make_episode_item(
        self,
        episode_dataset: datasets.Dataset,
        episode_index: int,
        frame_index: int,
        *,
        video_cache: EpisodeByteCache | None,
    ) -> dict:
        item = episode_dataset[frame_index]
        episode = self.meta.episodes[episode_index]
        episode_start = int(episode["dataset_from_index"])

        if self.delta_indices is not None:
            for key, delta_indices in self.delta_indices.items():
                target_indices = [
                    max(0, min(len(episode_dataset) - 1, frame_index + delta)) for delta in delta_indices
                ]
                item[f"{key}_is_pad"] = torch.BoolTensor(
                    [
                        frame_index + delta < 0 or frame_index + delta >= len(episode_dataset)
                        for delta in delta_indices
                    ]
                )
                if key not in self.meta.video_keys:
                    item[key] = torch.stack(episode_dataset[target_indices][key])

        if self.meta.video_keys:
            if video_cache is None:
                raise RuntimeError("Video dataset streaming requires an episode byte cache")
            episode_metadata = self.meta.episodes[episode_index]
            for video_key in self.meta.video_keys:
                if self.delta_indices is not None and video_key in self.delta_indices:
                    target_indices = [
                        max(0, min(len(episode_dataset) - 1, frame_index + delta))
                        for delta in self.delta_indices[video_key]
                    ]
                else:
                    target_indices = [frame_index]
                local_timestamps = [
                    float(episode_dataset[index]["timestamp"].item()) for index in target_indices
                ]
                from_timestamp = float(episode_metadata[f"videos/{video_key}/from_timestamp"])
                query_timestamps = [from_timestamp + timestamp for timestamp in local_timestamps]
                if video_key in self.meta.depth_keys:
                    source_start = video_cache.manifest.lookup(episode_index, video_key).source_start_pts
                    frames = decode_video_frames_pyav(
                        io.BytesIO(video_cache.get_bytes(episode_index, video_key)),
                        [timestamp - source_start for timestamp in query_timestamps],
                        self.tolerance_s,
                        return_uint8=False,
                        is_depth=True,
                    )
                    depth_encoder = self._depth_encoder_configs[video_key]
                    frames = dequantize_depth(
                        frames,
                        depth_min=depth_encoder.depth_min,
                        depth_max=depth_encoder.depth_max,
                        shift=depth_encoder.shift,
                        use_log=depth_encoder.use_log,
                        output_unit=self._depth_output_unit,
                    )
                else:
                    frames = video_cache.get_frames(episode_index, video_key, query_timestamps)
                    if not self._return_uint8:
                        frames = frames.to(torch.float32) / 255.0
                item[video_key] = frames.squeeze(0)

        if self.image_transforms is not None:
            for camera_key in self.meta.camera_keys:
                if camera_key in self.meta.depth_keys:
                    continue
                item[camera_key] = self.image_transforms(item[camera_key])

        for key, stored_unit in self._image_depth_units.items():
            if key in item and stored_unit is not None and stored_unit != self._depth_output_unit:
                item[key] = (
                    item[key] * MM_PER_METRE if stored_unit == DEPTH_METER_UNIT else item[key] / MM_PER_METRE
                )

        task_index = int(item["task_index"].item())
        item["task"] = self.meta.tasks.iloc[task_index].name
        if int(item["episode_index"].item()) != episode_index:
            raise RuntimeError(f"Episode reader returned episode {item['episode_index']} for {episode_index}")
        if int(item["index"].item()) != episode_start + frame_index:
            raise RuntimeError(
                f"Episode {episode_index} frame {frame_index} has unexpected absolute index {item['index']}"
            )
        return item

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch": self._active_epoch,
            "offset": self._state_offset,
            "batch_size": self._resume_batch_size,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
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
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._next_epoch = epoch
        self._active_epoch = epoch
        self._resume_offset = 0
        self._resume_batch_size = 1
        self._state_offset = 0

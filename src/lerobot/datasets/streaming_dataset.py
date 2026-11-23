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
import asyncio
import random
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import datasets
import fsspec
import numpy as np
import torch
from datasets import load_dataset

from lerobot.configs import DEFAULT_DEPTH_UNIT, DEPTH_METER_UNIT, DepthEncoderConfig
from lerobot.utils.constants import HF_LEROBOT_HOME

from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .depth_utils import MM_PER_METRE, dequantize_depth
from .feature_utils import get_delta_indices
from .io_utils import item_to_torch
from .utils import (
    check_version_compatibility,
    find_float_index,
    is_float_in_list,
)
from .video_utils import (
    decode_video_frames,
    decode_video_frames_torchcodec,
)


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
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        buffer_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        return_uint8: bool = False,
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
        *,
        repo_type: Literal["dataset", "bucket"] = "dataset",
        token: str | bool | None = None,
    ):
        """Initialize a StreamingLeRobotDataset.

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset.
            root (Path | None, optional): Local directory to use for local datasets. In bucket mode,
                this is an optional local metadata-cache directory; parquet and video data remain remote.
                When omitted, Hub metadata is resolved through the cache under ``$HF_LEROBOT_HOME/hub``.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list.
            image_transforms (Callable | None, optional): Transform to apply to image data.
            tolerance_s (float, optional): Tolerance in seconds for timestamp matching.
            revision (str, optional): Git revision id (branch name, tag, or commit hash).
            force_cache_sync (bool, optional): Flag to sync and refresh local files first.
            buffer_size (int, optional): Buffer size for shuffling when streaming. Defaults to 32.
            seed (int, optional): Reproducibility random seed.
            shuffle (bool, optional): Whether to shuffle the dataset across exhaustions. Defaults to True.
            depth_output_unit (str, optional): Physical unit depth maps are dequantized to ("m" or "mm").
                Defaults to "mm".
            repo_type: "dataset" (default) or "bucket" to stream from an HF Storage Bucket
                over ``hf://buckets/``.
            token: Authentication token used while streaming this dataset from
                the Hub. Pass a string token, ``True`` to require the locally
                stored token, ``False`` to disable authentication, or ``None``
                to use the Hugging Face Hub default. The token is not retained
                on the dataset instance after initialization.
        """
        super().__init__()
        if repo_type not in ("dataset", "bucket"):
            raise ValueError(f"repo_type must be 'dataset' or 'bucket', got {repo_type!r}")

        self.repo_id = repo_id
        self.repo_type = repo_type
        self._requested_root = Path(root) if root is not None else None
        self.root = self._requested_root if self._requested_root is not None else HF_LEROBOT_HOME / repo_id
        self.streaming_from_local = root is not None and self.repo_type in ("dataset", "bucket")

        self.image_transforms = image_transforms
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision: str | None = revision if revision else CODEBASE_VERSION
        self.seed = seed
        self.shuffle = shuffle

        self.buffer_size = buffer_size
        self._return_uint8 = return_uint8
        self._depth_output_unit = depth_output_unit

        if self._requested_root is not None:
            self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id,
            self._requested_root,
            self.revision,
            force_cache_sync=force_cache_sync,
            repo_type=self.repo_type,
            token=token,
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

        self.delta_timestamps = None
        self.delta_indices = None

        if delta_timestamps is not None:
            self._validate_delta_timestamp_keys(delta_timestamps)  # raises ValueError if invalid
            self.delta_timestamps = delta_timestamps
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        token_kwargs = {} if token is None and not self.streaming_from_local else {"token": token}
        if repo_type == "dataset":
            dataset_path = self.repo_id
        elif repo_type == "bucket":
            dataset_path = "buckets/" + self.repo_id
        else:
            dataset_path = str(self.root)
        self.hf_dataset: datasets.IterableDataset = load_dataset(
            dataset_path,
            split="train",
            streaming=True,
            data_files="data/*/*.parquet",
            revision=self.revision,
            **token_kwargs,
        )
        if self.hf_dataset.num_shards < 8:
            # reshard per row group instead of per file to get more shards
            self.hf_dataset = self.hf_dataset.reshard()

        self.num_shards = self.hf_dataset.num_shards

    @property
    def num_frames(self):
        return self.meta.total_frames

    @property
    def num_episodes(self):
        return self.meta.total_episodes

    @property
    def fps(self):
        return self.meta.fps

    @property
    def depth_output_unit(self) -> str:
        """Physical unit (``"m"`` or ``"mm"``) depth maps are returned in on read."""
        return self._depth_output_unit

    def __iter__(self) -> Iterator[dict[str, Any]]:

        ds = self.hf_dataset
        ds = ds.batch(by_column="episode_index")
        ds = ds.shuffle(seed=42, buffer_size=self.buffer_size)

        ds = ds.map(self._sample_pool_with_delta, batched=True, batch_size=self.buffer_size)

        decode_workers = 32
        decode_thread_pool = ThreadPoolExecutor(decode_workers)
        decode_semaphore = asyncio.Semaphore(decode_workers)

        async def decode_frames(item):
            async with decode_semaphore:
                return await asyncio.get_running_loop().run_in_executor(decode_thread_pool, self.query_frames, item)

        ds = ds.map(decode_frames)
        yield from ds

    def _sample_pool_with_delta(self, episodes_batch):
        episode_batches = [
            {col: episodes_batch[col][ep] for col in episodes_batch}
            for ep in range(len(episodes_batch["episode_index"]))
        ]
        for episode_batch in episode_batches:
            episode_batch = item_to_torch(episode_batch)
            if self.delta_indices is not None:
                query_results, paddings = self._get_delta_frames(episode_batch)
                episode_batch.update(query_results)
                episode_batch.update(paddings)
        samples = [(ep, frame) for ep in range(len(episode_batches)) for frame in range(len(episode_batches[ep]["episode_index"]))]
        random.Random("|".join(str(ep[0]) for ep in episodes_batch["episode_index"])).shuffle(samples)
        out = {col: [episode_batches[ep][col][frame] for ep, frame in samples] for col in episodes_batch}
        return out

    def _make_timestamps_from_indices(
        self, start_ts: float, indices: dict[str, list[int]] | None = None
    ) -> dict[str, list[float]]:
        if indices is not None:
            return {
                key: (
                    start_ts + torch.tensor(indices[key]) / self.fps
                ).tolist()  # NOTE: why not delta_timestamps directly?
                for key in indices
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

    def query_frames(self, item) -> dict:
        ep_idx = int(item["episode_index"])
        task_idx = int(item["task_index"])
        current_ts = int(item["frame_index"]) / self.fps

        episode_boundaries_ts = {
            key: (
                self.meta.episodes[ep_idx][f"videos/{key}/from_timestamp"],
                self.meta.episodes[ep_idx][f"videos/{key}/to_timestamp"],
            )
            for key in self.meta.video_keys
        }

        updates = []

        # Load video frames, when needed
        if len(self.meta.video_keys) > 0:
            original_timestamps = self._make_timestamps_from_indices(current_ts, self.delta_indices)

            # Some timestamps might not result available considering the episode's boundaries
            query_timestamps = self._get_query_timestamps(
                current_ts, self.delta_indices, episode_boundaries_ts
            )
            video_frames = self._query_videos(query_timestamps, ep_idx)

            if self.image_transforms is not None:
                image_keys = self.meta.camera_keys
                for cam in image_keys:
                    video_frames[cam] = self.image_transforms(video_frames[cam])

            updates.append(video_frames)

            if self.delta_indices is not None:
                # We always return the same number of frames. Unavailable frames are padded.
                padding_mask = self._get_video_frame_padding_mask(
                    video_frames, query_timestamps, original_timestamps
                )
                updates.append(padding_mask)

        result = item.copy()
        for update in updates:
            result.update(update)

        # Convert raw-image depth features to the output unit (video depth is already converted).
        for key, stored_unit in self._image_depth_units.items():
            if key in result and stored_unit is not None and stored_unit != self._depth_output_unit:
                result[key] = (
                    result[key] * MM_PER_METRE
                    if stored_unit == DEPTH_METER_UNIT
                    else result[key] / MM_PER_METRE
                )

        result["task"] = self.meta.tasks.iloc[task_idx].name

        return result

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None,
        episode_boundaries_ts: dict[str, tuple[float, float]],
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        keys_to_timestamps = self._make_timestamps_from_indices(current_ts, query_indices)
        for key in self.meta.video_keys:
            timestamps = keys_to_timestamps[key]
            # Clamp out timesteps outside of episode boundaries
            query_timestamps[key] = torch.clamp(
                torch.tensor(timestamps) + episode_boundaries_ts[key][0], *episode_boundaries_ts[key]
            ).tolist()
        return query_timestamps

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """

        item = {}
        for video_key, query_ts in query_timestamps.items():
            root = self.meta.url_root if not self.streaming_from_local else self.root
            video_path = f"{root}/{self.meta.get_video_file_path(ep_idx, video_key)}"
            if not self.streaming_from_local:
                # Caching video file blocks speeds-up streaming.
                # Contrary to fsspec's "blockcache", "sharedblockcache" is thread-safe
                # and shares the cache between threads.
                video_path = "sharedblockcache::" + video_path
            if video_key in self.meta.depth_keys:
                # Depth maps are 12-bit quantized and only decodable via pyav; dequantize back
                # to physical units to match the non-streaming reader.
                with fsspec.open(video_path) as video_file:
                    frames = decode_video_frames(
                        video_file,
                        query_ts,
                        self.tolerance_s,
                        backend="pyav",
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
                with fsspec.open(video_path) as video_file:
                    frames = decode_video_frames_torchcodec(
                        video_file,
                        query_ts,
                        self.tolerance_s,
                        return_uint8=self._return_uint8,
                    )

            item[video_key] = frames.squeeze(0) if len(query_ts) == 1 else frames

        return item

    def _get_delta_frames(
        self, episode_batch
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Get frames with delta offsets for every step.

        Args:
            episode_batch (dict): Batch containing all the data from one episode.
                Generally obtained using `ds.batch(by_column="episode_index")`.
                Values in the dictionary are lists of the same length (one item per step)

        Returns:
            tuple: (query_result, padding) - frames at delta offsets and padding info.
        """
        num_steps = len(episode_batch["episode_index"])
        if self.delta_indices is None:
            raise RuntimeError("Delta frames were requested but no delta_timestamps are configured.")

        # Prepare results
        query_results: dict[str, list[torch.Tensor]] = {key: [] for key in self.delta_indices}
        paddings: dict[str, list[torch.Tensor]] = {key: [] for key in self.delta_indices}

        for key, delta_indices in self.delta_indices.items():
            if key in self.meta.video_keys:
                continue  # visual frames are decoded separately

            for index in range(num_steps):
                current_item = {k: episode_batch[k][index] for k in episode_batch}
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
                        current_item,
                        False,
                    )

                # Process negative deltas in order of increasing difficulty
                lookback_failed = False

                last_successful_frame = current_item[key]

                for delta in negative_deltas:
                    if lookback_failed:
                        delta_results[delta] = (last_successful_frame, True)
                        continue

                    if index + delta > 0:
                        past_item = {k: episode_batch[k][index + delta] for k in episode_batch}
                        delta_results[delta] = (past_item[key], False)
                        last_successful_frame = past_item[key]
                    else:
                        delta_results[delta] = (last_successful_frame, True)
                        lookback_failed = True  # All subsequent negative deltas will also fail

                # Process positive deltas in order of increasing difficulty
                lookahead_failed = False
                last_successful_frame = current_item[key]

                for delta in positive_deltas:
                    if lookahead_failed:
                        delta_results[delta] = (last_successful_frame, True)
                        continue

                    if index + delta < num_steps:
                        future_item = {k: episode_batch[k][index + delta] for k in episode_batch}
                        delta_results[delta] = (future_item[key], False)
                        last_successful_frame = future_item[key]
                    else:
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
                    query_results[key].append(torch.stack(target_frames))
                    paddings[f"{key}_is_pad"].append(torch.BoolTensor(is_pad))

        return query_results, paddings

    def _validate_delta_timestamp_keys(self, delta_timestamps: dict[str, list[float]] | None) -> None:
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

import random
from pathlib import Path
from typing import Callable, Dict, Generator, Iterator, Tuple

import datasets
import numpy as np
import torch
from datasets import load_dataset
from line_profiler import profile

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    Backtrackable,
    LookAheadError,
    LookBackError,
    check_delta_timestamps,
    check_version_compatibility,
    get_delta_indices,
    item_to_torch,
)
from lerobot.common.datasets.video_utils import (
    VideoDecoderCache,
    decode_video_frames_torchcodec,
    get_safe_default_codec,
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
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        video_backend: str | None = "torchcodec",
        streaming: bool = True,
        buffer_size: int = 1000,
        max_num_shards: int = 16,
        seed: int = 42,
        rng: np.random.Generator | None = None,
    ):
        """Initialize a StreamingLeRobotDataset.

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset.
            root (Path | None, optional): Local directory to use for downloading/writing files.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list.
            image_transforms (Callable | None, optional): Transform to apply to image data.
            tolerance_s (float, optional): Tolerance in seconds for timestamp matching.
            revision (str, optional): Git revision id (branch name, tag, or commit hash).
            force_cache_sync (bool, optional): Flag to sync and refresh local files first.
            video_backend (str | None, optional): Video backend to use for decoding videos. Uses "torchcodec" by default.
            streaming (bool, optional): Whether to stream the dataset or load it all. Defaults to True.
            buffer_size (int, optional): Buffer size for shuffling when streaming. Defaults to 1000.
            max_num_shards (int, optional): Number of shards to re-shard the input dataset into. Defaults to 16.
            seed (int, optional): Reproducibility random seed.
            rng (np.random.Generator | None, optional): Random number generator.
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.seed = seed
        self.rng = rng if rng is not None else np.random.default_rng(seed)

        self.streaming = streaming
        self.buffer_size = buffer_size

        # We cache the video decoders to avoid re-initializing them at each frame (avoiding a ~10x slowdown)
        self.video_decoder_cache = VideoDecoderCache()

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        # Check version
        check_version_compatibility(self.repo_id, self.meta._version, CODEBASE_VERSION)

        if delta_timestamps is not None:
            self._validate_delta_timestamp_keys(delta_timestamps)  # raises ValueError if invalid
            self.delta_timestamps = delta_timestamps

        self.hf_dataset: datasets.IterableDataset = self.load_hf_dataset()
        self.num_shards = min(self.hf_dataset.num_shards, max_num_shards)

        max_backward_steps, max_forward_steps = self._get_window_steps()
        self.backtrackable_dataset: Backtrackable = Backtrackable(
            self.hf_dataset, history=max_backward_steps, lookahead=max_forward_steps
        )

    @property
    def fps(self):
        return self.meta.fps

    @staticmethod
    def _iter_random_indices(
        rng: np.random.Generator, buffer_size: int, random_batch_size=1000
    ) -> Iterator[int]:
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))

    @staticmethod
    def _infinite_generator_over_elements(elements: list[int]) -> Iterator[int]:
        return (random.choice(list(elements)) for _ in iter(int, 1))

    def load_hf_dataset(self) -> datasets.IterableDataset:
        dataset = load_dataset(self.repo_id, split="train", streaming=self.streaming)
        self.streaming_from_local = False

        # TODO(fracapuano): Add support for streaming from a local folder and not only from HF Hub
        return dataset

    def _get_window_steps(self) -> Tuple[int, int]:
        """
        Returns how many steps backward (& forward) should the backtrackable iterator maintain,
        based on the input delta_timestamps.
        """
        max_backward_steps = 1
        max_forward_steps = 1

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

            # Calculate maximum backward steps needed (i.e., history size)
            for delta_idx in self.delta_indices.values():
                min_delta = min(delta_idx)
                max_delta = max(delta_idx)
                if min_delta < 0:
                    max_backward_steps = max(max_backward_steps, abs(min_delta))
                if max_delta > 0:
                    max_forward_steps = max(max_forward_steps, max_delta)

        return max_backward_steps, max_forward_steps

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer_indices_generator = self._iter_random_indices(self.rng, self.buffer_size)
        idx_to_backtracktable_dataset = {
            idx: self._make_backtrackable_dataset(self.hf_dataset.shard(self.num_shards, index=idx))
            for idx in range(self.num_shards)
        }

        # This buffer is populated while iterating on the dataset's shards
        frames_buffer = []
        try:
            while available_shards := list(idx_to_backtracktable_dataset.keys()):
                shard_key = next(self._infinite_generator_over_elements(available_shards))
                dataset = idx_to_backtracktable_dataset[shard_key]  # selects which shard to iterate on
                for frame in self.make_frame(dataset):
                    if len(frames_buffer) == self.buffer_size:
                        i = next(buffer_indices_generator)
                        yield frames_buffer[i]
                        frames_buffer[i] = frame
                    else:
                        frames_buffer.append(frame)
                    break  # random shard sampled, switch shard

        except (
            RuntimeError,
            StopIteration,
        ):  # NOTE: StopIteration inside a generator throws a RuntimeError since 3.7
            del idx_to_backtracktable_dataset[shard_key]  # Remove exhausted shard, onto another shard

        # Once shards are all exhausted, shuffle the buffer and yield the remaining frames
        self.rng.shuffle(frames_buffer)
        yield from frames_buffer

    def _make_backtrackable_dataset(self, dataset: datasets.IterableDataset) -> Backtrackable:
        history, lookahead = self._get_window_steps()
        return Backtrackable(dataset, history=history, lookahead=lookahead)

    def _make_timestamps_from_indices(
        self, start_ts: float, indices: dict[str, list[int]] | None = None
    ) -> dict[str, list[float]]:
        if indices is not None:
            return {
                key: (start_ts + torch.tensor(indices[key]) / self.fps).tolist()
                for key in self.delta_timestamps
            }
        else:
            return dict.fromkeys(self.delta_timestamps, start_ts)

    def _make_padding_camera_frame(self, camera_key: str):
        """Variable-shape padding frame for given camera keys, given in (C, H, W)"""
        return torch.zeros(self.meta.info["features"][camera_key]["shape"]).permute(-1, 0, 1)

    def _pad_retrieved_video_frames(
        self,
        video_frames: dict[str, torch.Tensor],
        query_timestamps: dict[str, list[float]],
        original_timestamps: dict[str, list[float]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.BoolTensor]]:
        padded_video_frames = {}
        padding_mask = {}

        for video_key, timestamps in original_timestamps.items():
            if video_key not in video_frames:
                continue  # only padding on video keys that are available
            frames = []
            mask = []
            padding_frame = self._make_padding_camera_frame(video_key)
            for ts in timestamps:
                if ts in query_timestamps[video_key]:
                    idx = query_timestamps[video_key].index(ts)
                    frames.append(video_frames[video_key][idx, :])
                    mask.append(False)
                else:
                    frames.append(padding_frame)
                    mask.append(True)

            padded_video_frames[video_key] = torch.stack(frames)
            padding_mask[f"{video_key}.pad_masking"] = torch.BoolTensor(mask)

        return padded_video_frames, padding_mask

    @profile
    def make_frame(self, dataset_iterator: Backtrackable) -> Generator:
        """Makes a frame starting from a dataset iterator"""
        item = next(dataset_iterator)
        item = item_to_torch(item)

        updates = []  # list of updates to apply to the item

        # Get episode index from the item
        ep_idx = item["episode_index"]

        # "timestamp" restarts from 0 for each episode, whereas we need a global timestep within the single .mp4 file (given by index/fps)
        current_ts = item["index"] / self.fps

        episode_boundaries_ts = {
            key: (
                self.meta.episodes[ep_idx][f"videos/{key}/from_timestamp"],
                self.meta.episodes[ep_idx][f"videos/{key}/to_timestamp"],
            )
            for key in self.meta.video_keys
        }

        # Apply delta querying logic if necessary
        if self.delta_indices is not None:
            query_result, padding = self._get_delta_frames(dataset_iterator, item)
            updates.append(query_result)
            updates.append(padding)

        # Load video frames, when needed
        if len(self.meta.video_keys) > 0:
            original_timestamps = self._make_timestamps_from_indices(current_ts, self.delta_indices)

            # Some timestamps might not result available considering the episode's boundaries
            query_timestamps = self._get_query_timestamps(
                current_ts, self.delta_indices, episode_boundaries_ts
            )
            video_frames = self._query_videos(query_timestamps, ep_idx)

            # We always return the same number of frames. Unavailable frames are padded.
            padded_video_frames, padding_mask = self._pad_retrieved_video_frames(
                video_frames, query_timestamps, original_timestamps
            )

            updates.append(video_frames)
            updates.append(padded_video_frames)
            updates.append(padding_mask)

        result = item.copy()
        for update in updates:
            result.update(update)

        result["task"] = self.meta.tasks.iloc[item["task_index"]].name

        yield result

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
                # Filter out timesteps outside of episode boundaries
                query_timestamps[key] = [
                    ts
                    for ts in timestamps
                    if episode_boundaries_ts[key][0] <= ts <= episode_boundaries_ts[key][1]
                ]

                if len(query_timestamps[key]) == 0:
                    raise ValueError(f"No valid timestamps found for key {key} with {query_indices[key]}")

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
            root = self.meta.url_root if self.streaming and not self.streaming_from_local else self.root
            video_path = f"{root}/{self.meta.get_video_file_path(ep_idx, video_key)}"
            frames = decode_video_frames_torchcodec(
                video_path, query_ts, self.tolerance_s, decoder_cache=self.video_decoder_cache
            )

            item[video_key] = frames

        return item

    def _make_padding_frame(self, key: str) -> tuple[torch.Tensor, bool]:
        return torch.zeros(self.meta.info["features"][key]["shape"]), True

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
                )  # unsqueeze to add batch dimension for stacking

            # Process negative deltas in order of increasing difficulty
            lookback_failed = False
            for delta in negative_deltas:
                if lookback_failed:
                    delta_results[delta] = self._make_padding_frame(key)
                    continue

                try:
                    steps_back = abs(delta)
                    if dataset_iterator.can_peek_back(steps_back):
                        past_item = dataset_iterator.peek_back(steps_back)
                        past_item = item_to_torch(past_item)

                        if past_item["episode_index"] == current_episode_idx:
                            delta_results[delta] = (past_item[key], False)
                        else:
                            raise LookBackError("Retrieved frame is from different episode!")
                    else:
                        raise LookBackError("Cannot go back further than the history buffer!")

                except LookBackError:
                    delta_results[delta] = self._make_padding_frame(key)
                    lookback_failed = True  # All subsequent negative deltas will also fail

            # Process positive deltas in order of increasing difficulty
            lookahead_failed = False
            for delta in positive_deltas:
                if lookahead_failed:
                    delta_results[delta] = self._make_padding_frame(key)
                    continue

                try:
                    if dataset_iterator.can_peek_ahead(delta):
                        future_item = dataset_iterator.peek_ahead(delta)
                        future_item = item_to_torch(future_item)

                        if future_item["episode_index"] == current_episode_idx:
                            delta_results[delta] = (future_item[key], False)
                        else:
                            raise LookAheadError("Retrieved frame is from different episode!")
                    else:
                        raise LookAheadError("Cannot go ahead further than the lookahead buffer!")

                except LookAheadError:
                    delta_results[delta] = self._make_padding_frame(key)
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
                padding[f"{key}.pad_masking"] = torch.BoolTensor(is_pad)

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


# Example usage
if __name__ == "__main__":
    from tqdm import tqdm

    repo_id = "lerobot/aloha_mobile_cabinet"

    camera_key = "observation.images.cam_right_wrist"
    fps = 50

    delta_timestamps = {
        # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
        camera_key: [-1, -0.5, -0.20, 0],
        # loads 6 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
        "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],
        # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
        "action": [t / fps for t in range(64)],
    }

    dataset = StreamingLeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

    for _i, _frame in tqdm(enumerate(dataset)):
        pass

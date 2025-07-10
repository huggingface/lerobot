from pathlib import Path
from typing import Callable, Dict, Generator, Iterator

import datasets
import numpy as np
import torch
from datasets import load_dataset
from line_profiler import profile

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    check_version_compatibility,
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
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        video_backend: str | None = "torchcodec",
        streaming: bool = True,
        buffer_size: int = 1000,
        max_num_shards: int = 16,
        seed: int = 42,
        rng: np.random.Generator | None = None,
        shuffle: bool = True,
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
            shuffle (bool, optional): Whether to shuffle the dataset across exhaustions. Defaults to True.
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
        self.shuffle = shuffle

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

        self.hf_dataset = self.load_hf_dataset()
        self.num_shards = min(self.hf_dataset.num_shards, max_num_shards)

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
    def _infinite_generator_over_elements(rng: np.random.Generator, elements: list[int]) -> Iterator[int]:
        while True:
            yield rng.choice(elements)

    def load_hf_dataset(self) -> datasets.IterableDataset:
        dataset = load_dataset(self.repo_id, split="train", streaming=self.streaming)
        self.streaming_from_local = False

        # TODO(fracapuano): Add support for streaming from a local folder and not only from HF Hub
        return dataset

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # keep the same seed across exhaustions if shuffle is False, otherwise shuffle data across exhaustions
        rng = np.random.default_rng(self.seed) if not self.shuffle else self.rng

        buffer_indices_generator = self._iter_random_indices(rng, self.buffer_size)

        # This buffer is populated while iterating on the dataset's shards
        frames_buffer = []
        idx_to_iterable_dataset = {
            idx: self._make_iterable_dataset(self.hf_dataset.shard(self.num_shards, index=idx))
            for idx in range(self.num_shards)
        }

        try:
            while available_shards := list(idx_to_iterable_dataset.keys()):
                shard_key = next(self._infinite_generator_over_elements(rng, available_shards))
                dataset = idx_to_iterable_dataset[shard_key]  # selects which shard to iterate on
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
            # Remove exhausted shard
            del idx_to_iterable_dataset[shard_key]

        # Once shards are all exhausted, shuffle the buffer and yield the remaining frames
        rng.shuffle(frames_buffer)
        yield from frames_buffer

    def _make_iterable_dataset(self, dataset: datasets.IterableDataset) -> Iterator:
        return iter(dataset)

    @profile
    def make_frame(self, dataset_iterator: datasets.IterableDataset) -> Generator:
        """Makes a frame starting from a dataset iterator"""
        item = next(dataset_iterator)
        item = item_to_torch(item)

        # Get episode index from the item
        ep_idx = item["episode_index"]

        # Load video frames, when needed
        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"]
            query_timestamps = self._get_query_timestamps(current_ts, None)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        # Add task as a string
        task_idx = item["task_index"]
        item["task"] = self.meta.tasks.iloc[task_idx].name

        yield item

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
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
        for vid_key, query_ts in query_timestamps.items():
            root = self.meta.url_root if self.streaming and not self.streaming_from_local else self.root
            video_path = f"{root}/{self.meta.get_video_file_path(ep_idx, vid_key)}"
            frames = decode_video_frames_torchcodec(
                video_path, query_ts, self.tolerance_s, decoder_cache=self.video_decoder_cache
            )
            item[vid_key] = frames.squeeze(0)

        return item


# Example usage
if __name__ == "__main__":
    repo_id = "lerobot/aloha_mobile_cabinet"
    dataset = StreamingLeRobotDataset(repo_id)

    for i, frame in enumerate(dataset):
        print(frame)

        if i > 10:  # only stream first 10 frames
            break

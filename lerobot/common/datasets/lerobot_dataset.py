import os
from pathlib import Path

import datasets
import torch

from lerobot.common.datasets.utils import (
    load_episode_data_index,
    load_hf_dataset,
    load_info,
    load_previous_and_future_frames,
    load_stats,
    load_videos,
)
from lerobot.common.datasets.video_utils import VideoFrame, load_from_videos

DATA_DIR = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else None
CODEBASE_VERSION = "v1.3"


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        version: str | None = CODEBASE_VERSION,
        root: Path | None = DATA_DIR,
        split: str = "train",
        transform: callable = None,
        delta_timestamps: dict[list[float]] | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.version = version
        self.root = root
        self.split = split
        self.transform = transform
        self.delta_timestamps = delta_timestamps
        # load data from hub or locally when root is provided
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        self.hf_dataset = load_hf_dataset(repo_id, version, root, split)
        self.episode_data_index = load_episode_data_index(repo_id, version, root)
        self.stats = load_stats(repo_id, version, root)
        self.info = load_info(repo_id, version, root)
        if self.video:
            self.videos_dir = load_videos(repo_id, version, root)

    @property
    def fps(self) -> int:
        return self.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.
        Returns False if it only loads images from png files.
        """
        return self.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        return self.hf_dataset.features

    @property
    def image_keys(self) -> list[str]:
        image_keys = []
        for key, feats in self.hf_dataset.features.items():
            if isinstance(feats, datasets.Image):
                image_keys.append(key)
        return image_keys + self.video_frame_keys

    @property
    def video_frame_keys(self):
        video_frame_keys = []
        for key, feats in self.hf_dataset.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_samples(self) -> int:
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        return len(self.hf_dataset.unique("episode_index"))

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        if self.delta_timestamps is not None:
            item = load_previous_and_future_frames(
                item,
                self.hf_dataset,
                self.episode_data_index,
                self.delta_timestamps,
                self.tolerance_s,
            )

        if self.video:
            item = load_from_videos(
                item,
                self.video_frame_keys,
                self.videos_dir,
                self.tolerance_s,
            )

        if self.transform is not None:
            item = self.transform(item)

        return item

    @classmethod
    def from_preloaded(
        cls,
        repo_id: str,
        version: str | None = CODEBASE_VERSION,
        root: Path | None = None,
        split: str = "train",
        transform: callable = None,
        delta_timestamps: dict[list[float]] | None = None,
        # additional preloaded attributes
        hf_dataset=None,
        episode_data_index=None,
        stats=None,
        info=None,
        videos_dir=None,
    ):
        # create an empty object of type LeRobotDataset
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.version = version
        obj.root = root
        obj.split = split
        obj.transform = transform
        obj.delta_timestamps = delta_timestamps
        obj.hf_dataset = hf_dataset
        obj.episode_data_index = episode_data_index
        obj.stats = stats
        obj.info = info
        obj.videos_dir = videos_dir
        return obj

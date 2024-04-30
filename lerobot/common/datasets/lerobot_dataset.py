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
from lerobot.common.datasets.video_utils import load_from_videos


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        version: str | None = "v1.1",
        root: Path | None = None,
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
    def video(self) -> int:
        return self.info.get("video", False)

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
            if isinstance(feats, datasets.Value) and feats.id == "video_frame":
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_samples(self) -> int:
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        return len(self.hf_dataset.unique("episode_index"))

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
                tol=1 / self.fps - 1e-4,  # 1e-4 to account for possible numerical error
            )

        if self.video:
            item = load_from_videos(item, self.video_frame_keys, self.videos_dir)

        if self.transform is not None:
            item = self.transform(item)

        return item

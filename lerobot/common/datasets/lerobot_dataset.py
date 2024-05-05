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
        n_end_keyframes_dropped: int = 0,
    ):
        """
        Args:
            delta_timestamps: A dictionary mapping lists of relative times (Δt) to data keys. When a frame is
                sampled from the underlying dataset, we treat it as a "keyframe" and load multiple frames
                according to the list of Δt's. For example {"action": [-0.05, 0, 0.05]} indicates
                that we want to load the current keyframe's action, as well as one from 50 ms ago, and one
                50 ms into the future. The action key then contains a (3, action_dim) tensor (whereas without
                `delta_timestamps` there would just be a (action_dim,) tensor. When the Δt's demand that
                frames outside of an episode boundary are retrieved, a copy padding strategy is used. See
                `load_previous_and_future_frames` for more details.
            n_end_keyframes_dropped: Don't sample the last n items in each episode. This option is handy when
                used in combination with `delta_timestamps` when, for example, the Δt's demand multiple future
                frames, but we want to avoid introducing too much copy padding into the data distribution.
                For example if `delta_timestamps = {"action": [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]}`
                and we sample the last frame in the episode, we would end up padding with 6 frames worth of
                copies. Instead, we might want no padding (in which case we need n=6), or we might be okay
                with up to 2 frames of padding (in which case we need n=4).
        """
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
        # If `n_end_keyframes_dropped == 0`, `self.index` contains exactly the indices of the hf_dataset. If
        # `n_end_keyframes_dropped > 0`, `self.index` contains a subset of the indices of the hf_dataset where
        # we drop those indices pertaining to the last n frames of each episode.
        self.index = []
        for from_ix, to_ix in zip(*self.episode_data_index.values(), strict=True):
            self.index.extend(list(range(from_ix, to_ix - n_end_keyframes_dropped)))

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
        return len(self.index)

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
        item = self.hf_dataset[self.index[idx]]

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

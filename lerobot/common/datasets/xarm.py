from pathlib import Path

import torch

from lerobot.common.datasets.utils import (
    load_episode_data_index,
    load_hf_dataset,
    load_previous_and_future_frames,
    load_stats,
)


class XarmDataset(torch.utils.data.Dataset):
    """
    https://huggingface.co/datasets/lerobot/xarm_lift_medium
    https://huggingface.co/datasets/lerobot/xarm_lift_medium_replay
    https://huggingface.co/datasets/lerobot/xarm_push_medium
    https://huggingface.co/datasets/lerobot/xarm_push_medium_replay
    """

    # Copied from lerobot/__init__.py
    available_datasets = [
        "xarm_lift_medium",
        "xarm_lift_medium_replay",
        "xarm_push_medium",
        "xarm_push_medium_replay",
    ]
    fps = 15
    image_keys = ["observation.image"]

    def __init__(
        self,
        dataset_id: str,
        version: str | None = "v1.1",
        root: Path | None = None,
        split: str = "train",
        transform: callable = None,
        delta_timestamps: dict[list[float]] | None = None,
    ):
        super().__init__()
        self.dataset_id = dataset_id
        self.version = version
        self.root = root
        self.split = split
        self.transform = transform
        self.delta_timestamps = delta_timestamps
        # load data from hub or locally when root is provided
        self.hf_dataset = load_hf_dataset(dataset_id, version, root, split)
        self.episode_data_index = load_episode_data_index(dataset_id, version, root)
        self.stats = load_stats(dataset_id, version, root)

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

        if self.transform is not None:
            item = self.transform(item)

        return item

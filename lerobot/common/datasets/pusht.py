from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk

from lerobot.common.datasets.utils import load_previous_and_future_frames


class PushtDataset(torch.utils.data.Dataset):
    """
    https://huggingface.co/datasets/lerobot/pusht

    Arguments
    ----------
    delta_timestamps : dict[list[float]] | None, optional
        Loads data from frames with a shift in timestamps with a different strategy for each data key (e.g. state, action or image)
        If `None`, no shift is applied to current timestamp and the data from the current frame is loaded.
    """

    # Copied from lerobot/__init__.py
    available_datasets = ["pusht"]
    fps = 10
    image_keys = ["observation.image"]

    def __init__(
        self,
        dataset_id: str = "pusht",
        version: str | None = "v1.0",
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
        if self.root is not None:
            self.hf_dataset = load_from_disk(Path(self.root) / self.dataset_id / self.split)
        else:
            self.hf_dataset = load_dataset(
                f"lerobot/{self.dataset_id}", revision=self.version, split=self.split
            )
        self.hf_dataset = self.hf_dataset.with_format("torch")

    @property
    def num_samples(self) -> int:
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        return len(self.hf_dataset.unique("episode_id"))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        if self.delta_timestamps is not None:
            item = load_previous_and_future_frames(
                item,
                self.hf_dataset,
                self.delta_timestamps,
            )

        # convert images from channel last (PIL) to channel first (pytorch)
        for key in self.image_keys:
            if item[key].ndim == 3:
                item[key] = item[key].permute((2, 0, 1))  # h w c -> c h w
            elif item[key].ndim == 4:
                item[key] = item[key].permute((0, 3, 1, 2))  # t h w c -> t c h w
            else:
                raise ValueError(item[key].ndim)

        if self.transform is not None:
            item = self.transform(item)

        return item

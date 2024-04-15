import torch
from datasets import load_dataset

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

    available_datasets = ["pusht"]
    fps = 10
    image_keys = ["observation.image"]

    def __init__(
        self,
        dataset_id: str,
        version: str | None = "v1.0",
        transform: callable = None,
        delta_timestamps: dict[list[float]] | None = None,
    ):
        super().__init__()
        self.dataset_id = dataset_id
        self.version = version
        self.transform = transform
        self.delta_timestamps = delta_timestamps
        # self.data_dict = load_dataset(f"lerobot/{self.dataset_id}", revision=self.version, split="train")
        self.data_dict = load_dataset(f"lerobot/{self.dataset_id}", split="train")
        self.data_dict = self.data_dict.with_format("torch")
        self.data_dict.push_to_hub(f"lerobot/{dataset_id}", token=True, revision="v1.0")

    @property
    def num_samples(self) -> int:
        return len(self.data_dict)

    @property
    def num_episodes(self) -> int:
        return len(self.data_dict.unique("episode_id"))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.data_dict[idx]

        if self.delta_timestamps is not None:
            item = load_previous_and_future_frames(
                item,
                self.data_dict,
                self.delta_timestamps,
            )

        if self.transform is not None:
            item = self.transform(item)

        return item

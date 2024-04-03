import logging
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Callable

import einops
import torch
import torchrl
import tqdm
from huggingface_hub import snapshot_download
from tensordict import TensorDict
from torchrl.data.replay_buffers.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import Sampler, SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import TensorStorage, _collate_id
from torchrl.data.replay_buffers.writers import ImmutableDatasetWriter, Writer
from torchrl.envs.transforms.transforms import Compose

HF_USER = "lerobot"


class AbstractDataset(TensorDictReplayBuffer):
    """
    AbstractDataset represents a dataset in the context of imitation learning or reinforcement learning.
    This class is designed to be subclassed by concrete implementations that specify particular types of datasets.
    These implementations can vary based on the source of the data, the environment the data pertains to,
    or the specific kind of data manipulation applied.

    Note:
        - `TensorDictReplayBuffer` is the base class from which `AbstractDataset` inherits. It provides the foundational
           functionality for storing and retrieving `TensorDict`-like data.
        - `available_datasets` should be overridden by concrete subclasses to list the specific dataset variants supported.
           It is expected that these variants correspond to a HuggingFace dataset on the hub.
           For instance, the `AlohaDataset` which inherites from `AbstractDataset` has 4 available dataset variants:
            - [aloha_sim_transfer_cube_scripted](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_scripted)
            - [aloha_sim_insertion_scripted](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_scripted)
            - [aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human)
            - [aloha_sim_insertion_human](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human)
        - When implementing a concrete class (e.g. `AlohaDataset`, `PushtEnv`, `DiffusionPolicy`), you need to:
            1. set the required class attributes:
                - for classes inheriting from `AbstractDataset`: `available_datasets`
                - for classes inheriting from `AbstractEnv`: `name`, `available_tasks`
                - for classes inheriting from `AbstractPolicy`: `name`
            2. update variables in `lerobot/__init__.py` (e.g. `available_envs`, `available_datasets_per_envs`, `available_policies`)
            3. update variables in `tests/test_available.py` by importing your new class
    """

    available_datasets: list[str] | None = None

    def __init__(
        self,
        dataset_id: str,
        version: str | None = None,
        batch_size: int | None = None,
        *,
        shuffle: bool = True,
        root: Path | None = None,
        pin_memory: bool = False,
        prefetch: int = None,
        sampler: Sampler | None = None,
        collate_fn: Callable | None = None,
        writer: Writer | None = None,
        transform: "torchrl.envs.Transform" = None,
    ):
        assert (
            self.available_datasets is not None
        ), "Subclasses of `AbstractDataset` should set the `available_datasets` class attribute."
        assert (
            dataset_id in self.available_datasets
        ), f"The provided dataset ({dataset_id}) is not on the list of available datasets {self.available_datasets}."

        self.dataset_id = dataset_id
        self.version = version
        self.shuffle = shuffle
        self.root = root if root is None else Path(root)

        if self.root is not None and self.version is not None:
            logging.warning(
                f"The version of the dataset ({self.version}) is not enforced when root is provided ({self.root})."
            )

        storage = self._download_or_load_dataset()

        super().__init__(
            storage=storage,
            sampler=sampler,
            writer=ImmutableDatasetWriter() if writer is None else writer,
            collate_fn=_collate_id if collate_fn is None else collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            batch_size=batch_size,
            transform=transform,
        )

    @property
    def stats_patterns(self) -> dict:
        return {
            ("observation", "state"): "b c -> c",
            ("observation", "image"): "b c h w -> c 1 1",
            ("action",): "b c -> c",
        }

    @property
    def image_keys(self) -> list:
        return [("observation", "image")]

    @property
    def num_cameras(self) -> int:
        return len(self.image_keys)

    @property
    def num_samples(self) -> int:
        return len(self)

    @property
    def num_episodes(self) -> int:
        return len(self._storage._storage["episode"].unique())

    @property
    def transform(self):
        return self._transform

    def set_transform(self, transform):
        if not isinstance(transform, Compose):
            # required since torchrl calls `len(self._transform)` downstream
            if isinstance(transform, list):
                self._transform = Compose(*transform)
            else:
                self._transform = Compose(transform)
        else:
            self._transform = transform

    def compute_or_load_stats(self, batch_size: int = 32) -> TensorDict:
        stats_path = self.data_dir / "stats.pth"
        if stats_path.exists():
            stats = torch.load(stats_path)
        else:
            logging.info(f"compute_stats and save to {stats_path}")
            stats = self._compute_stats(batch_size)
            torch.save(stats, stats_path)
        return stats

    def _download_or_load_dataset(self) -> torch.StorageBase:
        if self.root is None:
            self.data_dir = Path(
                snapshot_download(
                    repo_id=f"{HF_USER}/{self.dataset_id}", repo_type="dataset", revision=self.version
                )
            )
        else:
            self.data_dir = self.root / self.dataset_id
        return TensorStorage(TensorDict.load_memmap(self.data_dir / "replay_buffer"))

    def _compute_stats(self, batch_size: int = 32):
        """Compute dataset statistics including minimum, maximum, mean, and standard deviation."""
        rb = TensorDictReplayBuffer(
            storage=self._storage,
            batch_size=32,
            prefetch=True,
            # Note: Due to be refactored soon. The point is that we should go through the whole dataset.
            sampler=SamplerWithoutReplacement(drop_last=False, shuffle=False),
        )

        # mean and std will be computed incrementally while max and min will track the running value.
        mean, std, max, min = {}, {}, {}, {}
        for key in self.stats_patterns:
            mean[key] = torch.tensor(0.0).float()
            std[key] = torch.tensor(0.0).float()
            max[key] = torch.tensor(-float("inf")).float()
            min[key] = torch.tensor(float("inf")).float()

        # Compute mean, min, max.
        # Note: Due to be refactored soon. The point of storing `first_batch` is to make sure we don't get
        # surprises when rerunning the sampler.
        first_batch = None
        running_item_count = 0  # for online mean computation
        for _ in tqdm.tqdm(range(ceil(len(rb) / batch_size))):
            batch = rb.sample()
            this_batch_size = batch.batch_size[0]
            running_item_count += this_batch_size
            if first_batch is None:
                first_batch = deepcopy(batch)
            for key, pattern in self.stats_patterns.items():
                batch[key] = batch[key].float()
                # Numerically stable update step for mean computation.
                batch_mean = einops.reduce(batch[key], pattern, "mean")
                # Hint: to update the mean we need x̄ₙ = (Nₙ₋₁x̄ₙ₋₁ + Bₙxₙ) / Nₙ, where the subscript represents
                # the update step, N is the running item count, B is this batch size, x̄ is the running mean,
                # and x is the current batch mean. Some rearrangement is then required to avoid risking
                # numerical overflow. Another hint: Nₙ₋₁ = Nₙ - Bₙ.
                mean[key] = mean[key] + this_batch_size * (batch_mean - mean[key]) / running_item_count
                max[key] = torch.maximum(max[key], einops.reduce(batch[key], pattern, "max"))
                min[key] = torch.minimum(min[key], einops.reduce(batch[key], pattern, "min"))

        # Compute std.
        first_batch_ = None
        running_item_count = 0  # for online std computation
        for _ in tqdm.tqdm(range(ceil(len(rb) / batch_size))):
            batch = rb.sample()
            this_batch_size = batch.batch_size[0]
            running_item_count += this_batch_size
            # Sanity check to make sure the batches are still in the same order as before.
            if first_batch_ is None:
                first_batch_ = deepcopy(batch)
                for key in self.stats_patterns:
                    assert torch.equal(first_batch_[key], first_batch[key])
            for key, pattern in self.stats_patterns.items():
                batch[key] = batch[key].float()
                # Numerically stable update step for mean computation (where the mean is over squared
                # residuals).See notes in the mean computation loop above.
                batch_std = einops.reduce((batch[key] - mean[key]) ** 2, pattern, "mean")
                std[key] = std[key] + this_batch_size * (batch_std - std[key]) / running_item_count

        for key in self.stats_patterns:
            std[key] = torch.sqrt(std[key])

        stats = TensorDict({}, batch_size=[])
        for key in self.stats_patterns:
            stats[(*key, "mean")] = mean[key]
            stats[(*key, "std")] = std[key]
            stats[(*key, "max")] = max[key]
            stats[(*key, "min")] = min[key]

            if key[0] == "observation":
                # use same stats for the next observations
                stats[("next", *key)] = stats[key]
        return stats

import abc
import logging
from pathlib import Path
from typing import Callable

import einops
import torch
import torchrl
import tqdm
from tensordict import TensorDict
from torchrl.data.datasets.utils import _get_root_dir
from torchrl.data.replay_buffers.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.data.replay_buffers.storages import TensorStorage, _collate_id
from torchrl.data.replay_buffers.writers import ImmutableDatasetWriter, Writer
from torchrl.envs.transforms.transforms import Compose


class AbstractExperienceReplay(TensorDictReplayBuffer):
    def __init__(
        self,
        dataset_id: str,
        batch_size: int = None,
        *,
        shuffle: bool = True,
        root: Path = None,
        pin_memory: bool = False,
        prefetch: int = None,
        sampler: SliceSampler = None,
        collate_fn: Callable = None,
        writer: Writer = None,
        transform: "torchrl.envs.Transform" = None,
    ):
        self.dataset_id = dataset_id
        self.shuffle = shuffle
        self.root = _get_root_dir(self.dataset_id) if root is None else root
        self.root = Path(self.root)
        self.data_dir = self.root / self.dataset_id

        storage = self._download_or_load_storage()

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
            ("observation", "state"): "b c -> 1 c",
            ("observation", "image"): "b c h w -> 1 c 1 1",
            ("action",): "b c -> 1 c",
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
            self._transform = Compose(transform)
        else:
            self._transform = transform

    def compute_or_load_stats(self, num_batch=100, batch_size=32) -> TensorDict:
        stats_path = self.data_dir / "stats.pth"
        if stats_path.exists():
            stats = torch.load(stats_path)
        else:
            logging.info(f"compute_stats and save to {stats_path}")
            stats = self._compute_stats(num_batch, batch_size)
            torch.save(stats, stats_path)
        return stats

    @abc.abstractmethod
    def _download_and_preproc(self) -> torch.StorageBase:
        raise NotImplementedError()

    def _download_or_load_storage(self):
        if not self._is_downloaded():
            storage = self._download_and_preproc()
        else:
            storage = TensorStorage(TensorDict.load_memmap(self.data_dir))
        return storage

    def _is_downloaded(self) -> bool:
        return self.data_dir.is_dir()

    def _compute_stats(self, num_batch=100, batch_size=32):
        rb = TensorDictReplayBuffer(
            storage=self._storage,
            batch_size=batch_size,
            prefetch=True,
        )

        mean, std, max, min = {}, {}, {}, {}

        # compute mean, min, max
        for _ in tqdm.tqdm(range(num_batch)):
            batch = rb.sample()
            for key, pattern in self.stats_patterns.items():
                batch[key] = batch[key].float()
                if key not in mean:
                    # first batch initialize mean, min, max
                    mean[key] = einops.reduce(batch[key], pattern, "mean")
                    max[key] = einops.reduce(batch[key], pattern, "max")
                    min[key] = einops.reduce(batch[key], pattern, "min")
                else:
                    mean[key] += einops.reduce(batch[key], pattern, "mean")
                    max[key] = torch.maximum(max[key], einops.reduce(batch[key], pattern, "max"))
                    min[key] = torch.minimum(min[key], einops.reduce(batch[key], pattern, "min"))
                batch = rb.sample()

        for key in self.stats_patterns:
            mean[key] /= num_batch

        # compute std, min, max
        for _ in tqdm.tqdm(range(num_batch)):
            batch = rb.sample()
            for key, pattern in self.stats_patterns.items():
                batch[key] = batch[key].float()
                batch_mean = einops.reduce(batch[key], pattern, "mean")
                if key not in std:
                    # first batch initialize std
                    std[key] = (batch_mean - mean[key]) ** 2
                else:
                    std[key] += (batch_mean - mean[key]) ** 2
                max[key] = torch.maximum(max[key], einops.reduce(batch[key], pattern, "max"))
                min[key] = torch.minimum(min[key], einops.reduce(batch[key], pattern, "min"))

        for key in self.stats_patterns:
            std[key] = torch.sqrt(std[key] / num_batch)

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

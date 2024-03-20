import logging
from pathlib import Path
from typing import Callable

import einops
import torch
import torchrl
import tqdm
from huggingface_hub import snapshot_download
from tensordict import TensorDict
from torchrl.data.replay_buffers.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.data.replay_buffers.storages import TensorStorage, _collate_id
from torchrl.data.replay_buffers.writers import ImmutableDatasetWriter, Writer
from torchrl.envs.transforms.transforms import Compose


class AbstractExperienceReplay(TensorDictReplayBuffer):
    def __init__(
        self,
        dataset_id: str,
        version: str | None = None,
        batch_size: int = None,
        *,
        shuffle: bool = True,
        root: Path | None = None,
        pin_memory: bool = False,
        prefetch: int = None,
        sampler: SliceSampler = None,
        collate_fn: Callable = None,
        writer: Writer = None,
        transform: "torchrl.envs.Transform" = None,
    ):
        self.dataset_id = dataset_id
        self.version = version
        self.shuffle = shuffle
        self.root = root

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

    def compute_or_load_stats(self, num_batch=100, batch_size=32) -> TensorDict:
        stats_path = self.data_dir / "stats.pth"
        if stats_path.exists():
            stats = torch.load(stats_path)
        else:
            logging.info(f"compute_stats and save to {stats_path}")
            stats = self._compute_stats(num_batch, batch_size)
            torch.save(stats, stats_path)
        return stats

    def _download_or_load_dataset(self) -> torch.StorageBase:
        if self.root is None:
            self.data_dir = Path(
                snapshot_download(
                    repo_id=f"cadene/{self.dataset_id}", repo_type="dataset", revision=self.version
                )
            )
        else:
            self.data_dir = self.root / self.dataset_id
        return TensorStorage(TensorDict.load_memmap(self.data_dir / "replay_buffer"))

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

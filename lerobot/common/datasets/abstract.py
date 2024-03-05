import abc
import logging
import math
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
        transform: "torchrl.envs.Transform" = None,  # noqa-F821
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
    def num_samples(self) -> int:
        return len(self)

    @property
    def num_episodes(self) -> int:
        return len(self._storage._storage["episode"].unique())

    def set_transform(self, transform):
        self.transform = transform

    def compute_or_load_stats(self, num_batch=100, batch_size=32) -> TensorDict:
        stats_path = self.data_dir / "stats.pth"
        if stats_path.exists():
            stats = torch.load(stats_path)
        else:
            logging.info(f"compute_stats and save to {stats_path}")
            stats = self._compute_stats(self._storage._storage, num_batch, batch_size)
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

    def _compute_stats(self, storage, num_batch=100, batch_size=32):
        rb = TensorDictReplayBuffer(
            storage=storage,
            batch_size=batch_size,
            prefetch=True,
        )
        batch = rb.sample()

        image_channels = batch["observation", "image"].shape[1]
        image_mean = torch.zeros(image_channels)
        image_std = torch.zeros(image_channels)
        image_max = torch.tensor([-math.inf] * image_channels)
        image_min = torch.tensor([math.inf] * image_channels)

        state_channels = batch["observation", "state"].shape[1]
        state_mean = torch.zeros(state_channels)
        state_std = torch.zeros(state_channels)
        state_max = torch.tensor([-math.inf] * state_channels)
        state_min = torch.tensor([math.inf] * state_channels)

        action_channels = batch["action"].shape[1]
        action_mean = torch.zeros(action_channels)
        action_std = torch.zeros(action_channels)
        action_max = torch.tensor([-math.inf] * action_channels)
        action_min = torch.tensor([math.inf] * action_channels)

        for _ in tqdm.tqdm(range(num_batch)):
            image_mean += einops.reduce(batch["observation", "image"], "b c h w -> c", "mean")
            state_mean += einops.reduce(batch["observation", "state"], "b c -> c", "mean")
            action_mean += einops.reduce(batch["action"], "b c -> c", "mean")

            b_image_max = einops.reduce(batch["observation", "image"], "b c h w -> c", "max")
            b_image_min = einops.reduce(batch["observation", "image"], "b c h w -> c", "min")
            b_state_max = einops.reduce(batch["observation", "state"], "b c -> c", "max")
            b_state_min = einops.reduce(batch["observation", "state"], "b c -> c", "min")
            b_action_max = einops.reduce(batch["action"], "b c -> c", "max")
            b_action_min = einops.reduce(batch["action"], "b c -> c", "min")
            image_max = torch.maximum(image_max, b_image_max)
            image_min = torch.maximum(image_min, b_image_min)
            state_max = torch.maximum(state_max, b_state_max)
            state_min = torch.maximum(state_min, b_state_min)
            action_max = torch.maximum(action_max, b_action_max)
            action_min = torch.maximum(action_min, b_action_min)

            batch = rb.sample()

        image_mean /= num_batch
        state_mean /= num_batch
        action_mean /= num_batch

        for i in tqdm.tqdm(range(num_batch)):
            b_image_mean = einops.reduce(batch["observation", "image"], "b c h w -> c", "mean")
            b_state_mean = einops.reduce(batch["observation", "state"], "b c -> c", "mean")
            b_action_mean = einops.reduce(batch["action"], "b c -> c", "mean")
            image_std += (b_image_mean - image_mean) ** 2
            state_std += (b_state_mean - state_mean) ** 2
            action_std += (b_action_mean - action_mean) ** 2

            b_image_max = einops.reduce(batch["observation", "image"], "b c h w -> c", "max")
            b_image_min = einops.reduce(batch["observation", "image"], "b c h w -> c", "min")
            b_state_max = einops.reduce(batch["observation", "state"], "b c -> c", "max")
            b_state_min = einops.reduce(batch["observation", "state"], "b c -> c", "min")
            b_action_max = einops.reduce(batch["action"], "b c -> c", "max")
            b_action_min = einops.reduce(batch["action"], "b c -> c", "min")
            image_max = torch.maximum(image_max, b_image_max)
            image_min = torch.maximum(image_min, b_image_min)
            state_max = torch.maximum(state_max, b_state_max)
            state_min = torch.maximum(state_min, b_state_min)
            action_max = torch.maximum(action_max, b_action_max)
            action_min = torch.maximum(action_min, b_action_min)

            if i < num_batch - 1:
                batch = rb.sample()

        image_std = torch.sqrt(image_std / num_batch)
        state_std = torch.sqrt(state_std / num_batch)
        action_std = torch.sqrt(action_std / num_batch)

        stats = TensorDict(
            {
                ("observation", "image", "mean"): image_mean[None, :, None, None],
                ("observation", "image", "std"): image_std[None, :, None, None],
                ("observation", "image", "max"): image_max[None, :, None, None],
                ("observation", "image", "min"): image_min[None, :, None, None],
                ("observation", "state", "mean"): state_mean[None, :],
                ("observation", "state", "std"): state_std[None, :],
                ("observation", "state", "max"): state_max[None, :],
                ("observation", "state", "min"): state_min[None, :],
                ("action", "mean"): action_mean[None, :],
                ("action", "std"): action_std[None, :],
                ("action", "max"): action_max[None, :],
                ("action", "min"): action_min[None, :],
            },
            batch_size=[],
        )
        stats["next", "observation", "image"] = stats["observation", "image"]
        stats["next", "observation", "state"] = stats["observation", "state"]
        return stats

import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch
import torchrl
import tqdm
from tensordict import TensorDict
from torchrl.data.datasets.utils import _get_root_dir
from torchrl.data.replay_buffers.replay_buffers import (
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.samplers import (
    Sampler,
    SliceSampler,
    SliceSamplerWithoutReplacement,
)
from torchrl.data.replay_buffers.storages import TensorStorage, _collate_id
from torchrl.data.replay_buffers.writers import ImmutableDatasetWriter, Writer


class PushtExperienceReplay(TensorDictReplayBuffer):

    available_datasets = [
        "xarm_lift_medium",
    ]

    def __init__(
        self,
        dataset_id,
        batch_size: int = None,
        *,
        shuffle: bool = True,
        num_slices: int = None,
        slice_len: int = None,
        pad: float = None,
        replacement: bool = None,
        streaming: bool = False,
        root: Path = None,
        download: bool = False,
        sampler: Sampler = None,
        writer: Writer = None,
        collate_fn: Callable = None,
        pin_memory: bool = False,
        prefetch: int = None,
        transform: "torchrl.envs.Transform" = None,  # noqa-F821
        split_trajs: bool = False,
        strict_length: bool = True,
    ):
        # TODO
        raise NotImplementedError()
        self.download = download
        if streaming:
            raise NotImplementedError
        self.streaming = streaming
        self.dataset_id = dataset_id
        self.split_trajs = split_trajs
        self.shuffle = shuffle
        self.num_slices = num_slices
        self.slice_len = slice_len
        self.pad = pad

        self.strict_length = strict_length
        if (self.num_slices is not None) and (self.slice_len is not None):
            raise ValueError("num_slices or slice_len can be not None, but not both.")
        if split_trajs:
            raise NotImplementedError

        if root is None:
            root = _get_root_dir("simxarm")
            os.makedirs(root, exist_ok=True)
        self.root = Path(root)
        if self.download == "force" or (self.download and not self._is_downloaded()):
            storage = self._download_and_preproc()
        else:
            storage = TensorStorage(TensorDict.load_memmap(self.root / dataset_id))

        if num_slices is not None or slice_len is not None:
            if sampler is not None:
                raise ValueError(
                    "`num_slices` and `slice_len` are exclusive with the `sampler` argument."
                )

            if replacement:
                if not self.shuffle:
                    raise RuntimeError(
                        "shuffle=False can only be used when replacement=False."
                    )
                sampler = SliceSampler(
                    num_slices=num_slices,
                    slice_len=slice_len,
                    strict_length=strict_length,
                )
            else:
                sampler = SliceSamplerWithoutReplacement(
                    num_slices=num_slices,
                    slice_len=slice_len,
                    strict_length=strict_length,
                    shuffle=self.shuffle,
                )

        if writer is None:
            writer = ImmutableDatasetWriter()
        if collate_fn is None:
            collate_fn = _collate_id

        super().__init__(
            storage=storage,
            sampler=sampler,
            writer=writer,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            batch_size=batch_size,
            transform=transform,
        )

    @property
    def data_path_root(self):
        if self.streaming:
            return None
        return self.root / self.dataset_id

    def _is_downloaded(self):
        return os.path.exists(self.data_path_root)

    def _download_and_preproc(self):
        # download
        # TODO(rcadene)

        # load
        dataset_dir = Path("data") / self.dataset_id
        dataset_path = dataset_dir / f"buffer.pkl"
        print(f"Using offline dataset '{dataset_path}'")
        with open(dataset_path, "rb") as f:
            dataset_dict = pickle.load(f)

        total_frames = dataset_dict["actions"].shape[0]

        idx0 = 0
        idx1 = 0
        episode_id = 0
        for i in tqdm.tqdm(range(total_frames)):
            idx1 += 1

            if not dataset_dict["dones"][i]:
                continue

            num_frames = idx1 - idx0

            image = torch.tensor(dataset_dict["observations"]["rgb"][idx0:idx1])
            state = torch.tensor(dataset_dict["observations"]["state"][idx0:idx1])
            next_image = torch.tensor(
                dataset_dict["next_observations"]["rgb"][idx0:idx1]
            )
            next_state = torch.tensor(
                dataset_dict["next_observations"]["state"][idx0:idx1]
            )
            next_reward = torch.tensor(dataset_dict["rewards"][idx0:idx1])
            next_done = torch.tensor(dataset_dict["dones"][idx0:idx1])

            episode = TensorDict(
                {
                    ("observation", "image"): image,
                    ("observation", "state"): state,
                    "action": torch.tensor(dataset_dict["actions"][idx0:idx1]),
                    "episode": torch.tensor([episode_id] * num_frames, dtype=torch.int),
                    "frame_id": torch.arange(0, num_frames, 1),
                    ("next", "observation", "image"): next_image,
                    ("next", "observation", "state"): next_state,
                    ("next", "observation", "reward"): next_reward,
                    ("next", "observation", "done"): next_done,
                },
                batch_size=num_frames,
            )

            if episode_id == 0:
                # hack to initialize tensordict data structure to store episodes
                td_data = (
                    episode[0]
                    .expand(total_frames)
                    .memmap_like(self.root / self.dataset_id)
                )

            td_data[idx0:idx1] = episode

            episode_id += 1
            idx0 = idx1

        return TensorStorage(td_data.lock_())

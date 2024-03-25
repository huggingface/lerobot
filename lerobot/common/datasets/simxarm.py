import pickle
import zipfile
from pathlib import Path
from typing import Callable

import torch
import torchrl
import tqdm
from tensordict import TensorDict
from torchrl.data.replay_buffers.samplers import (
    SliceSampler,
)
from torchrl.data.replay_buffers.storages import TensorStorage
from torchrl.data.replay_buffers.writers import Writer

from lerobot.common.datasets.abstract import AbstractExperienceReplay


def download():
    raise NotImplementedError()
    import gdown

    url = "https://drive.google.com/uc?id=1nhxpykGtPDhmQKm-_B8zBSywVRdgeVya"
    download_path = "data.zip"
    gdown.download(url, download_path, quiet=False)
    print("Extracting...")
    with zipfile.ZipFile(download_path, "r") as zip_f:
        for member in zip_f.namelist():
            if member.startswith("data/xarm") and member.endswith(".pkl"):
                print(member)
                zip_f.extract(member=member)
    Path(download_path).unlink()


class SimxarmExperienceReplay(AbstractExperienceReplay):
    available_datasets = [
        "xarm_lift_medium",
    ]

    def __init__(
        self,
        dataset_id: str,
        version: str | None = "v1.1",
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
        super().__init__(
            dataset_id,
            version,
            batch_size,
            shuffle=shuffle,
            root=root,
            pin_memory=pin_memory,
            prefetch=prefetch,
            sampler=sampler,
            collate_fn=collate_fn,
            writer=writer,
            transform=transform,
        )

    def _download_and_preproc_obsolete(self):
        # assert self.root is not None
        # TODO(rcadene): finish download
        # download()

        dataset_path = self.root / f"{self.dataset_id}" / "buffer.pkl"
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
            next_image = torch.tensor(dataset_dict["next_observations"]["rgb"][idx0:idx1])
            next_state = torch.tensor(dataset_dict["next_observations"]["state"][idx0:idx1])
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
                    ("next", "reward"): next_reward,
                    ("next", "done"): next_done,
                },
                batch_size=num_frames,
            )

            if episode_id == 0:
                # hack to initialize tensordict data structure to store episodes
                td_data = (
                    episode[0]
                    .expand(total_frames)
                    .memmap_like(self.root / f"{self.dataset_id}" / "replay_buffer")
                )

            td_data[idx0:idx1] = episode

            episode_id += 1
            idx0 = idx1

        return TensorStorage(td_data.lock_())

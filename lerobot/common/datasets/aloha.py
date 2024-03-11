import logging
from pathlib import Path
from typing import Callable

import einops
import gdown
import h5py
import torch
import torchrl
import tqdm
from tensordict import TensorDict
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.data.replay_buffers.storages import TensorStorage
from torchrl.data.replay_buffers.writers import Writer

from lerobot.common.datasets.abstract import AbstractExperienceReplay

DATASET_IDS = [
    "aloha_sim_insertion_human",
    "aloha_sim_insertion_scripted",
    "aloha_sim_transfer_cube_human",
    "aloha_sim_transfer_cube_scripted",
]

FOLDER_URLS = {
    "aloha_sim_insertion_human": "https://drive.google.com/drive/folders/1RgyD0JgTX30H4IM5XZn8I3zSV_mr8pyF",
    "aloha_sim_insertion_scripted": "https://drive.google.com/drive/folders/1TsojQQSXtHEoGnqgJ3gmpPQR2DPLtS2N",
    "aloha_sim_transfer_cube_human": "https://drive.google.com/drive/folders/1sc-E4QYW7A0o23m1u2VWNGVq5smAsfCo",
    "aloha_sim_transfer_cube_scripted": "https://drive.google.com/drive/folders/1aRyoOhQwxhyt1J8XgEig4s6kzaw__LXj",
}

EP48_URLS = {
    "aloha_sim_insertion_human": "https://drive.google.com/file/d/18Cudl6nikDtgRolea7je8iF_gGKzynOP/view?usp=drive_link",
    "aloha_sim_insertion_scripted": "https://drive.google.com/file/d/1wfMSZ24oOh5KR_0aaP3Cnu_c4ZCveduB/view?usp=drive_link",
    "aloha_sim_transfer_cube_human": "https://drive.google.com/file/d/18smMymtr8tIxaNUQ61gW6dG50pt3MvGq/view?usp=drive_link",
    "aloha_sim_transfer_cube_scripted": "https://drive.google.com/file/d/1pnGIOd-E4-rhz2P3VxpknMKRZCoKt6eI/view?usp=drive_link",
}

EP49_URLS = {
    "aloha_sim_insertion_human": "https://drive.google.com/file/d/1C1kZYyROzs-PrLc0SkDgUgMi4-L3lauE/view?usp=drive_link",
    "aloha_sim_insertion_scripted": "https://drive.google.com/file/d/17EuCUWS6uCCr6yyNzpXdcdE-_TTNCKtf/view?usp=drive_link",
    "aloha_sim_transfer_cube_human": "https://drive.google.com/file/d/1Nk7l53d9sJoGDBKAOnNrExX5nLacATc6/view?usp=drive_link",
    "aloha_sim_transfer_cube_scripted": "https://drive.google.com/file/d/1GKReZHrXU73NMiC5zKCq_UtqPVtYq8eo/view?usp=drive_link",
}

NUM_EPISODES = {
    "aloha_sim_insertion_human": 50,
    "aloha_sim_insertion_scripted": 50,
    "aloha_sim_transfer_cube_human": 50,
    "aloha_sim_transfer_cube_scripted": 50,
}

EPISODE_LEN = {
    "aloha_sim_insertion_human": 500,
    "aloha_sim_insertion_scripted": 400,
    "aloha_sim_transfer_cube_human": 400,
    "aloha_sim_transfer_cube_scripted": 400,
}

CAMERAS = {
    "aloha_sim_insertion_human": ["top"],
    "aloha_sim_insertion_scripted": ["top"],
    "aloha_sim_transfer_cube_human": ["top"],
    "aloha_sim_transfer_cube_scripted": ["top"],
}


def download(data_dir, dataset_id):
    assert dataset_id in DATASET_IDS
    assert dataset_id in FOLDER_URLS
    assert dataset_id in EP48_URLS
    assert dataset_id in EP49_URLS

    data_dir.mkdir(parents=True, exist_ok=True)

    gdown.download_folder(FOLDER_URLS[dataset_id], output=data_dir)

    # because of the 50 files limit per directory, two files episode 48 and 49 were missing
    gdown.download(EP48_URLS[dataset_id], output=data_dir / "episode_48.hdf5", fuzzy=True)
    gdown.download(EP49_URLS[dataset_id], output=data_dir / "episode_49.hdf5", fuzzy=True)


class AlohaExperienceReplay(AbstractExperienceReplay):
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
        assert dataset_id in DATASET_IDS

        super().__init__(
            dataset_id,
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

    @property
    def stats_patterns(self) -> dict:
        d = {
            ("observation", "state"): "b c -> 1 c",
            ("action",): "b c -> 1 c",
        }
        for cam in CAMERAS[self.dataset_id]:
            d[("observation", "image", cam)] = "b c h w -> 1 c 1 1"
        return d

    @property
    def image_keys(self) -> list:
        return [("observation", "image", cam) for cam in CAMERAS[self.dataset_id]]

    # def _is_downloaded(self) -> bool:
    #     return False

    def _download_and_preproc(self):
        raw_dir = self.data_dir.parent / f"{self.data_dir.name}_raw"
        if not raw_dir.is_dir():
            download(raw_dir, self.dataset_id)

        total_num_frames = 0
        logging.info("Compute total number of frames to initialize offline buffer")
        for ep_id in range(NUM_EPISODES[self.dataset_id]):
            ep_path = raw_dir / f"episode_{ep_id}.hdf5"
            with h5py.File(ep_path, "r") as ep:
                total_num_frames += ep["/action"].shape[0] - 1
        logging.info(f"{total_num_frames=}")

        logging.info("Initialize and feed offline buffer")
        idxtd = 0
        for ep_id in tqdm.tqdm(range(NUM_EPISODES[self.dataset_id])):
            ep_path = raw_dir / f"episode_{ep_id}.hdf5"
            with h5py.File(ep_path, "r") as ep:
                ep_num_frames = ep["/action"].shape[0]

                # last step of demonstration is considered done
                done = torch.zeros(ep_num_frames, 1, dtype=torch.bool)
                done[-1] = True

                state = torch.from_numpy(ep["/observations/qpos"][:])
                action = torch.from_numpy(ep["/action"][:])

                ep_td = TensorDict(
                    {
                        ("observation", "state"): state[:-1],
                        "action": action[:-1],
                        "episode": torch.tensor([ep_id] * (ep_num_frames - 1)),
                        "frame_id": torch.arange(0, ep_num_frames - 1, 1),
                        ("next", "observation", "state"): state[1:],
                        # TODO: compute reward and success
                        # ("next", "reward"): reward[1:],
                        ("next", "done"): done[1:],
                        # ("next", "success"): success[1:],
                    },
                    batch_size=ep_num_frames - 1,
                )

                for cam in CAMERAS[self.dataset_id]:
                    image = torch.from_numpy(ep[f"/observations/images/{cam}"][:])
                    image = einops.rearrange(image, "b h w c -> b c h w").contiguous()
                    ep_td["observation", "image", cam] = image[:-1]
                    ep_td["next", "observation", "image", cam] = image[1:]

                if ep_id == 0:
                    # hack to initialize tensordict data structure to store episodes
                    td_data = ep_td[0].expand(total_num_frames).memmap_like(self.data_dir)

                td_data[idxtd : idxtd + len(ep_td)] = ep_td
                idxtd = idxtd + len(ep_td)

        return TensorStorage(td_data.lock_())

import logging
from pathlib import Path

import einops
import gdown
import h5py
import torch
import tqdm

from lerobot.common.datasets.utils import load_data_with_delta_timestamps

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
    assert dataset_id in FOLDER_URLS
    assert dataset_id in EP48_URLS
    assert dataset_id in EP49_URLS

    data_dir.mkdir(parents=True, exist_ok=True)

    gdown.download_folder(FOLDER_URLS[dataset_id], output=str(data_dir))

    # because of the 50 files limit per directory, two files episode 48 and 49 were missing
    gdown.download(EP48_URLS[dataset_id], output=str(data_dir / "episode_48.hdf5"), fuzzy=True)
    gdown.download(EP49_URLS[dataset_id], output=str(data_dir / "episode_49.hdf5"), fuzzy=True)


class AlohaDataset(torch.utils.data.Dataset):
    available_datasets = [
        "aloha_sim_insertion_human",
        "aloha_sim_insertion_scripted",
        "aloha_sim_transfer_cube_human",
        "aloha_sim_transfer_cube_scripted",
    ]
    fps = 50
    image_keys = ["observation.images.top"]

    def __init__(
        self,
        dataset_id: str,
        version: str | None = "v1.2",
        root: Path | None = None,
        transform: callable = None,
        delta_timestamps: dict[list[float]] | None = None,
    ):
        super().__init__()
        self.dataset_id = dataset_id
        self.version = version
        self.root = root
        self.transform = transform
        self.delta_timestamps = delta_timestamps

        data_dir = self.root / f"{self.dataset_id}"
        if (data_dir / "data_dict.pth").exists() and (data_dir / "data_ids_per_episode.pth").exists():
            self.data_dict = torch.load(data_dir / "data_dict.pth")
            self.data_ids_per_episode = torch.load(data_dir / "data_ids_per_episode.pth")
        else:
            self._download_and_preproc_obsolete()
            data_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.data_dict, data_dir / "data_dict.pth")
            torch.save(self.data_ids_per_episode, data_dir / "data_ids_per_episode.pth")

    @property
    def num_samples(self) -> int:
        return len(self.data_dict["index"])

    @property
    def num_episodes(self) -> int:
        return len(self.data_ids_per_episode)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = {}

        # get episode id and timestamp of the sampled frame
        current_ts = self.data_dict["timestamp"][idx].item()
        episode = self.data_dict["episode"][idx].item()

        for key in self.data_dict:
            if self.delta_timestamps is not None and key in self.delta_timestamps:
                data, is_pad = load_data_with_delta_timestamps(
                    self.data_dict,
                    self.data_ids_per_episode,
                    self.delta_timestamps,
                    key,
                    current_ts,
                    episode,
                )
                item[key] = data
                item[f"{key}_is_pad"] = is_pad
            else:
                item[key] = self.data_dict[key][idx]

        if self.transform is not None:
            item = self.transform(item)

        return item

    def _download_and_preproc_obsolete(self):
        assert self.root is not None
        raw_dir = self.root / f"{self.dataset_id}_raw"
        if not raw_dir.is_dir():
            download(raw_dir, self.dataset_id)

        total_frames = 0
        logging.info("Compute total number of frames to initialize offline buffer")
        for ep_id in range(NUM_EPISODES[self.dataset_id]):
            ep_path = raw_dir / f"episode_{ep_id}.hdf5"
            with h5py.File(ep_path, "r") as ep:
                total_frames += ep["/action"].shape[0] - 1
        logging.info(f"{total_frames=}")

        self.data_ids_per_episode = {}
        ep_dicts = []

        logging.info("Initialize and feed offline buffer")
        for ep_id in tqdm.tqdm(range(NUM_EPISODES[self.dataset_id])):
            ep_path = raw_dir / f"episode_{ep_id}.hdf5"
            with h5py.File(ep_path, "r") as ep:
                num_frames = ep["/action"].shape[0]

                # last step of demonstration is considered done
                done = torch.zeros(num_frames, 1, dtype=torch.bool)
                done[-1] = True

                state = torch.from_numpy(ep["/observations/qpos"][:])
                action = torch.from_numpy(ep["/action"][:])

                ep_dict = {
                    "observation.state": state,
                    "action": action,
                    "episode": torch.tensor([ep_id] * num_frames),
                    "frame_id": torch.arange(0, num_frames, 1),
                    "timestamp": torch.arange(0, num_frames, 1) / self.fps,
                    # "next.observation.state": state,
                    # TODO(rcadene): compute reward and success
                    # "next.reward": reward[1:],
                    "next.done": done[1:],
                    # "next.success": success[1:],
                }

                for cam in CAMERAS[self.dataset_id]:
                    image = torch.from_numpy(ep[f"/observations/images/{cam}"][:])
                    image = einops.rearrange(image, "b h w c -> b c h w").contiguous()
                    ep_dict[f"observation.images.{cam}"] = image[:-1]
                    # ep_dict[f"next.observation.images.{cam}"] = image[1:]

                ep_dicts.append(ep_dict)

        self.data_dict = {}

        keys = ep_dicts[0].keys()
        for key in keys:
            self.data_dict[key] = torch.cat([x[key] for x in ep_dicts])

        self.data_dict["index"] = torch.arange(0, total_frames, 1)

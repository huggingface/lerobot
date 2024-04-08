import pickle
import zipfile
from pathlib import Path

import torch
import tqdm

from lerobot.common.datasets.utils import load_data_with_delta_timestamps


def download(raw_dir):
    import gdown

    raw_dir.mkdir(parents=True, exist_ok=True)
    url = "https://drive.google.com/uc?id=1nhxpykGtPDhmQKm-_B8zBSywVRdgeVya"
    zip_path = raw_dir / "data.zip"
    gdown.download(url, str(zip_path), quiet=False)
    print("Extracting...")
    with zipfile.ZipFile(str(zip_path), "r") as zip_f:
        for member in zip_f.namelist():
            if member.startswith("data/xarm") and member.endswith(".pkl"):
                print(member)
                zip_f.extract(member=member)
    zip_path.unlink()


class XarmDataset(torch.utils.data.Dataset):
    available_datasets = [
        "xarm_lift_medium",
    ]
    fps = 15
    image_keys = ["observation.image"]

    def __init__(
        self,
        dataset_id: str,
        version: str | None = "v1.1",
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

        self.data_dir = self.root / f"{self.dataset_id}"
        if (self.data_dir / "data_dict.pth").exists() and (
            self.data_dir / "data_ids_per_episode.pth"
        ).exists():
            self.data_dict = torch.load(self.data_dir / "data_dict.pth")
            self.data_ids_per_episode = torch.load(self.data_dir / "data_ids_per_episode.pth")
        else:
            self._download_and_preproc_obsolete()
            self.data_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.data_dict, self.data_dir / "data_dict.pth")
            torch.save(self.data_ids_per_episode, self.data_dir / "data_ids_per_episode.pth")

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
        if not raw_dir.exists():
            download(raw_dir)

        dataset_path = self.root / f"{self.dataset_id}" / "buffer.pkl"
        print(f"Using offline dataset '{dataset_path}'")
        with open(dataset_path, "rb") as f:
            dataset_dict = pickle.load(f)

        total_frames = dataset_dict["actions"].shape[0]

        self.data_ids_per_episode = {}
        ep_dicts = []

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
            action = torch.tensor(dataset_dict["actions"][idx0:idx1])
            # TODO(rcadene): concat the last "next_observations" to "observations"
            # next_image = torch.tensor(dataset_dict["next_observations"]["rgb"][idx0:idx1])
            # next_state = torch.tensor(dataset_dict["next_observations"]["state"][idx0:idx1])
            next_reward = torch.tensor(dataset_dict["rewards"][idx0:idx1])
            next_done = torch.tensor(dataset_dict["dones"][idx0:idx1])

            ep_dict = {
                "observation.image": image,
                "observation.state": state,
                "action": action,
                "episode": torch.tensor([episode_id] * num_frames, dtype=torch.int),
                "frame_id": torch.arange(0, num_frames, 1),
                "timestamp": torch.arange(0, num_frames, 1) / self.fps,
                # "next.observation.image": next_image,
                # "next.observation.state": next_state,
                "next.reward": next_reward,
                "next.done": next_done,
            }
            ep_dicts.append(ep_dict)

            assert isinstance(episode_id, int)
            self.data_ids_per_episode[episode_id] = torch.arange(idx0, idx1, 1)
            assert len(self.data_ids_per_episode[episode_id]) == num_frames

            idx0 = idx1
            episode_id += 1

        self.data_dict = {}

        keys = ep_dicts[0].keys()
        for key in keys:
            self.data_dict[key] = torch.cat([x[key] for x in ep_dicts])

        self.data_dict["index"] = torch.arange(0, total_frames, 1)

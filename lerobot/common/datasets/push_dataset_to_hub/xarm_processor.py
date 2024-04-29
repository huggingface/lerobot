import pickle
from pathlib import Path

import einops
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)


class XarmProcessor:
    """Process pickle files formatted like in: https://github.com/fyhMer/fowm"""

    def __init__(self, folder_path: str, fps: int | None = None):
        self.folder_path = Path(folder_path)
        self.keys = {"actions", "rewards", "dones", "masks"}
        self.nested_keys = {"observations": {"rgb", "state"}, "next_observations": {"rgb", "state"}}
        if fps is None:
            fps = 15
        self._fps = fps

    @property
    def fps(self) -> int:
        return self._fps

    def is_valid(self) -> bool:
        # get all .pkl files
        xarm_files = list(self.folder_path.glob("*.pkl"))
        if len(xarm_files) != 1:
            return False

        try:
            with open(xarm_files[0], "rb") as f:
                dataset_dict = pickle.load(f)
        except Exception:
            return False

        if not isinstance(dataset_dict, dict):
            return False

        if not all(k in dataset_dict for k in self.keys):
            return False

        # Check for consistent lengths in nested keys
        try:
            expected_len = len(dataset_dict["actions"])
            if any(len(dataset_dict[key]) != expected_len for key in self.keys if key in dataset_dict):
                return False

            for key, subkeys in self.nested_keys.items():
                nested_dict = dataset_dict.get(key, {})
                if any(
                    len(nested_dict[subkey]) != expected_len for subkey in subkeys if subkey in nested_dict
                ):
                    return False
        except KeyError:  # If any expected key or subkey is missing
            return False

        return True  # All checks passed

    def preprocess(self):
        if not self.is_valid():
            raise ValueError("The Xarm file is invalid or does not contain the required datasets.")

        xarm_files = list(self.folder_path.glob("*.pkl"))

        with open(xarm_files[0], "rb") as f:
            dataset_dict = pickle.load(f)
        ep_dicts = []
        episode_data_index = {"from": [], "to": []}

        id_from = 0
        id_to = 0
        episode_id = 0
        total_frames = dataset_dict["actions"].shape[0]
        for i in tqdm.tqdm(range(total_frames)):
            id_to += 1

            if not dataset_dict["dones"][i]:
                continue

            num_frames = id_to - id_from

            image = torch.tensor(dataset_dict["observations"]["rgb"][id_from:id_to])
            image = einops.rearrange(image, "b c h w -> b h w c")
            state = torch.tensor(dataset_dict["observations"]["state"][id_from:id_to])
            action = torch.tensor(dataset_dict["actions"][id_from:id_to])
            # TODO(rcadene): we have a missing last frame which is the observation when the env is done
            # it is critical to have this frame for tdmpc to predict a "done observation/state"
            # next_image = torch.tensor(dataset_dict["next_observations"]["rgb"][id_from:id_to])
            # next_state = torch.tensor(dataset_dict["next_observations"]["state"][id_from:id_to])
            next_reward = torch.tensor(dataset_dict["rewards"][id_from:id_to])
            next_done = torch.tensor(dataset_dict["dones"][id_from:id_to])

            ep_dict = {
                "observation.image": [PILImage.fromarray(x.numpy()) for x in image],
                "observation.state": state,
                "action": action,
                "episode_index": torch.tensor([episode_id] * num_frames, dtype=torch.int),
                "frame_index": torch.arange(0, num_frames, 1),
                "timestamp": torch.arange(0, num_frames, 1) / self.fps,
                # "next.observation.image": next_image,
                # "next.observation.state": next_state,
                "next.reward": next_reward,
                "next.done": next_done,
            }
            ep_dicts.append(ep_dict)

            episode_data_index["from"].append(id_from)
            episode_data_index["to"].append(id_from + num_frames)

            id_from = id_to
            episode_id += 1

        data_dict = concatenate_episodes(ep_dicts)
        return data_dict, episode_data_index

    def to_hf_dataset(self, data_dict):
        features = {
            "observation.image": Image(),
            "observation.state": Sequence(
                length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
            ),
            "action": Sequence(length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)),
            "episode_index": Value(dtype="int64", id=None),
            "frame_index": Value(dtype="int64", id=None),
            "timestamp": Value(dtype="float32", id=None),
            "next.reward": Value(dtype="float32", id=None),
            "next.done": Value(dtype="bool", id=None),
            #'next.success': Value(dtype='bool', id=None),
            "index": Value(dtype="int64", id=None),
        }
        features = Features(features)
        hf_dataset = Dataset.from_dict(data_dict, features=features)
        hf_dataset.set_transform(hf_transform_to_torch)

        return hf_dataset

    def cleanup(self):
        pass

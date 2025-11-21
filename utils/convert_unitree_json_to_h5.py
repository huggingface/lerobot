"""
Script Json to h5.

# --data_dirs   Corresponds to the directory of your JSON dataset
# --output_dir  Save path to h5 file
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire)

python unitree_lerobot/utils/convert_unitree_json_to_h5.py \
    --data_dirs $HOME/datasets/json \
    --output_dir $HOME/datasets/h5 \
    --robot_type Unitree_G1_Dex3
"""

import os
import tyro
import json
import h5py
import cv2
import tqdm
import glob
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from unitree_lerobot.utils.constants import ROBOT_CONFIGS


class JsonDataset:
    def __init__(self, data_dirs: Path, robot_type: str) -> None:
        """
        Initialize the dataset for loading and processing HDF5 files containing robot manipulation data.

        Args:
            data_dirs: Path to directory containing training data
        """
        assert data_dirs is not None, "Data directory cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        self.data_dirs = data_dirs
        self.json_file = "data.json"

        # Initialize paths and cache
        self._init_paths()
        self._init_cache()
        self.json_state_data_name = ROBOT_CONFIGS[robot_type].json_state_data_name
        self.json_action_data_name = ROBOT_CONFIGS[robot_type].json_action_data_name
        self.camera_to_image_key = ROBOT_CONFIGS[robot_type].camera_to_image_key

    def _init_paths(self) -> None:
        """Initialize episode and task paths."""

        self.episode_paths = []
        self.task_paths = []

        for task_path in glob.glob(os.path.join(self.data_dirs, "*")):
            if os.path.isdir(task_path):
                episode_paths = glob.glob(os.path.join(task_path, "*"))
                if episode_paths:
                    self.task_paths.append(task_path)
                    self.episode_paths.extend(episode_paths)

        self.episode_paths = sorted(self.episode_paths)
        self.episode_ids = list(range(len(self.episode_paths)))

    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episode_paths)

    def _init_cache(self) -> List:
        """Initialize data cache if enabled."""

        self.episodes_data_cached = []
        for episode_path in tqdm.tqdm(self.episode_paths, desc="Loading Cache Json"):
            json_path = os.path.join(episode_path, self.json_file)
            with open(json_path, "r", encoding="utf-8") as jsonf:
                self.episodes_data_cached.append(json.load(jsonf))

        print(f"==> Cached {len(self.episodes_data_cached)} episodes")

        return self.episodes_data_cached

    def _extract_data(self, episode_data: Dict, key: str, parts: List[str]) -> np.ndarray:
        """
        Extract data from episode dictionary for specified parts.

        Args:
            episode_data: Dictionary containing episode data
            key: Data key to extract ('states' or 'actions')
            parts: List of parts to include ('left_arm', 'right_arm')

        Returns:
            Concatenated numpy array of the requested data
        """
        result = []
        for sample_data in episode_data["data"]:
            data_array = np.array([], dtype=np.float32)
            for part in parts:
                if part in sample_data[key] and sample_data[key][part] is not None:
                    qpos = np.array(sample_data[key][part]["qpos"], dtype=np.float32)
                    data_array = np.concatenate([data_array, qpos])
            result.append(data_array)
        return np.array(result)

    def _parse_images(self, episode_path: str, episode_data) -> dict[str, list[np.ndarray]]:
        """Load and stack images for a given camera key."""

        images = defaultdict(list)

        keys = episode_data["data"][0]["colors"].keys()
        cameras = [key for key in keys if "depth" not in key]

        for camera in cameras:
            image_key = self.camera_to_image_key.get(camera)

            for sample_data in episode_data["data"]:
                image_path = os.path.join(episode_path, sample_data["colors"].get(camera))
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image path does not exist: {image_path}")

                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[image_key].append(image_rgb)

        return images

    def get_item(
        self,
        index: Optional[int] = None,
    ) -> Dict:
        """Get a training sample from the dataset."""

        file_path = np.random.choice(self.episode_paths) if index is None else self.episode_paths[index]
        episode_data = self.episodes_data_cached[index]

        # Load state and action data
        action = self._extract_data(episode_data, "actions", self.json_action_data_name)
        state = self._extract_data(episode_data, "states", self.json_state_data_name)
        episode_length = len(state)
        state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        action_dim = action.shape[1] if len(action.shape) == 2 else state.shape[0]

        # Load task description
        task = episode_data.get("text", {}).get("goal", "")

        # Load camera images
        cameras = self._parse_images(file_path, episode_data)

        # Extract camera configuration
        cam_height, cam_width = next(img for imgs in cameras.values() if imgs for img in imgs).shape[:2]
        data_cfg = {
            "camera_names": list(cameras.keys()),
            "cam_height": cam_height,
            "cam_width": cam_width,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }

        return {
            "episode_index": index,
            "episode_length": episode_length,
            "state": state,
            "action": action,
            "cameras": cameras,
            "task": task,
            "data_cfg": data_cfg,
        }


class H5Writer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_to_h5(self, episode: dict) -> None:
        """Write episode data to HDF5 file."""

        episode_length = episode["episode_length"]
        episode_index = episode["episode_index"]
        state = episode["state"]
        action = episode["action"]
        qvel = np.zeros_like(episode["state"])
        cameras = episode["cameras"]
        task = episode["task"]
        data_cfg = episode["data_cfg"]

        # Prepare data dictionary
        data_dict = {
            "/observations/qpos": [state],
            "/observations/qvel": [qvel],
            "/action": [action],
            **{f"/observations/images/{k}": [v] for k, v in cameras.items()},
        }

        h5_path = os.path.join(self.output_dir, f"episode_{episode_index}.hdf5")

        with h5py.File(h5_path, "w", rdcc_nbytes=1024**2 * 2, libver="latest") as root:
            # Set attributes
            root.attrs["sim"] = False

            # Create datasets
            obs = root.create_group("observations")
            image = obs.create_group("images")

            # Write camera images
            for cam_name, images in cameras.items():
                image.create_dataset(
                    cam_name,
                    shape=(episode_length, data_cfg["cam_height"], data_cfg["cam_width"], 3),
                    dtype="uint8",
                    chunks=(1, data_cfg["cam_height"], data_cfg["cam_width"], 3),
                    compression="gzip",
                )
                # root[f'/observations/images/{cam_name}'][...] = images

            # Write state and action data
            obs.create_dataset("qpos", (episode_length, data_cfg["state_dim"]), dtype="float32", compression="gzip")
            obs.create_dataset("qvel", (episode_length, data_cfg["state_dim"]), dtype="float32", compression="gzip")
            root.create_dataset("action", (episode_length, data_cfg["action_dim"]), dtype="float32", compression="gzip")

            # Write metadata
            root.create_dataset("is_edited", (1,), dtype="uint8")
            substep_reasonings = root.create_dataset(
                "substep_reasonings", (episode_length,), dtype=h5py.string_dtype(encoding="utf-8"), compression="gzip"
            )
            root.create_dataset("language_raw", data=task)
            substep_reasonings[:] = [task] * episode_length

            # Write additional data
            for name, array in data_dict.items():
                root[name][...] = array


def json_to_h5(
    data_dirs: Path,
    output_dir: Path,
    robot_type: str,
) -> None:
    """Convert JSON episode data to HDF5 format."""
    dataset = JsonDataset(data_dirs, robot_type)
    h5_writer = H5Writer(output_dir)

    for i in tqdm.tqdm(range(len(dataset))):
        episode = dataset.get_item(i)
        h5_writer.write_to_h5(episode)


if __name__ == "__main__":
    tyro.cli(json_to_h5)

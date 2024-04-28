import re
from pathlib import Path

import h5py
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)


class AlohaProcessor:
    """
    Process HDF5 files formatted like in: https://github.com/tonyzhaozh/act

    Attributes:
        folder_path (Path): Path to the directory containing HDF5 files.
        cameras (list[str]): List of camera identifiers to check in the files.
        fps (int): Frames per second used in timestamp calculations.

    Methods:
        is_valid() -> bool:
            Validates if each HDF5 file within the folder contains all required datasets.
        preprocess() -> dict:
            Processes the files and returns structured data suitable for further analysis.
        to_hf_dataset(data_dict: dict) -> Dataset:
            Converts processed data into a Hugging Face Dataset object.
    """

    def __init__(self, folder_path: Path, cameras: list[str] | None = None, fps: int | None = None):
        """
        Initializes the AlohaProcessor with a specified directory path containing HDF5 files,
        an optional list of cameras, and a frame rate.

        Args:
            folder_path (Path): The directory path where HDF5 files are stored.
            cameras (list[str] | None): Optional list of cameras to validate within the files. Defaults to ['top'] if None.
            fps (int): Frame rate for the datasets, used in time calculations. Default is 50.

        Examples:
            >>> processor = AlohaProcessor(Path("path_to_hdf5_directory"), ["camera1", "camera2"])
            >>> processor.is_valid()
            True
        """
        self.folder_path = folder_path
        if cameras is None:
            cameras = ["top"]
        self.cameras = cameras
        if fps is None:
            fps = 50
        self._fps = fps

    @property
    def fps(self) -> int:
        return self._fps

    def is_valid(self) -> bool:
        """
        Validates the HDF5 files in the specified folder to ensure they contain the required datasets
        for actions, positions, and images for each specified camera.

        Returns:
            bool: True if all files are valid HDF5 files with all required datasets, False otherwise.
        """
        hdf5_files: list[Path] = list(self.folder_path.glob("episode_*.hdf5"))
        if len(hdf5_files) == 0:
            return False
        try:
            hdf5_files = sorted(
                hdf5_files, key=lambda x: int(re.search(r"episode_(\d+).hdf5", x.name).group(1))
            )
        except AttributeError:
            # All file names must contain a numerical identifier matching 'episode_(\\d+).hdf5
            return False

        # Check if the sequence is consecutive eg episode_0, episode_1, episode_2, etc.
        # If not, return False
        previous_number = None
        for file in hdf5_files:
            current_number = int(re.search(r"episode_(\d+).hdf5", file.name).group(1))
            if previous_number is not None and current_number - previous_number != 1:
                return False
            previous_number = current_number

        for file in hdf5_files:
            try:
                with h5py.File(file, "r") as file:
                    # Check for the expected datasets within the HDF5 file
                    required_datasets = ["/action", "/observations/qpos"]
                    # Add camera-specific image datasets to the required datasets
                    camera_datasets = [f"/observations/images/{cam}" for cam in self.cameras]
                    required_datasets.extend(camera_datasets)

                    if not all(dataset in file for dataset in required_datasets):
                        return False
            except OSError:
                return False
        return True

    def preprocess(self):
        """
        Collects episode data from the HDF5 file and returns it as an AlohaStep named tuple.

        Returns:
            AlohaStep: Named tuple containing episode data.

        Raises:
            ValueError: If the file is not valid.
        """
        if not self.is_valid():
            raise ValueError("The HDF5 file is invalid or does not contain the required datasets.")

        hdf5_files = list(self.folder_path.glob("*.hdf5"))
        hdf5_files = sorted(hdf5_files, key=lambda x: int(re.search(r"episode_(\d+)", x.name).group(1)))
        ep_dicts = []
        episode_data_index = {"from": [], "to": []}

        id_from = 0

        for ep_path in tqdm.tqdm(hdf5_files):
            with h5py.File(ep_path, "r") as ep:
                ep_id = int(re.search(r"episode_(\d+)", ep_path.name).group(1))
                num_frames = ep["/action"].shape[0]

                # last step of demonstration is considered done
                done = torch.zeros(num_frames, dtype=torch.bool)
                done[-1] = True

                state = torch.from_numpy(ep["/observations/qpos"][:])
                action = torch.from_numpy(ep["/action"][:])

                ep_dict = {}

                for cam in self.cameras:
                    image = torch.from_numpy(ep[f"/observations/images/{cam}"][:])  # b h w c
                    ep_dict[f"observation.images.{cam}"] = [PILImage.fromarray(x.numpy()) for x in image]

                ep_dict.update(
                    {
                        "observation.state": state,
                        "action": action,
                        "episode_index": torch.tensor([ep_id] * num_frames),
                        "frame_index": torch.arange(0, num_frames, 1),
                        "timestamp": torch.arange(0, num_frames, 1) / self.fps,
                        # TODO(rcadene): compute reward and success
                        # "next.reward": reward,
                        "next.done": done,
                        # "next.success": success,
                    }
                )

                assert isinstance(ep_id, int)
                ep_dicts.append(ep_dict)

                episode_data_index["from"].append(id_from)
                episode_data_index["to"].append(id_from + num_frames)

            id_from += num_frames

        data_dict = concatenate_episodes(ep_dicts)
        return data_dict, episode_data_index

    def to_hf_dataset(self, data_dict) -> Dataset:
        """
        Converts a dictionary of data into a Hugging Face Dataset object.

        Args:
            data_dict (dict): A dictionary containing the data to be converted.

        Returns:
            Dataset: The converted Hugging Face Dataset object.
        """
        image_features = {f"observation.images.{cam}": Image() for cam in self.cameras}
        features = {
            "observation.state": Sequence(
                length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
            ),
            "action": Sequence(length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)),
            "episode_index": Value(dtype="int64", id=None),
            "frame_index": Value(dtype="int64", id=None),
            "timestamp": Value(dtype="float32", id=None),
            # "next.reward": Value(dtype="float32", id=None),
            "next.done": Value(dtype="bool", id=None),
            # "next.success": Value(dtype="bool", id=None),
            "index": Value(dtype="int64", id=None),
        }
        update_features = {**image_features, **features}
        features = Features(update_features)
        hf_dataset = Dataset.from_dict(data_dict, features=features)
        hf_dataset.set_transform(hf_transform_to_torch)

        return hf_dataset

    def cleanup(self):
        pass

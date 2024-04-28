import os
import re
import shutil
from glob import glob

import numpy as np
import torch
import tqdm
import zarr
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub._umi_imagecodecs_numcodecs import register_codecs
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)


class UmiProcessor:
    """
     Process UMI (Universal Manipulation Interface) data stored in Zarr format like in: https://github.com/real-stanford/universal_manipulation_interface

    Attributes:
        folder_path (str): The path to the folder containing Zarr datasets.
        fps (int): Frames per second, used to calculate timestamps for frames.

    """

    def __init__(self, folder_path: str, fps: int | None = None):
        self.zarr_path = folder_path
        if fps is None:
            # TODO (azouitine): Add reference to the paper
            fps = 15
        self._fps = fps
        register_codecs()

    @property
    def fps(self) -> int:
        return self._fps

    def is_valid(self) -> bool:
        """
        Validates the Zarr folder to ensure it contains all required datasets with consistent frame counts.

        Returns:
            bool: True if all required datasets are present and have consistent frame counts, False otherwise.
        """
        # Check if the Zarr folder is valid
        try:
            zarr_data = zarr.open(self.zarr_path, mode="r")
        except Exception:
            # TODO (azouitine): Handle the exception properly
            return False
        required_datasets = {
            "data/robot0_demo_end_pose",
            "data/robot0_demo_start_pose",
            "data/robot0_eef_pos",
            "data/robot0_eef_rot_axis_angle",
            "data/robot0_gripper_width",
            "meta/episode_ends",
            "data/camera0_rgb",
        }
        for dataset in required_datasets:
            if dataset not in zarr_data:
                return False
        nb_frames = zarr_data["data/camera0_rgb"].shape[0]

        required_datasets.remove("meta/episode_ends")

        return all(nb_frames == zarr_data[dataset].shape[0] for dataset in required_datasets)

    def preprocess(self):
        """
        Collects and processes all episodes from the Zarr dataset into structured data dictionaries.

        Returns:
            Tuple[Dict, Dict]: A tuple containing the structured episode data and episode index mappings.
        """
        zarr_data = zarr.open(self.zarr_path, mode="r")

        # We process the image data separately because it is too large to fit in memory
        end_pose = torch.from_numpy(zarr_data["data/robot0_demo_end_pose"][:])
        start_pos = torch.from_numpy(zarr_data["data/robot0_demo_start_pose"][:])
        eff_pos = torch.from_numpy(zarr_data["data/robot0_eef_pos"][:])
        eff_rot_axis_angle = torch.from_numpy(zarr_data["data/robot0_eef_rot_axis_angle"][:])
        gripper_width = torch.from_numpy(zarr_data["data/robot0_gripper_width"][:])

        states_pos = torch.cat([eff_pos, eff_rot_axis_angle], dim=1)
        states = torch.cat([states_pos, gripper_width], dim=1)

        episode_ends = zarr_data["meta/episode_ends"][:]
        num_episodes: int = episode_ends.shape[0]

        episode_ids = torch.from_numpy(self.get_episode_idxs(episode_ends))

        # We convert it in torch tensor later because the jit function does not support torch tensors
        episode_ends = torch.from_numpy(episode_ends)

        ep_dicts = []
        episode_data_index = {"from": [], "to": []}
        id_from = 0

        for episode_id in tqdm.tqdm(range(num_episodes)):
            id_to = episode_ends[episode_id]

            num_frames = id_to - id_from

            assert (
                episode_ids[id_from:id_to] == episode_id
            ).all(), f"episode_ids[{id_from}:{id_to}] != {episode_id}"

            state = states[id_from:id_to]
            ep_dict = {
                # observation.image will be filled later
                "observation.state": state,
                "episode_index": torch.tensor([episode_id] * num_frames, dtype=torch.int),
                "frame_index": torch.arange(0, num_frames, 1),
                "timestamp": torch.arange(0, num_frames, 1) / self.fps,
                "episode_data_index_from": torch.tensor([id_from] * num_frames),
                "episode_data_index_to": torch.tensor([id_from + num_frames] * num_frames),
                "end_pose": end_pose[id_from:id_to],
                "start_pos": start_pos[id_from:id_to],
                "gripper_width": gripper_width[id_from:id_to],
            }
            ep_dicts.append(ep_dict)
            episode_data_index["from"].append(id_from)
            episode_data_index["to"].append(id_from + num_frames)
            id_from += num_frames

        data_dict = concatenate_episodes(ep_dicts)

        total_frames = id_from
        data_dict["index"] = torch.arange(0, total_frames, 1)

        print("Saving images to disk in temporary folder...")
        # datasets.Image() can take a list of paths to images, so we save the images to a temporary folder
        # to avoid loading them all in memory
        _save_images_concurrently(
            data=zarr_data, image_key="data/camera0_rgb", folder_path="tmp_umi_images", max_workers=12
        )
        print("Saving images to disk in temporary folder... Done")

        # Sort files by number eg. 1.png, 2.png, 3.png, 9.png, 10.png instead of 1.png, 10.png, 2.png, 3.png, 9.png
        # to correctly match the images with the data
        images_path = sorted(
            glob("tmp_umi_images/*"), key=lambda x: int(re.search(r"(\d+)\.png$", x).group(1))
        )
        data_dict["observation.image"] = images_path
        print("Images saved to disk, do not forget to delete the folder tmp_umi_images/")

        # Cleanup
        return data_dict, episode_data_index

    def to_hf_dataset(self, data_dict):
        """
        Converts the processed data dictionary into a Hugging Face dataset with defined features.

        Args:
            data_dict (Dict): The data dictionary containing tensors and episode information.

        Returns:
            Dataset: A Hugging Face dataset constructed from the provided data dictionary.
        """
        features = {
            "observation.image": Image(),
            "observation.state": Sequence(
                length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
            ),
            "episode_index": Value(dtype="int64", id=None),
            "frame_index": Value(dtype="int64", id=None),
            "timestamp": Value(dtype="float32", id=None),
            "index": Value(dtype="int64", id=None),
            "episode_data_index_from": Value(dtype="int64", id=None),
            "episode_data_index_to": Value(dtype="int64", id=None),
            # `start_pos` and `end_pos` respectively represent the positions of the end-effector
            # at the beginning and the end of the episode.
            # `gripper_width` indicates the distance between the grippers, and this value is included
            # in the state vector, which comprises the concatenation of the end-effector position
            # and gripper width.
            "end_pose": Sequence(
                length=data_dict["end_pose"].shape[1], feature=Value(dtype="float32", id=None)
            ),
            "start_pos": Sequence(
                length=data_dict["start_pos"].shape[1], feature=Value(dtype="float32", id=None)
            ),
            "gripper_width": Sequence(
                length=data_dict["gripper_width"].shape[1], feature=Value(dtype="float32", id=None)
            ),
        }
        features = Features(features)
        hf_dataset = Dataset.from_dict(data_dict, features=features)
        hf_dataset.set_transform(hf_transform_to_torch)

        return hf_dataset

    def cleanup(self):
        # Cleanup
        if os.path.exists("tmp_umi_images"):
            print("Removing temporary images folder")
            shutil.rmtree("tmp_umi_images")
            print("Cleanup done")

    @classmethod
    def get_episode_idxs(cls, episode_ends: np.ndarray) -> np.ndarray:
        # Optimized and simplified version of this function: https://github.com/real-stanford/universal_manipulation_interface/blob/298776ce251f33b6b3185a98d6e7d1f9ad49168b/diffusion_policy/common/replay_buffer.py#L374
        from numba import jit

        @jit(nopython=True)
        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            start_idx = 0
            for episode_number, end_idx in enumerate(episode_ends):
                result[start_idx:end_idx] = episode_number
                start_idx = end_idx
            return result

        return _get_episode_idxs(episode_ends)


def _clear_folder(folder_path: str):
    """
    Clears all the content of the specified folder. Creates the folder if it does not exist.

    Args:
    folder_path (str): Path to the folder to clear.

    Examples:
    >>> import os
    >>> os.makedirs('example_folder', exist_ok=True)
    >>> with open('example_folder/temp_file.txt', 'w') as f:
    ...     f.write('example')
    >>> clear_folder('example_folder')
    >>> os.listdir('example_folder')
    []
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(folder_path)


def _save_image(img_array: np.array, i: int, folder_path: str):
    """
    Saves a single image to the specified folder.

    Args:
    img_array (ndarray): The numpy array of the image.
    i (int): Index of the image, used for naming.
    folder_path (str): Path to the folder where the image will be saved.
    """
    img = PILImage.fromarray(img_array)
    img_format = "PNG" if img_array.dtype == np.uint8 else "JPEG"
    img.save(os.path.join(folder_path, f"{i}.{img_format.lower()}"), quality=100)


def _save_images_concurrently(data: dict, image_key: str, folder_path: str, max_workers: int = 4):
    from concurrent.futures import ThreadPoolExecutor

    """
    Saves images from the zarr_data to the specified folder using multithreading.

    Args:
    zarr_data (dict): A dictionary containing image data in an array format.
    folder_path (str): Path to the folder where images will be saved.
    max_workers (int): The maximum number of threads to use for saving images.
    """
    num_images = len(data["data/camera0_rgb"])
    _clear_folder(folder_path)  # Clear or create folder first

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        [executor.submit(_save_image, data[image_key][i], i, folder_path) for i in range(num_images)]

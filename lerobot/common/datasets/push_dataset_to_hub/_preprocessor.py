# import os
# import pickle
# import re
# import shutil
# from glob import glob
# from pathlib import Path
# from typing import Protocol

# import einops
# import h5py
# import numpy as np
# import torch
# import tqdm
# import zarr
# from datasets import Dataset, Features, Image, Sequence, Value
# from PIL import Image as PILImage

# from lerobot.common.datasets.push_dataset_to_hub._umi_imagecodecs_numcodecs import register_codecs
# from lerobot.common.datasets.utils import (
#     hf_transform_to_torch,
# )


# class DatasetProcessor(Protocol):
#     def is_valid(self) -> bool: ...
#     def preprocess(self) -> tuple[dict, dict]: ...
#     def to_hf_dataset(self, data_dict: dict) -> Dataset: ...
#     @property
#     def fps(self) -> int: ...
#     def cleanup(self): ...


# class AlohaProcessor:
#     """
#     A class to process and validate HDF5 files containing episodic data related to specific camera datasets.

#     This processor handles the management and validation of data stored in HDF5 file format,
#     specifically designed for episodic content from different camera sources. The validation
#     involves checking the integrity and presence of required datasets within the files.

#     Attributes:
#         folder_path (Path): Path to the directory containing HDF5 files.
#         cameras (list[str]): List of camera identifiers to check in the files.
#         fps (int): Frames per second for time-series data, used in timestamp calculations.

#     Methods:
#         is_valid() -> bool:
#             Validates if each HDF5 file within the folder contains all required datasets.
#         preprocess() -> dict:
#             Processes the files and returns structured data suitable for further analysis.
#         to_hf_dataset(data_dict: dict) -> Dataset:
#             Converts processed data into a Hugging Face Dataset object.
#     """

#     def __init__(self, folder_path: Path, cameras: list[str] | None = None, fps: int | None = None):
#         """
#         Initializes the AlohaProcessor with a specified directory path containing HDF5 files,
#         an optional list of cameras, and a frame rate.

#         Args:
#             folder_path (Path): The directory path where HDF5 files are stored.
#             cameras (list[str] | None): Optional list of cameras to validate within the files. Defaults to ['top'] if None.
#             fps (int): Frame rate for the datasets, used in time calculations. Default is 50.

#         Examples:
#             >>> processor = AlohaProcessor(Path("path_to_hdf5_directory"), ["camera1", "camera2"])
#             >>> processor.is_valid()
#             True
#         """
#         self.folder_path = folder_path
#         if cameras is None:
#             cameras = ["top"]
#         self.cameras = cameras
#         if fps is None:
#             fps = 50
#         self.fps = fps

#     @property
#     def fps(self) -> int:
#         return self.fps

#     def is_valid(self) -> bool:
#         """
#         Validates the HDF5 files in the specified folder to ensure they contain the required datasets
#         for actions, positions, and images for each specified camera.

#         Returns:
#             bool: True if all files are valid HDF5 files with all required datasets, False otherwise.
#         """
#         hdf5_files: list[Path] = list(self.folder_path.glob("episode_*.hdf5"))
#         if len(hdf5_files) == 0:
#             return False
#         try:
#             hdf5_files = sorted(
#                 hdf5_files, key=lambda x: int(re.search(r"episode_(\d+).hdf5", x.name).group(1))
#             )
#         except AttributeError:
#             # All file names must contain a numerical identifier matching 'episode_(\\d+).hdf5
#             return False

#         # Check if the sequence is consecutive eg episode_0, episode_1, episode_2, etc.
#         # If not, return False
#         previous_number = None
#         for file in hdf5_files:
#             current_number = int(re.search(r"episode_(\d+).hdf5", file.name).group(1))
#             if previous_number is not None and current_number - previous_number != 1:
#                 return False
#             previous_number = current_number

#         for file in hdf5_files:
#             try:
#                 with h5py.File(file, "r") as file:
#                     # Check for the expected datasets within the HDF5 file
#                     required_datasets = ["/action", "/observations/qpos"]
#                     # Add camera-specific image datasets to the required datasets
#                     camera_datasets = [f"/observations/images/{cam}" for cam in self.cameras]
#                     required_datasets.extend(camera_datasets)

#                     if not all(dataset in file for dataset in required_datasets):
#                         return False
#             except OSError:
#                 return False
#         return True

#     def preprocess(self):
#         """
#         Collects episode data from the HDF5 file and returns it as an AlohaStep named tuple.

#         Returns:
#             AlohaStep: Named tuple containing episode data.

#         Raises:
#             ValueError: If the file is not valid.
#         """
#         if not self.is_valid():
#             raise ValueError("The HDF5 file is invalid or does not contain the required datasets.")

#         hdf5_files = list(self.folder_path.glob("*.hdf5"))
#         hdf5_files = sorted(hdf5_files, key=lambda x: int(re.search(r"episode_(\d+)", x.name).group(1)))
#         ep_dicts = []
#         episode_data_index = {"from": [], "to": []}

#         id_from = 0

#         for ep_path in tqdm.tqdm(hdf5_files):
#             # for ep_id in tqdm.tqdm(range(num_episodes)):
#             # ep_path = raw_dir / f"episode_{ep_id}.hdf5"
#             with h5py.File(ep_path, "r") as ep:
#                 ep_id = int(re.search(r"episode_(\d+)", ep_path.name).group(1))
#                 num_frames = ep["/action"].shape[0]
#                 # assert episode_len[dataset_id] == num_frames

#                 # last step of demonstration is considered done
#                 done = torch.zeros(num_frames, dtype=torch.bool)
#                 done[-1] = True

#                 state = torch.from_numpy(ep["/observations/qpos"][:])
#                 action = torch.from_numpy(ep["/action"][:])

#                 ep_dict = {}

#                 for cam in self.cameras:
#                     image = torch.from_numpy(ep[f"/observations/images/{cam}"][:])  # b h w c
#                     # image = einops.rearrange(image, "b h w c -> b c h w").contiguous()
#                     ep_dict[f"observation.images.{cam}"] = [PILImage.fromarray(x.numpy()) for x in image]
#                     # ep_dict[f"next.observation.images.{cam}"] = image

#                 ep_dict.update(
#                     {
#                         "observation.state": state,
#                         "action": action,
#                         "episode_index": torch.tensor([ep_id] * num_frames),
#                         "frame_index": torch.arange(0, num_frames, 1),
#                         "timestamp": torch.arange(0, num_frames, 1) / self.fps,
#                         # "next.observation.state": state,
#                         # TODO(rcadene): compute reward and success
#                         # "next.reward": reward,
#                         "next.done": done,
#                         # "next.success": success,
#                     }
#                 )

#                 assert isinstance(ep_id, int)
#                 ep_dicts.append(ep_dict)

#                 episode_data_index["from"].append(id_from)
#                 episode_data_index["to"].append(id_from + num_frames)

#             id_from += num_frames

#         data_dict = concatenate_episodes(ep_dicts)
#         return data_dict, episode_data_index

#     def to_hf_dataset(self, data_dict) -> Dataset:
#         """
#         Converts a dictionary of data into a Hugging Face Dataset object.

#         Args:
#             data_dict (dict): A dictionary containing the data to be converted.

#         Returns:
#             Dataset: The converted Hugging Face Dataset object.
#         """
#         image_features = {f"observation.images.{cam}": Image() for cam in self.cameras}
#         features = {
#             "observation.state": Sequence(
#                 length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
#             ),
#             "action": Sequence(length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)),
#             "episode_index": Value(dtype="int64", id=None),
#             "frame_index": Value(dtype="int64", id=None),
#             "timestamp": Value(dtype="float32", id=None),
#             # "next.reward": Value(dtype="float32", id=None),
#             "next.done": Value(dtype="bool", id=None),
#             # "next.success": Value(dtype="bool", id=None),
#             "index": Value(dtype="int64", id=None),
#         }
#         update_features = {**image_features, **features}
#         features = Features(update_features)
#         hf_dataset = Dataset.from_dict(data_dict, features=features)
#         hf_dataset.set_transform(hf_transform_to_torch)

#         return hf_dataset

#     def clenup(self):
#         pass


# class UmiProcessor:
#     """
#     A processor class for handling and validating UMI (Universal Manipulation Interface) data stored in Zarr format.

#     Attributes:
#         folder_path (str): The path to the folder containing Zarr datasets.
#         fps (int): Frames per second, used to calculate timestamps for frames.

#     Methods:
#         is_valid() -> bool: Check the validity of the Zarr datasets.
#         preprocess() -> Tuple[Dict, Dict]: Collect and process data from the Zarr datasets.
#         to_hf_dataset(data_dict: Dict) -> Dataset: Convert processed data into a Hugging Face dataset format.
#     """

#     def __init__(self, folder_path: str, fps: int | None = None):
#         self.zarr_path = folder_path
#         if self.fps is None:
#             fps = 15
#         self.fps = fps
#         register_codecs()

#     @property
#     def fps(self) -> int:
#         return self.fps

#     def is_valid(self) -> bool:
#         """
#         Validates the Zarr folder to ensure it contains all required datasets with consistent frame counts.

#         Returns:
#             bool: True if all required datasets are present and have consistent frame counts, False otherwise.
#         """
#         # Check if the Zarr folder is valid
#         try:
#             zarr_data = zarr.open(self.zarr_path, mode="r")
#         except Exception:
#             # TODO (azouitine): Handle the exception properly
#             return False
#         required_datasets = {
#             "data/robot0_demo_end_pose",
#             "data/robot0_demo_start_pose",
#             "data/robot0_eef_pos",
#             "data/robot0_eef_rot_axis_angle",
#             "data/robot0_gripper_width",
#             "meta/episode_ends",
#             "data/camera0_rgb",
#         }
#         for dataset in required_datasets:
#             if dataset not in zarr_data:
#                 return False
#         nb_frames = zarr_data["data/camera0_rgb"].shape[0]

#         required_datasets.remove("meta/episode_ends")

#         return all(nb_frames == zarr_data[dataset].shape[0] for dataset in required_datasets)

#     def preprocess(self):
#         """
#         Collects and processes all episodes from the Zarr dataset into structured data dictionaries.

#         Returns:
#             Tuple[Dict, Dict]: A tuple containing the structured episode data and episode index mappings.
#         """
#         zarr_data = zarr.open(self.zarr_path, mode="r")

#         # We process the image data separately because it is too large to fit in memory
#         end_pose = torch.from_numpy(zarr_data["data/robot0_demo_end_pose"][:])
#         start_pos = torch.from_numpy(zarr_data["data/robot0_demo_start_pose"][:])
#         eff_pos = torch.from_numpy(zarr_data["data/robot0_eef_pos"][:])
#         eff_rot_axis_angle = torch.from_numpy(zarr_data["data/robot0_eef_rot_axis_angle"][:])
#         gripper_width = torch.from_numpy(zarr_data["data/robot0_gripper_width"][:])

#         states_pos = torch.cat([eff_pos, eff_rot_axis_angle], dim=1)
#         states = torch.cat([states_pos, gripper_width], dim=1)

#         episode_ends = zarr_data["meta/episode_ends"][:]
#         num_episodes: int = episode_ends.shape[0]

#         episode_ids = torch.from_numpy(self.get_episode_idxs(episode_ends))

#         # We convert it in torch tensor later because the jit function does not support torch tensors
#         episode_ends = torch.from_numpy(episode_ends)

#         ep_dicts = []
#         episode_data_index = {"from": [], "to": []}
#         id_from = 0

#         for episode_id in tqdm.tqdm(range(num_episodes)):
#             id_to = episode_ends[episode_id]

#             num_frames = id_to - id_from

#             assert (
#                 episode_ids[id_from:id_to] == episode_id
#             ).all(), f"episode_ids[{id_from}:{id_to}] != {episode_id}"

#             state = states[id_from:id_to]
#             ep_dict = {
#                 # observation.image will be filled later
#                 "observation.state": state,
#                 "episode_index": torch.tensor([episode_id] * num_frames, dtype=torch.int),
#                 "frame_index": torch.arange(0, num_frames, 1),
#                 "timestamp": torch.arange(0, num_frames, 1) / self.fps,
#                 "episode_data_index_from": torch.tensor([id_from] * num_frames),
#                 "episode_data_index_to": torch.tensor([id_from + num_frames] * num_frames),
#                 "end_pose": end_pose[id_from:id_to],
#                 "start_pos": start_pos[id_from:id_to],
#                 "gripper_width": gripper_width[id_from:id_to],
#             }
#             ep_dicts.append(ep_dict)
#             episode_data_index["from"].append(id_from)
#             episode_data_index["to"].append(id_from + num_frames)
#             id_from += num_frames

#         data_dict = concatenate_episodes(ep_dicts)

#         total_frames = id_from
#         data_dict["index"] = torch.arange(0, total_frames, 1)

#         print("Saving images to disk in temporary folder...")
#         # datasets.Image() can take a list of paths to images, so we save the images to a temporary folder
#         # to avoid loading them all in memory
#         _save_images_concurrently(
#             data=zarr_data, image_key="data/camera0_rgb", folder_path="tmp_umi_images", max_workers=12
#         )
#         print("Saving images to disk in temporary folder... Done")

#         # Sort files by number eg. 1.png, 2.png, 3.png, 9.png, 10.png instead of 1.png, 10.png, 2.png, 3.png, 9.png
#         # to correctly match the images with the data
#         images_path = sorted(
#             glob("tmp_umi_images/*"), key=lambda x: int(re.search(r"(\d+)\.png$", x).group(1))
#         )
#         data_dict["observation.image"] = images_path
#         print("Images saved to disk, do not forget to delete the folder tmp_umi_images/")

#         # Cleanup
#         return data_dict, episode_data_index

#     def to_hf_dataset(self, data_dict):
#         """
#         Converts the processed data dictionary into a Hugging Face dataset with defined features.

#         Args:
#             data_dict (Dict): The data dictionary containing tensors and episode information.

#         Returns:
#             Dataset: A Hugging Face dataset constructed from the provided data dictionary.
#         """
#         features = {
#             "observation.image": Image(),
#             "observation.state": Sequence(
#                 length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
#             ),
#             "episode_index": Value(dtype="int64", id=None),
#             "frame_index": Value(dtype="int64", id=None),
#             "timestamp": Value(dtype="float32", id=None),
#             "index": Value(dtype="int64", id=None),
#             "episode_data_index_from": Value(dtype="int64", id=None),
#             "episode_data_index_to": Value(dtype="int64", id=None),
#             "end_pose": Sequence(
#                 length=data_dict["end_pose"].shape[1], feature=Value(dtype="float32", id=None)
#             ),
#             "start_pos": Sequence(
#                 length=data_dict["start_pos"].shape[1], feature=Value(dtype="float32", id=None)
#             ),
#             "gripper_width": Sequence(
#                 length=data_dict["gripper_width"].shape[1], feature=Value(dtype="float32", id=None)
#             ),
#         }
#         features = Features(features)
#         hf_dataset = Dataset.from_dict(data_dict, features=features)
#         hf_dataset.set_transform(hf_transform_to_torch)

#         return hf_dataset

#     def cleanup(self):
#         # Cleanup
#         if os.path.exists("tmp_umi_images"):
#             print("Removing temporary images folder")
#             shutil.rmtree("tmp_umi_images")
#             print("Cleanup done")

#     @classmethod
#     def get_episode_idxs(cls, episode_ends: np.ndarray) -> np.ndarray:
#         # Optimized and simplified version of this function: https://github.com/real-stanford/universal_manipulation_interface/blob/298776ce251f33b6b3185a98d6e7d1f9ad49168b/diffusion_policy/common/replay_buffer.py#L374
#         from numba import jit

#         @jit(nopython=True)
#         def _get_episode_idxs(episode_ends):
#             result = np.zeros((episode_ends[-1],), dtype=np.int64)
#             start_idx = 0
#             for episode_number, end_idx in enumerate(episode_ends):
#                 result[start_idx:end_idx] = episode_number
#                 start_idx = end_idx
#             return result

#         return _get_episode_idxs(episode_ends)


# class XarmProcessor:
#     def __init__(self, folder_path: str, fps: int | None = None):
#         self.folder_path = Path(folder_path)
#         self.keys = {"actions", "rewards", "dones", "masks"}
#         self.nested_keys = {"observations": {"rgb", "state"}, "next_observations": {"rgb", "state"}}
#         if fps is None:
#             fps = 15
#         self.fps = fps

#     @property
#     def fps(self) -> int:
#         return self.fps

#     def is_valid(self) -> bool:
#         # get all .pkl files
#         xarm_files = list(self.folder_path.glob("*.pkl"))
#         if len(xarm_files) != 1:
#             return False

#         try:
#             with open(xarm_files[0], "rb") as f:
#                 dataset_dict = pickle.load(f)
#         except Exception:
#             return False

#         if not isinstance(dataset_dict, dict):
#             return False

#         if not all(k in dataset_dict for k in self.keys):
#             return False

#         # Check for consistent lengths in nested keys
#         try:
#             expected_len = len(dataset_dict["actions"])
#             if any(len(dataset_dict[key]) != expected_len for key in self.keys if key in dataset_dict):
#                 return False

#             for key, subkeys in self.nested_keys.items():
#                 nested_dict = dataset_dict.get(key, {})
#                 if any(
#                     len(nested_dict[subkey]) != expected_len for subkey in subkeys if subkey in nested_dict
#                 ):
#                     return False
#         except KeyError:  # If any expected key or subkey is missing
#             return False

#         return True  # All checks passed

#     def preprocess(self):
#         if not self.is_valid():
#             raise ValueError("The Xarm file is invalid or does not contain the required datasets.")

#         xarm_files = list(self.folder_path.glob("*.pkl"))

#         with open(xarm_files[0], "rb") as f:
#             dataset_dict = pickle.load(f)
#         ep_dicts = []
#         episode_data_index = {"from": [], "to": []}

#         id_from = 0
#         id_to = 0
#         episode_id = 0
#         total_frames = dataset_dict["actions"].shape[0]
#         for i in tqdm.tqdm(range(total_frames)):
#             id_to += 1

#             if not dataset_dict["dones"][i]:
#                 continue

#             num_frames = id_to - id_from

#             image = torch.tensor(dataset_dict["observations"]["rgb"][id_from:id_to])
#             image = einops.rearrange(image, "b c h w -> b h w c")
#             state = torch.tensor(dataset_dict["observations"]["state"][id_from:id_to])
#             action = torch.tensor(dataset_dict["actions"][id_from:id_to])
#             # TODO(rcadene): we have a missing last frame which is the observation when the env is done
#             # it is critical to have this frame for tdmpc to predict a "done observation/state"
#             # next_image = torch.tensor(dataset_dict["next_observations"]["rgb"][id_from:id_to])
#             # next_state = torch.tensor(dataset_dict["next_observations"]["state"][id_from:id_to])
#             next_reward = torch.tensor(dataset_dict["rewards"][id_from:id_to])
#             next_done = torch.tensor(dataset_dict["dones"][id_from:id_to])

#             ep_dict = {
#                 "observation.image": [PILImage.fromarray(x.numpy()) for x in image],
#                 "observation.state": state,
#                 "action": action,
#                 "episode_index": torch.tensor([episode_id] * num_frames, dtype=torch.int),
#                 "frame_index": torch.arange(0, num_frames, 1),
#                 "timestamp": torch.arange(0, num_frames, 1) / self.fps,
#                 # "next.observation.image": next_image,
#                 # "next.observation.state": next_state,
#                 "next.reward": next_reward,
#                 "next.done": next_done,
#             }
#             ep_dicts.append(ep_dict)

#             episode_data_index["from"].append(id_from)
#             episode_data_index["to"].append(id_from + num_frames)

#             id_from = id_to
#             episode_id += 1

#         data_dict = concatenate_episodes(ep_dicts)
#         return data_dict, episode_data_index

#     def to_hf_dataset(self, data_dict):
#         features = {
#             "observation.image": Image(),
#             "observation.state": Sequence(
#                 length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
#             ),
#             "action": Sequence(length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)),
#             "episode_index": Value(dtype="int64", id=None),
#             "frame_index": Value(dtype="int64", id=None),
#             "timestamp": Value(dtype="float32", id=None),
#             "next.reward": Value(dtype="float32", id=None),
#             "next.done": Value(dtype="bool", id=None),
#             #'next.success': Value(dtype='bool', id=None),
#             "index": Value(dtype="int64", id=None),
#         }
#         features = Features(features)
#         hf_dataset = Dataset.from_dict(data_dict, features=features)
#         hf_dataset.set_transform(hf_transform_to_torch)

#         return hf_dataset

#     def cleanup(self):
#         pass


# class PushTProcessor:
#     def __init__(self, folder_path: Path, fps: int | None = None):
#         self.zarr_path = folder_path
#         if fps is None:
#             fps = 10
#         self.fps = fps

#     @property
#     def fps(self) -> int:
#         return self.fps

#     def is_valid(self):
#         try:
#             zarr_data = zarr.open(self.zarr_path, mode="r")
#         except Exception:
#             # TODO (azouitine): Handle the exception properly
#             return False
#         required_datasets = {
#             "data/action",
#             "data/img",
#             "data/keypoint",
#             "data/n_contacts",
#             "data/state",
#             "meta/episode_ends",
#         }
#         for dataset in required_datasets:
#             if dataset not in zarr_data:
#                 return False
#         nb_frames = zarr_data["data/img"].shape[0]

#         required_datasets.remove("meta/episode_ends")

#         return all(nb_frames == zarr_data[dataset].shape[0] for dataset in required_datasets)

#     def preprocess(self):
#         try:
#             import pymunk
#             from gym_pusht.envs.pusht import PushTEnv, pymunk_to_shapely

#             from lerobot.common.datasets.push_dataset_to_hub._diffusion_policy_replay_buffer import (
#                 ReplayBuffer as DiffusionPolicyReplayBuffer,
#             )
#         except ModuleNotFoundError as e:
#             print("`gym_pusht` is not installed. Please install it with `pip install 'lerobot[gym_pusht]'`")
#             raise e

#         # as define in env
#         success_threshold = 0.95  # 95% coverage,

#         dataset_dict = DiffusionPolicyReplayBuffer.copy_from_path(
#             self.zarr_path
#         )  # , keys=['img', 'state', 'action'])

#         episode_ids = torch.from_numpy(dataset_dict.get_episode_idxs())
#         num_episodes = dataset_dict.meta["episode_ends"].shape[0]
#         assert len(
#             {dataset_dict[key].shape[0] for key in dataset_dict.keys()}  # noqa: SIM118
#         ), "Some data type dont have the same number of total frames."

#         # TODO: verify that goal pose is expected to be fixed
#         goal_pos_angle = np.array([256, 256, np.pi / 4])  # x, y, theta (in radians)
#         goal_body = PushTEnv.get_goal_pose_body(goal_pos_angle)

#         imgs = torch.from_numpy(dataset_dict["img"])  # b h w c
#         states = torch.from_numpy(dataset_dict["state"])
#         actions = torch.from_numpy(dataset_dict["action"])

#         ep_dicts = []
#         episode_data_index = {"from": [], "to": []}

#         id_from = 0
#         for episode_id in tqdm.tqdm(range(num_episodes)):
#             id_to = dataset_dict.meta["episode_ends"][episode_id]

#             num_frames = id_to - id_from

#             assert (episode_ids[id_from:id_to] == episode_id).all()

#             image = imgs[id_from:id_to]
#             assert image.min() >= 0.0
#             assert image.max() <= 255.0
#             image = image.type(torch.uint8)

#             state = states[id_from:id_to]
#             agent_pos = state[:, :2]
#             block_pos = state[:, 2:4]
#             block_angle = state[:, 4]

#             reward = torch.zeros(num_frames)
#             success = torch.zeros(num_frames, dtype=torch.bool)
#             done = torch.zeros(num_frames, dtype=torch.bool)
#             for i in range(num_frames):
#                 space = pymunk.Space()
#                 space.gravity = 0, 0
#                 space.damping = 0

#                 # Add walls.
#                 walls = [
#                     PushTEnv.add_segment(space, (5, 506), (5, 5), 2),
#                     PushTEnv.add_segment(space, (5, 5), (506, 5), 2),
#                     PushTEnv.add_segment(space, (506, 5), (506, 506), 2),
#                     PushTEnv.add_segment(space, (5, 506), (506, 506), 2),
#                 ]
#                 space.add(*walls)

#                 block_body = PushTEnv.add_tee(space, block_pos[i].tolist(), block_angle[i].item())
#                 goal_geom = pymunk_to_shapely(goal_body, block_body.shapes)
#                 block_geom = pymunk_to_shapely(block_body, block_body.shapes)
#                 intersection_area = goal_geom.intersection(block_geom).area
#                 goal_area = goal_geom.area
#                 coverage = intersection_area / goal_area
#                 reward[i] = np.clip(coverage / success_threshold, 0, 1)
#                 success[i] = coverage > success_threshold

#             # last step of demonstration is considered done
#             done[-1] = True

#             ep_dict = {
#                 "observation.image": [PILImage.fromarray(x.numpy()) for x in image],
#                 "observation.state": agent_pos,
#                 "action": actions[id_from:id_to],
#                 "episode_index": torch.tensor([episode_id] * num_frames, dtype=torch.int),
#                 "frame_index": torch.arange(0, num_frames, 1),
#                 "timestamp": torch.arange(0, num_frames, 1) / self.fps,
#                 # "next.observation.image": image[1:],
#                 # "next.observation.state": agent_pos[1:],
#                 # TODO(rcadene): verify that reward and done are aligned with image and agent_pos
#                 "next.reward": torch.cat([reward[1:], reward[[-1]]]),
#                 "next.done": torch.cat([done[1:], done[[-1]]]),
#                 "next.success": torch.cat([success[1:], success[[-1]]]),
#             }
#             ep_dicts.append(ep_dict)

#             episode_data_index["from"].append(id_from)
#             episode_data_index["to"].append(id_from + num_frames)

#             id_from += num_frames

#         data_dict = concatenate_episodes(ep_dicts)
#         return data_dict, episode_data_index

#     def to_hf_dataset(self, data_dict, episode_data_index):
#         features = {
#             "observation.image": Image(),
#             "observation.state": Sequence(
#                 length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
#             ),
#             "action": Sequence(length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)),
#             "episode_index": Value(dtype="int64", id=None),
#             "frame_index": Value(dtype="int64", id=None),
#             "timestamp": Value(dtype="float32", id=None),
#             "next.reward": Value(dtype="float32", id=None),
#             "next.done": Value(dtype="bool", id=None),
#             "next.success": Value(dtype="bool", id=None),
#             "index": Value(dtype="int64", id=None),
#         }
#         features = Features(features)
#         hf_dataset = Dataset.from_dict(data_dict, features=features)
#         hf_dataset.set_transform(hf_transform_to_torch)
#         return hf_dataset

#     def cleanup(self):
#         pass


# def concatenate_episodes(ep_dicts):
#     data_dict = {}

#     keys = ep_dicts[0].keys()
#     for key in keys:
#         if torch.is_tensor(ep_dicts[0][key][0]):
#             data_dict[key] = torch.cat([ep_dict[key] for ep_dict in ep_dicts])
#         else:
#             if key not in data_dict:
#                 data_dict[key] = []
#             for ep_dict in ep_dicts:
#                 for x in ep_dict[key]:
#                     data_dict[key].append(x)

#     total_frames = data_dict["frame_index"].shape[0]
#     data_dict["index"] = torch.arange(0, total_frames, 1)
#     return data_dict


# def _clear_folder(folder_path: str):
#     """
#     Clears all the content of the specified folder. Creates the folder if it does not exist.

#     Args:
#     folder_path (str): Path to the folder to clear.

#     Examples:
#     >>> import os
#     >>> os.makedirs('example_folder', exist_ok=True)
#     >>> with open('example_folder/temp_file.txt', 'w') as f:
#     ...     f.write('example')
#     >>> clear_folder('example_folder')
#     >>> os.listdir('example_folder')
#     []
#     """
#     if os.path.exists(folder_path):
#         for filename in os.listdir(folder_path):
#             file_path = os.path.join(folder_path, filename)
#             try:
#                 if os.path.isfile(file_path) or os.path.islink(file_path):
#                     os.unlink(file_path)
#                 elif os.path.isdir(file_path):
#                     shutil.rmtree(file_path)
#             except Exception as e:
#                 print(f"Failed to delete {file_path}. Reason: {e}")
#     else:
#         os.makedirs(folder_path)


# def _save_image(img_array: np.array, i: int, folder_path: str):
#     """
#     Saves a single image to the specified folder.

#     Args:
#     img_array (ndarray): The numpy array of the image.
#     i (int): Index of the image, used for naming.
#     folder_path (str): Path to the folder where the image will be saved.
#     """
#     img = PILImage.fromarray(img_array)
#     img_format = "PNG" if img_array.dtype == np.uint8 else "JPEG"
#     img.save(os.path.join(folder_path, f"{i}.{img_format.lower()}"), quality=100)


# def _save_images_concurrently(data: dict, image_key: str, folder_path: str, max_workers: int = 4):
#     from concurrent.futures import ThreadPoolExecutor

#     """
#     Saves images from the zarr_data to the specified folder using multithreading.

#     Args:
#     zarr_data (dict): A dictionary containing image data in an array format.
#     folder_path (str): Path to the folder where images will be saved.
#     max_workers (int): The maximum number of threads to use for saving images.
#     """
#     num_images = len(data["data/camera0_rgb"])
#     _clear_folder(folder_path)  # Clear or create folder first

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         [executor.submit(_save_image, data[image_key][i], i, folder_path) for i in range(num_images)]

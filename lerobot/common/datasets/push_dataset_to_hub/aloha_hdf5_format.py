"""
Contains utilities to process raw data format of HDF5 files like in: https://github.com/tonyzhaozh/act
"""

import re
import shutil
from pathlib import Path

import h5py
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)

# TODO(rcadene): enable for PR video dataset
# from lerobot.common.datasets.video_utils import encode_video_frames


def check_format(raw_dir) -> bool:
    cameras = ["top"]

    hdf5_files: list[Path] = list(raw_dir.glob("episode_*.hdf5"))
    assert len(hdf5_files) != 0
    hdf5_files = sorted(hdf5_files, key=lambda x: int(re.search(r"episode_(\d+).hdf5", x.name).group(1)))

    # Check if the sequence is consecutive eg episode_0, episode_1, episode_2, etc.
    previous_number = None
    for file in hdf5_files:
        current_number = int(re.search(r"episode_(\d+).hdf5", file.name).group(1))
        if previous_number is not None:
            assert current_number - previous_number == 1
        previous_number = current_number

    for file in hdf5_files:
        with h5py.File(file, "r") as file:
            # Check for the expected datasets within the HDF5 file
            required_datasets = ["/action", "/observations/qpos"]
            # Add camera-specific image datasets to the required datasets
            camera_datasets = [f"/observations/images/{cam}" for cam in cameras]
            required_datasets.extend(camera_datasets)

            assert all(dataset in file for dataset in required_datasets)


def load_from_raw(raw_dir, out_dir, fps, video, debug):
    hdf5_files = list(raw_dir.glob("*.hdf5"))
    hdf5_files = sorted(hdf5_files, key=lambda x: int(re.search(r"episode_(\d+)", x.name).group(1)))
    ep_dicts = []
    episode_data_index = {"from": [], "to": []}

    id_from = 0

    for ep_path in tqdm.tqdm(hdf5_files):
        with h5py.File(ep_path, "r") as ep:
            ep_idx = int(re.search(r"episode_(\d+)", ep_path.name).group(1))
            num_frames = ep["/action"].shape[0]

            # last step of demonstration is considered done
            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True

            state = torch.from_numpy(ep["/observations/qpos"][:])
            action = torch.from_numpy(ep["/action"][:])

            ep_dict = {}

            cameras = list(ep["/observations/images"].keys())
            for cam in cameras:
                img_key = f"observation.images.{cam}"
                imgs_array = ep[f"/observations/images/{cam}"][:]  # b h w c
                if video:
                    # save png images in temporary directory
                    tmp_imgs_dir = out_dir / "tmp_images"
                    save_images_concurrently(imgs_array, tmp_imgs_dir)

                    # encode images to a mp4 video
                    video_path = out_dir / "videos" / f"{img_key}_episode_{ep_idx:06d}.mp4"
                    encode_video_frames(tmp_imgs_dir, video_path, fps)  # noqa: F821

                    # clean temporary images directory
                    shutil.rmtree(tmp_imgs_dir)

                    # store the episode idx
                    ep_dict[img_key] = torch.tensor([ep_idx] * num_frames, dtype=torch.int)
                else:
                    ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

            ep_dict["observation.state"] = state
            ep_dict["action"] = action
            ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
            ep_dict["next.done"] = done
            # TODO(rcadene): compute reward and success
            # ep_dict[""next.reward"] = reward
            # ep_dict[""next.success"] = success

            assert isinstance(ep_idx, int)
            ep_dicts.append(ep_dict)

            episode_data_index["from"].append(id_from)
            episode_data_index["to"].append(id_from + num_frames)

        id_from += num_frames

        # process first episode only
        if debug:
            break

    data_dict = concatenate_episodes(ep_dicts)
    return data_dict, episode_data_index


def to_hf_dataset(data_dict, video) -> Dataset:
    features = {}

    image_keys = [key for key in data_dict if "observation.images." in key]
    for image_key in image_keys:
        if video:
            features[image_key] = Value(dtype="int64", id="video")
        else:
            features[image_key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)
    # TODO(rcadene): add reward and success
    # features["next.reward"] = Value(dtype="float32", id=None)
    # features["next.success"] = Value(dtype="bool", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(raw_dir: Path, out_dir: Path, fps=None, video=True, debug=False):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 50

    data_dir, episode_data_index = load_from_raw(raw_dir, out_dir, fps, video, debug)
    hf_dataset = to_hf_dataset(data_dir, video)

    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info

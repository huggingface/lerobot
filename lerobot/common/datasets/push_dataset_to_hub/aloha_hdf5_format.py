#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains utilities to process raw data format of HDF5 files like in: https://github.com/tonyzhaozh/act
"""

import gc
import os
import shutil
from pathlib import Path
import pickle
import tempfile

import h5py
import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    get_default_encoding,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames


def get_cameras(hdf5_data):
    # ignore depth channel, not currently handled
    # TODO(rcadene): add depth
    rgb_cameras = [key for key in hdf5_data["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118
    return rgb_cameras


def check_format(raw_dir) -> bool:
    # only frames from simulation are uncompressed
    compressed_images = "sim" not in raw_dir.name
    print(f"Checking format of {raw_dir} with compressed images: {compressed_images}")

    hdf5_paths = sorted(list(raw_dir.rglob("episode_*.hdf5")))
    assert len(hdf5_paths) != 0
    for hdf5_path in hdf5_paths:
        print(f"Checking {hdf5_path}")
        with h5py.File(hdf5_path, "r") as data:
            assert "/action" in data
            assert "/observations/qpos" in data

            assert data["/action"].ndim == 2
            assert data["/observations/qpos"].ndim == 2

            num_frames = data["/action"].shape[0]
            assert num_frames == data["/observations/qpos"].shape[0]

            for camera in get_cameras(data):
                assert num_frames == data[f"/observations/images/{camera}"].shape[0]

                if compressed_images:
                    assert data[f"/observations/images/{camera}"].ndim == 2
                else:
                    assert data[f"/observations/images/{camera}"].ndim == 4
                    b, h, w, c = data[f"/observations/images/{camera}"].shape
                    assert c < h and c < w, f"Expect (h,w,c) image format but ({h=},{w=},{c=}) provided."


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
    chunk_size: int = 100, # episodes
):
    # Only frames from simulation are uncompressed
    compressed_images = "sim" not in raw_dir.name

    hdf5_files = sorted(raw_dir.rglob("episode_*.hdf5"))
    num_episodes = len(hdf5_files)

    print("Found", num_episodes, "episodes")

    ep_ids = episodes if episodes else range(num_episodes)
    pickle_file_names = []
    features = None  # Will define features based on the first episode
    chunk_ep_dicts = []  # Accumulator for episodes in the current chunk
    chunk_count = 0
    global_index = 0  # Initialize global index

    for idx, ep_idx in enumerate(tqdm.tqdm(ep_ids)):
        ep_path = hdf5_files[ep_idx]
        with h5py.File(ep_path, "r") as ep:
            num_frames = ep["/action"].shape[0]

            # last step of demonstration is considered done
            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True

            state = torch.from_numpy(ep["/observations/qpos"][:])
            action = torch.from_numpy(ep["/action"][:])
            if "/observations/qvel" in ep:
                velocity = torch.from_numpy(ep["/observations/qvel"][:])
            if "/observations/effort" in ep:
                effort = torch.from_numpy(ep["/observations/effort"][:])

            ep_dict = {}

            for camera in get_cameras(ep):
                img_key = f"observation.images.{camera}"

                if compressed_images:
                    import cv2

                    # load one compressed image after the other in RAM and uncompress
                    imgs_array = []
                    for data in ep[f"/observations/images/{camera}"]:
                        imgs_array.append(cv2.imdecode(data, 1))
                    imgs_array = np.array(imgs_array)

                else:
                    # load all images in RAM
                    imgs_array = ep[f"/observations/images/{camera}"][:]

                if video:
                    # save png images in temporary directory
                    tmp_imgs_dir = videos_dir / "tmp_images"
                    save_images_concurrently(imgs_array, tmp_imgs_dir)

                    # encode images to a mp4 video
                    fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
                    video_path = videos_dir / fname
                    encode_video_frames(tmp_imgs_dir, video_path, fps, **(encoding or {}))

                    # clean temporary images directory
                    shutil.rmtree(tmp_imgs_dir)

                    # store the reference to the video frame
                    ep_dict[img_key] = [
                        {"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)
                    ]
                else:
                    ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

            # Convert tensors to lists for pickling
            ep_dict["observation.state"] = state.numpy().tolist()
            if "/observations/velocity" in ep:
                ep_dict["observation.velocity"] = velocity.numpy().tolist()
            if "/observations/effort" in ep:
                ep_dict["observation.effort"] = effort.numpy().tolist()
            ep_dict["action"] = action.numpy().tolist()
            ep_dict["episode_index"] = [int(ep_idx)] * num_frames
            ep_dict["frame_index"] = list(range(num_frames))
            ep_dict["timestamp"] = (np.arange(0, num_frames, 1) / fps).tolist()
            ep_dict["next.done"] = done.tolist()
            # We no longer set 'index' here; it will be set in concatenate_episodes

            # Accumulate episodes in the current chunk
            chunk_ep_dicts.append(ep_dict)
            chunk_count += 1

            # Define features based on the first episode
            if idx == 0:
                features = {}
                keys = [key for key in ep_dict if "observation.images." in key]
                for key in keys:
                    if video:
                        features[key] = VideoFrame()
                    else:
                        features[key] = Image()

                features["observation.state"] = Sequence(
                    feature=Value(dtype="float32"), length=len(ep_dict["observation.state"][0])
                )
                if "observation.velocity" in ep_dict:
                    features["observation.velocity"] = Sequence(
                        feature=Value(dtype="float32"), length=len(ep_dict["observation.velocity"][0])
                    )
                if "observation.effort" in ep_dict:
                    features["observation.effort"] = Sequence(
                        feature=Value(dtype="float32"), length=len(ep_dict["observation.effort"][0])
                    )
                features["action"] = Sequence(
                    feature=Value(dtype="float32"), length=len(ep_dict["action"][0])
                )
                features["episode_index"] = Value(dtype="int64")
                features["frame_index"] = Value(dtype="int64")
                features["timestamp"] = Value(dtype="float32")
                features["next.done"] = Value(dtype="bool")
                features["index"] = Value(dtype="int64")

            # When chunk_size is reached, save the chunk to a pickle file
            if chunk_count >= chunk_size:
                # Concatenate episodes in the chunk, passing the global_index
                chunk_data_dict = concatenate_episodes(chunk_ep_dicts, starting_index=global_index)
                # Update the global_index
                global_index += len(chunk_data_dict["index"])
                # Save chunk_data_dict to a temporary pickle file
                with tempfile.NamedTemporaryFile('wb', delete=False, suffix='.pkl') as tmp_file:
                    pickle.dump(chunk_data_dict, tmp_file)
                    pickle_file_name = tmp_file.name
                pickle_file_names.append(pickle_file_name)
                # Reset the chunk
                chunk_ep_dicts = []
                chunk_count = 0

            gc.collect()

    # Save any remaining episodes in the last chunk
    if chunk_ep_dicts:
        chunk_data_dict = concatenate_episodes(chunk_ep_dicts, starting_index=global_index)
        global_index += len(chunk_data_dict["index"])
        with tempfile.NamedTemporaryFile('wb', delete=False, suffix='.pkl') as tmp_file:
            pickle.dump(chunk_data_dict, tmp_file)
            pickle_file_name = tmp_file.name
        pickle_file_names.append(pickle_file_name)

    return pickle_file_names, features


def to_hf_dataset(pickle_file_names, features) -> Dataset:
    from datasets import Dataset, Features

    def generator():
        for pickle_file_name in pickle_file_names:
            with open(pickle_file_name, "rb") as f:
                chunk_data_dict = pickle.load(f)
                num_examples = len(chunk_data_dict["index"])
                for idx in range(num_examples):
                    example = {
                        key: value[idx] for key, value in chunk_data_dict.items()
                    }
                    yield example

            # After processing, delete the temporary file
            os.remove(pickle_file_name)

    # Create the dataset using Dataset.from_generator
    hf_dataset = Dataset.from_generator(generator, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 50

    pickle_file_names, features = load_from_raw(
        raw_dir, videos_dir, fps, video, episodes, encoding
    )
    hf_dataset = to_hf_dataset(pickle_file_names, features)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    return hf_dataset, episode_data_index, info

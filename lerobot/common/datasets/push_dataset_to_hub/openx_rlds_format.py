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
For all datasets in the RLDS format.
For https://github.com/google-deepmind/open_x_embodiment (OPENX) datasets.

NOTE: You need to install tensorflow and tensorflow_datsets before running this script.

Example:
    python lerobot/scripts/push_dataset_to_hub.py \
        --raw-dir /path/to/data/bridge_dataset/1.0.0/ \
        --repo-id your_hub/sampled_bridge_data_v2 \
        --raw-format rlds \
        --episodes 3 4 5 8 9

Exact dataset fps defined in openx/config.py, obtained from:
    https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit?gid=0#gid=0&range=R:R
"""

import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    calculate_episode_data_index,
    concatenate_episodes,
    get_default_encoding,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames

np.set_printoptions(precision=2)


def tf_to_torch(data):
    return torch.from_numpy(data.numpy())


def tf_img_convert(img):
    if img.dtype == tf.string:
        img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    elif img.dtype != tf.uint8:
        raise ValueError(f"Unsupported image dtype: found with dtype {img.dtype}")
    return img.numpy()


def _broadcast_metadata_rlds(i: tf.Tensor, traj: dict) -> dict:
    """
    In the RLDS format, each trajectory has some top-level metadata that is explicitly separated out, and a "steps"
    entry. This function moves the "steps" entry to the top level, broadcasting any metadata to the length of the
    trajectory. This function also adds the extra metadata fields `_len`, `_traj_index`, and `_frame_index`.

    NOTE: adapted from DLimp library https://github.com/kvablack/dlimp/
    """
    steps = traj.pop("steps")

    traj_len = tf.shape(tf.nest.flatten(steps)[0])[0]

    # broadcast metadata to the length of the trajectory
    metadata = tf.nest.map_structure(lambda x: tf.repeat(x, traj_len), traj)

    # put steps back in
    assert "traj_metadata" not in steps
    traj = {**steps, "traj_metadata": metadata}

    assert "_len" not in traj
    assert "_traj_index" not in traj
    assert "_frame_index" not in traj
    traj["_len"] = tf.repeat(traj_len, traj_len)
    traj["_traj_index"] = tf.repeat(i, traj_len)
    traj["_frame_index"] = tf.range(traj_len)

    return traj


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    """
    Args:
        raw_dir (Path): _description_
        videos_dir (Path): _description_
        fps (int): _description_
        video (bool): _description_
        episodes (list[int] | None, optional): _description_. Defaults to None.
    """
    ds_builder = tfds.builder_from_directory(str(raw_dir))
    dataset = ds_builder.as_dataset(
        split="all",
        decoders={"steps": tfds.decode.SkipDecoding()},
    )

    dataset_info = ds_builder.info
    print("dataset_info: ", dataset_info)

    ds_length = len(dataset)
    dataset = dataset.take(ds_length)
    # "flatten" the dataset as such we can apply trajectory level map() easily
    # each [obs][key] has a shape of (frame_size, ...)
    dataset = dataset.enumerate().map(_broadcast_metadata_rlds)

    # we will apply the standardization transform if the dataset_name is provided
    # if the dataset name is not provided and the goal is to convert any rlds formatted dataset
    # search for 'image' keys in the observations
    image_keys = []
    state_keys = []
    observation_info = dataset_info.features["steps"]["observation"]
    for key in observation_info:
        # check whether the key is for an image or a vector observation
        if len(observation_info[key].shape) == 3:
            # only adding uint8 images discards depth images
            if observation_info[key].dtype == tf.uint8:
                image_keys.append(key)
        else:
            state_keys.append(key)

    lang_key = "language_instruction" if "language_instruction" in dataset.element_spec else None

    print(" - image_keys: ", image_keys)
    print(" - lang_key: ", lang_key)

    it = iter(dataset)

    ep_dicts = []
    # Init temp path to save ep_dicts in case of crash
    tmp_ep_dicts_dir = videos_dir.parent.joinpath("ep_dicts")
    tmp_ep_dicts_dir.mkdir(parents=True, exist_ok=True)

    # check if ep_dicts have already been saved in /tmp
    starting_ep_idx = 0
    saved_ep_dicts = [ep.__str__() for ep in tmp_ep_dicts_dir.iterdir()]
    if len(saved_ep_dicts) > 0:
        saved_ep_dicts.sort()
        # get last ep_idx number
        starting_ep_idx = int(saved_ep_dicts[-1][-13:-3]) + 1
        for i in range(starting_ep_idx):
            episode = next(it)
            ep_dicts.append(torch.load(saved_ep_dicts[i]))

    # if we user specified episodes, skip the ones not in the list
    if episodes is not None:
        if ds_length == 0:
            raise ValueError("No episodes found.")
        # convert episodes index to sorted list
        episodes = sorted(episodes)

    for ep_idx in tqdm.tqdm(range(starting_ep_idx, ds_length)):
        episode = next(it)

        # if user specified episodes, skip the ones not in the list
        if episodes is not None:
            if len(episodes) == 0:
                break
            if ep_idx == episodes[0]:
                # process this episode
                print(" selecting episode idx: ", ep_idx)
                episodes.pop(0)
            else:
                continue  # skip

        num_frames = episode["action"].shape[0]

        ep_dict = {}
        for key in state_keys:
            ep_dict[f"observation.{key}"] = tf_to_torch(episode["observation"][key])

        ep_dict["action"] = tf_to_torch(episode["action"])
        ep_dict["next.reward"] = tf_to_torch(episode["reward"]).float()
        ep_dict["next.done"] = tf_to_torch(episode["is_last"])
        ep_dict["is_terminal"] = tf_to_torch(episode["is_terminal"])
        ep_dict["is_first"] = tf_to_torch(episode["is_first"])
        ep_dict["discount"] = tf_to_torch(episode["discount"])

        # If lang_key is present, convert the entire tensor at once
        if lang_key is not None:
            ep_dict["language_instruction"] = [x.numpy().decode("utf-8") for x in episode[lang_key]]

        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)

        image_array_dict = {key: [] for key in image_keys}

        for im_key in image_keys:
            imgs = episode["observation"][im_key]
            image_array_dict[im_key] = [tf_img_convert(img) for img in imgs]

        # loop through all cameras
        for im_key in image_keys:
            img_key = f"observation.images.{im_key}"
            imgs_array = image_array_dict[im_key]
            imgs_array = np.array(imgs_array)
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

        path_ep_dict = tmp_ep_dicts_dir.joinpath(
            "ep_dict_" + "0" * (10 - len(str(ep_idx))) + str(ep_idx) + ".pt"
        )
        torch.save(ep_dict, path_ep_dict)

        ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video) -> Dataset:
    features = {}

    for key in data_dict:
        # check if vector state obs
        if key.startswith("observation.") and "observation.images." not in key:
            features[key] = Sequence(length=data_dict[key].shape[1], feature=Value(dtype="float32", id=None))
        # check if image obs
        elif "observation.images." in key:
            if video:
                features[key] = VideoFrame()
            else:
                features[key] = Image()

    if "language_instruction" in data_dict:
        features["language_instruction"] = Value(dtype="string", id=None)

    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )

    features["is_terminal"] = Value(dtype="bool", id=None)
    features["is_first"] = Value(dtype="bool", id=None)
    features["discount"] = Value(dtype="float32", id=None)

    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.reward"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
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
    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, encoding)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    return hf_dataset, episode_data_index, info

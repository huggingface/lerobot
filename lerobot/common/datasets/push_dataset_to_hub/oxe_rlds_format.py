#!/usr/bin/env python

"""
For https://github.com/google-deepmind/open_x_embodiment (OXE) datasets.

Example:
    python lerobot/scripts/push_dataset_to_hub.py \
        --raw-dir /hdd/tensorflow_datasets/bridge_dataset/1.0.0/ \
        --repo-id youliangtan/sampled_bridge_data_v2 \
        --raw-format oxe_rlds.bridge_orig \
        --episodes 3 4 5 8 9

Exact dataset fps defined in oxe/config.py, obtained from:
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

from lerobot.common.datasets.push_dataset_to_hub.oxe.configs import OXE_DATASET_CONFIGS
from lerobot.common.datasets.push_dataset_to_hub.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames

np.set_printoptions(precision=2)


def get_cameras_keys(obs_keys):
    return [key for key in obs_keys if "image" in key]


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
    episodes: list[int],
    oxe_dataset_name: str | None = None,
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
    if oxe_dataset_name is not None:
        print(" - applying standardization transform for dataset: ", oxe_dataset_name)
        assert oxe_dataset_name in OXE_STANDARDIZATION_TRANSFORMS
        transform_fn = OXE_STANDARDIZATION_TRANSFORMS[oxe_dataset_name]
        dataset = dataset.map(transform_fn)

    image_keys = get_cameras_keys(dataset_info.features["steps"]["observation"].keys())
    lang_key = "language_instruction" if "language_instruction" in dataset.element_spec else None
    print(" - image_keys: ", image_keys)
    print(" - lang_key: ", lang_key)

    it = iter(dataset)

    ep_dicts = []

    # if we user specified episodes, skip the ones not in the list
    if episodes is not None:
        if ds_length == 0:
            raise ValueError("No episodes found.")
        # convert episodes index to sorted list
        episodes = sorted(episodes)

    for ep_idx in tqdm.tqdm(range(ds_length)):
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

        ###########################################################
        # Handle the episodic data

        # last step of demonstration is considered done
        done = torch.zeros(num_frames, dtype=torch.bool)
        done[-1] = True
        ep_dict = {}
        langs = []  # TODO: might be located in "observation"

        image_array_dict = {key: [] for key in image_keys}

        # We will create the state observation tensor by stacking the state
        # obs keys defined in the oxe/configs.py
        if oxe_dataset_name is not None:
            state_obs_keys = OXE_DATASET_CONFIGS[oxe_dataset_name]["state_obs_keys"]
            # stack the state observations, if is None, pad with zeros
            states = []
            for key in state_obs_keys:
                if key in episode["observation"]:
                    states.append(tf_to_torch(episode["observation"][key]))
                else:
                    states.append(torch.zeros(num_frames, 1))  # pad with zeros
            states = torch.cat(states, dim=1)
            assert states.shape == (num_frames, 8)
        else:
            states = tf_to_torch(episode["observation"]["state"])

        actions = tf_to_torch(episode["action"])
        rewards = tf_to_torch(episode["reward"]).float()

        # If lang_key is present, convert the entire tensor at once
        if lang_key is not None:
            langs = [str(x) for x in episode[lang_key]]

        for im_key in image_keys:
            imgs = episode["observation"][im_key]
            image_array_dict[im_key] = [tf_img_convert(img) for img in imgs]

        # simple assertions
        for item in [states, actions, rewards, done]:
            assert len(item) == num_frames

        ###########################################################

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
                encode_video_frames(tmp_imgs_dir, video_path, fps)

                # clean temporary images directory
                shutil.rmtree(tmp_imgs_dir)

                # store the reference to the video frame
                ep_dict[img_key] = [
                    {"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)
                ]
            else:
                ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

        if lang_key is not None:
            ep_dict["language_instruction"] = langs

        ep_dict["observation.state"] = states
        ep_dict["action"] = actions
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["next.reward"] = rewards
        ep_dict["next.done"] = done

        ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video) -> Dataset:
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    if "observation.velocity" in data_dict:
        features["observation.velocity"] = Sequence(
            length=data_dict["observation.velocity"].shape[1], feature=Value(dtype="float32", id=None)
        )
    if "observation.effort" in data_dict:
        features["observation.effort"] = Sequence(
            length=data_dict["observation.effort"].shape[1], feature=Value(dtype="float32", id=None)
        )
    if "language_instruction" in data_dict:
        features["language_instruction"] = Value(dtype="string", id=None)

    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
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
    oxe_dataset_name: str | None = None,
):
    """This is a test impl for rlds conversion"""
    if fps is None:
        if oxe_dataset_name is not None:
            if "fps" not in OXE_DATASET_CONFIGS[oxe_dataset_name]:
                raise ValueError(
                    "fps for this dataset is not specified in oxe/configs.py yet,"
                    "means it is not yet tested"
                )
            fps = OXE_DATASET_CONFIGS[oxe_dataset_name]["fps"]
        else:
            print(" - WARNING: fps is not provided, using default value of 5 fps")
            fps = 5

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, oxe_dataset_name)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info


if __name__ == "__main__":
    # NOTE (YL): This mainly serves as a unit test
    # austin_buds_dataset_converted_externally_to_rlds is a smaller dataset in
    # open x embodiment datasets.
    raw_dir = Path("/hdd/tensorflow_datasets/austin_buds_dataset_converted_externally_to_rlds/0.1.0/")
    videos_dir = Path("/hdd/tmp/")
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(
        raw_dir,
        videos_dir,
        fps=50,
        video=True,
        episodes=[2, 3],
    )
    print(hf_dataset)
    print(episode_data_index)
    print(info)

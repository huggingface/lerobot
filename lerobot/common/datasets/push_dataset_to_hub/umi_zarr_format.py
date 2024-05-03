"""Process UMI (Universal Manipulation Interface) data stored in Zarr format like in: https://github.com/real-stanford/universal_manipulation_interface"""

import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
import zarr
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub._umi_imagecodecs_numcodecs import register_codecs
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames


def check_format(raw_dir) -> bool:
    zarr_path = raw_dir / "cup_in_the_wild.zarr"
    zarr_data = zarr.open(zarr_path, mode="r")

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

    # mandatory to access zarr_data
    register_codecs()
    nb_frames = zarr_data["data/camera0_rgb"].shape[0]

    required_datasets.remove("meta/episode_ends")
    assert all(nb_frames == zarr_data[dataset].shape[0] for dataset in required_datasets)


def get_episode_idxs(episode_ends: np.ndarray) -> np.ndarray:
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


def load_from_raw(raw_dir, out_dir, fps, video, debug):
    zarr_path = raw_dir / "cup_in_the_wild.zarr"
    zarr_data = zarr.open(zarr_path, mode="r")

    # We process the image data separately because it is too large to fit in memory
    end_pose = torch.from_numpy(zarr_data["data/robot0_demo_end_pose"][:])
    start_pos = torch.from_numpy(zarr_data["data/robot0_demo_start_pose"][:])
    eff_pos = torch.from_numpy(zarr_data["data/robot0_eef_pos"][:])
    eff_rot_axis_angle = torch.from_numpy(zarr_data["data/robot0_eef_rot_axis_angle"][:])
    gripper_width = torch.from_numpy(zarr_data["data/robot0_gripper_width"][:])

    states_pos = torch.cat([eff_pos, eff_rot_axis_angle], dim=1)
    states = torch.cat([states_pos, gripper_width], dim=1)

    episode_ends = zarr_data["meta/episode_ends"][:]
    num_episodes = episode_ends.shape[0]

    episode_ids = torch.from_numpy(get_episode_idxs(episode_ends))

    # We convert it in torch tensor later because the jit function does not support torch tensors
    episode_ends = torch.from_numpy(episode_ends)

    ep_dicts = []
    episode_data_index = {"from": [], "to": []}

    id_from = 0
    for ep_idx in tqdm.tqdm(range(num_episodes)):
        id_to = episode_ends[ep_idx]
        num_frames = id_to - id_from

        # sanity heck
        assert (episode_ids[id_from:id_to] == ep_idx).all()

        # TODO(rcadene): save temporary images of the episode?

        state = states[id_from:id_to]

        ep_dict = {}

        # load 57MB of images in RAM (400x224x224x3 uint8)
        imgs_array = zarr_data["data/camera0_rgb"][id_from:id_to]
        img_key = "observation.image"
        if video:
            # save png images in temporary directory
            tmp_imgs_dir = out_dir / "tmp_images"
            save_images_concurrently(imgs_array, tmp_imgs_dir)

            # encode images to a mp4 video
            fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
            video_path = out_dir / "videos" / fname
            encode_video_frames(tmp_imgs_dir, video_path, fps)

            # clean temporary images directory
            shutil.rmtree(tmp_imgs_dir)

            # store the reference to the video frame
            ep_dict[img_key] = [{"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)]
        else:
            ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

        ep_dict["observation.state"] = state
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames, dtype=torch.int64)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        ep_dict["episode_data_index_from"] = torch.tensor([id_from] * num_frames)
        ep_dict["episode_data_index_to"] = torch.tensor([id_from + num_frames] * num_frames)
        ep_dict["end_pose"] = end_pose[id_from:id_to]
        ep_dict["start_pos"] = start_pos[id_from:id_to]
        ep_dict["gripper_width"] = gripper_width[id_from:id_to]
        ep_dicts.append(ep_dict)

        episode_data_index["from"].append(id_from)
        episode_data_index["to"].append(id_from + num_frames)
        id_from += num_frames

        # process first episode only
        if debug:
            break

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = id_from
    data_dict["index"] = torch.arange(0, total_frames, 1)

    return data_dict, episode_data_index


def to_hf_dataset(data_dict, video):
    features = {}

    if video:
        features["observation.image"] = VideoFrame()
    else:
        features["observation.image"] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["index"] = Value(dtype="int64", id=None)
    features["episode_data_index_from"] = Value(dtype="int64", id=None)
    features["episode_data_index_to"] = Value(dtype="int64", id=None)
    # `start_pos` and `end_pos` respectively represent the positions of the end-effector
    # at the beginning and the end of the episode.
    # `gripper_width` indicates the distance between the grippers, and this value is included
    # in the state vector, which comprises the concatenation of the end-effector position
    # and gripper width.
    features["end_pose"] = Sequence(
        length=data_dict["end_pose"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["start_pos"] = Sequence(
        length=data_dict["start_pos"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["gripper_width"] = Sequence(
        length=data_dict["gripper_width"].shape[1], feature=Value(dtype="float32", id=None)
    )

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(raw_dir: Path, out_dir: Path, fps=None, video=True, debug=False):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        # For umi cup in the wild: https://arxiv.org/pdf/2402.10329#table.caption.16
        fps = 10

    if not video:
        logging.warning(
            "Generating UMI dataset without `video=True` creates ~150GB on disk and requires ~80GB in RAM."
        )

    data_dict, episode_data_index = load_from_raw(raw_dir, out_dir, fps, video, debug)
    hf_dataset = to_hf_dataset(data_dict, video)

    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info

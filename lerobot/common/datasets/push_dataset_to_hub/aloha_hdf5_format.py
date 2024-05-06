"""
Contains utilities to process raw data format of HDF5 files like in: https://github.com/tonyzhaozh/act
"""

import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
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

    hdf5_paths = list(raw_dir.glob("episode_*.hdf5"))
    assert len(hdf5_paths) != 0
    for hdf5_path in hdf5_paths:
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


def load_from_raw(raw_dir, out_dir, fps, video, debug):
    # only frames from simulation are uncompressed
    compressed_images = "sim" not in raw_dir.name

    hdf5_files = list(raw_dir.glob("*.hdf5"))
    ep_dicts = []
    episode_data_index = {"from": [], "to": []}

    id_from = 0

    for ep_path in tqdm.tqdm(hdf5_files, total=len(hdf5_files)):
        with h5py.File(ep_path, "r") as ep:
            ep_idx = int(re.search(r"episode_(\d+)", ep_path.name).group(1))
            num_frames = ep["/action"].shape[0]

            # last step of demonstration is considered done
            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True

            state = torch.from_numpy(ep["/observations/qpos"][:])
            action = torch.from_numpy(ep["/action"][:])

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
                    tmp_imgs_dir = out_dir / "tmp_images"
                    save_images_concurrently(imgs_array, tmp_imgs_dir)

                    # encode images to a mp4 video
                    fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
                    video_path = out_dir / "videos" / fname
                    encode_video_frames(tmp_imgs_dir, video_path, fps)

                    # clean temporary images directory
                    shutil.rmtree(tmp_imgs_dir)

                    # store the reference to the video frame
                    ep_dict[img_key] = [
                        {"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)
                    ]
                else:
                    ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

            ep_dict["observation.state"] = state
            ep_dict["action"] = action
            ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
            ep_dict["next.done"] = done
            # TODO(rcadene): add reward and success by computing them in sim

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

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

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

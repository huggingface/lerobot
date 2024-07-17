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
"""Process zarr files formatted like in: https://github.com/real-stanford/diffusion_policy"""

import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
import zarr
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


def check_format(raw_dir):
    zarr_path = raw_dir / "pusht_cchi_v7_replay.zarr"
    zarr_data = zarr.open(zarr_path, mode="r")

    required_datasets = {
        "data/action",
        "data/img",
        "data/keypoint",
        "data/n_contacts",
        "data/state",
        "meta/episode_ends",
    }
    for dataset in required_datasets:
        assert dataset in zarr_data
    nb_frames = zarr_data["data/img"].shape[0]

    required_datasets.remove("meta/episode_ends")

    assert all(nb_frames == zarr_data[dataset].shape[0] for dataset in required_datasets)


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    keypoints_instead_of_image: bool = False,
    encoding: dict | None = None,
):
    try:
        import pymunk
        from gym_pusht.envs.pusht import PushTEnv, pymunk_to_shapely

        from lerobot.common.datasets.push_dataset_to_hub._diffusion_policy_replay_buffer import (
            ReplayBuffer as DiffusionPolicyReplayBuffer,
        )
    except ModuleNotFoundError as e:
        print("`gym_pusht` is not installed. Please install it with `pip install 'lerobot[gym_pusht]'`")
        raise e
    # as define in gmy-pusht env: https://github.com/huggingface/gym-pusht/blob/e0684ff988d223808c0a9dcfaba9dc4991791370/gym_pusht/envs/pusht.py#L174
    success_threshold = 0.95  # 95% coverage,

    zarr_path = raw_dir / "pusht_cchi_v7_replay.zarr"
    zarr_data = DiffusionPolicyReplayBuffer.copy_from_path(zarr_path)

    episode_ids = torch.from_numpy(zarr_data.get_episode_idxs())
    assert len(
        {zarr_data[key].shape[0] for key in zarr_data.keys()}  # noqa: SIM118
    ), "Some data type dont have the same number of total frames."

    # TODO(rcadene): verify that goal pose is expected to be fixed
    goal_pos_angle = np.array([256, 256, np.pi / 4])  # x, y, theta (in radians)
    goal_body = PushTEnv.get_goal_pose_body(goal_pos_angle)

    imgs = torch.from_numpy(zarr_data["img"])  # b h w c
    states = torch.from_numpy(zarr_data["state"])
    actions = torch.from_numpy(zarr_data["action"])

    # load data indices from which each episode starts and ends
    from_ids, to_ids = [], []
    from_idx = 0
    for to_idx in zarr_data.meta["episode_ends"]:
        from_ids.append(from_idx)
        to_ids.append(to_idx)
        from_idx = to_idx

    num_episodes = len(from_ids)

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    for ep_idx, selected_ep_idx in tqdm.tqdm(enumerate(ep_ids)):
        from_idx = from_ids[selected_ep_idx]
        to_idx = to_ids[selected_ep_idx]
        num_frames = to_idx - from_idx

        # sanity check
        assert (episode_ids[from_idx:to_idx] == ep_idx).all()

        # get image
        if not keypoints_instead_of_image:
            image = imgs[from_idx:to_idx]
            assert image.min() >= 0.0
            assert image.max() <= 255.0
            image = image.type(torch.uint8)

        # get state
        state = states[from_idx:to_idx]
        agent_pos = state[:, :2]
        block_pos = state[:, 2:4]
        block_angle = state[:, 4]

        # get reward, success, done, and (maybe) keypoints
        reward = torch.zeros(num_frames)
        success = torch.zeros(num_frames, dtype=torch.bool)
        if keypoints_instead_of_image:
            keypoints = torch.zeros(num_frames, 16)  # 8 keypoints each with 2 coords
        done = torch.zeros(num_frames, dtype=torch.bool)
        for i in range(num_frames):
            space = pymunk.Space()
            space.gravity = 0, 0
            space.damping = 0

            # Add walls.
            walls = [
                PushTEnv.add_segment(space, (5, 506), (5, 5), 2),
                PushTEnv.add_segment(space, (5, 5), (506, 5), 2),
                PushTEnv.add_segment(space, (506, 5), (506, 506), 2),
                PushTEnv.add_segment(space, (5, 506), (506, 506), 2),
            ]
            space.add(*walls)

            block_body, block_shapes = PushTEnv.add_tee(space, block_pos[i].tolist(), block_angle[i].item())
            goal_geom = pymunk_to_shapely(goal_body, block_body.shapes)
            block_geom = pymunk_to_shapely(block_body, block_body.shapes)
            intersection_area = goal_geom.intersection(block_geom).area
            goal_area = goal_geom.area
            coverage = intersection_area / goal_area
            reward[i] = np.clip(coverage / success_threshold, 0, 1)
            success[i] = coverage > success_threshold
            if keypoints_instead_of_image:
                keypoints[i] = torch.from_numpy(PushTEnv.get_keypoints(block_shapes).flatten())

        # last step of demonstration is considered done
        done[-1] = True

        ep_dict = {}

        if not keypoints_instead_of_image:
            imgs_array = [x.numpy() for x in image]
            img_key = "observation.image"
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

        ep_dict["observation.state"] = agent_pos
        if keypoints_instead_of_image:
            ep_dict["observation.environment_state"] = keypoints
        ep_dict["action"] = actions[from_idx:to_idx]
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames, dtype=torch.int64)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        # ep_dict["next.observation.image"] = image[1:],
        # ep_dict["next.observation.state"] = agent_pos[1:],
        # TODO(rcadene)] = verify that reward and done are aligned with image and agent_pos
        ep_dict["next.reward"] = torch.cat([reward[1:], reward[[-1]]])
        ep_dict["next.done"] = torch.cat([done[1:], done[[-1]]])
        ep_dict["next.success"] = torch.cat([success[1:], success[[-1]]])
        ep_dicts.append(ep_dict)
    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video, keypoints_instead_of_image: bool = False):
    features = {}

    if not keypoints_instead_of_image:
        if video:
            features["observation.image"] = VideoFrame()
        else:
            features["observation.image"] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    if keypoints_instead_of_image:
        features["observation.environment_state"] = Sequence(
            length=data_dict["observation.environment_state"].shape[1],
            feature=Value(dtype="float32", id=None),
        )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.reward"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["next.success"] = Value(dtype="bool", id=None)
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
    # Manually change this to True to use keypoints of the T instead of an image observation (but don't merge
    # with True). Also make sure to use video = 0 in the `push_dataset_to_hub.py` script.
    keypoints_instead_of_image = False

    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 10

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, keypoints_instead_of_image, encoding)
    hf_dataset = to_hf_dataset(data_dict, video, keypoints_instead_of_image)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video if not keypoints_instead_of_image else 0,
    }
    if video:
        info["encoding"] = get_default_encoding()

    return hf_dataset, episode_data_index, info

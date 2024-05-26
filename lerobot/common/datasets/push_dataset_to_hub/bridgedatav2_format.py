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
"""Process BridgeData V2 files"""

import gc
import os
import pickle
import shutil
from pathlib import Path

import torch
import tqdm
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.video_utils import encode_video_frames


def check_format(raw_dir) -> bool:
    traj_search_path = os.path.join("**", "traj_group*", "traj*")
    traj_folders = list(raw_dir.glob(traj_search_path))
    assert len(traj_folders) != 0
    for traj in traj_folders:
        if len(list(traj.glob("*.pkl"))) == 2:
            actions = load_actions(traj)
            num_frames = len(list(traj.glob(os.path.join("images0", "*.jpg"))))
            assert len(actions) == num_frames - 1
            state = load_state(traj)
            assert len(state) == num_frames


def load_actions(path):
    fp = os.path.join(path, "policy_out.pkl")
    with open(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return act_list


def load_state(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"]


def load_from_raw(raw_dir, out_dir, fps, video, debug):
    traj_search_path = os.path.join("**", "traj_group*", "traj*")
    traj_folders = list(raw_dir.glob(traj_search_path))
    ep_dicts = []
    episode_data_index = {"from": [], "to": []}

    id_from = 0
    for ep_idx, ep_path in tqdm.tqdm(enumerate(traj_folders), total=len(traj_folders)):
        if len(list(ep_path.glob("*.pkl"))) != 2:
            continue

        image_paths = list(ep_path.glob(os.path.join("images0", "*.jpg")))
        num_frames = len(image_paths)

        done = torch.zeros(num_frames, dtype=torch.bool)
        done[-1] = True

        actions = load_actions(ep_path)
        state = load_state(ep_path)
        ep_dict = {}
        img_key = "observation.image"
        imgs_array = [PILImage.open(x) for x in image_paths]

        if video:
            # save png images in temporary directory
            tmp_imgs_dir = out_dir / "tmp_images"
            save_images_concurrently(imgs_array, tmp_imgs_dir)  # TODO: this isn't working

            # encode images to a mp4 video
            fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
            video_path = out_dir / "videos" / fname
            encode_video_frames(tmp_imgs_dir, video_path, fps)

            # clean temporary images directory
            shutil.rmtree(tmp_imgs_dir)

            # store the reference to the video frame
            ep_dict[img_key] = [{"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)]
        else:
            ep_dict[img_key] = imgs_array

        ep_dict["action"] = actions
        ep_dict["observation.state"] = state
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        ep_dict["next.done"] = done

        assert isinstance(ep_idx, int)
        ep_dicts.append(ep_dict)

        episode_data_index["from"].append(id_from)
        episode_data_index["to"].append(id_from + num_frames)

        id_from += num_frames

        gc.collect()

        # process first episode only
        if debug:
            break
    data_dict = concatenate_episodes(ep_dicts)
    return data_dict, episode_data_index


def from_raw_to_lerobot_format(raw_dir: Path, out_dir: Path, fps=None, video=True, debug=False):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 12

    data_dir, episode_data_index = load_from_raw(raw_dir, out_dir, fps, video, debug)

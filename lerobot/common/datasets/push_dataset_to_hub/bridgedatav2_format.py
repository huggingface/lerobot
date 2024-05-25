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

import os
import pickle
from pathlib import Path


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


def from_raw_to_lerobot_format(raw_dir: Path, out_dir: Path, fps=None, video=True, debug=False):
    # sanity check
    check_format(raw_dir)

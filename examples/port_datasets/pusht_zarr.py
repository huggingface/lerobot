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

import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw

PUSHT_TASK = "Push the T-shaped blue block onto the T-shaped green target surface."
PUSHT_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (2,),
        "names": {
            "axes": ["x", "y"],
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": {
            "axes": ["x", "y"],
        },
    },
    "next.reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "next.success": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    "observation.environment_state": {
        "dtype": "float32",
        "shape": (16,),
        "names": [
            "keypoints",
        ],
    },
    "observation.image": {
        "dtype": None,
        "shape": (3, 96, 96),
        "names": [
            "channels",
            "height",
            "width",
        ],
    },
}


def build_features(mode: str) -> dict:
    features = PUSHT_FEATURES
    if mode == "keypoints":
        features.pop("observation.image")
    else:
        features.pop("observation.environment_state")
        features["observation.image"]["dtype"] = mode

    return features


def load_raw_dataset(zarr_path: Path):
    try:
        from lerobot.common.datasets.push_dataset_to_hub._diffusion_policy_replay_buffer import (
            ReplayBuffer as DiffusionPolicyReplayBuffer,
        )
    except ModuleNotFoundError as e:
        print("`gym_pusht` is not installed. Please install it with `pip install 'lerobot[gym_pusht]'`")
        raise e

    zarr_data = DiffusionPolicyReplayBuffer.copy_from_path(zarr_path)
    return zarr_data


def calculate_coverage(zarr_data):
    try:
        import pymunk
        from gym_pusht.envs.pusht import PushTEnv, pymunk_to_shapely
    except ModuleNotFoundError as e:
        print("`gym_pusht` is not installed. Please install it with `pip install 'lerobot[gym_pusht]'`")
        raise e

    block_pos = zarr_data["state"][:, 2:4]
    block_angle = zarr_data["state"][:, 4]

    num_frames = len(block_pos)

    coverage = np.zeros((num_frames,), dtype=np.float32)
    # 8 keypoints with 2 coords each
    keypoints = np.zeros((num_frames, 16), dtype=np.float32)

    # Set x, y, theta (in radians)
    goal_pos_angle = np.array([256, 256, np.pi / 4])
    goal_body = PushTEnv.get_goal_pose_body(goal_pos_angle)

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
        coverage[i] = intersection_area / goal_area
        keypoints[i] = PushTEnv.get_keypoints(block_shapes).flatten()

    return coverage, keypoints


def calculate_success(coverage: float, success_threshold: float):
    return coverage > success_threshold


def calculate_reward(coverage: float, success_threshold: float):
    return np.clip(coverage / success_threshold, 0, 1)


def main(raw_dir: Path, repo_id: str, mode: str = "video", push_to_hub: bool = True):
    if mode not in ["video", "image", "keypoints"]:
        raise ValueError(mode)

    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        download_raw(raw_dir, repo_id="lerobot-raw/pusht_raw")

    zarr_data = load_raw_dataset(zarr_path=raw_dir / "pusht_cchi_v7_replay.zarr")

    env_state = zarr_data["state"][:]
    agent_pos = env_state[:, :2]

    action = zarr_data["action"][:]
    image = zarr_data["img"]  # (b, h, w, c)

    if image.dtype == np.float32 and image.max() == np.float32(255):
        # HACK: images are loaded as float32 but they actually encode uint8 data
        image = image.astype(np.uint8)

    episode_data_index = {
        "from": np.concatenate(([0], zarr_data.meta["episode_ends"][:-1])),
        "to": zarr_data.meta["episode_ends"],
    }

    # Calculate success and reward based on the overlapping area
    # of the T-object and the T-area.
    coverage, keypoints = calculate_coverage(zarr_data)
    success = calculate_success(coverage, success_threshold=0.95)
    reward = calculate_reward(coverage, success_threshold=0.95)

    features = build_features(mode)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        robot_type="2d pointer",
        features=features,
        image_writer_threads=4,
    )
    episodes = range(len(episode_data_index["from"]))
    for ep_idx in episodes:
        from_idx = episode_data_index["from"][ep_idx]
        to_idx = episode_data_index["to"][ep_idx]
        num_frames = to_idx - from_idx

        for frame_idx in range(num_frames):
            i = from_idx + frame_idx
            idx = i + (frame_idx < num_frames - 1)
            frame = {
                "action": action[i],
                # Shift reward and success by +1 until the last item of the episode
                "next.reward": reward[idx : idx + 1],
                "next.success": success[idx : idx + 1],
                "task": PUSHT_TASK,
            }

            frame["observation.state"] = agent_pos[i]

            if mode == "keypoints":
                frame["observation.environment_state"] = keypoints[i]
            else:
                frame["observation.image"] = image[i]

            dataset.add_frame(frame)

        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub()
        hub_api = HfApi()
        hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")


if __name__ == "__main__":
    # To try this script, modify the repo id with your own HuggingFace user (e.g cadene/pusht)
    repo_id = "lerobot/pusht"

    modes = ["video", "image", "keypoints"]
    # Uncomment if you want to try with a specific mode
    # modes = ["video"]
    # modes = ["image"]
    # modes = ["keypoints"]

    raw_dir = Path("data/lerobot-raw/pusht_raw")
    for mode in modes:
        if mode in ["image", "keypoints"]:
            repo_id += f"_{mode}"

        # download and load raw dataset, create LeRobotDataset, populate it, push to hub
        main(raw_dir, repo_id=repo_id, mode=mode)

        # Uncomment if you want to load the local dataset and explore it
        # dataset = LeRobotDataset(repo_id=repo_id)
        # breakpoint()

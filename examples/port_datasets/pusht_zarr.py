import shutil
from pathlib import Path

import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw


def create_empty_dataset(repo_id, mode):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": [
                ["x", "y"],
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": [
                ["x", "y"],
            ],
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
    }

    if mode == "keypoints":
        features["observation.environment_state"] = {
            "dtype": "float32",
            "shape": (16,),
            "names": [
                "keypoints",
            ],
        }
    else:
        features["observation.image"] = {
            "dtype": mode,
            "shape": (3, 96, 96),
            "names": [
                "channel",
                "height",
                "width",
            ],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        robot_type="2d pointer",
        features=features,
        image_writer_threads=4,
    )
    return dataset


def load_raw_dataset(zarr_path, load_images=True):
    try:
        from lerobot.common.datasets.push_dataset_to_hub._diffusion_policy_replay_buffer import (
            ReplayBuffer as DiffusionPolicyReplayBuffer,
        )
    except ModuleNotFoundError as e:
        print("`gym_pusht` is not installed. Please install it with `pip install 'lerobot[gym_pusht]'`")
        raise e

    zarr_data = DiffusionPolicyReplayBuffer.copy_from_path(zarr_path)

    env_state = zarr_data["state"][:]
    agent_pos = env_state[:, :2]
    block_pos = env_state[:, 2:4]
    block_angle = env_state[:, 4]

    action = zarr_data["action"][:]

    image = None
    if load_images:
        # b h w c
        image = zarr_data["img"]

    episode_data_index = {
        "from": np.array([0] + zarr_data.meta["episode_ends"][:-1].tolist()),
        "to": zarr_data.meta["episode_ends"],
    }

    return image, agent_pos, block_pos, block_angle, action, episode_data_index


def calculate_coverage(block_pos, block_angle):
    try:
        import pymunk
        from gym_pusht.envs.pusht import PushTEnv, pymunk_to_shapely
    except ModuleNotFoundError as e:
        print("`gym_pusht` is not installed. Please install it with `pip install 'lerobot[gym_pusht]'`")
        raise e

    num_frames = len(block_pos)

    coverage = np.zeros((num_frames,))
    # 8 keypoints with 2 coords each
    keypoints = np.zeros((num_frames, 16))

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
        keypoints[i] = torch.from_numpy(PushTEnv.get_keypoints(block_shapes).flatten())

    return coverage, keypoints


def calculate_success(coverage, success_threshold):
    return coverage > success_threshold


def calculate_reward(coverage, success_threshold):
    return np.clip(coverage / success_threshold, 0, 1)


def populate_dataset(dataset, episode_data_index, episodes, image, state, env_state, action, reward, success):
    if episodes is None:
        episodes = range(len(episode_data_index["from"]))

    for ep_idx in episodes:
        from_idx = episode_data_index["from"][ep_idx]
        to_idx = episode_data_index["to"][ep_idx]
        num_frames = to_idx - from_idx

        for frame_idx in range(num_frames):
            i = from_idx + frame_idx

            frame = {
                "action": torch.from_numpy(action[i]),
                # Shift reward and success by +1 until the last item of the episode
                "next.reward": reward[i + (frame_idx < num_frames - 1)],
                "next.success": success[i + (frame_idx < num_frames - 1)],
            }

            frame["observation.state"] = torch.from_numpy(state[i])

            if env_state is not None:
                frame["observation.environment_state"] = torch.from_numpy(env_state[i])

            if image is not None:
                frame["observation.image"] = torch.from_numpy(image[i])

            dataset.add_frame(frame)

        dataset.save_episode(task="Push the T-shaped blue block onto the T-shaped green target surface.")

    return dataset


def port_pusht(raw_dir, repo_id, episodes=None, mode="video", push_to_hub=True):
    if mode not in ["video", "image", "keypoints"]:
        raise ValueError(mode)

    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        download_raw(raw_dir, repo_id="lerobot-raw/pusht_raw")

    image, agent_pos, block_pos, block_angle, action, episode_data_index = load_raw_dataset(
        zarr_path=raw_dir / "pusht_cchi_v7_replay.zarr"
    )

    # Calculate success and reward based on the overlapping area
    # of the T-object and the T-area.
    coverage, keypoints = calculate_coverage(block_pos, block_angle)
    success = calculate_success(coverage, success_threshold=0.95)
    reward = calculate_reward(coverage, success_threshold=0.95)

    dataset = create_empty_dataset(repo_id, mode)
    dataset = populate_dataset(
        dataset,
        episode_data_index,
        episodes,
        image=None if mode == "keypoints" else image,
        state=agent_pos,
        env_state=keypoints if mode == "keypoints" else None,
        action=action,
        reward=reward,
        success=success,
    )
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    # To try this script, modify the repo id with your own HuggingFace user (e.g cadene/pusht)
    repo_id = "lerobot/pusht"

    episodes = None
    # Uncomment if you want to try with a subset (episode 0 and 1)
    # episodes = [0, 1]

    modes = ["video", "image", "keypoints"]
    # Uncomment if you want to try with a specific mode
    # modes = ["video"]
    # modes = ["image"]
    # modes = ["keypoints"]

    for mode in ["video", "image", "keypoints"]:
        if mode in ["image", "keypoints"]:
            repo_id += f"_{mode}"

        # download and load raw dataset, create LeRobotDataset, populate it, push to hub
        port_pusht("data/lerobot-raw/pusht_raw", repo_id=repo_id, mode=mode, episodes=episodes)

        # Uncomment if you want to loal the local dataset and explore it
        # dataset = LeRobotDataset(repo_id=repo_id, local_files_only=True)
        # breakpoint()

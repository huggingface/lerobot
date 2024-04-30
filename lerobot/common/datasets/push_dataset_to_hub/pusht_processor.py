from pathlib import Path

import numpy as np
import torch
import tqdm
import zarr
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)


class PushTProcessor:
    """Process zarr files formatted like in: https://github.com/real-stanford/diffusion_policy"""

    def __init__(self, folder_path: Path, fps: int | None = None):
        self.zarr_path = folder_path
        if fps is None:
            fps = 10
        self._fps = fps

    @property
    def fps(self) -> int:
        return self._fps

    def is_valid(self):
        try:
            zarr_data = zarr.open(self.zarr_path, mode="r")
        except Exception:
            # TODO (azouitine): Handle the exception properly
            return False
        required_datasets = {
            "data/action",
            "data/img",
            "data/keypoint",
            "data/n_contacts",
            "data/state",
            "meta/episode_ends",
        }
        for dataset in required_datasets:
            if dataset not in zarr_data:
                return False
        nb_frames = zarr_data["data/img"].shape[0]

        required_datasets.remove("meta/episode_ends")

        return all(nb_frames == zarr_data[dataset].shape[0] for dataset in required_datasets)

    def preprocess(self):
        try:
            import pymunk
            from gym_pusht.envs.pusht import PushTEnv, pymunk_to_shapely

            from lerobot.common.datasets.push_dataset_to_hub._diffusion_policy_replay_buffer import (
                ReplayBuffer as DiffusionPolicyReplayBuffer,
            )
        except ModuleNotFoundError as e:
            print("`gym_pusht` is not installed. Please install it with `pip install 'lerobot[gym_pusht]'`")
            raise e

        # as define in env
        success_threshold = 0.95  # 95% coverage,

        dataset_dict = DiffusionPolicyReplayBuffer.copy_from_path(
            self.zarr_path
        )  # , keys=['img', 'state', 'action'])

        episode_ids = torch.from_numpy(dataset_dict.get_episode_idxs())
        num_episodes = dataset_dict.meta["episode_ends"].shape[0]
        assert len(
            {dataset_dict[key].shape[0] for key in dataset_dict.keys()}  # noqa: SIM118
        ), "Some data type dont have the same number of total frames."

        # TODO: verify that goal pose is expected to be fixed
        goal_pos_angle = np.array([256, 256, np.pi / 4])  # x, y, theta (in radians)
        goal_body = PushTEnv.get_goal_pose_body(goal_pos_angle)

        imgs = torch.from_numpy(dataset_dict["img"])  # b h w c
        states = torch.from_numpy(dataset_dict["state"])
        actions = torch.from_numpy(dataset_dict["action"])

        ep_dicts = []
        episode_data_index = {"from": [], "to": []}

        id_from = 0
        for episode_id in tqdm.tqdm(range(num_episodes)):
            id_to = dataset_dict.meta["episode_ends"][episode_id]

            num_frames = id_to - id_from

            assert (episode_ids[id_from:id_to] == episode_id).all()

            image = imgs[id_from:id_to]
            assert image.min() >= 0.0
            assert image.max() <= 255.0
            image = image.type(torch.uint8)

            state = states[id_from:id_to]
            agent_pos = state[:, :2]
            block_pos = state[:, 2:4]
            block_angle = state[:, 4]

            reward = torch.zeros(num_frames)
            success = torch.zeros(num_frames, dtype=torch.bool)
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

                block_body = PushTEnv.add_tee(space, block_pos[i].tolist(), block_angle[i].item())
                goal_geom = pymunk_to_shapely(goal_body, block_body.shapes)
                block_geom = pymunk_to_shapely(block_body, block_body.shapes)
                intersection_area = goal_geom.intersection(block_geom).area
                goal_area = goal_geom.area
                coverage = intersection_area / goal_area
                reward[i] = np.clip(coverage / success_threshold, 0, 1)
                success[i] = coverage > success_threshold

            # last step of demonstration is considered done
            done[-1] = True

            ep_dict = {
                "observation.image": [PILImage.fromarray(x.numpy()) for x in image],
                "observation.state": agent_pos,
                "action": actions[id_from:id_to],
                "episode_index": torch.tensor([episode_id] * num_frames, dtype=torch.int),
                "frame_index": torch.arange(0, num_frames, 1),
                "timestamp": torch.arange(0, num_frames, 1) / self.fps,
                # "next.observation.image": image[1:],
                # "next.observation.state": agent_pos[1:],
                # TODO(rcadene): verify that reward and done are aligned with image and agent_pos
                "next.reward": torch.cat([reward[1:], reward[[-1]]]),
                "next.done": torch.cat([done[1:], done[[-1]]]),
                "next.success": torch.cat([success[1:], success[[-1]]]),
            }
            ep_dicts.append(ep_dict)

            episode_data_index["from"].append(id_from)
            episode_data_index["to"].append(id_from + num_frames)

            id_from += num_frames

        data_dict = concatenate_episodes(ep_dicts)
        return data_dict, episode_data_index

    def to_hf_dataset(self, data_dict):
        features = {
            "observation.image": Image(),
            "observation.state": Sequence(
                length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
            ),
            "action": Sequence(length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)),
            "episode_index": Value(dtype="int64", id=None),
            "frame_index": Value(dtype="int64", id=None),
            "timestamp": Value(dtype="float32", id=None),
            "next.reward": Value(dtype="float32", id=None),
            "next.done": Value(dtype="bool", id=None),
            "next.success": Value(dtype="bool", id=None),
            "index": Value(dtype="int64", id=None),
        }
        features = Features(features)
        hf_dataset = Dataset.from_dict(data_dict, features=features)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def cleanup(self):
        pass

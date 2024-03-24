from pathlib import Path
from typing import Callable

import einops
import numpy as np
import pygame
import pymunk
import torch
import torchrl
import tqdm
from tensordict import TensorDict
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.data.replay_buffers.storages import TensorStorage
from torchrl.data.replay_buffers.writers import Writer

from lerobot.common.datasets.abstract import AbstractExperienceReplay
from lerobot.common.datasets.utils import download_and_extract_zip
from lerobot.common.envs.pusht.pusht_env import pymunk_to_shapely
from lerobot.common.policies.diffusion.replay_buffer import ReplayBuffer as DiffusionPolicyReplayBuffer

# as define in env
SUCCESS_THRESHOLD = 0.95  # 95% coverage,

PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
PUSHT_ZARR = Path("pusht/pusht_cchi_v7_replay.zarr")


def get_goal_pose_body(pose):
    mass = 1
    inertia = pymunk.moment_for_box(mass, (50, 100))
    body = pymunk.Body(mass, inertia)
    # preserving the legacy assignment order for compatibility
    # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
    body.position = pose[:2].tolist()
    body.angle = pose[2]
    return body


def add_segment(space, a, b, radius):
    shape = pymunk.Segment(space.static_body, a, b, radius)
    shape.color = pygame.Color("LightGray")  # https://htmlcolorcodes.com/color-names
    return shape


def add_tee(
    space,
    position,
    angle,
    scale=30,
    color="LightSlateGray",
    mask=None,
):
    if mask is None:
        mask = pymunk.ShapeFilter.ALL_MASKS()
    mass = 1
    length = 4
    vertices1 = [
        (-length * scale / 2, scale),
        (length * scale / 2, scale),
        (length * scale / 2, 0),
        (-length * scale / 2, 0),
    ]
    inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
    vertices2 = [
        (-scale / 2, scale),
        (-scale / 2, length * scale),
        (scale / 2, length * scale),
        (scale / 2, scale),
    ]
    inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
    body = pymunk.Body(mass, inertia1 + inertia2)
    shape1 = pymunk.Poly(body, vertices1)
    shape2 = pymunk.Poly(body, vertices2)
    shape1.color = pygame.Color(color)
    shape2.color = pygame.Color(color)
    shape1.filter = pymunk.ShapeFilter(mask=mask)
    shape2.filter = pymunk.ShapeFilter(mask=mask)
    body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
    body.position = position
    body.angle = angle
    body.friction = 1
    space.add(body, shape1, shape2)
    return body


class PushtExperienceReplay(AbstractExperienceReplay):
    def __init__(
        self,
        dataset_id: str,
        version: str | None = "v1.2",
        batch_size: int = None,
        *,
        shuffle: bool = True,
        root: Path | None = None,
        pin_memory: bool = False,
        prefetch: int = None,
        sampler: SliceSampler = None,
        collate_fn: Callable = None,
        writer: Writer = None,
        transform: "torchrl.envs.Transform" = None,
    ):
        super().__init__(
            dataset_id,
            version,
            batch_size,
            shuffle=shuffle,
            root=root,
            pin_memory=pin_memory,
            prefetch=prefetch,
            sampler=sampler,
            collate_fn=collate_fn,
            writer=writer,
            transform=transform,
        )

    def _download_and_preproc_obsolete(self):
        assert self.root is not None
        raw_dir = self.root / f"{self.dataset_id}_raw"
        zarr_path = (raw_dir / PUSHT_ZARR).resolve()
        if not zarr_path.is_dir():
            raw_dir.mkdir(parents=True, exist_ok=True)
            download_and_extract_zip(PUSHT_URL, raw_dir)

        # load
        dataset_dict = DiffusionPolicyReplayBuffer.copy_from_path(
            zarr_path
        )  # , keys=['img', 'state', 'action'])

        episode_ids = torch.from_numpy(dataset_dict.get_episode_idxs())
        num_episodes = dataset_dict.meta["episode_ends"].shape[0]
        total_frames = dataset_dict["action"].shape[0]
        # to create test artifact
        # num_episodes = 1
        # total_frames = 50
        assert len(
            {dataset_dict[key].shape[0] for key in dataset_dict.keys()}  # noqa: SIM118
        ), "Some data type dont have the same number of total frames."

        # TODO: verify that goal pose is expected to be fixed
        goal_pos_angle = np.array([256, 256, np.pi / 4])  # x, y, theta (in radians)
        goal_body = get_goal_pose_body(goal_pos_angle)

        imgs = torch.from_numpy(dataset_dict["img"])
        imgs = einops.rearrange(imgs, "b h w c -> b c h w")
        states = torch.from_numpy(dataset_dict["state"])
        actions = torch.from_numpy(dataset_dict["action"])

        idx0 = 0
        idxtd = 0
        for episode_id in tqdm.tqdm(range(num_episodes)):
            idx1 = dataset_dict.meta["episode_ends"][episode_id]
            # to create test artifact
            # idx1 = 51

            num_frames = idx1 - idx0

            assert (episode_ids[idx0:idx1] == episode_id).all()

            image = imgs[idx0:idx1]

            state = states[idx0:idx1]
            agent_pos = state[:, :2]
            block_pos = state[:, 2:4]
            block_angle = state[:, 4]

            reward = torch.zeros(num_frames, 1)
            success = torch.zeros(num_frames, 1, dtype=torch.bool)
            done = torch.zeros(num_frames, 1, dtype=torch.bool)
            for i in range(num_frames):
                space = pymunk.Space()
                space.gravity = 0, 0
                space.damping = 0

                # Add walls.
                walls = [
                    add_segment(space, (5, 506), (5, 5), 2),
                    add_segment(space, (5, 5), (506, 5), 2),
                    add_segment(space, (506, 5), (506, 506), 2),
                    add_segment(space, (5, 506), (506, 506), 2),
                ]
                space.add(*walls)

                block_body = add_tee(space, block_pos[i].tolist(), block_angle[i].item())
                goal_geom = pymunk_to_shapely(goal_body, block_body.shapes)
                block_geom = pymunk_to_shapely(block_body, block_body.shapes)
                intersection_area = goal_geom.intersection(block_geom).area
                goal_area = goal_geom.area
                coverage = intersection_area / goal_area
                reward[i] = np.clip(coverage / SUCCESS_THRESHOLD, 0, 1)
                success[i] = coverage > SUCCESS_THRESHOLD

            # last step of demonstration is considered done
            done[-1] = True

            ep_td = TensorDict(
                {
                    ("observation", "image"): image[:-1],
                    ("observation", "state"): agent_pos[:-1],
                    "action": actions[idx0:idx1][:-1],
                    "episode": episode_ids[idx0:idx1][:-1],
                    "frame_id": torch.arange(0, num_frames - 1, 1),
                    ("next", "observation", "image"): image[1:],
                    ("next", "observation", "state"): agent_pos[1:],
                    # TODO: verify that reward and done are aligned with image and agent_pos
                    ("next", "reward"): reward[1:],
                    ("next", "done"): done[1:],
                    ("next", "success"): success[1:],
                },
                batch_size=num_frames - 1,
            )

            if episode_id == 0:
                # hack to initialize tensordict data structure to store episodes
                td_data = ep_td[0].expand(total_frames).memmap_like(self.root / f"{self.dataset_id}")

            td_data[idxtd : idxtd + len(ep_td)] = ep_td

            idx0 = idx1
            idxtd = idxtd + len(ep_td)

        return TensorStorage(td_data.lock_())

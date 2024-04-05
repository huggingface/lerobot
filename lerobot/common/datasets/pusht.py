from pathlib import Path

import einops
import numpy as np
import pygame
import pymunk
import torch
import tqdm

from lerobot.common.datasets.utils import download_and_extract_zip, load_data_with_delta_timestamps
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


class PushtDataset(torch.utils.data.Dataset):
    """

    Arguments
    ----------
    delta_timestamps : dict[list[float]] | None, optional
        Loads data from frames with a shift in timestamps with a different strategy for each data key (e.g. state, action or image)
        If `None`, no shift is applied to current timestamp and the data from the current frame is loaded.
    """

    available_datasets = ["pusht"]
    fps = 10
    image_keys = ["observation.image"]

    def __init__(
        self,
        dataset_id: str,
        version: str | None = "v1.2",
        root: Path | None = None,
        transform: callable = None,
        delta_timestamps: dict[list[float]] | None = None,
    ):
        super().__init__()
        self.dataset_id = dataset_id
        self.version = version
        self.root = root
        self.transform = transform
        self.delta_timestamps = delta_timestamps

        self.data_dir = self.root / f"{self.dataset_id}"
        if (self.data_dir / "data_dict.pth").exists() and (
            self.data_dir / "data_ids_per_episode.pth"
        ).exists():
            self.data_dict = torch.load(self.data_dir / "data_dict.pth")
            self.data_ids_per_episode = torch.load(self.data_dir / "data_ids_per_episode.pth")
        else:
            self._download_and_preproc_obsolete()
            self.data_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.data_dict, self.data_dir / "data_dict.pth")
            torch.save(self.data_ids_per_episode, self.data_dir / "data_ids_per_episode.pth")

    @property
    def num_samples(self) -> int:
        return len(self.data_dict["index"])

    @property
    def num_episodes(self) -> int:
        return len(self.data_ids_per_episode)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = {}

        # get episode id and timestamp of the sampled frame
        current_ts = self.data_dict["timestamp"][idx].item()
        episode = self.data_dict["episode"][idx].item()

        for key in self.data_dict:
            if self.delta_timestamps is not None and key in self.delta_timestamps:
                data, is_pad = load_data_with_delta_timestamps(
                    self.data_dict,
                    self.data_ids_per_episode,
                    self.delta_timestamps,
                    key,
                    current_ts,
                    episode,
                )
                item[key] = data
                item[f"{key}_is_pad"] = is_pad
            else:
                item[key] = self.data_dict[key][idx]

        if self.transform is not None:
            item = self.transform(item)

        return item

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

        self.data_ids_per_episode = {}
        ep_dicts = []

        idx0 = 0
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

            ep_dict = {
                "observation.image": image,
                "observation.state": agent_pos,
                "action": actions[idx0:idx1],
                "episode": torch.tensor([episode_id] * num_frames, dtype=torch.int),
                "frame_id": torch.arange(0, num_frames, 1),
                "timestamp": torch.arange(0, num_frames, 1) / self.fps,
                # "next.observation.image": image[1:],
                # "next.observation.state": agent_pos[1:],
                # TODO(rcadene): verify that reward and done are aligned with image and agent_pos
                "next.reward": torch.cat([reward[1:], reward[[-1]]]),
                "next.done": torch.cat([done[1:], done[[-1]]]),
                "next.success": torch.cat([success[1:], success[[-1]]]),
            }
            ep_dicts.append(ep_dict)

            assert isinstance(episode_id, int)
            self.data_ids_per_episode[episode_id] = torch.arange(idx0, idx1, 1)
            assert len(self.data_ids_per_episode[episode_id]) == num_frames

            idx0 = idx1

        self.data_dict = {}

        keys = ep_dicts[0].keys()
        for key in keys:
            self.data_dict[key] = torch.cat([x[key] for x in ep_dicts])

        self.data_dict["index"] = torch.arange(0, total_frames, 1)


if __name__ == "__main__":
    dataset = PushtDataset(
        "pusht",
        root=Path("data"),
        delta_timestamps={
            "observation.image": [0, -1, -0.2, -0.1],
            "observation.state": [0, -1, -0.2, -0.1],
            "action": [-0.1, 0, 1, 2, 3],
        },
    )
    dataset[10]

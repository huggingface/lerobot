"""
This file contains all obsolete download scripts. They are centralized here to not have to load
useless dependencies when using datasets.
"""

import io
import pickle
import shutil
from pathlib import Path

import einops
import h5py
import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage


def download_and_extract_zip(url: str, destination_folder: Path) -> bool:
    import zipfile

    import requests

    print(f"downloading from {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        progress_bar = tqdm.tqdm(total=total_size, unit="B", unit_scale=True)

        zip_file = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                zip_file.write(chunk)
                progress_bar.update(len(chunk))

        progress_bar.close()

        zip_file.seek(0)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(destination_folder)
        return True
    else:
        return False


def download_and_upload_pusht(root, root_tests, dataset_id="pusht", fps=10):
    try:
        import pymunk
        from gym_pusht.envs.pusht import PushTEnv, pymunk_to_shapely

        from lerobot.common.policies.diffusion.replay_buffer import (
            ReplayBuffer as DiffusionPolicyReplayBuffer,
        )
    except ModuleNotFoundError as e:
        print("`gym_pusht` is not installed. Please install it with `pip install 'lerobot[gym_pusht]'`")
        raise e

    # as define in env
    success_threshold = 0.95  # 95% coverage,

    pusht_url = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
    pusht_zarr = Path("pusht/pusht_cchi_v7_replay.zarr")

    root = Path(root)
    raw_dir = root / f"{dataset_id}_raw"
    zarr_path = (raw_dir / pusht_zarr).resolve()
    if not zarr_path.is_dir():
        raw_dir.mkdir(parents=True, exist_ok=True)
        download_and_extract_zip(pusht_url, raw_dir)

    # load
    dataset_dict = DiffusionPolicyReplayBuffer.copy_from_path(zarr_path)  # , keys=['img', 'state', 'action'])

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

    data_ids_per_episode = {}
    ep_dicts = []

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
            "episode_id": torch.tensor([episode_id] * num_frames, dtype=torch.int),
            "frame_id": torch.arange(0, num_frames, 1),
            "timestamp": torch.arange(0, num_frames, 1) / fps,
            # "next.observation.image": image[1:],
            # "next.observation.state": agent_pos[1:],
            # TODO(rcadene): verify that reward and done are aligned with image and agent_pos
            "next.reward": torch.cat([reward[1:], reward[[-1]]]),
            "next.done": torch.cat([done[1:], done[[-1]]]),
            "next.success": torch.cat([success[1:], success[[-1]]]),
            "episode_data_id_from": torch.tensor([id_from] * num_frames),
            "episode_data_id_to": torch.tensor([id_from + num_frames - 1] * num_frames),
        }
        ep_dicts.append(ep_dict)

        assert isinstance(episode_id, int)
        data_ids_per_episode[episode_id] = torch.arange(id_from, id_to, 1)
        assert len(data_ids_per_episode[episode_id]) == num_frames

        id_from += num_frames

    data_dict = {}

    keys = ep_dicts[0].keys()
    for key in keys:
        if torch.is_tensor(ep_dicts[0][key][0]):
            data_dict[key] = torch.cat([ep_dict[key] for ep_dict in ep_dicts])
        else:
            if key not in data_dict:
                data_dict[key] = []
            for ep_dict in ep_dicts:
                for x in ep_dict[key]:
                    data_dict[key].append(x)

    total_frames = id_from
    data_dict["index"] = torch.arange(0, total_frames, 1)

    features = {
        "observation.image": Image(),
        "observation.state": Sequence(
            length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
        ),
        "action": Sequence(length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)),
        "episode_id": Value(dtype="int64", id=None),
        "frame_id": Value(dtype="int64", id=None),
        "timestamp": Value(dtype="float32", id=None),
        "next.reward": Value(dtype="float32", id=None),
        "next.done": Value(dtype="bool", id=None),
        "next.success": Value(dtype="bool", id=None),
        "index": Value(dtype="int64", id=None),
        "episode_data_id_from": Value(dtype="int64", id=None),
        "episode_data_id_to": Value(dtype="int64", id=None),
    }
    features = Features(features)
    dataset = Dataset.from_dict(data_dict, features=features)
    dataset = dataset.with_format("torch")

    num_items_first_ep = ep_dicts[0]["frame_id"].shape[0]
    dataset.select(range(num_items_first_ep)).save_to_disk(f"{root_tests}/{dataset_id}/train")
    dataset.push_to_hub(f"lerobot/{dataset_id}", token=True)
    dataset.push_to_hub(f"lerobot/{dataset_id}", token=True, revision="v1.0")


def download_and_upload_xarm(root, root_tests, dataset_id, fps=15):
    root = Path(root)
    raw_dir = root / f"{dataset_id}_raw"
    if not raw_dir.exists():
        import zipfile

        import gdown

        raw_dir.mkdir(parents=True, exist_ok=True)
        url = "https://drive.google.com/uc?id=1nhxpykGtPDhmQKm-_B8zBSywVRdgeVya"
        zip_path = raw_dir / "data.zip"
        gdown.download(url, str(zip_path), quiet=False)
        print("Extracting...")
        with zipfile.ZipFile(str(zip_path), "r") as zip_f:
            for member in zip_f.namelist():
                if member.startswith("data/xarm") and member.endswith(".pkl"):
                    print(member)
                    zip_f.extract(member=member)
        zip_path.unlink()

    dataset_path = root / f"{dataset_id}" / "buffer.pkl"
    print(f"Using offline dataset '{dataset_path}'")
    with open(dataset_path, "rb") as f:
        dataset_dict = pickle.load(f)

    total_frames = dataset_dict["actions"].shape[0]

    ep_dicts = []

    id_from = 0
    id_to = 0
    episode_id = 0
    for i in tqdm.tqdm(range(total_frames)):
        id_to += 1

        if not dataset_dict["dones"][i]:
            continue

        num_frames = id_to - id_from

        image = torch.tensor(dataset_dict["observations"]["rgb"][id_from:id_to])
        image = einops.rearrange(image, "b c h w -> b h w c")
        state = torch.tensor(dataset_dict["observations"]["state"][id_from:id_to])
        action = torch.tensor(dataset_dict["actions"][id_from:id_to])
        # TODO(rcadene): we have a missing last frame which is the observation when the env is done
        # it is critical to have this frame for tdmpc to predict a "done observation/state"
        # next_image = torch.tensor(dataset_dict["next_observations"]["rgb"][id_from:id_to])
        # next_state = torch.tensor(dataset_dict["next_observations"]["state"][id_from:id_to])
        next_reward = torch.tensor(dataset_dict["rewards"][id_from:id_to])
        next_done = torch.tensor(dataset_dict["dones"][id_from:id_to])

        ep_dict = {
            "observation.image": [PILImage.fromarray(x.numpy()) for x in image],
            "observation.state": state,
            "action": action,
            "episode_id": torch.tensor([episode_id] * num_frames, dtype=torch.int),
            "frame_id": torch.arange(0, num_frames, 1),
            "timestamp": torch.arange(0, num_frames, 1) / fps,
            # "next.observation.image": next_image,
            # "next.observation.state": next_state,
            "next.reward": next_reward,
            "next.done": next_done,
            "episode_data_id_from": torch.tensor([id_from] * num_frames),
            "episode_data_id_to": torch.tensor([id_from + num_frames - 1] * num_frames),
        }
        ep_dicts.append(ep_dict)

        id_from = id_to
        episode_id += 1

    data_dict = {}
    keys = ep_dicts[0].keys()
    for key in keys:
        if torch.is_tensor(ep_dicts[0][key][0]):
            data_dict[key] = torch.cat([ep_dict[key] for ep_dict in ep_dicts])
        else:
            if key not in data_dict:
                data_dict[key] = []
            for ep_dict in ep_dicts:
                for x in ep_dict[key]:
                    data_dict[key].append(x)

    total_frames = id_from
    data_dict["index"] = torch.arange(0, total_frames, 1)

    features = {
        "observation.image": Image(),
        "observation.state": Sequence(
            length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
        ),
        "action": Sequence(length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)),
        "episode_id": Value(dtype="int64", id=None),
        "frame_id": Value(dtype="int64", id=None),
        "timestamp": Value(dtype="float32", id=None),
        "next.reward": Value(dtype="float32", id=None),
        "next.done": Value(dtype="bool", id=None),
        #'next.success': Value(dtype='bool', id=None),
        "index": Value(dtype="int64", id=None),
        "episode_data_id_from": Value(dtype="int64", id=None),
        "episode_data_id_to": Value(dtype="int64", id=None),
    }
    features = Features(features)
    dataset = Dataset.from_dict(data_dict, features=features)
    dataset = dataset.with_format("torch")

    num_items_first_ep = ep_dicts[0]["frame_id"].shape[0]
    dataset.select(range(num_items_first_ep)).save_to_disk(f"{root_tests}/{dataset_id}/train")
    dataset.push_to_hub(f"lerobot/{dataset_id}", token=True)
    dataset.push_to_hub(f"lerobot/{dataset_id}", token=True, revision="v1.0")


def download_and_upload_aloha(root, root_tests, dataset_id, fps=50):
    folder_urls = {
        "aloha_sim_insertion_human": "https://drive.google.com/drive/folders/1RgyD0JgTX30H4IM5XZn8I3zSV_mr8pyF",
        "aloha_sim_insertion_scripted": "https://drive.google.com/drive/folders/1TsojQQSXtHEoGnqgJ3gmpPQR2DPLtS2N",
        "aloha_sim_transfer_cube_human": "https://drive.google.com/drive/folders/1sc-E4QYW7A0o23m1u2VWNGVq5smAsfCo",
        "aloha_sim_transfer_cube_scripted": "https://drive.google.com/drive/folders/1aRyoOhQwxhyt1J8XgEig4s6kzaw__LXj",
    }

    ep48_urls = {
        "aloha_sim_insertion_human": "https://drive.google.com/file/d/18Cudl6nikDtgRolea7je8iF_gGKzynOP/view?usp=drive_link",
        "aloha_sim_insertion_scripted": "https://drive.google.com/file/d/1wfMSZ24oOh5KR_0aaP3Cnu_c4ZCveduB/view?usp=drive_link",
        "aloha_sim_transfer_cube_human": "https://drive.google.com/file/d/18smMymtr8tIxaNUQ61gW6dG50pt3MvGq/view?usp=drive_link",
        "aloha_sim_transfer_cube_scripted": "https://drive.google.com/file/d/1pnGIOd-E4-rhz2P3VxpknMKRZCoKt6eI/view?usp=drive_link",
    }

    ep49_urls = {
        "aloha_sim_insertion_human": "https://drive.google.com/file/d/1C1kZYyROzs-PrLc0SkDgUgMi4-L3lauE/view?usp=drive_link",
        "aloha_sim_insertion_scripted": "https://drive.google.com/file/d/17EuCUWS6uCCr6yyNzpXdcdE-_TTNCKtf/view?usp=drive_link",
        "aloha_sim_transfer_cube_human": "https://drive.google.com/file/d/1Nk7l53d9sJoGDBKAOnNrExX5nLacATc6/view?usp=drive_link",
        "aloha_sim_transfer_cube_scripted": "https://drive.google.com/file/d/1GKReZHrXU73NMiC5zKCq_UtqPVtYq8eo/view?usp=drive_link",
    }

    num_episodes = {
        "aloha_sim_insertion_human": 50,
        "aloha_sim_insertion_scripted": 50,
        "aloha_sim_transfer_cube_human": 50,
        "aloha_sim_transfer_cube_scripted": 50,
    }

    episode_len = {
        "aloha_sim_insertion_human": 500,
        "aloha_sim_insertion_scripted": 400,
        "aloha_sim_transfer_cube_human": 400,
        "aloha_sim_transfer_cube_scripted": 400,
    }

    cameras = {
        "aloha_sim_insertion_human": ["top"],
        "aloha_sim_insertion_scripted": ["top"],
        "aloha_sim_transfer_cube_human": ["top"],
        "aloha_sim_transfer_cube_scripted": ["top"],
    }

    root = Path(root)
    raw_dir = root / f"{dataset_id}_raw"
    if not raw_dir.is_dir():
        import gdown

        assert dataset_id in folder_urls
        assert dataset_id in ep48_urls
        assert dataset_id in ep49_urls

        raw_dir.mkdir(parents=True, exist_ok=True)

        gdown.download_folder(folder_urls[dataset_id], output=str(raw_dir))

        # because of the 50 files limit per directory, two files episode 48 and 49 were missing
        gdown.download(ep48_urls[dataset_id], output=str(raw_dir / "episode_48.hdf5"), fuzzy=True)
        gdown.download(ep49_urls[dataset_id], output=str(raw_dir / "episode_49.hdf5"), fuzzy=True)

    ep_dicts = []

    id_from = 0
    for ep_id in tqdm.tqdm(range(num_episodes[dataset_id])):
        ep_path = raw_dir / f"episode_{ep_id}.hdf5"
        with h5py.File(ep_path, "r") as ep:
            num_frames = ep["/action"].shape[0]
            assert episode_len[dataset_id] == num_frames

            # last step of demonstration is considered done
            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True

            state = torch.from_numpy(ep["/observations/qpos"][:])
            action = torch.from_numpy(ep["/action"][:])

            ep_dict = {
                "observation.state": state,
                "action": action,
                "episode_id": torch.tensor([ep_id] * num_frames),
                "frame_id": torch.arange(0, num_frames, 1),
                "timestamp": torch.arange(0, num_frames, 1) / fps,
                # "next.observation.state": state,
                # TODO(rcadene): compute reward and success
                # "next.reward": reward,
                "next.done": done,
                # "next.success": success,
                "episode_data_id_from": torch.tensor([id_from] * num_frames),
                "episode_data_id_to": torch.tensor([id_from + num_frames - 1] * num_frames),
            }

            for cam in cameras[dataset_id]:
                image = torch.from_numpy(ep[f"/observations/images/{cam}"][:])  # b h w c
                # image = einops.rearrange(image, "b h w c -> b c h w").contiguous()
                ep_dict[f"observation.images.{cam}"] = [PILImage.fromarray(x.numpy()) for x in image]
                # ep_dict[f"next.observation.images.{cam}"] = image

            assert isinstance(ep_id, int)
            ep_dicts.append(ep_dict)

        id_from += num_frames

    data_dict = {}

    data_dict = {}
    keys = ep_dicts[0].keys()
    for key in keys:
        if torch.is_tensor(ep_dicts[0][key][0]):
            data_dict[key] = torch.cat([ep_dict[key] for ep_dict in ep_dicts])
        else:
            if key not in data_dict:
                data_dict[key] = []
            for ep_dict in ep_dicts:
                for x in ep_dict[key]:
                    data_dict[key].append(x)

    total_frames = id_from
    data_dict["index"] = torch.arange(0, total_frames, 1)

    features = {
        "observation.images.top": Image(),
        "observation.state": Sequence(
            length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
        ),
        "action": Sequence(length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)),
        "episode_id": Value(dtype="int64", id=None),
        "frame_id": Value(dtype="int64", id=None),
        "timestamp": Value(dtype="float32", id=None),
        #'next.reward': Value(dtype='float32', id=None),
        "next.done": Value(dtype="bool", id=None),
        #'next.success': Value(dtype='bool', id=None),
        "index": Value(dtype="int64", id=None),
        "episode_data_id_from": Value(dtype="int64", id=None),
        "episode_data_id_to": Value(dtype="int64", id=None),
    }
    features = Features(features)
    dataset = Dataset.from_dict(data_dict, features=features)
    dataset = dataset.with_format("torch")

    num_items_first_ep = ep_dicts[0]["frame_id"].shape[0]
    dataset.select(range(num_items_first_ep)).save_to_disk(f"{root_tests}/{dataset_id}/train")
    dataset.push_to_hub(f"lerobot/{dataset_id}", token=True)
    dataset.push_to_hub(f"lerobot/{dataset_id}", token=True, revision="v1.0")


if __name__ == "__main__":
    root = "data"
    root_tests = "{root_tests}"

    download_and_upload_pusht(root, root_tests, dataset_id="pusht")
    download_and_upload_xarm(root, root_tests, dataset_id="xarm_lift_medium")
    download_and_upload_aloha(root, root_tests, dataset_id="aloha_sim_insertion_human")
    download_and_upload_aloha(root, root_tests, dataset_id="aloha_sim_insertion_scripted")
    download_and_upload_aloha(root, root_tests, dataset_id="aloha_sim_transfer_cube_human")
    download_and_upload_aloha(root, root_tests, dataset_id="aloha_sim_transfer_cube_scripted")

    dataset_ids = [
        "pusht",
        "xarm_lift_medium",
        "aloha_sim_insertion_human",
        "aloha_sim_insertion_scripted",
        "aloha_sim_transfer_cube_human",
        "aloha_sim_transfer_cube_scripted",
    ]
    for dataset_id in dataset_ids:
        # assume stats have been precomputed
        shutil.copy(f"{root}/{dataset_id}/stats.pth", f"{root_tests}/{dataset_id}/stats.pth")

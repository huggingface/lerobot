"""
This file contains all obsolete download scripts. They are centralized here to not have to load
useless dependencies when using datasets.
"""

import io
import json
import pickle
import shutil
from pathlib import Path

import einops
import h5py
import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from huggingface_hub import HfApi
from PIL import Image as PILImage
from safetensors.torch import save_file

from lerobot.common.datasets.utils import compute_stats, flatten_dict, hf_transform_to_torch


def download_and_upload(root, revision, dataset_id):
    # TODO(rcadene, adilzouitine): add community_id/user_id (e.g. "lerobot", "cadene") or repo_id (e.g. "lerobot/pusht")
    if "pusht" in dataset_id:
        download_and_upload_pusht(root, revision, dataset_id)
    elif "xarm" in dataset_id:
        download_and_upload_xarm(root, revision, dataset_id)
    elif "aloha" in dataset_id:
        download_and_upload_aloha(root, revision, dataset_id)
    elif "umi" in dataset_id:
        download_and_upload_umi(root, revision, dataset_id)
    else:
        raise ValueError(dataset_id)


def concatenate_episodes(ep_dicts):
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

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


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


def push_to_hub(hf_dataset, episode_data_index, info, stats, root, revision, dataset_id):
    # push to main to indicate latest version
    hf_dataset.push_to_hub(f"lerobot/{dataset_id}", token=True)

    # push to version branch
    hf_dataset.push_to_hub(f"lerobot/{dataset_id}", token=True, revision=revision)

    # create and store meta_data
    meta_data_dir = root / dataset_id / "meta_data"
    meta_data_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    # info
    info_path = meta_data_dir / "info.json"
    with open(str(info_path), "w") as f:
        json.dump(info, f, indent=4)
    api.upload_file(
        path_or_fileobj=info_path,
        path_in_repo=str(info_path).replace(f"{root}/{dataset_id}", ""),
        repo_id=f"lerobot/{dataset_id}",
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=info_path,
        path_in_repo=str(info_path).replace(f"{root}/{dataset_id}", ""),
        repo_id=f"lerobot/{dataset_id}",
        repo_type="dataset",
        revision=revision,
    )

    # stats
    stats_path = meta_data_dir / "stats.safetensors"
    save_file(flatten_dict(stats), stats_path)
    api.upload_file(
        path_or_fileobj=stats_path,
        path_in_repo=str(stats_path).replace(f"{root}/{dataset_id}", ""),
        repo_id=f"lerobot/{dataset_id}",
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=stats_path,
        path_in_repo=str(stats_path).replace(f"{root}/{dataset_id}", ""),
        repo_id=f"lerobot/{dataset_id}",
        repo_type="dataset",
        revision=revision,
    )

    # episode_data_index
    episode_data_index = {key: torch.tensor(episode_data_index[key]) for key in episode_data_index}
    ep_data_idx_path = meta_data_dir / "episode_data_index.safetensors"
    save_file(episode_data_index, ep_data_idx_path)
    api.upload_file(
        path_or_fileobj=ep_data_idx_path,
        path_in_repo=str(ep_data_idx_path).replace(f"{root}/{dataset_id}", ""),
        repo_id=f"lerobot/{dataset_id}",
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=ep_data_idx_path,
        path_in_repo=str(ep_data_idx_path).replace(f"{root}/{dataset_id}", ""),
        repo_id=f"lerobot/{dataset_id}",
        repo_type="dataset",
        revision=revision,
    )

    # copy in tests folder, the first episode and the meta_data directory
    num_items_first_ep = episode_data_index["to"][0] - episode_data_index["from"][0]
    hf_dataset.select(range(num_items_first_ep)).with_format("torch").save_to_disk(
        f"tests/data/lerobot/{dataset_id}/train"
    )
    if Path(f"tests/data/lerobot/{dataset_id}/meta_data").exists():
        shutil.rmtree(f"tests/data/lerobot/{dataset_id}/meta_data")
    shutil.copytree(meta_data_dir, f"tests/data/lerobot/{dataset_id}/meta_data")


def download_and_upload_pusht(root, revision, dataset_id="pusht", fps=10):
    try:
        import pymunk
        from gym_pusht.envs.pusht import PushTEnv, pymunk_to_shapely

        from lerobot.common.datasets._diffusion_policy_replay_buffer import (
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
            "timestamp": torch.arange(0, num_frames, 1) / fps,
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

    info = {
        "fps": fps,
    }
    stats = compute_stats(hf_dataset)
    push_to_hub(hf_dataset, episode_data_index, info, stats, root, revision, dataset_id)


def download_and_upload_xarm(root, revision, dataset_id, fps=15):
    root = Path(root)
    raw_dir = root / "xarm_datasets_raw"
    if not raw_dir.exists():
        import zipfile

        import gdown

        raw_dir.mkdir(parents=True, exist_ok=True)
        # from https://github.com/fyhMer/fowm/blob/main/scripts/download_datasets.py
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

    ep_dicts = []
    episode_data_index = {"from": [], "to": []}

    id_from = 0
    id_to = 0
    episode_id = 0
    total_frames = dataset_dict["actions"].shape[0]
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
            "episode_index": torch.tensor([episode_id] * num_frames, dtype=torch.int),
            "frame_index": torch.arange(0, num_frames, 1),
            "timestamp": torch.arange(0, num_frames, 1) / fps,
            # "next.observation.image": next_image,
            # "next.observation.state": next_state,
            "next.reward": next_reward,
            "next.done": next_done,
        }
        ep_dicts.append(ep_dict)

        episode_data_index["from"].append(id_from)
        episode_data_index["to"].append(id_from + num_frames)

        id_from = id_to
        episode_id += 1

    data_dict = concatenate_episodes(ep_dicts)

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
        #'next.success': Value(dtype='bool', id=None),
        "index": Value(dtype="int64", id=None),
    }
    features = Features(features)
    hf_dataset = Dataset.from_dict(data_dict, features=features)
    hf_dataset.set_transform(hf_transform_to_torch)

    info = {
        "fps": fps,
    }
    stats = compute_stats(hf_dataset)
    push_to_hub(hf_dataset, episode_data_index, info, stats, root, revision, dataset_id)


def download_and_upload_aloha(root, revision, dataset_id, fps=50):
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
    episode_data_index = {"from": [], "to": []}

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

            ep_dict = {}

            for cam in cameras[dataset_id]:
                image = torch.from_numpy(ep[f"/observations/images/{cam}"][:])  # b h w c
                # image = einops.rearrange(image, "b h w c -> b c h w").contiguous()
                ep_dict[f"observation.images.{cam}"] = [PILImage.fromarray(x.numpy()) for x in image]
                # ep_dict[f"next.observation.images.{cam}"] = image

            ep_dict.update(
                {
                    "observation.state": state,
                    "action": action,
                    "episode_index": torch.tensor([ep_id] * num_frames),
                    "frame_index": torch.arange(0, num_frames, 1),
                    "timestamp": torch.arange(0, num_frames, 1) / fps,
                    # "next.observation.state": state,
                    # TODO(rcadene): compute reward and success
                    # "next.reward": reward,
                    "next.done": done,
                    # "next.success": success,
                }
            )

            assert isinstance(ep_id, int)
            ep_dicts.append(ep_dict)

            episode_data_index["from"].append(id_from)
            episode_data_index["to"].append(id_from + num_frames)

        id_from += num_frames

    data_dict = concatenate_episodes(ep_dicts)

    features = {
        "observation.images.top": Image(),
        "observation.state": Sequence(
            length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
        ),
        "action": Sequence(length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)),
        "episode_index": Value(dtype="int64", id=None),
        "frame_index": Value(dtype="int64", id=None),
        "timestamp": Value(dtype="float32", id=None),
        # "next.reward": Value(dtype="float32", id=None),
        "next.done": Value(dtype="bool", id=None),
        # "next.success": Value(dtype="bool", id=None),
        "index": Value(dtype="int64", id=None),
    }
    features = Features(features)
    hf_dataset = Dataset.from_dict(data_dict, features=features)
    hf_dataset.set_transform(hf_transform_to_torch)

    info = {
        "fps": fps,
    }
    stats = compute_stats(hf_dataset)
    push_to_hub(hf_dataset, episode_data_index, info, stats, root, revision, dataset_id)


def download_and_upload_umi(root, revision, dataset_id, fps=10):
    # fps is equal to 10 source:https://arxiv.org/pdf/2402.10329.pdf#table.caption.16
    import os
    import re
    import shutil
    from glob import glob

    import numpy as np
    import torch
    import tqdm
    import zarr
    from datasets import Dataset, Features, Image, Sequence, Value

    from lerobot.common.datasets._umi_imagecodecs_numcodecs import register_codecs

    # NOTE: This is critical otherwise ValueError: codec not available: 'imagecodecs_jpegxl'
    # will be raised
    register_codecs()

    url_cup_in_the_wild = "https://real.stanford.edu/umi/data/zarr_datasets/cup_in_the_wild.zarr.zip"
    cup_in_the_wild_zarr = Path("umi/cup_in_the_wild/cup_in_the_wild.zarr")

    root = Path(root)
    raw_dir = root / f"{dataset_id}_raw"
    zarr_path = (raw_dir / cup_in_the_wild_zarr).resolve()
    if not zarr_path.is_dir():
        raw_dir.mkdir(parents=True, exist_ok=True)
        download_and_extract_zip(url_cup_in_the_wild, zarr_path)
    zarr_data = zarr.open(zarr_path, mode="r")

    # We process the image data separately because it is too large to fit in memory
    end_pose = torch.from_numpy(zarr_data["data/robot0_demo_end_pose"][:])
    start_pos = torch.from_numpy(zarr_data["data/robot0_demo_start_pose"][:])
    eff_pos = torch.from_numpy(zarr_data["data/robot0_eef_pos"][:])
    eff_rot_axis_angle = torch.from_numpy(zarr_data["data/robot0_eef_rot_axis_angle"][:])
    gripper_width = torch.from_numpy(zarr_data["data/robot0_gripper_width"][:])

    states_pos = torch.cat([eff_pos, eff_rot_axis_angle], dim=1)
    states = torch.cat([states_pos, gripper_width], dim=1)

    def get_episode_idxs(episode_ends: np.ndarray) -> np.ndarray:
        # Optimized and simplified version of this function: https://github.com/real-stanford/universal_manipulation_interface/blob/298776ce251f33b6b3185a98d6e7d1f9ad49168b/diffusion_policy/common/replay_buffer.py#L374
        from numba import jit

        @jit(nopython=True)
        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            start_idx = 0
            for episode_number, end_idx in enumerate(episode_ends):
                result[start_idx:end_idx] = episode_number
                start_idx = end_idx
            return result

        return _get_episode_idxs(episode_ends)

    episode_ends = zarr_data["meta/episode_ends"][:]
    num_episodes: int = episode_ends.shape[0]

    episode_ids = torch.from_numpy(get_episode_idxs(episode_ends))

    # We convert it in torch tensor later because the jit function does not support torch tensors
    episode_ends = torch.from_numpy(episode_ends)

    ep_dicts = []
    episode_data_index = {"from": [], "to": []}
    id_from = 0

    for episode_id in tqdm.tqdm(range(num_episodes)):
        id_to = episode_ends[episode_id]

        num_frames = id_to - id_from

        assert (
            episode_ids[id_from:id_to] == episode_id
        ).all(), f"episode_ids[{id_from}:{id_to}] != {episode_id}"

        state = states[id_from:id_to]
        ep_dict = {
            # observation.image will be filled later
            "observation.state": state,
            "episode_index": torch.tensor([episode_id] * num_frames, dtype=torch.int),
            "frame_index": torch.arange(0, num_frames, 1),
            "timestamp": torch.arange(0, num_frames, 1) / fps,
            "episode_data_index_from": torch.tensor([id_from] * num_frames),
            "episode_data_index_to": torch.tensor([id_from + num_frames] * num_frames),
            "end_pose": end_pose[id_from:id_to],
            "start_pos": start_pos[id_from:id_to],
            "gripper_width": gripper_width[id_from:id_to],
        }
        ep_dicts.append(ep_dict)
        episode_data_index["from"].append(id_from)
        episode_data_index["to"].append(id_from + num_frames)
        id_from += num_frames

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = id_from
    data_dict["index"] = torch.arange(0, total_frames, 1)

    print("Saving images to disk in temporary folder...")
    # datasets.Image() can take a list of paths to images, so we save the images to a temporary folder
    # to avoid loading them all in memory
    _umi_save_images_concurrently(zarr_data, "tmp_umi_images", max_workers=12)
    print("Saving images to disk in temporary folder... Done")

    # Sort files by number eg. 1.png, 2.png, 3.png, 9.png, 10.png instead of 1.png, 10.png, 2.png, 3.png, 9.png
    # to correctly match the images with the data
    images_path = sorted(glob("tmp_umi_images/*"), key=lambda x: int(re.search(r"(\d+)\.png$", x).group(1)))
    data_dict["observation.image"] = images_path

    features = {
        "observation.image": Image(),
        "observation.state": Sequence(
            length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
        ),
        "episode_index": Value(dtype="int64", id=None),
        "frame_index": Value(dtype="int64", id=None),
        "timestamp": Value(dtype="float32", id=None),
        "index": Value(dtype="int64", id=None),
        "episode_data_index_from": Value(dtype="int64", id=None),
        "episode_data_index_to": Value(dtype="int64", id=None),
        # `start_pos` and `end_pos` respectively represent the positions of the end-effector
        # at the beginning and the end of the episode.
        # `gripper_width` indicates the distance between the grippers, and this value is included
        # in the state vector, which comprises the concatenation of the end-effector position
        # and gripper width.
        "end_pose": Sequence(length=data_dict["end_pose"].shape[1], feature=Value(dtype="float32", id=None)),
        "start_pos": Sequence(
            length=data_dict["start_pos"].shape[1], feature=Value(dtype="float32", id=None)
        ),
        "gripper_width": Sequence(
            length=data_dict["gripper_width"].shape[1], feature=Value(dtype="float32", id=None)
        ),
    }
    features = Features(features)
    hf_dataset = Dataset.from_dict(data_dict, features=features)
    hf_dataset.set_transform(hf_transform_to_torch)

    info = {
        "fps": fps,
    }
    stats = compute_stats(hf_dataset)
    push_to_hub(
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        stats=stats,
        root=root,
        revision=revision,
        dataset_id=dataset_id,
    )
    # Cleanup
    if os.path.exists("tmp_umi_images"):
        print("Removing temporary images folder")
        shutil.rmtree("tmp_umi_images")
        print("Cleanup done")


def _umi_clear_folder(folder_path: str):
    import os

    """
    Clears all the content of the specified folder. Creates the folder if it does not exist.

    Args:
    folder_path (str): Path to the folder to clear.

    Examples:
    >>> import os
    >>> os.makedirs('example_folder', exist_ok=True)
    >>> with open('example_folder/temp_file.txt', 'w') as f:
    ...     f.write('example')
    >>> clear_folder('example_folder')
    >>> os.listdir('example_folder')
    []
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(folder_path)


def _umi_save_image(img_array: np.array, i: int, folder_path: str):
    import os

    """
    Saves a single image to the specified folder.

    Args:
    img_array (ndarray): The numpy array of the image.
    i (int): Index of the image, used for naming.
    folder_path (str): Path to the folder where the image will be saved.
    """
    img = PILImage.fromarray(img_array)
    img_format = "PNG" if img_array.dtype == np.uint8 else "JPEG"
    img.save(os.path.join(folder_path, f"{i}.{img_format.lower()}"), quality=100)


def _umi_save_images_concurrently(zarr_data: dict, folder_path: str, max_workers: int = 4):
    from concurrent.futures import ThreadPoolExecutor

    """
    Saves images from the zarr_data to the specified folder using multithreading.

    Args:
    zarr_data (dict): A dictionary containing image data in an array format.
    folder_path (str): Path to the folder where images will be saved.
    max_workers (int): The maximum number of threads to use for saving images.
    """
    num_images = len(zarr_data["data/camera0_rgb"])
    _umi_clear_folder(folder_path)  # Clear or create folder first

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        [
            executor.submit(_umi_save_image, zarr_data["data/camera0_rgb"][i], i, folder_path)
            for i in range(num_images)
        ]


if __name__ == "__main__":
    root = "data"
    revision = "v1.1"
    dataset_ids = [
        "pusht",
        "xarm_lift_medium",
        "xarm_lift_medium_replay",
        "xarm_push_medium",
        "xarm_push_medium_replay",
        "aloha_sim_insertion_human",
        "aloha_sim_insertion_scripted",
        "aloha_sim_transfer_cube_human",
        "aloha_sim_transfer_cube_scripted",
        "umi_cup_in_the_wild",
    ]
    for dataset_id in dataset_ids:
        download_and_upload(root, revision, dataset_id)

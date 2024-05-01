"""Process pickle files formatted like in: https://github.com/fyhMer/fowm"""

import pickle
import shutil
from pathlib import Path

import einops
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames


def check_format(raw_dir):
    keys = {"actions", "rewards", "dones"}
    nested_keys = {"observations": {"rgb", "state"}, "next_observations": {"rgb", "state"}}

    xarm_files = list(raw_dir.glob("*.pkl"))
    assert len(xarm_files) > 0

    with open(xarm_files[0], "rb") as f:
        dataset_dict = pickle.load(f)

    assert isinstance(dataset_dict, dict)
    assert all(k in dataset_dict for k in keys)

    # Check for consistent lengths in nested keys
    expected_len = len(dataset_dict["actions"])
    assert all(len(dataset_dict[key]) == expected_len for key in keys if key in dataset_dict)

    for key, subkeys in nested_keys.items():
        nested_dict = dataset_dict.get(key, {})
        assert all(len(nested_dict[subkey]) == expected_len for subkey in subkeys if subkey in nested_dict)


def load_from_raw(raw_dir, out_dir, fps, video, debug):
    pkl_path = raw_dir / "buffer.pkl"

    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    ep_dicts = []
    episode_data_index = {"from": [], "to": []}

    id_from = 0
    id_to = 0
    ep_idx = 0
    total_frames = pkl_data["actions"].shape[0]
    for i in tqdm.tqdm(range(total_frames)):
        id_to += 1

        if not pkl_data["dones"][i]:
            continue

        num_frames = id_to - id_from

        image = torch.tensor(pkl_data["observations"]["rgb"][id_from:id_to])
        image = einops.rearrange(image, "b c h w -> b h w c")
        state = torch.tensor(pkl_data["observations"]["state"][id_from:id_to])
        action = torch.tensor(pkl_data["actions"][id_from:id_to])
        # TODO(rcadene): we have a missing last frame which is the observation when the env is done
        # it is critical to have this frame for tdmpc to predict a "done observation/state"
        # next_image = torch.tensor(pkl_data["next_observations"]["rgb"][id_from:id_to])
        # next_state = torch.tensor(pkl_data["next_observations"]["state"][id_from:id_to])
        next_reward = torch.tensor(pkl_data["rewards"][id_from:id_to])
        next_done = torch.tensor(pkl_data["dones"][id_from:id_to])

        ep_dict = {}

        imgs_array = [x.numpy() for x in image]
        img_key = "observation.image"
        if video:
            # save png images in temporary directory
            tmp_imgs_dir = out_dir / "tmp_images"
            save_images_concurrently(imgs_array, tmp_imgs_dir)

            # encode images to a mp4 video
            fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
            video_path = out_dir / "videos" / fname
            encode_video_frames(tmp_imgs_dir, video_path, fps)

            # clean temporary images directory
            shutil.rmtree(tmp_imgs_dir)

            # store the reference to the video frame
            ep_dict[img_key] = [{"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)]
        else:
            ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

        ep_dict["observation.state"] = state
        ep_dict["action"] = action
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames, dtype=torch.int64)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        # ep_dict["next.observation.image"] = next_image
        # ep_dict["next.observation.state"] = next_state
        ep_dict["next.reward"] = next_reward
        ep_dict["next.done"] = next_done
        ep_dicts.append(ep_dict)

        episode_data_index["from"].append(id_from)
        episode_data_index["to"].append(id_from + num_frames)

        id_from = id_to
        ep_idx += 1

        # process first episode only
        if debug:
            break

    data_dict = concatenate_episodes(ep_dicts)
    return data_dict, episode_data_index


def to_hf_dataset(data_dict, video):
    features = {}

    if video:
        features["observation.image"] = VideoFrame()
    else:
        features["observation.image"] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.reward"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)
    # TODO(rcadene): add success
    # features["next.success"] = Value(dtype='bool', id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(raw_dir: Path, out_dir: Path, fps=None, video=True, debug=False):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 15

    data_dict, episode_data_index = load_from_raw(raw_dir, out_dir, fps, video, debug)
    hf_dataset = to_hf_dataset(data_dict, video)

    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info

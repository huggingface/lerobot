import argparse
import concurrent.futures
import json
import logging
import os
import platform
import shutil
import time
import traceback
from contextlib import nullcontext
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig
from PIL import Image
from termcolor import colored


# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, get_default_encoding
from lerobot.common.datasets.utils import calculate_episode_data_index, create_branch
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed
from lerobot.scripts.eval import get_pretrained_policy_path
from lerobot.scripts.push_dataset_to_hub import (
    push_dataset_card_to_hub,
    push_meta_data_to_hub,
    push_videos_to_hub,
    save_meta_data,
)


logging.info("Encoding videos")
# Use ffmpeg to convert frames stored as png into mp4 videos
for episode_index in tqdm.tqdm(range(num_episodes)):
    for key in image_keys:
        tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
        fname = f"{key}_episode_{episode_index:06d}.mp4"
        video_path = local_dir / "videos" / fname
        if video_path.exists():
            # Skip if video is already encoded. Could be the case when resuming data recording.
            continue
        # note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        # since video encoding with ffmpeg is already using multithreading.
        encode_video_frames(tmp_imgs_dir, video_path, fps, overwrite=True)
        shutil.rmtree(tmp_imgs_dir)

logging.info("Concatenating episodes")
ep_dicts = []
for episode_index in tqdm.tqdm(range(num_episodes)):
    ep_path = episodes_dir / f"episode_{episode_index}.pth"
    ep_dict = torch.load(ep_path)
    ep_dicts.append(ep_dict)
data_dict = concatenate_episodes(ep_dicts)

total_frames = data_dict["frame_index"].shape[0]
data_dict["index"] = torch.arange(0, total_frames, 1)

hf_dataset = to_hf_dataset(data_dict, video)
episode_data_index = calculate_episode_data_index(hf_dataset)
info = {
    "codebase_version": CODEBASE_VERSION,
    "fps": fps,
    "video": video,
}
if video:
    info["encoding"] = get_default_encoding()

lerobot_dataset = LeRobotDataset.from_preloaded(
    repo_id=repo_id,
    hf_dataset=hf_dataset,
    episode_data_index=episode_data_index,
    info=info,
    videos_dir=videos_dir,
)
if run_compute_stats:
    logging.info("Computing dataset statistics")
    stats = compute_stats(lerobot_dataset)
    lerobot_dataset.stats = stats
else:
    stats = {}
    logging.info("Skipping computation of the dataset statistics")

hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
hf_dataset.save_to_disk(str(local_dir / "train"))

meta_data_dir = local_dir / "meta_data"
save_meta_data(info, stats, episode_data_index, meta_data_dir)

if push_to_hub:
    hf_dataset.push_to_hub(repo_id, revision="main")
    push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
    push_dataset_card_to_hub(repo_id, revision="main", tags=tags)
    if video:
        push_videos_to_hub(repo_id, videos_dir, revision="main")
    create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION)

logging.info("Exiting")

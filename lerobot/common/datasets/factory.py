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
import logging

import torch
from omegaconf import ListConfig, OmegaConf

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.common.datasets.transforms import get_image_transforms


def resolve_delta_timestamps(cfg):
    """Resolves delta_timestamps config key (in-place) by using `eval`.

    Doesn't do anything if delta_timestamps is not specified or has already been resolve (as evidenced by
    the data type of its values).
    """
    delta_timestamps = cfg.training.get("delta_timestamps")
    if delta_timestamps is not None:
        for key in delta_timestamps:
            if isinstance(delta_timestamps[key], str):
                # TODO(rcadene, alexander-soare): remove `eval` to avoid exploit
                cfg.training.delta_timestamps[key] = eval(delta_timestamps[key])


def make_dataset(cfg, split: str = "train") -> LeRobotDataset | MultiLeRobotDataset:
    """
    Args:
        cfg: A Hydra config as per the LeRobot config scheme.
        split: Select the data subset used to create an instance of LeRobotDataset.
            All datasets hosted on [lerobot](https://huggingface.co/lerobot) contain only one subset: "train".
            Thus, by default, `split="train"` selects all the available data. `split` aims to work like the
            slicer in the hugging face datasets:
            https://huggingface.co/docs/datasets/v2.19.0/loading#slice-splits
            As of now, it only supports `split="train[:n]"` to load the first n frames of the dataset or
            `split="train[n:]"` to load the last n frames. For instance `split="train[:1000]"`.
    Returns:
        The LeRobotDataset.
    """
    if not isinstance(cfg.dataset_repo_id, (str, ListConfig)):
        raise ValueError(
            "Expected cfg.dataset_repo_id to be either a single string to load one dataset or a list of "
            "strings to load multiple datasets."
        )

    # A soft check to warn if the environment matches the dataset. Don't check if we are using a real world env (dora).
    if cfg.env.name != "dora":
        if isinstance(cfg.dataset_repo_id, str):
            dataset_repo_ids = [cfg.dataset_repo_id]  # single dataset
        else:
            dataset_repo_ids = cfg.dataset_repo_id  # multiple datasets

        for dataset_repo_id in dataset_repo_ids:
            if cfg.env.name not in dataset_repo_id:
                logging.warning(
                    f"There might be a mismatch between your training dataset ({dataset_repo_id=}) and your "
                    f"environment ({cfg.env.name=})."
                )

    resolve_delta_timestamps(cfg)

    image_transforms = None
    if cfg.training.image_transforms.enable:
        cfg_tf = cfg.training.image_transforms
        image_transforms = get_image_transforms(
            brightness_weight=cfg_tf.brightness.weight,
            brightness_min_max=cfg_tf.brightness.min_max,
            contrast_weight=cfg_tf.contrast.weight,
            contrast_min_max=cfg_tf.contrast.min_max,
            saturation_weight=cfg_tf.saturation.weight,
            saturation_min_max=cfg_tf.saturation.min_max,
            hue_weight=cfg_tf.hue.weight,
            hue_min_max=cfg_tf.hue.min_max,
            sharpness_weight=cfg_tf.sharpness.weight,
            sharpness_min_max=cfg_tf.sharpness.min_max,
            max_num_transforms=cfg_tf.max_num_transforms,
            random_order=cfg_tf.random_order,
        )

    if cfg.hdf5_data_dir:
        from pathlib import Path
        from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import from_raw_to_lerobot_format
        from lerobot.common.datasets.compute_stats import compute_stats

        data_dir = Path(cfg.hdf5_data_dir)

        if data_dir.exists():
            logging.info(f"HDF5 data dir: {data_dir}")
            dataset, episode_data_index, info = from_raw_to_lerobot_format(raw_dir=data_dir, videos_dir=None, fps=cfg.env.fps, video=False)
            stats = compute_stats(dataset, batch_size=16, num_workers=16, max_num_samples=None)
        else:
            raise ValueError(f"HDF5 data directory: {data_dir} does not exist!")

        dataset = LeRobotDataset.from_preloaded(
            root=data_dir,
            hf_dataset=dataset,
            episode_data_index=episode_data_index,
            stats=stats,
            # delta_timestamps=None,
            transform=image_transforms,
            split=split,
            delta_timestamps=cfg.training.get("delta_timestamps"),
            info=info,
        )
    elif isinstance(cfg.dataset_repo_id, str):
        dataset = LeRobotDataset(
            cfg.dataset_repo_id,
            split=split,
            delta_timestamps=cfg.training.get("delta_timestamps"),
            image_transforms=image_transforms,
            video_backend=cfg.video_backend,
        )
    else:
        dataset = MultiLeRobotDataset(
            cfg.dataset_repo_id,
            split=split,
            delta_timestamps=cfg.training.get("delta_timestamps"),
            image_transforms=image_transforms,
            video_backend=cfg.video_backend,
        )

    if cfg.get("override_dataset_stats"):
        for key, stats_dict in cfg.override_dataset_stats.items():
            for stats_type, listconfig in stats_dict.items():
                # example of stats_type: min, max, mean, std
                stats = OmegaConf.to_container(listconfig, resolve=True)
                dataset.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset

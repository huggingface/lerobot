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
from omegaconf import OmegaConf

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


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


def make_dataset(
    cfg,
    split="train",
):
    if cfg.env.name not in cfg.dataset_repo_id:
        logging.warning(
            f"There might be a mismatch between your training dataset ({cfg.dataset_repo_id=}) and your "
            f"environment ({cfg.env.name=})."
        )

    resolve_delta_timestamps(cfg)

    # TODO(rcadene): add data augmentations

    dataset = LeRobotDataset(
        cfg.dataset_repo_id,
        split=split,
        delta_timestamps=cfg.training.get("delta_timestamps"),
    )

    if cfg.get("override_dataset_stats"):
        for key, stats_dict in cfg.override_dataset_stats.items():
            for stats_type, listconfig in stats_dict.items():
                # example of stats_type: min, max, mean, std
                stats = OmegaConf.to_container(listconfig, resolve=True)
                dataset.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset

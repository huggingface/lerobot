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
""" Visualize effects of image transforms for a given configuration.

This script will generate examples of transformed images as they are output by LeRobot dataset.
Additionally, each individual transform can be visualized separately as well as examples of combined transforms

Example:
```bash
python lerobot/scripts/visualize_image_transforms.py \
    --repo_id=lerobot/pusht \
    --episodes='[0]' \
    --image_transforms.enable=True
```
"""

import logging
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import draccus
from torchvision.transforms import ToPILImage

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.transforms import (
    ImageTransforms,
    ImageTransformsConfig,
    make_transform_from_config,
)
from lerobot.configs.default import DatasetConfig

OUTPUT_DIR = Path("outputs/image_transforms")
to_pil = ToPILImage()


def save_all_transforms(cfg: ImageTransformsConfig, original_frame, output_dir, n_examples):
    output_dir_all = output_dir / "all"
    output_dir_all.mkdir(parents=True, exist_ok=True)

    tfs = ImageTransforms(cfg)
    for i in range(1, n_examples + 1):
        transformed_frame = tfs(original_frame)
        to_pil(transformed_frame).save(output_dir_all / f"{i}.png", quality=100)

    print("Combined transforms examples saved to:")
    print(f"    {output_dir_all}")


def save_each_transform(cfg: ImageTransformsConfig, original_frame, output_dir, n_examples):
    if not cfg.enable:
        logging.warning(
            "No single transforms will be saved, because `image_transforms.enable=False`. To enable, set `enable` to True in `ImageTransformsConfig` or in the command line with `--image_transforms.enable=True`."
        )
        return

    print("Individual transforms examples saved to:")
    for tf_name, tf_cfg in cfg.tfs.items():
        # Apply a few transformation with random value in min_max range
        output_dir_single = output_dir / tf_name
        output_dir_single.mkdir(parents=True, exist_ok=True)

        tf = make_transform_from_config(tf_cfg)
        for i in range(1, n_examples + 1):
            transformed_frame = tf(original_frame)
            to_pil(transformed_frame).save(output_dir_single / f"{i}.png", quality=100)

        # Apply min, max, average transformations
        tf_cfg_kwgs_min = deepcopy(tf_cfg.kwargs)
        tf_cfg_kwgs_max = deepcopy(tf_cfg.kwargs)
        tf_cfg_kwgs_avg = deepcopy(tf_cfg.kwargs)

        for key, (min_, max_) in tf_cfg.kwargs.items():
            avg = (min_ + max_) / 2
            tf_cfg_kwgs_min[key] = [min_, min_]
            tf_cfg_kwgs_max[key] = [max_, max_]
            tf_cfg_kwgs_avg[key] = [avg, avg]

        tf_min = make_transform_from_config(replace(tf_cfg, **{"kwargs": tf_cfg_kwgs_min}))
        tf_max = make_transform_from_config(replace(tf_cfg, **{"kwargs": tf_cfg_kwgs_max}))
        tf_avg = make_transform_from_config(replace(tf_cfg, **{"kwargs": tf_cfg_kwgs_avg}))

        tf_frame_min = tf_min(original_frame)
        tf_frame_max = tf_max(original_frame)
        tf_frame_avg = tf_avg(original_frame)

        to_pil(tf_frame_min).save(output_dir_single / "min.png", quality=100)
        to_pil(tf_frame_max).save(output_dir_single / "max.png", quality=100)
        to_pil(tf_frame_avg).save(output_dir_single / "mean.png", quality=100)

        print(f"    {output_dir_single}")


@draccus.wrap()
def visualize_image_transforms(cfg: DatasetConfig, output_dir: Path = OUTPUT_DIR, n_examples: int = 5):
    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        episodes=cfg.episodes,
        revision=cfg.revision,
        video_backend=cfg.video_backend,
    )

    output_dir = output_dir / cfg.repo_id.split("/")[-1]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get 1st frame from 1st camera of 1st episode
    original_frame = dataset[0][dataset.meta.camera_keys[0]]
    to_pil(original_frame).save(output_dir / "original_frame.png", quality=100)
    print("\nOriginal frame saved to:")
    print(f"    {output_dir / 'original_frame.png'}.")

    save_all_transforms(cfg.image_transforms, original_frame, output_dir, n_examples)
    save_each_transform(cfg.image_transforms, original_frame, output_dir, n_examples)


if __name__ == "__main__":
    visualize_image_transforms()

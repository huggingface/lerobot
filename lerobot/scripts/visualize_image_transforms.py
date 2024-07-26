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


--- Usage Examples ---

Increase hue jitter
```
python lerobot/scripts/visualize_image_transforms.py \
    dataset_repo_id=lerobot/aloha_mobile_shrimp \
    training.image_transforms.hue.min_max="[-0.25,0.25]"
```

Increase brightness & brightness weight
```
python lerobot/scripts/visualize_image_transforms.py \
    dataset_repo_id=lerobot/aloha_mobile_shrimp \
    training.image_transforms.brightness.weight=10.0 \
    training.image_transforms.brightness.min_max="[1.0,2.0]"
```

Blur images and disable saturation & hue
```
python lerobot/scripts/visualize_image_transforms.py \
    dataset_repo_id=lerobot/aloha_mobile_shrimp \
    training.image_transforms.sharpness.weight=10.0 \
    training.image_transforms.sharpness.min_max="[0.0,1.0]" \
    training.image_transforms.saturation.weight=0.0 \
    training.image_transforms.hue.weight=0.0
```

Use all transforms with random order
```
python lerobot/scripts/visualize_image_transforms.py \
    dataset_repo_id=lerobot/aloha_mobile_shrimp \
    training.image_transforms.max_num_transforms=5 \
    training.image_transforms.random_order=true
```

"""

from pathlib import Path

import hydra
from torchvision.transforms import ToPILImage

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.transforms import get_image_transforms

OUTPUT_DIR = Path("outputs/image_transforms")
to_pil = ToPILImage()


def save_config_all_transforms(cfg, original_frame, output_dir, n_examples):
    tf = get_image_transforms(
        brightness_weight=cfg.brightness.weight,
        brightness_min_max=cfg.brightness.min_max,
        contrast_weight=cfg.contrast.weight,
        contrast_min_max=cfg.contrast.min_max,
        saturation_weight=cfg.saturation.weight,
        saturation_min_max=cfg.saturation.min_max,
        hue_weight=cfg.hue.weight,
        hue_min_max=cfg.hue.min_max,
        sharpness_weight=cfg.sharpness.weight,
        sharpness_min_max=cfg.sharpness.min_max,
        max_num_transforms=cfg.max_num_transforms,
        random_order=cfg.random_order,
    )

    output_dir_all = output_dir / "all"
    output_dir_all.mkdir(parents=True, exist_ok=True)

    for i in range(1, n_examples + 1):
        transformed_frame = tf(original_frame)
        to_pil(transformed_frame).save(output_dir_all / f"{i}.png", quality=100)

    print("Combined transforms examples saved to:")
    print(f"    {output_dir_all}")


def save_config_single_transforms(cfg, original_frame, output_dir, n_examples):
    transforms = [
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "sharpness",
    ]
    print("Individual transforms examples saved to:")
    for transform in transforms:
        # Apply one transformation with random value in min_max range
        kwargs = {
            f"{transform}_weight": cfg[f"{transform}"].weight,
            f"{transform}_min_max": cfg[f"{transform}"].min_max,
        }
        tf = get_image_transforms(**kwargs)
        output_dir_single = output_dir / f"{transform}"
        output_dir_single.mkdir(parents=True, exist_ok=True)

        for i in range(1, n_examples + 1):
            transformed_frame = tf(original_frame)
            to_pil(transformed_frame).save(output_dir_single / f"{i}.png", quality=100)

        # Apply min transformation
        min_value, max_value = cfg[f"{transform}"].min_max
        kwargs = {
            f"{transform}_weight": cfg[f"{transform}"].weight,
            f"{transform}_min_max": (min_value, min_value),
        }
        tf = get_image_transforms(**kwargs)
        transformed_frame = tf(original_frame)
        to_pil(transformed_frame).save(output_dir_single / "min.png", quality=100)

        # Apply max transformation
        kwargs = {
            f"{transform}_weight": cfg[f"{transform}"].weight,
            f"{transform}_min_max": (max_value, max_value),
        }
        tf = get_image_transforms(**kwargs)
        transformed_frame = tf(original_frame)
        to_pil(transformed_frame).save(output_dir_single / "max.png", quality=100)

        # Apply mean transformation
        mean_value = (min_value + max_value) / 2
        kwargs = {
            f"{transform}_weight": cfg[f"{transform}"].weight,
            f"{transform}_min_max": (mean_value, mean_value),
        }
        tf = get_image_transforms(**kwargs)
        transformed_frame = tf(original_frame)
        to_pil(transformed_frame).save(output_dir_single / "mean.png", quality=100)

        print(f"    {output_dir_single}")


def visualize_transforms(cfg, output_dir: Path, n_examples: int = 5):
    dataset = LeRobotDataset(cfg.dataset_repo_id)

    output_dir = output_dir / cfg.dataset_repo_id.split("/")[-1]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get 1st frame from 1st camera of 1st episode
    original_frame = dataset[0][dataset.camera_keys[0]]
    to_pil(original_frame).save(output_dir / "original_frame.png", quality=100)
    print("\nOriginal frame saved to:")
    print(f"    {output_dir / 'original_frame.png'}.")

    save_config_all_transforms(cfg.training.image_transforms, original_frame, output_dir, n_examples)
    save_config_single_transforms(cfg.training.image_transforms, original_frame, output_dir, n_examples)


@hydra.main(version_base="1.2", config_name="default", config_path="../configs")
def visualize_transforms_cli(cfg):
    visualize_transforms(cfg, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    visualize_transforms_cli()

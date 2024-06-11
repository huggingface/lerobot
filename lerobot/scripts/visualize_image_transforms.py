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
from pathlib import Path

import hydra
from torchvision.transforms import ToPILImage

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.transforms import make_image_transforms

to_pil = ToPILImage()


def main(cfg, output_dir=Path("outputs/image_transforms")):
    dataset = LeRobotDataset(cfg.dataset_repo_id, image_transforms=None)

    output_dir = Path(output_dir) / Path(cfg.dataset_repo_id.split("/")[-1])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get first frame of 1st episode
    first_idx = dataset.episode_data_index["from"][0].item()
    frame = dataset[first_idx][dataset.camera_keys[0]]
    to_pil(frame).save(output_dir / "original_frame.png", quality=100)

    transforms = ["brightness", "contrast", "saturation", "hue", "sharpness"]

    # Apply each single transformation
    for transform_name in transforms:
        for t in transforms:
            if t == transform_name:
                cfg.training.image_transforms[t].weight = 1
            else:
                cfg.training.image_transforms[t].weight = 0

        transform = make_image_transforms(cfg.training.image_transforms)
        img = transform(frame)
        to_pil(img).save(output_dir / f"{transform_name}.png", quality=100)


@hydra.main(version_base="1.2", config_name="default", config_path="../configs")
def visualize_transforms_cli(cfg: dict):
    main(
        cfg,
    )


if __name__ == "__main__":
    visualize_transforms_cli()

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

import pytest

from lerobot.common.utils.utils import init_hydra_config
from lerobot.scripts.visualize_image_transforms import visualize_transforms
from tests.utils import DEFAULT_CONFIG_PATH

OUTPUT_DIR = Path("outputs/image_transforms")
N_EXAMPLES = 5


@pytest.mark.parametrize(
    "repo_id",
    [
        ("lerobot/aloha_sim_transfer_cube_human"),
    ],
)
def test_visualize_image_transforms_correct_files_exist(repo_id):
    cfg = init_hydra_config(DEFAULT_CONFIG_PATH)
    cfg.dataset_repo_id = repo_id
    output_dir = Path(OUTPUT_DIR) / repo_id.split("/")[-1]
    visualize_transforms(cfg)

    # Check if the original frame image exists
    assert (output_dir / "original_frame.png").exists(), "Original frame image was not saved."

    # Check if the transformed images exist for each transform type
    transforms = ["brightness", "contrast", "saturation", "hue", "sharpness"]
    for transform in transforms:
        transform_dir = output_dir / transform
        assert transform_dir.exists(), f"{transform} directory was not created."
        assert any(transform_dir.iterdir()), f"No transformed images found in {transform} directory."

        # Check for specific files within each transform directory
        expected_files = [f"{i}.png" for i in range(1, N_EXAMPLES + 1)] + ["min.png", "max.png", "mean.png"]
        for file_name in expected_files:
            assert (
                transform_dir / file_name
            ).exists(), f"{file_name} was not found in {transform} directory."

    # Check if the combined transforms directory exists and contains the right files
    combined_transforms_dir = output_dir / "all"
    assert combined_transforms_dir.exists(), "Combined transforms directory was not created."
    assert any(
        combined_transforms_dir.iterdir()
    ), "No transformed images found in combined transforms directory."
    for i in range(1, N_EXAMPLES + 1):
        assert (
            combined_transforms_dir / f"{i}.png"
        ).exists(), f"Combined transform image {i}.png was not found."

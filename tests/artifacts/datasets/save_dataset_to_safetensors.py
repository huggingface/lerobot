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
"""
This script provides a utility for saving a dataset as safetensors files for the purpose of testing backward compatibility
when updating the data format. It uses the `PushtDataset` to create a DataLoader and saves selected frame from the
dataset into a corresponding safetensors file in a specified output directory.

If you know that your change will break backward compatibility, you should write a shortlived test by modifying
`tests/test_datasets.py::test_backward_compatibility` accordingly, and make sure this custom test pass. Your custom test
doesnt need to be merged into the `main` branch. Then you need to run this script and update the tests artifacts.

Example usage:
    `python tests/artifacts/datasets/save_dataset_to_safetensors.py`
"""

import shutil
from pathlib import Path

from safetensors.torch import save_file

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def save_dataset_to_safetensors(output_dir, repo_id="lerobot/pusht"):
    repo_dir = Path(output_dir) / repo_id

    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    repo_dir.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        episodes=[0],
    )

    # save 2 first frames of first episode
    i = dataset.episode_data_index["from"][0].item()
    save_file(dataset[i], repo_dir / f"frame_{i}.safetensors")
    save_file(dataset[i + 1], repo_dir / f"frame_{i + 1}.safetensors")

    # save 2 frames at the middle of first episode
    i = int((dataset.episode_data_index["to"][0].item() - dataset.episode_data_index["from"][0].item()) / 2)
    save_file(dataset[i], repo_dir / f"frame_{i}.safetensors")
    save_file(dataset[i + 1], repo_dir / f"frame_{i + 1}.safetensors")

    # save 2 last frames of first episode
    i = dataset.episode_data_index["to"][0].item()
    save_file(dataset[i - 2], repo_dir / f"frame_{i - 2}.safetensors")
    save_file(dataset[i - 1], repo_dir / f"frame_{i - 1}.safetensors")

    # TODO(rcadene): Enable testing on second and last episode
    # We currently cant because our test dataset only contains the first episode

    # # save 2 first frames of second episode
    # i = dataset.episode_data_index["from"][1].item()
    # save_file(dataset[i], repo_dir / f"frame_{i}.safetensors")
    # save_file(dataset[i + 1], repo_dir / f"frame_{i+1}.safetensors")

    # # save 2 last frames of second episode
    # i = dataset.episode_data_index["to"][1].item()
    # save_file(dataset[i - 2], repo_dir / f"frame_{i-2}.safetensors")
    # save_file(dataset[i - 1], repo_dir / f"frame_{i-1}.safetensors")

    # # save 2 last frames of last episode
    # i = dataset.episode_data_index["to"][-1].item()
    # save_file(dataset[i - 2], repo_dir / f"frame_{i-2}.safetensors")
    # save_file(dataset[i - 1], repo_dir / f"frame_{i-1}.safetensors")


if __name__ == "__main__":
    for dataset in [
        "lerobot/pusht",
        "lerobot/aloha_sim_insertion_human",
        "lerobot/xarm_lift_medium",
        "lerobot/nyu_franka_play_dataset",
        "lerobot/cmu_stretch",
    ]:
        save_dataset_to_safetensors("tests/artifacts/datasets", repo_id=dataset)

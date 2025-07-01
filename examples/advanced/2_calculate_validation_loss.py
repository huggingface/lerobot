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

"""This script demonstrates how to slice a dataset and calculate the loss on a subset of the data.

This technique can be useful for debugging and testing purposes, as well as identifying whether a policy
is learning effectively.

Furthermore, relying on validation loss to evaluate performance is generally not considered a good practice,
especially in the context of imitation learning. The most reliable approach is to evaluate the policy directly
on the target environment, whether that be in simulation or the real world.
"""

import math

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def main():
    device = torch.device("cuda")

    # Download the diffusion policy for pusht environment
    pretrained_policy_path = "lerobot/diffusion_pusht"
    # OR uncomment the following to evaluate a policy from the local outputs/train folder.
    # pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    policy.eval()
    policy.to(device)

    # Set up the dataset.
    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to calculate the loss.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    # Load the last 10% of episodes of the dataset as a validation set.
    # - Load dataset metadata
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    # - Calculate train and val episodes
    total_episodes = dataset_metadata.total_episodes
    episodes = list(range(dataset_metadata.total_episodes))
    num_train_episodes = math.floor(total_episodes * 90 / 100)
    train_episodes = episodes[:num_train_episodes]
    val_episodes = episodes[num_train_episodes:]
    print(f"Number of episodes in full dataset: {total_episodes}")
    print(f"Number of episodes in training dataset (90% subset): {len(train_episodes)}")
    print(f"Number of episodes in validation dataset (10% subset): {len(val_episodes)}")
    # - Load train and val datasets
    train_dataset = LeRobotDataset(
        "lerobot/pusht", episodes=train_episodes, delta_timestamps=delta_timestamps
    )
    val_dataset = LeRobotDataset("lerobot/pusht", episodes=val_episodes, delta_timestamps=delta_timestamps)
    print(f"Number of frames in training dataset (90% subset): {len(train_dataset)}")
    print(f"Number of frames in validation dataset (10% subset): {len(val_dataset)}")

    # Create dataloader for evaluation.
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=4,
        batch_size=64,
        shuffle=False,
        pin_memory=device != torch.device("cpu"),
        drop_last=False,
    )

    # Run validation loop.
    loss_cumsum = 0
    n_examples_evaluated = 0
    for batch in val_dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        loss, _ = policy.forward(batch)

        loss_cumsum += loss.item()
        n_examples_evaluated += batch["index"].shape[0]

    # Calculate the average loss over the validation set.
    average_loss = loss_cumsum / n_examples_evaluated

    print(f"Average loss on validation set: {average_loss:.4f}")


if __name__ == "__main__":
    main()

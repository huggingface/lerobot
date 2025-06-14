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

"""This script demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType

def inject_normalization_stats(policy, dataset):
    """Manually loads normalization stats from the dataset into the policy's state dictionary."""
    stats = dataset.meta.stats
    pol_state_dict = policy.state_dict()

    keys_to_update = {
        "normalize_inputs.buffer_observation_state.mean": ("observation.state", "mean"),
        "normalize_inputs.buffer_observation_state.std": ("observation.state", "std"),
        "normalize_targets.buffer_action.mean": ("action", "mean"),
        "normalize_targets.buffer_action.std": ("action", "std"),
        "unnormalize_outputs.buffer_action.mean": ("action", "mean"),
        "unnormalize_outputs.buffer_action.std": ("action", "std"),
    }

    for pol_key, (stat_key, stat_type) in keys_to_update.items():
        pol_state_dict[pol_key] = torch.from_numpy(stats[stat_key][stat_type])

    policy.load_state_dict(pol_state_dict)
    print("Normalization stats injected into the policy.")

def prepare_batch(batch, device):
    """
    Prepares a batch of samples from the dataset for inference.
    This involves moving tensors to the correct device,
    and remapping image keys to match the policy's expectations.
    """
    batch = {
        "observation.state": batch["observation.state"].to(device),
        "observation.image": batch["observation.images.top"].to(device),
        "observation.image2": batch["observation.images.wrist"].to(device),
        "action": batch["action"].to(device),
        "task": batch["task"],
    }
    return batch

def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train/smolvlaplus_training")
    output_directory.mkdir(parents=True, exist_ok=True)

    # # Select your device
    device = torch.device("mps")

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 10_000
    log_freq = 1
    batch_size = 32

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset = LeRobotDataset("lerobot/svla_so100_stacking")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

    # fix absence of normalization stats in the policy
    inject_normalization_stats(policy, dataset)
    policy.train()
    policy.to(device)

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = prepare_batch(batch, device)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)


if __name__ == "__main__":
    main()

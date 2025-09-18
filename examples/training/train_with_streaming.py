# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""This script demonstrates how to train a Diffusion Policy on the PushT environment,
using a dataset processed in streaming mode.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.constants import ACTION
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train/example_streaming_dataset")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Selects the "best" device available
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    training_steps = 10
    log_freq = 1

    dataset_id = (
        "aractingi/droid_1.0.1"  # 26M frames! Would require 4TB of disk space if installed locally (:
    )
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # We can now instantiate our policy with this config and the dataset stats.
    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # Delta timestamps are used to (1) augment frames used during training and (2) supervise the policy.
    # Here, we use delta-timestamps to only provide ground truth actions for supervision
    delta_timestamps = {
        ACTION: [t / dataset_metadata.fps for t in range(cfg.n_action_steps)],
    }

    # Instantiating the training dataset in streaming mode allows to not consume up memory as the data is fetched
    # iteratively rather than being load into memory all at once. Retrieved frames are shuffled across epochs
    dataset = StreamingLeRobotDataset(dataset_id, delta_timestamps=delta_timestamps, tolerance_s=1e-3)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=16,
        pin_memory=device.type != "cpu",
        drop_last=True,
        prefetch_factor=2,  # loads batches with multiprocessing while policy trains
    )

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {
                k: (v.type(torch.float32) if isinstance(v, torch.Tensor) and v.dtype != torch.bool else v)
                for k, v in batch.items()
            }
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
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

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

"""Train directly from episode-scoped Parquet and MP4 streams.

For normal training, prefer ``lerobot-train --dataset.streaming=true`` so distributed sharding,
checkpoint resume, and device placement are configured by the training pipeline. This lower-level
example shows the underlying Python API.
"""

from pathlib import Path

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDatasetMetadata, StreamingLeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.feature_utils import dataset_to_policy_features


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

    dataset_id = "lerobot/droid_1.0.1"  # 26M frames! Would require 4TB of disk space if installed locally (:
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # We can now instantiate our policy with this config and the dataset stats.
    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # Delta timestamps are used to (1) augment frames used during training and (2) supervise the policy.
    # Here, we use delta-timestamps to only provide ground truth actions for supervision
    delta_timestamps = {
        ACTION: [t / dataset_metadata.fps for t in range(cfg.n_action_steps)],
    }

    # The first run resolves or locally builds a revision-safe MP4 index sidecar. It is never
    # uploaded implicitly. Episode rows and video byte ranges are then prefetched together.
    dataset = StreamingLeRobotDataset(
        dataset_id,
        delta_timestamps=delta_timestamps,
        tolerance_s=1e-3,
        episode_pool_size=16,
        prefetch_episodes=4,
        byte_budget_gb=4,
        repeat=True,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        # One worker owns the rank-level pool. Internal fetch concurrency is configured through
        # StreamingLeRobotDataset.max_num_shards (the lerobot-train CLI derives it from num_workers).
        num_workers=1,
        batch_size=16,
        pin_memory=device.type != "cpu",
        prefetch_factor=2,  # bounded decoded-batch queue while the policy trains
        persistent_workers=True,
    )

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
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
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)


if __name__ == "__main__":
    main()

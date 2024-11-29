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
from copy import deepcopy
from math import ceil

import einops
import torch
import tqdm


def get_stats_einops_patterns(dataset, num_workers=0):
    """These einops patterns will be used to aggregate batches and compute statistics.

    Note: We assume the images are in channel first format
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    stats_patterns = {}

    for key in dataset.features:
        # sanity check that tensors are not float64
        assert batch[key].dtype != torch.float64

        # if isinstance(feats_type, (VideoFrame, Image)):
        if key in dataset.meta.camera_keys:
            # sanity check that images are channel first
            _, c, h, w = batch[key].shape
            assert c < h and c < w, f"expect channel first images, but instead {batch[key].shape}"

            # sanity check that images are float32 in range [0,1]
            assert batch[key].dtype == torch.float32, f"expect torch.float32, but instead {batch[key].dtype=}"
            assert batch[key].max() <= 1, f"expect pixels lower than 1, but instead {batch[key].max()=}"
            assert batch[key].min() >= 0, f"expect pixels greater than 1, but instead {batch[key].min()=}"

            stats_patterns[key] = "b c h w -> c 1 1"
        elif batch[key].ndim == 2:
            stats_patterns[key] = "b c -> c "
        elif batch[key].ndim == 1:
            stats_patterns[key] = "b -> 1"
        else:
            raise ValueError(f"{key}, {batch[key].shape}")

    return stats_patterns


def compute_stats(dataset, batch_size=8, num_workers=8, max_num_samples=None):
    """Compute mean/std and min/max statistics of all data keys in a LeRobotDataset."""
    if max_num_samples is None:
        max_num_samples = len(dataset)

    # for more info on why we need to set the same number of workers, see `load_from_videos`
    stats_patterns = get_stats_einops_patterns(dataset, num_workers)

    # mean and std will be computed incrementally while max and min will track the running value.
    mean, std, max, min = {}, {}, {}, {}
    for key in stats_patterns:
        mean[key] = torch.tensor(0.0).float()
        std[key] = torch.tensor(0.0).float()
        max[key] = torch.tensor(-float("inf")).float()
        min[key] = torch.tensor(float("inf")).float()

    def create_seeded_dataloader(dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return dataloader

    # Note: Due to be refactored soon. The point of storing `first_batch` is to make sure we don't get
    # surprises when rerunning the sampler.
    first_batch = None
    running_item_count = 0  # for online mean computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm.tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute mean, min, max")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        if first_batch is None:
            first_batch = deepcopy(batch)
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation.
            batch_mean = einops.reduce(batch[key], pattern, "mean")
            # Hint: to update the mean we need x̄ₙ = (Nₙ₋₁x̄ₙ₋₁ + Bₙxₙ) / Nₙ, where the subscript represents
            # the update step, N is the running item count, B is this batch size, x̄ is the running mean,
            # and x is the current batch mean. Some rearrangement is then required to avoid risking
            # numerical overflow. Another hint: Nₙ₋₁ = Nₙ - Bₙ. Rearrangement yields
            # x̄ₙ = x̄ₙ₋₁ + Bₙ * (xₙ - x̄ₙ₋₁) / Nₙ
            mean[key] = mean[key] + this_batch_size * (batch_mean - mean[key]) / running_item_count
            max[key] = torch.maximum(max[key], einops.reduce(batch[key], pattern, "max"))
            min[key] = torch.minimum(min[key], einops.reduce(batch[key], pattern, "min"))

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    first_batch_ = None
    running_item_count = 0  # for online std computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm.tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute std")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        # Sanity check to make sure the batches are still in the same order as before.
        if first_batch_ is None:
            first_batch_ = deepcopy(batch)
            for key in stats_patterns:
                assert torch.equal(first_batch_[key], first_batch[key])
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation (where the mean is over squared
            # residuals).See notes in the mean computation loop above.
            batch_std = einops.reduce((batch[key] - mean[key]) ** 2, pattern, "mean")
            std[key] = std[key] + this_batch_size * (batch_std - std[key]) / running_item_count

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    for key in stats_patterns:
        std[key] = torch.sqrt(std[key])

    stats = {}
    for key in stats_patterns:
        stats[key] = {
            "mean": mean[key],
            "std": std[key],
            "max": max[key],
            "min": min[key],
        }
    return stats


def aggregate_stats(ls_datasets) -> dict[str, torch.Tensor]:
    """Aggregate stats of multiple LeRobot datasets into one set of stats without recomputing from scratch.

    The final stats will have the union of all data keys from each of the datasets.

    The final stats will have the union of all data keys from each of the datasets. For instance:
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_mean = (mean of all data)
    - new_std = (std of all data)
    """
    data_keys = set()
    for dataset in ls_datasets:
        data_keys.update(dataset.meta.stats.keys())
    stats = {k: {} for k in data_keys}
    for data_key in data_keys:
        for stat_key in ["min", "max"]:
            # compute `max(dataset_0["max"], dataset_1["max"], ...)`
            stats[data_key][stat_key] = einops.reduce(
                torch.stack(
                    [ds.meta.stats[data_key][stat_key] for ds in ls_datasets if data_key in ds.meta.stats],
                    dim=0,
                ),
                "n ... -> ...",
                stat_key,
            )
        total_samples = sum(d.num_frames for d in ls_datasets if data_key in d.meta.stats)
        # Compute the "sum" statistic by multiplying each mean by the number of samples in the respective
        # dataset, then divide by total_samples to get the overall "mean".
        # NOTE: the brackets around (d.num_frames / total_samples) are needed tor minimize the risk of
        # numerical overflow!
        stats[data_key]["mean"] = sum(
            d.meta.stats[data_key]["mean"] * (d.num_frames / total_samples)
            for d in ls_datasets
            if data_key in d.meta.stats
        )
        # The derivation for standard deviation is a little more involved but is much in the same spirit as
        # the computation of the mean.
        # Given two sets of data where the statistics are known:
        # σ_combined = sqrt[ (n1 * (σ1^2 + d1^2) + n2 * (σ2^2 + d2^2)) / (n1 + n2) ]
        # where d1 = μ1 - μ_combined, d2 = μ2 - μ_combined
        # NOTE: the brackets around (d.num_frames / total_samples) are needed tor minimize the risk of
        # numerical overflow!
        stats[data_key]["std"] = torch.sqrt(
            sum(
                (
                    d.meta.stats[data_key]["std"] ** 2
                    + (d.meta.stats[data_key]["mean"] - stats[data_key]["mean"]) ** 2
                )
                * (d.num_frames / total_samples)
                for d in ls_datasets
                if data_key in d.meta.stats
            )
        )
    return stats

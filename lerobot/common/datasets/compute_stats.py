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
import numpy as np

from lerobot.common.datasets.utils import load_image_as_numpy


def compute_episode_stats(episode_buffer: dict, features: dict, num_image_samples: int | None = None) -> dict:
    stats = {}
    for key, data in episode_buffer.items():
        if features[key]["dtype"] in ["image", "video"]:
            stats[key] = compute_image_stats(data, num_samples=num_image_samples)
        else:
            axes_to_reduce = 0  # Compute stats over the first axis
            stats[key] = {
                "min": np.min(data, axis=axes_to_reduce),
                "max": np.max(data, axis=axes_to_reduce),
                "mean": np.mean(data, axis=axes_to_reduce),
                "std": np.std(data, axis=axes_to_reduce),
                "count": np.array([data.shape[0]]),
            }
    return stats


def estimate_num_samples(dataset_len: int, min_num_samples=100, max_num_samples=10_000, power=0.75) -> int:
    """Heuristic to estimate the number of samples based on dataset size.
    The power controls the sample growth relative to dataset sizelower the power for less number of samples.

    For default arguments, we have:
    - from 1 to ~500, num_samples=100
    - at 1000, num_samples=178
    - at 2000, num_samples=299
    - at 5000, num_samples=594
    - at 10000, num_samples=1000
    - at 20000, num_samples=1681
    """
    return max(min_num_samples, min(dataset_len**power, max_num_samples))


def compute_image_stats(image_paths: list[str], num_samples: int | None = None) -> dict:
    num_samples = estimate_num_samples(len(image_paths)) if num_samples is None else num_samples
    num_samples = min(num_samples, len(image_paths))

    step_size = len(image_paths) / num_samples
    sampled_indices = np.arange(0, len(image_paths), step_size).astype(int).tolist()

    images = []
    for idx in sampled_indices:
        path = image_paths[idx]
        img = load_image_as_numpy(path, channel_first=True)
        images.append(img)

    images = np.stack(images)
    axes_to_reduce = (0, 2, 3)  # keep channel dim
    image_stats = {
        "min": np.min(images, axis=axes_to_reduce, keepdims=True),
        "max": np.max(images, axis=axes_to_reduce, keepdims=True),
        "mean": np.mean(images, axis=axes_to_reduce, keepdims=True),
        "std": np.std(images, axis=axes_to_reduce, keepdims=True),
    }
    for key in image_stats:  # squeeze batch dim
        image_stats[key] = np.squeeze(image_stats[key], axis=0)

    image_stats["count"] = np.array([len(images)])
    return image_stats


def aggregate_stats(stats_list: list[dict[str, dict]]) -> dict:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    def _assert_type_and_shape(stats_list):
        for i in range(len(stats_list)):
            for fkey in stats_list[i]:
                for k, v in stats_list[i][fkey].items():
                    if not isinstance(v, np.ndarray):
                        raise ValueError(
                            f"Stats must be composed of numpy array, but key '{k}' of feature '{fkey}' is of type '{type(v)}' instead."
                        )
                    if v.ndim == 0:
                        raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")
                    if k == "count" and v.shape != (1,):
                        raise ValueError(f"Shape of 'count' must be (1), but is {v.shape} instead.")
                    if "image" in k and v.shape != (3, 1, 1):
                        raise ValueError(f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead.")

    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        # Collect stats for the current key from all datasets where it exists
        stats_with_key = [stats[key] for stats in stats_list if key in stats]

        # Aggregate 'min' and 'max' using np.minimum and np.maximum
        aggregated_stats[key]["min"] = np.min(np.stack([s["min"] for s in stats_with_key]), axis=0)
        aggregated_stats[key]["max"] = np.max(np.stack([s["max"] for s in stats_with_key]), axis=0)

        # Extract means, variances (std^2), and counts
        means = np.stack([s["mean"] for s in stats_with_key])
        variances = np.stack([s["std"] ** 2 for s in stats_with_key])
        counts = np.stack([s["count"] for s in stats_with_key])

        # Compute total counts
        total_count = counts.sum(axis=0)

        # Prepare weighted mean by matching number of dimensions
        while counts.ndim < means.ndim:
            counts = np.expand_dims(counts, axis=-1)

        # Compute the weighted mean
        weighted_means = means * counts
        total_mean = weighted_means.sum(axis=0) / total_count

        # Compute the variance using the parallel algorithm
        delta_means = means - total_mean
        weighted_variances = (variances + delta_means**2) * counts
        total_variance = weighted_variances.sum(axis=0) / total_count

        # Store the aggregated stats
        aggregated_stats[key]["mean"] = total_mean
        aggregated_stats[key]["std"] = np.sqrt(total_variance)
        aggregated_stats[key]["count"] = total_count

    return aggregated_stats

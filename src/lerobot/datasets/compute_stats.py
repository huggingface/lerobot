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

from lerobot.datasets.utils import load_image_as_numpy


def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    """Heuristic to estimate the number of samples based on dataset size.
    The power controls the sample growth relative to dataset size.
    Lower the power for less number of samples.

    For default arguments, we have:
    - from 1 to ~500, num_samples=100
    - at 1000, num_samples=177
    - at 2000, num_samples=299
    - at 5000, num_samples=594
    - at 10000, num_samples=1000
    - at 20000, num_samples=1681
    """
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    _, height, width = img.shape

    if max(width, height) < max_size_threshold:
        # no downsampling needed
        return img

    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def sample_images(image_paths: list[str]) -> np.ndarray:
    sampled_indices = sample_indices(len(image_paths))

    images = None
    for i, idx in enumerate(sampled_indices):
        path = image_paths[idx]
        # we load as uint8 to reduce memory usage
        img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
        img = auto_downsample_height_width(img)

        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

        images[i] = img

    return images


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue  # HACK: we should receive np.arrays of strings
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)  # data is a list of image paths
            axes_to_reduce = (0, 2, 3)  # keep channel dim
            keepdims = True
        else:
            ep_ft_array = data  # data is already a np.ndarray
            axes_to_reduce = 0  # compute stats over the first axis
            keepdims = data.ndim == 1  # keep as np.array

        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

        # finally, we normalize and remove batch dim for images
        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats


def _assert_type_and_shape(stats_list: list[dict[str, dict]]):
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
                if "image" in fkey and k != "count" and v.shape != (3, 1, 1):
                    raise ValueError(f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead.")


def aggregate_feature_stats(stats_ft_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregates stats for a single feature."""
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
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

    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def aggregate_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats

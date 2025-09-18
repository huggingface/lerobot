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


class RunningQuantileStats:
    """Compute running statistics including quantiles for a batch of vectors."""

    def __init__(self, num_quantile_bins: int = 5000):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = num_quantile_bins

    def update(self, batch: np.ndarray) -> None:
        """Update the running statistics with a batch of vectors.
        
        Args:
            batch: An array where all dimensions except the last are batch dimensions.
        """
        original_shape = batch.shape
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self, quantiles: list[float] | None = None) -> dict[str, np.ndarray]:
        """Compute and return the statistics of the vectors processed so far.
        
        Args:
            quantiles: List of quantiles to compute (e.g., [0.01, 0.99]). If None, no quantiles computed.
            
        Returns:
            Dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        
        stats = {
            "min": self._min.copy(),
            "max": self._max.copy(), 
            "mean": self._mean.copy(),
            "std": stddev,
            "count": np.array([self._count])
        }
        
        if quantiles is not None:
            quantile_results = self._compute_quantiles(quantiles)
            for i, q in enumerate(quantiles):
                # Format as q01, q99, etc.
                q_key = f"q{int(q*100):02d}"
                stats[q_key] = quantile_results[i]
                
        return stats

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            old_hist = self._histograms[i]
            
            # Create new edges with small padding to ensure range coverage
            padding = (self._max[i] - self._min[i]) * 1e-10
            new_edges = np.linspace(self._min[i] - padding, self._max[i] + padding, self._num_quantile_bins + 1)

            # Redistribute existing histogram counts to new bins
            # We need to map each old bin center to the new bins
            old_centers = (old_edges[:-1] + old_edges[1:]) / 2
            new_hist = np.zeros(self._num_quantile_bins)
            
            for j, (old_center, count) in enumerate(zip(old_centers, old_hist)):
                if count > 0:
                    # Find which new bin this old center belongs to
                    bin_idx = np.searchsorted(new_edges, old_center) - 1
                    bin_idx = max(0, min(bin_idx, self._num_quantile_bins - 1))
                    new_hist[bin_idx] += count

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles: list[float]) -> list[np.ndarray]:
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


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


def get_feature_stats(
    array: np.ndarray, 
    axis: tuple, 
    keepdims: bool, 
    compute_quantiles: bool = False,
    quantiles: list[float] | None = None
) -> dict[str, np.ndarray]:
    """Compute feature statistics, optionally including quantiles.
    
    Args:
        array: Input data array
        axis: Axes along which to compute statistics  
        keepdims: Whether to keep reduced dimensions
        compute_quantiles: Whether to compute quantiles using running stats
        quantiles: List of quantiles to compute (e.g., [0.01, 0.99])
        
    Returns:
        Dictionary containing computed statistics
    """
    if not compute_quantiles:
        # Use original fast computation for basic stats
        return {
            "min": np.min(array, axis=axis, keepdims=keepdims),
            "max": np.max(array, axis=axis, keepdims=keepdims),
            "mean": np.mean(array, axis=axis, keepdims=keepdims),
            "std": np.std(array, axis=axis, keepdims=keepdims),
            "count": np.array([len(array)]),
        }
    
    # Use running stats for quantile computation
    if quantiles is None:
        quantiles = [0.01, 0.99]  # Default to 1st and 99th percentiles
        
    # Reshape array to match RunningQuantileStats expectation
    if axis == (0, 2, 3):  # Image case: (batch, channels, height, width) -> (batch*height*width, channels) 
        batch_size, channels, height, width = array.shape
        reshaped = array.transpose(0, 2, 3, 1).reshape(-1, channels)  # (batch*height*width, channels)
    elif axis == 0:  # Vector case
        if array.ndim == 1:
            # 1D array: reshape to (n_samples, 1) 
            reshaped = array.reshape(-1, 1)
        else:
            # Multi-dimensional: should already be (n_samples, n_features)
            reshaped = array
    else:
        raise ValueError(f"Unsupported axis configuration for quantile computation: {axis}")
    
    # Compute stats using RunningQuantileStats
    running_stats = RunningQuantileStats()
    running_stats.update(reshaped)
    stats = running_stats.get_statistics(quantiles)
    
    # Apply keepdims if needed
    if keepdims and axis == (0, 2, 3):
        # For images, reshape to (1, channels, 1, 1)
        for key in stats:
            if key != "count" and stats[key].ndim == 1:
                stats[key] = stats[key].reshape(1, -1, 1, 1)
    elif keepdims and axis == 0:
        if array.ndim == 1:
            # For 1D vectors with keepdims, stats should be shape (1,) 
            for key in stats:
                if key != "count":
                    stats[key] = stats[key].reshape(1) if stats[key].ndim > 0 else stats[key]
        # For multi-dimensional vectors, keepdims doesn't change shape when axis=0
                
    return stats


def compute_episode_stats(
    episode_data: dict[str, list[str] | np.ndarray], 
    features: dict, 
    compute_quantiles: bool = False,
    quantiles: list[float] | None = None
) -> dict:
    """Compute episode statistics, optionally including quantiles.
    
    Args:
        episode_data: Dictionary containing episode data
        features: Dictionary describing feature types and shapes
        compute_quantiles: Whether to compute quantiles (slower but more informative)
        quantiles: List of quantiles to compute (e.g., [0.01, 0.99])
        
    Returns:
        Dictionary containing computed statistics for each feature
    """
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

        ep_stats[key] = get_feature_stats(
            ep_ft_array, 
            axis=axes_to_reduce, 
            keepdims=keepdims,
            compute_quantiles=compute_quantiles,
            quantiles=quantiles
        )

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
                if "image" in fkey and k != "count" and not k.startswith('q') and v.shape != (3, 1, 1):
                    raise ValueError(f"Shape of '{k}' must be (3,1,1), but is {v.shape} instead.")
                # Allow quantile keys (q01, q99, etc.) to have same shape as other stats
                if "image" in fkey and k.startswith('q') and k[1:].isdigit() and v.shape != (3, 1, 1):
                    raise ValueError(f"Shape of quantile '{k}' must be (3,1,1), but is {v.shape} instead.")


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

    aggregated = {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }
    
    # Handle quantiles if present
    quantile_keys = [k for k in stats_ft_list[0].keys() if k.startswith('q') and k[1:].isdigit()]
    for q_key in quantile_keys:
        # For quantiles, use weighted average as approximation
        # This is not mathematically exact but provides a reasonable estimate
        quantile_values = np.stack([s[q_key] for s in stats_ft_list])
        weighted_quantiles = quantile_values * counts
        aggregated[q_key] = weighted_quantiles.sum(axis=0) / total_count
    
    return aggregated


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

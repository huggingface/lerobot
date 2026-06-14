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
from __future__ import annotations

import logging

import numpy as np

from lerobot.processor import RelativeActionsProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

from .io_utils import load_image_as_numpy

DEFAULT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]

# Emitted at most once per process: see `_warn_quantile_aggregation_is_approximate`.
_QUANTILE_AGGREGATION_WARNED = False


class RunningQuantileStats:
    """
    Maintains running statistics for batches of vectors, including mean,
    standard deviation, min, max, and approximate quantiles.

    Statistics are computed per feature dimension and updated incrementally
    as new batches are observed. Quantiles are estimated using histograms,
    which adapt dynamically if the observed data range expands.
    """

    def __init__(self, quantile_list: list[float] | None = None, num_quantile_bins: int = 5000):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = num_quantile_bins

        self._quantile_list = quantile_list
        if self._quantile_list is None:
            self._quantile_list = DEFAULT_QUANTILES
        self._quantile_keys = [f"q{int(q * 100):02d}" for q in self._quantile_list]

    def update(self, batch: np.ndarray) -> None:
        """Update the running statistics with a batch of vectors.

        Args:
            batch: An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        # Promote integer and low-precision inputs before computing squared statistics.
        batch = batch.astype(np.result_type(batch.dtype, np.float32), copy=False)
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
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )

        self._update_histograms(batch)

    def get_statistics(self) -> dict[str, np.ndarray]:
        """Compute and return the statistics of the vectors processed so far.

        Args:
            quantiles: List of quantiles to compute (e.g., [0.01, 0.10, 0.50, 0.90, 0.99]). If None, no quantiles computed.

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
            "count": np.array([self._count]),
        }

        quantile_results = self._compute_quantiles()
        for i, q in enumerate(self._quantile_keys):
            stats[q] = quantile_results[i]

        return stats

    @staticmethod
    def _make_bin_edges(low: float, high: float, num_bins: int) -> np.ndarray:
        """Build ``num_bins + 1`` edges spanning ``[low, high]`` with tiny padding.

        The padding guarantees strictly increasing edges even for a constant
        feature (``low == high``), so downstream histogram/quantile math is safe.
        """
        span = high - low
        padding = span * 1e-10 if span > 0 else 1e-10
        return np.linspace(low - padding, high + padding, num_bins + 1)

    @staticmethod
    def _rebin_histogram(hist: np.ndarray, old_edges: np.ndarray, new_edges: np.ndarray) -> np.ndarray:
        """Redistribute ``hist`` counts from ``old_edges`` onto ``new_edges``.

        Each old bin's count is assigned to the new bin that contains the old
        bin's center. This is the mapping used both when a single histogram's
        range expands and when two histograms are merged.
        """
        num_bins = len(new_edges) - 1
        old_centers = (old_edges[:-1] + old_edges[1:]) / 2
        bin_idx = np.searchsorted(new_edges, old_centers) - 1
        np.clip(bin_idx, 0, num_bins - 1, out=bin_idx)
        new_hist = np.zeros(num_bins)
        np.add.at(new_hist, bin_idx, hist)
        return new_hist

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            new_edges = self._make_bin_edges(self._min[i], self._max[i], self._num_quantile_bins)
            self._histograms[i] = self._rebin_histogram(self._histograms[i], self._bin_edges[i], new_edges)
            self._bin_edges[i] = new_edges

    def merge(self, other: RunningQuantileStats) -> RunningQuantileStats:
        """Merge another ``RunningQuantileStats`` into this one, in place.

        Combines counts, means, and per-dimension histograms so that the merged
        object yields the same statistics as if every batch seen by ``other``
        had instead been passed to ``self.update``. Crucially, the resulting
        quantiles are *global* over the union of all data (accurate to histogram
        resolution) -- not a count-weighted average of the two objects'
        quantiles, which biases distribution tails inward.

        ``mean``/``std``/``min``/``max`` remain exact; only quantiles carry the
        usual histogram discretization error. Returns ``self`` for chaining.
        """
        if other._count == 0:
            return self
        if self._count == 0:
            self._count = other._count
            self._mean = other._mean.copy()
            self._mean_of_squares = other._mean_of_squares.copy()
            self._min = other._min.copy()
            self._max = other._max.copy()
            self._histograms = [h.copy() for h in other._histograms]
            self._bin_edges = [e.copy() for e in other._bin_edges]
            return self

        if self._num_quantile_bins != other._num_quantile_bins:
            raise ValueError(
                "Cannot merge RunningQuantileStats with different num_quantile_bins "
                f"({self._num_quantile_bins} != {other._num_quantile_bins})."
            )
        if self._mean.size != other._mean.size:
            raise ValueError(
                "Cannot merge RunningQuantileStats with different vector lengths "
                f"({self._mean.size} != {other._mean.size})."
            )

        total_count = self._count + other._count
        # Mean and mean-of-squares are exact count-weighted averages of the two parts.
        self._mean = (self._mean * self._count + other._mean * other._count) / total_count
        self._mean_of_squares = (
            self._mean_of_squares * self._count + other._mean_of_squares * other._count
        ) / total_count

        new_min = np.minimum(self._min, other._min)
        new_max = np.maximum(self._max, other._max)

        # Re-bin both histograms onto the union range, then sum the counts.
        for i in range(len(self._histograms)):
            new_edges = self._make_bin_edges(new_min[i], new_max[i], self._num_quantile_bins)
            self._histograms[i] = self._rebin_histogram(
                self._histograms[i], self._bin_edges[i], new_edges
            ) + self._rebin_histogram(other._histograms[i], other._bin_edges[i], new_edges)
            self._bin_edges[i] = new_edges

        self._min = new_min
        self._max = new_max
        self._count = total_count
        return self

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self) -> list[np.ndarray]:
        """Compute quantiles based on histograms."""
        results = []
        for q in self._quantile_list:
            target_count = q * self._count
            q_values = []

            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                q_value = self._compute_single_quantile(hist, edges, target_count)
                q_values.append(q_value)

            results.append(np.array(q_values))
        return results

    def _compute_single_quantile(self, hist: np.ndarray, edges: np.ndarray, target_count: float) -> float:
        """Compute a single quantile value from histogram and bin edges."""
        cumsum = np.cumsum(hist)
        idx = np.searchsorted(cumsum, target_count)

        if idx == 0:
            return edges[0]
        if idx >= len(cumsum):
            return edges[-1]

        # If not edge case, interpolate within the bin
        count_before = cumsum[idx - 1]
        count_in_bin = cumsum[idx] - count_before

        # If no samples in this bin, use the bin edge
        if count_in_bin == 0:
            return edges[idx]

        # Linear interpolation within the bin
        fraction = (target_count - count_before) / count_in_bin
        return edges[idx] + fraction * (edges[idx + 1] - edges[idx])


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


def _reshape_stats_by_axis(
    stats: dict[str, np.ndarray],
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    original_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Reshape all statistics to match NumPy's output conventions.

    Applies consistent reshaping to all statistics (except 'count') based on the
    axis and keepdims parameters. This ensures statistics have the correct shape
    for broadcasting with the original data.

    Args:
        stats: Dictionary of computed statistics
        axis: Axis or axes along which statistics were computed
        keepdims: Whether to keep reduced dimensions as size-1 dimensions
        original_shape: Shape of the original array

    Returns:
        Dictionary with reshaped statistics

    Note:
        The 'count' statistic is never reshaped as it represents metadata
        rather than per-feature statistics.
    """
    if axis == (1,) and not keepdims:
        return stats

    result = {}
    for key, value in stats.items():
        if key == "count":
            result[key] = value
        else:
            result[key] = _reshape_single_stat(value, axis, keepdims, original_shape)

    return result


def _reshape_for_image_stats(value: np.ndarray, keepdims: bool) -> np.ndarray:
    """Reshape statistics for image data (axis=(0,2,3))."""
    if keepdims and value.ndim == 1:
        return value.reshape(1, -1, 1, 1)
    return value


def _reshape_for_vector_stats(
    value: np.ndarray, keepdims: bool, original_shape: tuple[int, ...]
) -> np.ndarray:
    """Reshape statistics for vector data (axis=0 or axis=(0,))."""
    if not keepdims:
        return value

    if len(original_shape) == 1 and value.ndim > 0:
        return value.reshape(1)
    elif len(original_shape) >= 2 and value.ndim == 1:
        return value.reshape(1, -1)
    return value


def _reshape_for_feature_stats(value: np.ndarray, keepdims: bool) -> np.ndarray:
    """Reshape statistics for feature-wise computation (axis=(1,))."""
    if not keepdims:
        return value

    if value.ndim == 0:
        return value.reshape(1, 1)
    elif value.ndim == 1:
        return value.reshape(-1, 1)
    return value


def _reshape_for_global_stats(
    value: np.ndarray, keepdims: bool, original_shape: tuple[int, ...]
) -> np.ndarray | float:
    """Reshape statistics for global reduction (axis=None)."""
    if keepdims:
        target_shape = tuple(1 for _ in original_shape)
        return value.reshape(target_shape)
    # Keep at least 1-D arrays to satisfy validator
    return np.atleast_1d(value)


def _reshape_single_stat(
    value: np.ndarray, axis: int | tuple[int, ...] | None, keepdims: bool, original_shape: tuple[int, ...]
) -> np.ndarray | float:
    """Apply appropriate reshaping to a single statistic array.

    This function transforms statistic arrays to match expected output shapes
    based on the axis configuration and keepdims parameter.

    Args:
        value: The statistic array to reshape
        axis: Axis or axes that were reduced during computation
        keepdims: Whether to maintain reduced dimensions as size-1 dimensions
        original_shape: Shape of the original data before reduction

    Returns:
        Reshaped array following NumPy broadcasting conventions

    """
    if axis == (0, 2, 3):
        return _reshape_for_image_stats(value, keepdims)

    if axis in [0, (0,)]:
        return _reshape_for_vector_stats(value, keepdims, original_shape)

    if axis == (1,):
        return _reshape_for_feature_stats(value, keepdims)

    if axis is None:
        return _reshape_for_global_stats(value, keepdims, original_shape)

    return value


def _prepare_array_for_stats(array: np.ndarray, axis: int | tuple[int, ...] | None) -> tuple[np.ndarray, int]:
    """Prepare array for statistics computation by reshaping according to axis.

    Args:
        array: Input data array
        axis: Axis or axes along which to compute statistics

    Returns:
        Tuple of (reshaped_array, sample_count)
    """
    if axis == (0, 2, 3):  # Image data
        batch_size, channels, height, width = array.shape
        reshaped = array.transpose(0, 2, 3, 1).reshape(-1, channels)
        return reshaped, batch_size

    if axis == 0 or axis == (0,):  # Vector data
        reshaped = array
        if array.ndim == 1:
            reshaped = array.reshape(-1, 1)
        return reshaped, array.shape[0]

    if axis == (1,):  # Feature-wise statistics
        return array.T, array.shape[1]

    if axis is None:  # Global statistics
        reshaped = array.reshape(-1, 1)
        # For backward compatibility, count represents the first dimension size
        return reshaped, array.shape[0] if array.ndim > 0 else 1

    raise ValueError(f"Unsupported axis configuration: {axis}")


def _compute_basic_stats(
    array: np.ndarray, sample_count: int, quantile_list: list[float] | None = None
) -> dict[str, np.ndarray]:
    """Compute basic statistics for arrays with insufficient samples for quantiles.

    Args:
        array: Reshaped array ready for statistics computation
        sample_count: Number of samples represented in the data

    Returns:
        Dictionary with basic statistics and quantiles set to mean values
    """
    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES
    quantile_list_keys = [f"q{int(q * 100):02d}" for q in quantile_list]

    stats = {
        "min": np.min(array, axis=0),
        "max": np.max(array, axis=0),
        "mean": np.mean(array, axis=0),
        "std": np.std(array, axis=0),
        "count": np.array([sample_count]),
    }

    for q in quantile_list_keys:
        stats[q] = stats["mean"].copy()

    return stats


def get_feature_stats(
    array: np.ndarray,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    quantile_list: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute comprehensive statistics for array features along specified axes.

    This function calculates min, max, mean, std, and quantiles (1%, 10%, 50%, 90%, 99%)
    for the input array along the specified axes. It handles different data layouts:
    - Image data: axis=(0,2,3) computes per-channel statistics
    - Vector data: axis=0 computes per-feature statistics
    - Feature-wise: axis=1 computes statistics across features
    - Global: axis=None computes statistics over entire array

    Args:
        array: Input data array with shape appropriate for the specified axis
        axis: Axis or axes along which to compute statistics
            - (0, 2, 3): For image data (batch, channels, height, width)
            - 0 or (0,): For vector/tabular data (samples, features)
            - (1,): For computing across features
            - None: For global statistics over entire array
        keepdims: If True, reduced axes are kept as dimensions with size 1

    Returns:
        Dictionary containing:
            - 'min': Minimum values
            - 'max': Maximum values
            - 'mean': Mean values
            - 'std': Standard deviation
            - 'count': Number of samples (always shape (1,))
            - 'q01', 'q10', 'q50', 'q90', 'q99': Quantile values

    """
    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES

    original_shape = array.shape
    running_stats, sample_count = compute_feature_running_stats(array, axis, quantile_list=quantile_list)

    if running_stats is None:
        reshaped, _ = _prepare_array_for_stats(array, axis)
        stats = _compute_basic_stats(reshaped, sample_count, quantile_list)
        return _reshape_stats_by_axis(stats, axis, keepdims, original_shape)

    return feature_stats_from_running(running_stats, sample_count, axis, keepdims, original_shape)


def compute_feature_running_stats(
    array: np.ndarray,
    axis: int | tuple[int, ...] | None,
    quantile_list: list[float] | None = None,
) -> tuple[RunningQuantileStats | None, int]:
    """Build a :class:`RunningQuantileStats` accumulator for one feature array.

    The array is reshaped according to ``axis`` (same conventions as
    :func:`get_feature_stats`) and fed to a single accumulator. The returned
    accumulator can be merged with others from different shards/episodes via
    :meth:`RunningQuantileStats.merge` to obtain *global* statistics in a
    single streaming pass, without holding all data in memory at once.

    Args:
        array: Input data array.
        axis: Axis or axes along which to reduce (see :func:`get_feature_stats`).
        quantile_list: Quantiles to track. Defaults to :data:`DEFAULT_QUANTILES`.

    Returns:
        ``(running_stats, sample_count)``. ``running_stats`` is ``None`` when
        fewer than 2 vectors are available (too few for histogram quantiles);
        callers should fall back to :func:`_compute_basic_stats` in that case.
    """
    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES

    reshaped, sample_count = _prepare_array_for_stats(array, axis)
    if reshaped.shape[0] < 2:
        return None, sample_count

    running_stats = RunningQuantileStats(quantile_list=quantile_list)
    running_stats.update(reshaped)
    return running_stats, sample_count


def feature_stats_from_running(
    running_stats: RunningQuantileStats,
    sample_count: int,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    original_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Finalize a (possibly merged) accumulator into a reshaped stats dict.

    Produces the same output layout as :func:`get_feature_stats`. ``count`` is
    overridden with ``sample_count`` (number of samples, e.g. frames) rather
    than the accumulator's internal element count, matching the existing
    convention used throughout the stats pipeline.
    """
    stats = running_stats.get_statistics()
    stats["count"] = np.array([sample_count])
    return _reshape_stats_by_axis(stats, axis, keepdims, original_shape)


def aggregate_quantile_stats(running_stats_list: list[RunningQuantileStats]) -> RunningQuantileStats:
    """Merge per-shard accumulators into one with correct global quantiles.

    This is the statistically correct counterpart to the quantile handling in
    :func:`aggregate_feature_stats`: instead of averaging already-computed
    per-shard quantiles (which biases tails), it merges the underlying
    histograms so the result reflects the true global distribution.
    """
    if not running_stats_list:
        raise ValueError("Cannot aggregate an empty list of RunningQuantileStats.")

    merged = RunningQuantileStats(
        quantile_list=running_stats_list[0]._quantile_list,
        num_quantile_bins=running_stats_list[0]._num_quantile_bins,
    )
    for running_stats in running_stats_list:
        merged.merge(running_stats)
    return merged


def compute_episode_stats(
    episode_data: dict[str, list[str] | np.ndarray],
    features: dict,
    quantile_list: list[float] | None = None,
) -> dict:
    """Compute comprehensive statistics for all features in an episode.

    Processes different data types appropriately:
    - Images/videos: Samples from paths, computes per-channel stats, normalizes to [0,1]
    - Numerical arrays: Computes per-feature statistics
    - Strings: Skipped (no statistics computed)

    Args:
        episode_data: Dictionary mapping feature names to data
            - For images/videos: list of file paths
            - For numerical data: numpy arrays
        features: Dictionary describing each feature's dtype and shape

    Returns:
        Dictionary mapping feature names to their statistics dictionaries.
        Each statistics dictionary contains min, max, mean, std, count, and quantiles.

    Note:
        Image statistics are normalized to [0,1] range and have shape (3,1,1) for
        per-channel values when dtype is 'image' or 'video'.
    """
    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES

    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] in {"string", "language"}:
            continue

        if features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            ep_ft_array = data
            axes_to_reduce = 0
            keepdims = data.ndim == 1

        ep_stats[key] = get_feature_stats(
            ep_ft_array, axis=axes_to_reduce, keepdims=keepdims, quantile_list=quantile_list
        )

        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats


def _validate_stat_value(value: np.ndarray, key: str, feature_key: str) -> None:
    """Validate a single statistic value."""
    if not isinstance(value, np.ndarray):
        raise ValueError(
            f"Stats must be composed of numpy array, but key '{key}' of feature '{feature_key}' "
            f"is of type '{type(value)}' instead."
        )

    if value.ndim == 0:
        raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")

    if key == "count" and value.shape != (1,):
        raise ValueError(f"Shape of 'count' must be (1), but is {value.shape} instead.")

    if "image" in feature_key and key != "count" and value.shape != (3, 1, 1):
        raise ValueError(f"Shape of quantile '{key}' must be (3,1,1), but is {value.shape} instead.")


def _assert_type_and_shape(stats_list: list[dict[str, dict]]):
    """Validate that all statistics have correct types and shapes.

    Args:
        stats_list: List of statistics dictionaries to validate

    Raises:
        ValueError: If any statistic has incorrect type or shape
    """
    for stats in stats_list:
        for feature_key, feature_stats in stats.items():
            for stat_key, stat_value in feature_stats.items():
                _validate_stat_value(stat_value, stat_key, feature_key)


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

    if stats_ft_list:
        quantile_keys = [k for k in stats_ft_list[0] if k.startswith("q") and k[1:].isdigit()]

        if quantile_keys and len(stats_ft_list) > 1:
            _warn_quantile_aggregation_is_approximate()

        for q_key in quantile_keys:
            if all(q_key in s for s in stats_ft_list):
                # WARNING: this count-weighted average of per-shard quantiles is only an
                # APPROXIMATION. The mean of per-episode q01 values is not the global q01;
                # averaging pulls the distribution tails inward (q01 too high, q99 too low).
                # Exact global quantiles cannot be recovered from already-summarized stats
                # dicts -- they require the underlying data/histograms. For quantile-normalized
                # policies (e.g. pi0 / pi0.5) recompute true global quantiles by merging
                # histograms instead (see `aggregate_quantile_stats` /
                # `scripts/augment_dataset_quantile_stats.py`).
                quantile_values = np.stack([s[q_key] for s in stats_ft_list])
                weighted_quantiles = quantile_values * counts
                aggregated[q_key] = weighted_quantiles.sum(axis=0) / total_count

    return aggregated


def _warn_quantile_aggregation_is_approximate() -> None:
    """Warn once per process that aggregated quantiles are an approximation."""
    global _QUANTILE_AGGREGATION_WARNED
    if _QUANTILE_AGGREGATION_WARNED:
        return
    _QUANTILE_AGGREGATION_WARNED = True
    logging.warning(
        "aggregate_stats() approximates aggregated quantiles by count-weighted averaging of "
        "per-shard quantiles, which biases distribution tails inward (q01 too high, q99 too low). "
        "Quantile-normalized policies (e.g. pi0 / pi0.5) require TRUE global quantiles -- recompute "
        "them from data with `python -m lerobot.scripts.augment_dataset_quantile_stats "
        "--repo-id <repo_id> --overwrite`."
    )


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


def _get_valid_chunk_starts(episode_indices: np.ndarray, chunk_size: int) -> np.ndarray:
    """Return all start indices where a chunk of ``chunk_size`` stays within one episode."""
    total = len(episode_indices)
    if total < chunk_size:
        return np.array([], dtype=np.int64)
    max_start = total - chunk_size
    starts = np.arange(max_start + 1)
    valid = episode_indices[starts] == episode_indices[starts + chunk_size - 1]
    return starts[valid]


def _compute_relative_chunk_batch(
    start_indices: np.ndarray,
    all_actions: np.ndarray,
    all_states: np.ndarray,
    chunk_size: int,
    relative_mask: np.ndarray,
) -> np.ndarray:
    """Vectorised relative-action computation for a batch of start indices.

    Returns an ``(N * chunk_size, action_dim)`` float32 array.
    """
    if len(start_indices) == 0:
        return np.empty((0, all_actions.shape[1]), dtype=np.float32)
    offsets = np.arange(chunk_size)
    frame_idx = start_indices[:, None] + offsets[None, :]
    chunks = all_actions[frame_idx].copy()
    states = all_states[start_indices]
    mask_dim = len(relative_mask)
    chunks[:, :, :mask_dim] -= states[:, None, :mask_dim] * relative_mask[None, None, :]
    return chunks.reshape(-1, all_actions.shape[1])


def compute_relative_action_stats(
    hf_dataset,
    features: dict,
    chunk_size: int,
    exclude_joints: list[str] | None = None,
    num_workers: int = 0,
) -> dict[str, np.ndarray]:
    """Compute normalization statistics for relative actions over the full dataset.

    Iterates *all* valid action chunks (within single episodes), converts them to
    relative actions (action − current_state), and computes per-dimension
    statistics suitable for normalization.

    Args:
        hf_dataset: The underlying HuggingFace dataset with "action",
            "observation.state", and "episode_index" columns.
        features: Dataset feature metadata (must contain "action" with "shape"
            and optionally "names").
        chunk_size: Number of consecutive frames per action chunk.
        exclude_joints: Joint names whose dimensions should remain absolute
            (not converted to relative actions).
        num_workers: Number of parallel threads for computation. Values ≤1
            mean single-threaded. Numpy releases the GIL so threads give
            real parallelism here.

    Returns:
        Statistics dict with keys "mean", "std", "min", "max", "q01", …, "q99".

    Raises:
        ValueError: If the dataset has fewer frames than ``chunk_size``.
        RuntimeError: If no valid (single-episode) chunks are found.
    """
    if exclude_joints is None:
        exclude_joints = []

    action_dim = features[ACTION]["shape"][0]
    action_names = features.get(ACTION, {}).get("names")
    mask_step = RelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=exclude_joints,
        action_names=action_names,
    )
    relative_mask = np.array(mask_step._build_mask(action_dim), dtype=np.float32)

    logging.info("Loading action/state data for relative action stats...")
    all_actions = np.array(hf_dataset[ACTION], dtype=np.float32)
    all_states = np.array(hf_dataset[OBS_STATE], dtype=np.float32)
    episode_indices = np.array(hf_dataset["episode_index"])

    valid_starts = _get_valid_chunk_starts(episode_indices, chunk_size)
    if len(valid_starts) == 0:
        raise RuntimeError(
            f"No valid chunks found (total_frames={len(episode_indices)}, chunk_size={chunk_size})"
        )

    effective_workers = max(num_workers, 1)
    logging.info(
        f"Computing relative action stats from {len(valid_starts)} chunks "
        f"(chunk_size={chunk_size}, workers={effective_workers})"
    )

    batch_size = 50_000
    batches = [valid_starts[i : i + batch_size] for i in range(0, len(valid_starts), batch_size)]

    running_stats = RunningQuantileStats()

    if num_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [
                pool.submit(
                    _compute_relative_chunk_batch,
                    batch,
                    all_actions,
                    all_states,
                    chunk_size,
                    relative_mask,
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                running_stats.update(future.result())
    else:
        for batch in batches:
            running_stats.update(
                _compute_relative_chunk_batch(batch, all_actions, all_states, chunk_size, relative_mask)
            )

    stats = running_stats.get_statistics()

    excluded_dims = int(len(relative_mask) - relative_mask.sum())
    total_frames = len(valid_starts) * chunk_size
    logging.info(
        f"Relative action stats ({len(valid_starts)} chunks, {total_frames} frames): "
        f"relative_dims={int(relative_mask.sum())}/{len(relative_mask)} (excluded={excluded_dims}), "
        f"mean={np.abs(stats['mean']).mean():.4f}, std={stats['std'].mean():.4f}, "
        f"q01={stats['q01'].mean():.4f}, q99={stats['q99'].mean():.4f}"
    )

    return stats

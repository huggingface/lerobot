# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Online normalization-statistics accumulator, ported from upstream LingBot-VLA 2.0.

This is a line-for-line port of ``lingbotvla/utils/normalize.py`` (upstream repo
``Robbyant/lingbot-vla-v2``), with the numpydantic/pydantic serialization
replaced by plain dataclasses so this policy has no extra dependency. Numerical
behavior is intentionally identical — mean/variance use batchwise weighted
updates in float64, quantiles come from a 5000-bin online histogram that rebins
whenever the observed range grows, and ``merge`` redistributes per-shard
histograms onto unified edges. The test suite asserts parity against the
upstream implementation for fixed seeds.

Output shape: ``{"norm_stats": {key: {"mean": [...], "std": [...], "q01": ...,
"q99": ..., "q02": ..., "q98": ..., "min": ..., "max": ...}}}``, keys being the
canonical feature names (``action.arm.position`` etc.) produced by
``FeatureTransform.apply`` with ``return_item_before_padding=True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None  # 1st quantile
    q99: np.ndarray | None = None  # 99th quantile
    q02: np.ndarray | None = None  # 2nd quantile
    q98: np.ndarray | None = None  # 98th quantile
    min: np.ndarray | None = None
    max: np.ndarray | None = None


@dataclass
class RunningStatsState:
    """Serializable snapshot of a RunningStats accumulator (pydantic-free)."""

    count: int
    mean: np.ndarray
    mean_of_squares: np.ndarray
    min_val: np.ndarray
    max_val: np.ndarray
    histograms: np.ndarray  # Shape: (vector_length, num_bins)
    bin_edges: np.ndarray  # Shape: (vector_length, num_bins + 1)
    num_quantile_bins: int


class RunningStats:
    """Compute running statistics of a batch of vectors.

    Ported from upstream ``lingbotvla.utils.normalize.RunningStats`` — update /
    merge / quantile formulas kept identical so stats computed here match the
    upstream recipe bit-for-bit (given the same data order and float64 dtype).
    """

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors: A 2D array where each row is a new vector.
        """
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)

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

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self, chunk_size=None) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        q02, q98 = self._compute_quantiles([0.02, 0.98])

        if chunk_size is not None:
            assert isinstance(chunk_size, int)
            self._mean = self._mean.reshape(chunk_size, -1)
            self._min = self._min.reshape(chunk_size, -1)
            self._max = self._max.reshape(chunk_size, -1)
            stddev = stddev.reshape(chunk_size, -1)
            q01 = q01.reshape(chunk_size, -1)
            q99 = q99.reshape(chunk_size, -1)
            q02 = q02.reshape(chunk_size, -1)
            q98 = q98.reshape(chunk_size, -1)

        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99, q02=q02, q98=q98, min=self._min, max=self._max)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
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

    def get_state(self) -> RunningStatsState:
        """Get all current internal states"""
        if self._count == 0:
            raise ValueError("No data processed yet.")
        return RunningStatsState(
            count=self._count,
            mean=self._mean,
            mean_of_squares=self._mean_of_squares,
            min_val=self._min,
            max_val=self._max,
            histograms=np.stack(self._histograms, axis=0),
            bin_edges=np.stack(self._bin_edges, axis=0),
            num_quantile_bins=self._num_quantile_bins,
        )

    @classmethod
    def from_state(cls, state: RunningStatsState):
        """Restore a RunningStats object from its state"""
        instance = cls()
        instance._num_quantile_bins = state.num_quantile_bins
        instance._count = state.count
        instance._mean = np.asarray(state.mean)
        instance._mean_of_squares = np.asarray(state.mean_of_squares)
        instance._min = np.asarray(state.min_val)
        instance._max = np.asarray(state.max_val)
        # After numpydantic serialization, histograms/bin_edges become a single 2D array.
        # Internally we split it back into a list[1D-array] per dim, so that
        # _update_histograms / _adjust_histograms can be reused.
        hist = np.asarray(state.histograms)
        edges = np.asarray(state.bin_edges)
        instance._histograms = [hist[i] for i in range(hist.shape[0])]
        instance._bin_edges = [edges[i] for i in range(edges.shape[0])]
        return instance

    @classmethod
    def merge(cls, others: list[RunningStats]) -> RunningStats:
        """Merge multiple RunningStats (typical use: aggregating across ranks).

        Merge formula (per-dim):
            count = Σ cᵢ
            mean = Σ cᵢ·meanᵢ / count
            mean_of_squares = Σ cᵢ·msᵢ / count
            min/max = elementwise min/max
            histograms = rebin each shard's histogram onto unified new_edges, then sum
        """
        valid = [o for o in others if o is not None and o._count > 0]
        if not valid:
            raise ValueError("merge() requires at least one non-empty RunningStats.")
        if len(valid) == 1:
            return valid[0]

        num_bins = valid[0]._num_quantile_bins
        assert all(o._num_quantile_bins == num_bins for o in valid), (
            "All RunningStats must share the same num_quantile_bins to merge."
        )
        vector_length = valid[0]._mean.size
        assert all(o._mean.size == vector_length for o in valid), (
            "All RunningStats must share the same vector length to merge."
        )

        counts = np.array([o._count for o in valid], dtype=np.float64)
        total_count = counts.sum()
        weights = counts / total_count

        merged_mean = sum(w * o._mean for w, o in zip(weights, valid))
        merged_ms = sum(w * o._mean_of_squares for w, o in zip(weights, valid))
        merged_min = np.minimum.reduce([o._min for o in valid])
        merged_max = np.maximum.reduce([o._max for o in valid])

        # Leave a little padding for linspace, consistent with update() init logic
        merged_histograms = []
        merged_bin_edges = []
        for dim in range(vector_length):
            new_edges = np.linspace(merged_min[dim] - 1e-10, merged_max[dim] + 1e-10, num_bins + 1)
            acc = np.zeros(num_bins)
            for o in valid:
                old_edges = o._bin_edges[dim]
                old_hist = o._histograms[dim]
                # Same rebin approach as _adjust_histograms
                rebinned, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=old_hist)
                acc += rebinned
            merged_histograms.append(acc)
            merged_bin_edges.append(new_edges)

        instance = cls()
        instance._num_quantile_bins = num_bins
        instance._count = int(total_count)
        instance._mean = merged_mean
        instance._mean_of_squares = merged_ms
        instance._min = merged_min
        instance._max = merged_max
        instance._histograms = merged_histograms
        instance._bin_edges = merged_bin_edges
        return instance


def serialize_json(norm_stats: dict[str, NormStats], count: int) -> str:
    """Serialize the running statistics to a JSON string."""

    def _entry(stats: NormStats) -> dict:
        def _tolist(arr):
            return None if arr is None else np.asarray(arr).tolist()

        return {
            "mean": _tolist(stats.mean),
            "std": _tolist(stats.std),
            "q01": _tolist(stats.q01),
            "q99": _tolist(stats.q99),
            "q02": _tolist(stats.q02),
            "q98": _tolist(stats.q98),
            "min": _tolist(stats.min),
            "max": _tolist(stats.max),
        }

    payload = {"norm_stats": {key: _entry(stats) for key, stats in norm_stats.items()}, "count": count}
    return json.dumps(payload, indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    payload = json.loads(data)
    return {
        key: NormStats(
            mean=np.asarray(v["mean"], dtype=np.float64),
            std=np.asarray(v["std"], dtype=np.float64),
            q01=np.asarray(v["q01"], dtype=np.float64) if v.get("q01") is not None else None,
            q99=np.asarray(v["q99"], dtype=np.float64) if v.get("q99") is not None else None,
            q02=np.asarray(v["q02"], dtype=np.float64) if v.get("q02") is not None else None,
            q98=np.asarray(v["q98"], dtype=np.float64) if v.get("q98") is not None else None,
            min=np.asarray(v["min"], dtype=np.float64) if v.get("min") is not None else None,
            max=np.asarray(v["max"], dtype=np.float64) if v.get("max") is not None else None,
        )
        for key, v in payload["norm_stats"].items()
    }


def save(path: Path | str, norm_stats: dict[str, NormStats], count: int) -> None:
    """Save the normalization stats to a JSON file (upstream-compatible layout)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats, count))


def load(path: Path | str) -> dict[str, NormStats]:
    """Load normalization stats from a norm_stats.json file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())


def save_running_state(path: Path | str, stats: dict[str, RunningStats]):
    """Save the full computed intermediate state to JSON (per-key)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _state_to_dict(s: RunningStatsState) -> dict:
        return {
            "count": s.count,
            "mean": np.asarray(s.mean).tolist(),
            "mean_of_squares": np.asarray(s.mean_of_squares).tolist(),
            "min_val": np.asarray(s.min_val).tolist(),
            "max_val": np.asarray(s.max_val).tolist(),
            "histograms": np.asarray(s.histograms).tolist(),
            "bin_edges": np.asarray(s.bin_edges).tolist(),
            "num_quantile_bins": s.num_quantile_bins,
        }

    path.write_text(json.dumps({k: _state_to_dict(v.get_state()) for k, v in stats.items()}))


def load_running_state(path: Path | str) -> dict[str, RunningStats]:
    """Load intermediate state from JSON and restore a RunningStats object per key."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")
    data = json.loads(path.read_text())

    def _from_dict(d: dict) -> RunningStats:
        return RunningStats.from_state(
            RunningStatsState(
                count=d["count"],
                mean=np.asarray(d["mean"]),
                mean_of_squares=np.asarray(d["mean_of_squares"]),
                min_val=np.asarray(d["min_val"]),
                max_val=np.asarray(d["max_val"]),
                histograms=np.asarray(d["histograms"]),
                bin_edges=np.asarray(d["bin_edges"]),
                num_quantile_bins=d["num_quantile_bins"],
            )
        )

    return {key: _from_dict(state) for key, state in data.items()}

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

"""
RA-BC (Reward-Aligned Behavior Cloning) sample weighting implementation.

This module implements the SampleWeighter protocol for RA-BC training,
which weights training samples based on their task progress as measured
by the SARM reward model.

The weights are computed based on progress deltas:
    delta = progress[t + chunk_size] - progress[t]

High-quality samples (positive progress) get higher weights, while
samples with negative progress (going backwards) get zero weight.

See: https://arxiv.org/abs/2509.25358 for the SARM paper.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from lerobot.utils.import_utils import _av_available, _datasets_available, require_package
from lerobot.utils.sample_weighting import SampleWeighter

from ..scoring import load_signal_descriptors

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

if TYPE_CHECKING or (_datasets_available and _av_available):
    from lerobot.datasets.scores import load_scores


class RABCWeights(SampleWeighter):
    """
    Load precomputed SARM progress values and compute RA-BC weights during training.

    This class implements the SampleWeighter ABC for use with the generic
    sample weighting infrastructure in lerobot.

    Progress values are loaded from named dataset scores.
    During training, computes:
        - progress_delta = progress[t + chunk_size] - progress[t]
        - rabc_weight based on the delta (paper Eq. 8-9)

    Args:
        dataset: Dataset containing the scores.
        score_name: Name of the dataset scores.
        signal_name: Progress signal selected from the Parquet file of scores.
        chunk_size: Number of frames ahead for computing progress delta.
        kappa: Hard threshold for high-quality samples (default: 0.01).
        epsilon: Small constant for numerical stability (default: 1e-6).
        fallback_weight: Weight to use for frames without valid delta (default: 1.0).
        device: Device to return tensors on.
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        score_name: str,
        signal_name: str,
        chunk_size: int = 50,
        kappa: float = 0.01,
        epsilon: float = 1e-6,
        fallback_weight: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        require_package("datasets", extra="dataset")
        require_package("av", extra="dataset")
        self.score_name = score_name
        self.signal_name = signal_name
        self.chunk_size = chunk_size
        self.kappa = kappa
        self.epsilon = epsilon
        self.fallback_weight = fallback_weight
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info("Loading signal %s from dataset scores %s", signal_name, score_name)
        descriptors = load_signal_descriptors(dataset, score_name)

        if signal_name not in descriptors:
            raise ValueError(
                f"Signal '{signal_name}' not found in scores '{score_name}'. "
                f"Available signals: {sorted(descriptors)}"
            )
        descriptor = descriptors[signal_name]
        if descriptor.greater_is_better is not True:
            raise ValueError(
                f"RA-BC requires a signal where higher values indicate better progress, "
                f"but {signal_name!r} declares greater_is_better={descriptor.greater_is_better!r}"
            )

        # `load_scores` rejects scores whose rows no longer match the dataset's episodes.
        scores = load_scores(
            dataset, score_name, columns=["index", "episode_index", signal_name]
        ).with_format("numpy")
        # Whole-column reads in NumPy format; converting a column with `np.asarray` iterates in Python.
        global_indices = scores["index"][:]
        progress_values = scores[signal_name][:]
        episode_indices = scores["episode_index"][:]
        if progress_values.dtype.kind not in "iuf":
            raise ValueError(f"RA-BC signal {signal_name!r} must be numeric")
        # NaN marks a missing value, but one infinite value would make every weight NaN.
        if np.isinf(progress_values).any():
            raise ValueError(f"RA-BC signal {signal_name!r} in scores {score_name!r} has infinite values")

        self.progress_lookup: dict[int, float] = {}
        self.episode_lookup: dict[int, int] = {}

        for global_idx, progress, episode_idx in zip(
            global_indices, progress_values, episode_indices, strict=True
        ):
            global_idx = int(global_idx)
            episode_idx = int(episode_idx)

            if not np.isnan(progress):
                self.progress_lookup[global_idx] = float(progress)
            self.episode_lookup[global_idx] = episode_idx

        # Build episode boundaries for delta computation, in one pass over the rows sorted by episode.
        order = np.argsort(episode_indices, kind="stable")
        scored_episodes, first_rows = np.unique(episode_indices[order], return_index=True)
        sorted_indices = global_indices[order]
        starts = np.minimum.reduceat(sorted_indices, first_rows)
        ends = np.maximum.reduceat(sorted_indices, first_rows) + 1
        self.episode_boundaries: dict[int, dict[str, int]] = {
            int(episode_idx): {"start": int(start), "end": int(end)}
            for episode_idx, start, end in zip(scored_episodes, starts, ends, strict=True)
        }

        # Frames of unscored episodes get `fallback_weight`, for example after appending episodes
        # without scoring them and merging their scores.
        selected = range(dataset.meta.total_episodes) if dataset.episodes is None else dataset.episodes
        unscored = sorted(set(selected) - set(self.episode_boundaries))
        if unscored:
            logging.warning(
                "Scores %r have no values for %d episode(s) %s; their frames get weight %s",
                score_name,
                len(unscored),
                unscored[:10],
                fallback_weight,
            )

        logging.info(f"Loaded {len(self.progress_lookup)} frame progress values")
        logging.info(f"Chunk size for delta computation: {chunk_size}")

        # Compute global statistics for weight computation
        self._compute_global_stats()

    def _compute_global_stats(self) -> None:
        """Compute global mean and std of progress deltas for weight calculation."""
        all_deltas = []

        for global_idx, progress in self.progress_lookup.items():
            episode_idx = self.episode_lookup.get(global_idx)
            if episode_idx is None:
                continue

            bounds = self.episode_boundaries.get(episode_idx)
            if bounds is None:
                continue

            future_idx = global_idx + self.chunk_size
            if future_idx >= bounds["end"]:
                # Near end of episode: use last frame's progress
                future_idx = bounds["end"] - 1

            future_progress = self.progress_lookup.get(future_idx)
            if future_progress is not None:
                delta = future_progress - progress
                all_deltas.append(delta)

        if all_deltas:
            self.delta_mean = max(float(np.mean(all_deltas)), 0.0)
            self.delta_std = max(float(np.std(all_deltas)), self.epsilon)
            logging.info(f"Progress delta stats: mean={self.delta_mean:.4f}, std={self.delta_std:.4f}")
        else:
            self.delta_mean = 0.0
            self.delta_std = self.epsilon
            logging.warning("No valid progress deltas found, using default stats")

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        Compute RA-BC weights for a batch.

        For each sample:
        1. Get progress at current frame
        2. Get progress at frame + chunk_size (within same episode)
        3. Compute delta = future_progress - current_progress
        4. Compute weight using paper Eq. 8-9

        Args:
            batch: Training batch containing "index" key with global frame indices.

        Returns:
            Tuple of:
            - Weights tensor (batch_size,) normalized to sum to batch_size.
            - Stats dict with weighting statistics for logging.
        """
        indices = batch.get("index")
        if indices is None:
            logging.warning("RA-BC: Batch missing 'index' key, using uniform weights")
            batch_size = self._get_batch_size(batch)
            stats = {"mean_weight": 1.0, "num_zero_weight": 0, "num_full_weight": batch_size}
            return torch.ones(batch_size, device=self.device), stats

        # Convert to list of ints
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()

        # Compute deltas and weights for each sample
        deltas = []
        for idx in indices:
            idx = int(idx)
            delta = self._compute_delta(idx)
            deltas.append(delta)

        deltas_array = np.array(deltas, dtype=np.float32)

        # Compute weights from deltas
        weights = self._compute_weights(deltas_array)

        # Compute stats before normalization for logging
        raw_mean_weight = float(np.nanmean(weights))
        num_zero_weight = int(np.sum(weights == 0))
        num_full_weight = int(np.sum(weights == 1.0))
        batch_stats = {
            "mean_weight": raw_mean_weight,
            "num_zero_weight": num_zero_weight,
            "num_full_weight": num_full_weight,
        }

        weights_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32)

        # Normalize to sum to batch_size
        batch_size = len(weights_tensor)
        weight_sum = weights_tensor.sum() + self.epsilon
        weights_tensor = weights_tensor * batch_size / weight_sum

        return weights_tensor, batch_stats

    def _compute_delta(self, global_idx: int) -> float:
        """Compute progress delta for a single frame."""
        current_progress = self.progress_lookup.get(global_idx)
        if current_progress is None:
            return np.nan

        episode_idx = self.episode_lookup.get(global_idx)
        if episode_idx is None:
            return np.nan

        bounds = self.episode_boundaries.get(episode_idx)
        if bounds is None:
            return np.nan

        future_idx = global_idx + self.chunk_size  # Δ = chunk_size
        if future_idx >= bounds["end"]:
            # Near end of episode: use last frame's progress instead
            future_idx = bounds["end"] - 1

        future_progress = self.progress_lookup.get(future_idx)
        if future_progress is None:
            return np.nan

        return future_progress - current_progress

    def _compute_weights(self, deltas: np.ndarray) -> np.ndarray:
        """
        Compute RA-BC weights from progress deltas.

        Following paper Eq. 8-9:
        - Soft weight: ˜wi = clip((ri − (µ − 2σ)) / (4σ + ε), 0, 1)
        - Final weight: wi = 1{ri > κ} + 1{0 ≤ ri ≤ κ}˜wi

        Returns:
            Array of weights.
        """
        valid_mask = ~np.isnan(deltas)

        # Compute soft weights using global statistics
        lower_bound = self.delta_mean - 2 * self.delta_std
        soft_weights = (deltas - lower_bound) / (4 * self.delta_std + self.epsilon)
        soft_weights = np.clip(soft_weights, 0.0, 1.0)

        # Apply paper's Eq. 9
        weights = np.zeros_like(deltas, dtype=np.float32)

        # High quality: ri > kappa → weight = 1
        high_quality_mask = deltas > self.kappa
        weights[high_quality_mask] = 1.0

        # Moderate quality: 0 <= ri <= kappa → weight = soft_weight
        moderate_mask = (deltas >= 0) & (deltas <= self.kappa)
        weights[moderate_mask] = soft_weights[moderate_mask]

        # Negative progress: ri < 0 → weight = 0 (already 0)
        # Invalid (NaN): use fallback weight
        weights[~valid_mask] = self.fallback_weight

        return weights

    def _get_batch_size(self, batch: dict) -> int:
        """Determine batch size from batch."""
        for key in ["action", "index"]:
            if key in batch:
                val = batch[key]
                if isinstance(val, (torch.Tensor, np.ndarray)):
                    return int(val.shape[0])
        return 1

    def get_stats(self) -> dict:
        """Get global statistics about the RA-BC weighting."""
        return {
            "type": "rabc",
            "num_frames": len(self.progress_lookup),
            "chunk_size": self.chunk_size,
            "score_name": self.score_name,
            "signal_name": self.signal_name,
            "delta_mean": self.delta_mean,
            "delta_std": self.delta_std,
            "kappa": self.kappa,
        }

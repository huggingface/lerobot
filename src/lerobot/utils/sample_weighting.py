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
Sample weighting abstraction for training.

This module provides an abstract base class for sample weighting strategies (e.g., RA-BC)
that can be used during training without polluting the training script with
policy-specific code.

Example usage:
    # In training config
    sample_weighting:
        type: rabc
        progress_path: hf://datasets/my-dataset/sarm_progress.parquet
        head_mode: sparse
        kappa: 0.01

    # In training script
    sample_weighter = make_sample_weighter(cfg.sample_weighting, policy, device, dataset_root=cfg.dataset.root, dataset_repo_id=cfg.dataset.repo_id)
    ...
    weights, stats = sample_weighter.compute_batch_weights(batch)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy


class SampleWeighter(ABC):
    """
    Implementations compute per-sample weights that can be used to weight
    the loss during training. This enables techniques like:
    - RA-BC (Reward-Aligned Behavior Cloning)
    - Importance sampling
    - Curriculum learning
    - Quality-based filtering
    """

    @abstractmethod
    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        Compute per-sample weights for a training batch.

        Args:
            batch: Training batch dictionary containing at minimum an "index" key
                   with global frame indices.
        """

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Get global statistics about the weighting strategy.
        """


@dataclass
class SampleWeightingConfig:
    """
    Configuration for sample weighting during training.

    This is a generic config that supports multiple weighting strategies.
    The `type` field determines which implementation to use, and `extra_params`
    contains additional type-specific parameters.

    Attributes:
        type: Weighting strategy type ("rabc", "static", "uniform", etc.)
        progress_path: Path to precomputed progress values (for RABC)
        weights_path: Path to precomputed per-frame weights (for the static
            weighter): a parquet with an "index" column (global frame index,
            matching the batch's "index" key) and a "weight" column.
        head_mode: Which model head to use for progress ("sparse" or "dense")
        kappa: Hard threshold for high-quality samples (RABC-specific)
        epsilon: Small constant for numerical stability
        extra_params: Additional type-specific parameters passed to the weighter
    """

    type: str = "rabc"
    progress_path: str | None = None
    weights_path: str | None = None
    head_mode: str = "sparse"
    kappa: float = 0.01
    epsilon: float = 1e-6
    # Additional type-specific params can be added here or passed via extra_params
    extra_params: dict = field(default_factory=dict)


def make_sample_weighter(
    config: SampleWeightingConfig | None,
    policy: PreTrainedPolicy,
    device: torch.device,
    dataset_root: str | None = None,
    dataset_repo_id: str | None = None,
) -> SampleWeighter | None:
    """
    Factory function to create a SampleWeighter from config.

    This keeps policy-specific initialization logic out of the training script.

    Args:
        config: Sample weighting configuration, or None to disable weighting.
        policy: The policy being trained (used to extract chunk_size, etc.)
        device: Device to place weight tensors on.
        dataset_root: Local path to dataset root (for auto-detecting progress_path).
        dataset_repo_id: HuggingFace repo ID (for auto-detecting progress_path).
    """
    if config is None:
        return None

    if config.type == "rabc":
        return _make_rabc_weighter(config, policy, device, dataset_root, dataset_repo_id)

    if config.type == "static":
        if not config.weights_path:
            raise ValueError(
                "Static sample weighting requires 'weights_path' to be set: a "
                "parquet with 'index' (global frame index) and 'weight' columns."
            )
        return StaticWeights(
            weights_path=config.weights_path,
            device=device,
            epsilon=config.epsilon,
            **config.extra_params,
        )

    if config.type == "uniform":
        # No-op weighter that returns uniform weights
        return UniformWeighter(device=device)

    raise ValueError(
        f"Unknown sample weighting type: '{config.type}'. Supported types: 'rabc', 'static', 'uniform'"
    )


def _make_rabc_weighter(
    config: SampleWeightingConfig,
    policy: PreTrainedPolicy,
    device: torch.device,
    dataset_root: str | None = None,
    dataset_repo_id: str | None = None,
) -> SampleWeighter:
    """Create RABC weighter with policy-specific initialization.

    Args:
        config: Sample weighting configuration.
        policy: The policy being trained (used to extract chunk_size).
        device: Device to place weight tensors on.
        dataset_root: Local path to dataset root (for auto-detecting progress_path).
        dataset_repo_id: HuggingFace repo ID (for auto-detecting progress_path).
    """
    # Import here to avoid circular imports and keep RABC code in SARM module
    from lerobot.rewards.sarm.rabc import RABCWeights

    # Extract chunk_size from policy config
    chunk_size = getattr(policy.config, "chunk_size", None)
    if chunk_size is None:
        raise ValueError(
            "RABC sample weighting requires a policy with 'chunk_size' in its config. "
            "This is typically set for action-chunking policies like ACT, Diffusion, PI0, etc."
        )

    # Determine progress_path: use explicit config or auto-detect from dataset
    progress_path = config.progress_path
    if progress_path is None:
        if dataset_root:
            progress_path = str(Path(dataset_root) / "sarm_progress.parquet")
        elif dataset_repo_id:
            progress_path = f"hf://datasets/{dataset_repo_id}/sarm_progress.parquet"
        else:
            raise ValueError(
                "RABC sample weighting requires 'progress_path' to be set, "
                "or dataset_root/dataset_repo_id for auto-detection. "
                "Generate progress values using: "
                "python -m lerobot.rewards.sarm.compute_rabc_weights --help"
            )

    return RABCWeights(
        progress_path=progress_path,
        chunk_size=chunk_size,
        head_mode=config.head_mode,
        kappa=config.kappa,
        epsilon=config.epsilon,
        device=device,
        **config.extra_params,
    )


class UniformWeighter(SampleWeighter):
    """
    No-op sample weighter that returns uniform weights.

    Useful as a baseline or when you want to disable weighting without
    changing the training code structure.

    Note:
        Batch size is determined by looking for tensor values in the batch
        dictionary. The method checks common keys like "action", "index",
        and "observation.state" first, then falls back to scanning all values.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Return uniform weights (all ones)."""
        batch_size = self._determine_batch_size(batch)

        weights = torch.ones(batch_size, device=self.device)
        stats = {"mean_weight": 1.0, "type": "uniform"}
        return weights, stats

    def _determine_batch_size(self, batch: dict) -> int:
        """
        Determine batch size from the batch dictionary.

        Checks common keys first, then scans all values for tensors.

        Args:
            batch: Training batch dictionary.
        """
        if not batch:
            raise ValueError("Cannot determine batch size from empty batch")

        # Check common keys first
        for key in ["action", "index", "observation.state"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key].shape[0]

        # Scan all values for any tensor
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return value.shape[0]

        # Last resort: return 1 (this handles non-tensor batches)
        return 1

    def get_stats(self) -> dict:
        """Return empty stats for uniform weighting."""
        return {"type": "uniform"}


class StaticWeights(SampleWeighter):
    """
    Sample weighter that reads precomputed per-frame weights from a file.

    The generic half of the abstraction's stated motivation (importance
    sampling, curriculum learning, quality-based filtering): any pipeline that
    can score frames offline — data-quality classifiers, DAgger human/policy
    provenance, curriculum schedules — writes one parquet and trains with it,
    no new weighter class needed.

    The file is a parquet with two required columns:
        - "index": global frame index (the same value the training batch
          carries under its "index" key)
        - "weight": non-negative float weight for that frame

    Frames absent from the file receive ``fallback_weight`` (default 1.0), so
    a partial file up- or down-weights only the frames it names. Batch weights
    are normalized to sum to the batch size, matching the training loop's
    expectation (weighted loss keeps gradient scale when some weights are 0).
    """

    def __init__(
        self,
        weights_path: str,
        device: torch.device,
        epsilon: float = 1e-6,
        fallback_weight: float = 1.0,
    ):
        import pandas as pd

        self.device = device
        self.epsilon = epsilon
        self.fallback_weight = float(fallback_weight)

        logging.info(f"Loading static sample weights from {weights_path}")
        df = pd.read_parquet(weights_path)
        for column in ("index", "weight"):
            if column not in df.columns:
                raise ValueError(
                    f"Column '{column}' not found in {weights_path}. Available columns: {list(df.columns)}"
                )

        self.weight_lookup: dict[int, float] = {}
        for _, row in df.iterrows():
            weight = float(row["weight"])
            if weight < 0:
                raise ValueError(
                    f"Negative weight {weight} for frame index {int(row['index'])} "
                    f"in {weights_path}; weights must be >= 0."
                )
            self.weight_lookup[int(row["index"])] = weight

        logging.info(f"Loaded {len(self.weight_lookup)} static frame weights")

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Look up each sample's weight by its global frame index."""
        indices = batch.get("index")
        if indices is None:
            logging.warning("StaticWeights: batch missing 'index' key, using uniform weights")
            batch_size = self._determine_batch_size(batch)
            stats = {"mean_weight": 1.0, "num_zero_weight": 0, "num_missing": batch_size}
            return torch.ones(batch_size, device=self.device), stats

        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().tolist()

        weights = []
        num_missing = 0
        for idx in indices:
            weight = self.weight_lookup.get(int(idx))
            if weight is None:
                num_missing += 1
                weight = self.fallback_weight
            weights.append(weight)

        weights_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32)
        stats = {
            "mean_weight": float(weights_tensor.mean()),
            "num_zero_weight": int((weights_tensor == 0).sum()),
            "num_missing": num_missing,
        }

        # Normalize to sum to batch_size (see the training loop's weighted
        # loss: zero-weight samples shift their share to the rest).
        batch_size = len(weights)
        weights_tensor = weights_tensor * batch_size / (weights_tensor.sum() + self.epsilon)
        return weights_tensor, stats

    def _determine_batch_size(self, batch: dict) -> int:
        for key in ["action", "index", "observation.state"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key].shape[0]
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return value.shape[0]
        return 1

    def get_stats(self) -> dict:
        """Global statistics about the loaded weight table."""
        if not self.weight_lookup:
            return {"type": "static", "num_frames": 0}
        values = list(self.weight_lookup.values())
        return {
            "type": "static",
            "num_frames": len(values),
            "mean_weight": float(sum(values) / len(values)),
            "num_zero_weight": sum(1 for v in values if v == 0),
            "fallback_weight": self.fallback_weight,
        }

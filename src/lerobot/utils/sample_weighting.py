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
        score_name: robometer
        signal_name: reward.robometer.progress
        kappa: 0.01

    # In training script
    sample_weighter = make_sample_weighter(cfg.sample_weighting, policy, device, dataset=dataset)
    ...
    weights, stats = sample_weighter.compute_batch_weights(batch)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    import datasets

    from lerobot.datasets import SignalDescriptor
    from lerobot.policies.pretrained import PreTrainedPolicy


@runtime_checkable
class ScoreDataset(Protocol):
    def read_score(self, name: str) -> datasets.Dataset: ...

    def get_score_descriptors(self, name: str) -> Mapping[str, SignalDescriptor]: ...


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
        type: Weighting strategy type ("rabc", "uniform", etc.)
        score_name: Dataset score sidecar used by RA-BC
        signal_name: Progress signal selected from the sidecar
        kappa: Hard threshold for high-quality samples (RABC-specific)
        epsilon: Small constant for numerical stability
        extra_params: Additional type-specific parameters passed to the weighter
    """

    type: str = "rabc"
    score_name: str | None = None
    signal_name: str | None = None
    kappa: float = 0.01
    epsilon: float = 1e-6
    # Additional type-specific params can be added here or passed via extra_params
    extra_params: dict = field(default_factory=dict)


def make_sample_weighter(
    config: SampleWeightingConfig | None,
    policy: PreTrainedPolicy,
    device: torch.device,
    dataset: object | None = None,
) -> SampleWeighter | None:
    """
    Factory function to create a SampleWeighter from config.

    This keeps policy-specific initialization logic out of the training script.

    Args:
        config: Sample weighting configuration, or None to disable weighting.
        policy: The policy being trained (used to extract chunk_size, etc.)
        device: Device to place weight tensors on.
        dataset: Dataset exposing the score-reading API.
    """
    if config is None:
        return None

    if config.type == "rabc":
        return _make_rabc_weighter(config, policy, device, dataset)

    if config.type == "uniform":
        # No-op weighter that returns uniform weights
        return UniformWeighter(device=device)

    raise ValueError(f"Unknown sample weighting type: '{config.type}'. Supported types: 'rabc', 'uniform'")


def _make_rabc_weighter(
    config: SampleWeightingConfig,
    policy: PreTrainedPolicy,
    device: torch.device,
    dataset: object | None = None,
) -> SampleWeighter:
    """Create RABC weighter with policy-specific initialization.

    Args:
        config: Sample weighting configuration.
        policy: The policy being trained (used to extract chunk_size).
        device: Device to place weight tensors on.
        dataset: Dataset exposing the score-reading API.
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

    if not isinstance(dataset, ScoreDataset):
        raise ValueError(
            "RA-BC sample weighting needs a dataset that can read scores, such as LeRobotDataset; "
            f"got {type(dataset).__name__}"
        )
    if config.score_name is None or config.signal_name is None:
        raise ValueError("RA-BC sample weighting requires both 'score_name' and 'signal_name'")

    return RABCWeights(
        dataset=dataset,
        score_name=config.score_name,
        signal_name=config.signal_name,
        chunk_size=chunk_size,
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

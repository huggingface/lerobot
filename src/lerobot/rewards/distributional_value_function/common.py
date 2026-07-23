"""Shared distributional targets, loss, and metrics for VF experiments."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


class DistributionalValueMixin:
    """Mixin for models that expose ``_get_value_readout(batch)``."""

    config: Any
    value_head: Any
    hl_gauss_sigma: float

    def hl_gauss_target(self, target_value: Tensor) -> Tensor:
        target_value = target_value.reshape(-1).clamp(
            self.config.value_support_min, self.config.value_support_max
        )
        target_value = target_value.to(self.value_head.bin_centers.dtype)
        bin_width = (self.config.value_support_max - self.config.value_support_min) / (
            self.config.num_value_bins - 1
        )
        support_edges = torch.linspace(
            self.config.value_support_min - bin_width / 2,
            self.config.value_support_max + bin_width / 2,
            self.config.num_value_bins + 1,
            device=target_value.device,
            dtype=target_value.dtype,
        )
        cdf = 0.5 * (
            1.0
            + torch.erf((support_edges[None] - target_value[:, None]) / (self.hl_gauss_sigma * math.sqrt(2)))
        )
        normalization = (cdf[:, -1] - cdf[:, 0]).unsqueeze(-1).clamp_min(1e-10)
        return (cdf[:, 1:] - cdf[:, :-1]) / normalization

    def dirac_delta_target(self, target_value: Tensor) -> Tensor:
        target_value = target_value.reshape(-1).clamp(
            self.config.value_support_min, self.config.value_support_max
        )
        target_value = target_value.to(self.value_head.bin_centers.dtype)
        bin_width = self.value_head.bin_centers[1] - self.value_head.bin_centers[0]
        position = (target_value - self.config.value_support_min) / bin_width
        lower = position.floor().long().clamp(0, self.config.num_value_bins - 1)
        upper = position.ceil().long().clamp(0, self.config.num_value_bins - 1)
        weight_upper = position - lower.float()
        weight_lower = upper.float() - position
        same = lower == upper
        weight_upper = torch.where(same, torch.zeros_like(weight_upper), weight_upper)
        weight_lower = torch.where(same, torch.ones_like(weight_lower), weight_lower)
        distribution = torch.zeros(
            target_value.shape[0],
            self.config.num_value_bins,
            device=target_value.device,
            dtype=target_value.dtype,
        )
        rows = torch.arange(target_value.shape[0], device=target_value.device)
        distribution[rows, lower] += weight_lower
        distribution[rows, upper] += weight_upper
        return distribution

    def compute_target_distribution(self, target_value: Tensor, is_terminal: Tensor) -> Tensor:
        if self.config.target_method == "hl_gauss":
            base = self.hl_gauss_target(target_value)
        elif self.config.target_method == "dirac_delta":
            base = self.dirac_delta_target(target_value)
        else:
            raise ValueError(f"Unknown target method: {self.config.target_method}")
        if not self.config.use_one_hot_terminal:
            return base
        nearest = torch.argmin(
            torch.abs(
                self.value_head.bin_centers[None]
                - target_value.reshape(-1, 1).to(self.value_head.bin_centers.dtype)
            ),
            dim=-1,
        )
        terminal = F.one_hot(nearest, num_classes=self.config.num_value_bins).to(base.dtype)
        return torch.where(is_terminal.reshape(-1, 1).bool(), terminal, base)

    def _distributional_forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        readout = self._get_value_readout(batch)
        logits = self.value_head(readout)
        probabilities = logits.softmax(-1)
        predicted_value = (probabilities * self.value_head.bin_centers.to(probabilities.dtype)).sum(-1)
        targets = self.compute_target_distribution(batch["mc_return"], batch["is_terminal"])
        loss = -(targets * logits.log_softmax(-1)).sum(-1).mean()
        target_values = (
            batch["mc_return"].reshape(-1).clamp(self.config.value_support_min, self.config.value_support_max)
        )
        return loss, {
            "loss": loss.item(),
            "predicted_value_mean": predicted_value.mean().item(),
            "mc_return_mean": target_values.mean().item(),
            "mae": (predicted_value - target_values).abs().mean().item(),
            "acc_best": (logits.argmax(-1) == targets.argmax(-1)).float().mean().item(),
            "acc_neighbor": _neighbor_accuracy(
                logits.argmax(-1),
                target_values,
                self.value_head.bin_centers,
            ),
        }

    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        logits = self.value_head(self._get_value_readout(batch))
        probabilities = logits.softmax(-1)
        return (probabilities * self.value_head.bin_centers.to(probabilities.dtype)).sum(-1)


def _neighbor_accuracy(predicted_bin: Tensor, target: Tensor, centers: Tensor) -> float:
    width = centers[1] - centers[0]
    position = (target - centers[0]) / width
    lower = position.floor().long().clamp(0, len(centers) - 1)
    upper = position.ceil().long().clamp(0, len(centers) - 1)
    return ((predicted_bin == lower) | (predicted_bin == upper)).float().mean().item()

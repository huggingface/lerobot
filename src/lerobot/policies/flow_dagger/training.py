# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Regression of the FlowDAgger noise policy onto inverted noise targets.

The loss is ``E ||pi_w(s) - w*||^2``. Every batch draws ``intervention_fraction`` of its
samples from the intervention buffer and the rest from the autonomous buffer (50/50 in the
paper), so the sparse corrections stay prominent while the policy is anchored on the states
the base policy already handles.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from .inversion import InversionResult
from .noise_policy import NoisePolicy

logger = logging.getLogger(__name__)


@dataclass
class NoisePolicyTrainConfig:
    steps: int = 5000
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    grad_clip_norm: float = 1.0
    # Share of each batch drawn from the intervention buffer (0.5 in the paper).
    intervention_fraction: float = 0.5
    # Held-out share of each buffer, used to report validation MSE.
    val_fraction: float = 0.1
    log_freq: int = 100

    def __post_init__(self) -> None:
        if not 0.0 <= self.intervention_fraction <= 1.0:
            raise ValueError(f"intervention_fraction must be in [0, 1], got {self.intervention_fraction}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}")
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be >= 1")


def split_indices(n: int, val_fraction: float, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Random ``(train, val)`` split of ``range(n)``. Keeps at least one training sample."""
    perm = torch.randperm(n, generator=generator)
    n_val = min(int(round(n * val_fraction)), max(n - 1, 0))
    return perm[n_val:], perm[:n_val]


class BalancedSampler:
    """Draws batches with a fixed share from each of two index pools (with replacement)."""

    def __init__(
        self,
        intervention_idx: Tensor,
        autonomous_idx: Tensor,
        batch_size: int,
        intervention_fraction: float,
        generator: torch.Generator,
    ):
        if len(intervention_idx) == 0 and len(autonomous_idx) == 0:
            raise ValueError("Both FlowDAgger buffers are empty: no samples to train on")
        if len(autonomous_idx) == 0:
            intervention_fraction = 1.0
        elif len(intervention_idx) == 0:
            intervention_fraction = 0.0
        self.intervention_idx = intervention_idx
        self.autonomous_idx = autonomous_idx
        self.n_intervention = int(round(batch_size * intervention_fraction))
        self.n_autonomous = batch_size - self.n_intervention
        self.generator = generator

    def sample(self) -> Tensor:
        parts = []
        if self.n_intervention > 0:
            pick = torch.randint(len(self.intervention_idx), (self.n_intervention,), generator=self.generator)
            parts.append(self.intervention_idx[pick])
        if self.n_autonomous > 0:
            pick = torch.randint(len(self.autonomous_idx), (self.n_autonomous,), generator=self.generator)
            parts.append(self.autonomous_idx[pick])
        return torch.cat(parts)


def _lr_lambda(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return fn


@torch.no_grad()
def evaluate_noise_policy(
    noise_policy: NoisePolicy, features: Tensor, noise: Tensor, device: torch.device, batch_size: int = 1024
) -> float:
    """Mean squared error of ``noise_policy(features)`` against ``noise``."""
    if features.shape[0] == 0:
        return float("nan")
    was_training = noise_policy.training
    noise_policy.eval()
    total, count = 0.0, 0
    for start in range(0, features.shape[0], batch_size):
        f = features[start : start + batch_size].to(device)
        w = noise[start : start + batch_size].to(device)
        total += float(((noise_policy(f) - w) ** 2).sum())
        count += w.numel()
    noise_policy.train(was_training)
    return total / count


def train_noise_policy(
    noise_policy: NoisePolicy,
    data: InversionResult,
    cfg: NoisePolicyTrainConfig,
    device: torch.device,
    *,
    seed: int | None = None,
    log_fn: Callable[[dict[str, float], int], None] | None = None,
) -> dict[str, float]:
    """Fit ``noise_policy`` on the ``(features, noise)`` pairs in ``data``. Returns final metrics."""
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    intervention_all = torch.nonzero(data.is_intervention).flatten()
    autonomous_all = torch.nonzero(~data.is_intervention).flatten()
    int_split = split_indices(len(intervention_all), cfg.val_fraction, generator)
    auto_split = split_indices(len(autonomous_all), cfg.val_fraction, generator)
    int_train, int_val = intervention_all[int_split[0]], intervention_all[int_split[1]]
    auto_train, auto_val = autonomous_all[auto_split[0]], autonomous_all[auto_split[1]]
    logger.info(
        "FlowDAgger noise policy: %d/%d intervention and %d/%d autonomous train/val samples",
        len(int_train),
        len(int_val),
        len(auto_train),
        len(auto_val),
    )
    if len(int_train) == 0:
        logger.warning("No intervention samples: the noise policy only learns to reproduce the base policy")

    sampler = BalancedSampler(int_train, auto_train, cfg.batch_size, cfg.intervention_fraction, generator)
    features = data.features.to(device)
    targets = data.noise.to(device)

    noise_policy.set_feature_stats(data.features[torch.cat([int_train, auto_train])])
    noise_policy.to(device).train()
    optimizer = torch.optim.AdamW(noise_policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(cfg.warmup_steps, cfg.steps))

    def val_metrics() -> dict[str, float]:
        return {
            "val/intervention_mse": evaluate_noise_policy(
                noise_policy, data.features[int_val], data.noise[int_val], device
            ),
            "val/autonomous_mse": evaluate_noise_policy(
                noise_policy, data.features[auto_val], data.noise[auto_val], device
            ),
        }

    running = 0.0
    metrics: dict[str, float] = {}
    for step in range(1, cfg.steps + 1):
        idx = sampler.sample().to(device)
        loss = ((noise_policy(features[idx]) - targets[idx]) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(noise_policy.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        running += loss.item()

        if step % cfg.log_freq == 0 or step == cfg.steps:
            n = cfg.log_freq if step % cfg.log_freq == 0 else step % cfg.log_freq
            metrics = {
                "train/loss": running / n,
                "train/grad_norm": float(grad_norm),
                "train/lr": scheduler.get_last_lr()[0],
                **val_metrics(),
            }
            running = 0.0
            logger.info(
                "step %d/%d  " + "  ".join(f"{k}={v:.5f}" for k, v in metrics.items()), step, cfg.steps
            )
            if log_fn is not None:
                log_fn(metrics, step)

    noise_policy.eval()
    return metrics

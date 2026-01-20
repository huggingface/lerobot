#!/usr/bin/env python

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

from __future__ import annotations

import torch

from lerobot.configs.types import RTCTrainingDelayDistribution
from lerobot.policies.rtc.configuration_rtc import RTCTrainingConfig


def sample_rtc_delay(cfg: RTCTrainingConfig, batch_size: int, device: torch.device) -> torch.Tensor:
    if cfg.max_delay == cfg.min_delay:
        return torch.full((batch_size,), cfg.min_delay, device=device, dtype=torch.long)

    if cfg.delay_distribution == RTCTrainingDelayDistribution.UNIFORM:
        return torch.randint(
            cfg.min_delay, cfg.max_delay + 1, (batch_size,), device=device, dtype=torch.long
        )

    delay_values = torch.arange(cfg.min_delay, cfg.max_delay + 1, device=device, dtype=torch.long)
    weights = torch.exp(-cfg.exp_decay * delay_values.to(dtype=torch.float32))
    probs = weights / weights.sum()
    samples = torch.multinomial(probs, batch_size, replacement=True)
    return delay_values[samples]


def apply_rtc_training_time(
    time: torch.Tensor, delay: torch.Tensor, seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    device = time.device
    delay = torch.clamp(delay, max=seq_len)
    prefix_mask = torch.arange(seq_len, device=device)[None, :] < delay[:, None]
    time_tokens = time[:, None].expand(-1, seq_len)
    time_tokens = time_tokens.masked_fill(prefix_mask, 0.0)
    postfix_mask = ~prefix_mask
    return time_tokens, postfix_mask


def masked_mean(
    losses: torch.Tensor, mask: torch.Tensor | None, reduce_dims: tuple[int, ...], eps: float = 1e-8
) -> torch.Tensor:
    if mask is None:
        return losses.mean(dim=reduce_dims)

    mask = mask.to(dtype=losses.dtype)
    while mask.dim() < losses.dim():
        mask = mask.unsqueeze(-1)
    masked = losses * mask
    denom = mask.sum(dim=reduce_dims).clamp_min(eps)
    return masked.sum(dim=reduce_dims) / denom


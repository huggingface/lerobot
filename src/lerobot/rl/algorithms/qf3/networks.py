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

"""Scalar twin Q used by the QF3 MuJoCo-style critic."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class QNetwork(nn.Module):
    """MLP Q(s, a) with SiLU hidden layers.

    The MuJoCo supplement says the critic is identical to the CleanRL TD3 baseline.
    CleanRL's Q network uses ReLU. SiLU is a departure from that stated identity.
    The from-scratch appendix specifies SiLU for the actor.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int | list[int]) -> None:
        super().__init__()
        hidden_dims = [hidden, hidden] if isinstance(hidden, int) else list(hidden)
        if not hidden_dims:
            raise ValueError("hidden width list is empty.")
        layers: list[nn.Module] = []
        in_dim = state_dim + action_dim
        for width in hidden_dims:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat((state, action), dim=-1))


class TwinQ(nn.Module):
    """Two scalar critics. The QF3 actor uses q1 only."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int | list[int]) -> None:
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden)
        self.q2 = QNetwork(state_dim, action_dim, hidden)

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(state, action), self.q2(state, action)


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """``target <- (1 - tau) target + tau source``."""
    for src, dst in zip(source.parameters(), target.parameters(), strict=True):
        dst.mul_(1.0 - tau).add_(src, alpha=tau)

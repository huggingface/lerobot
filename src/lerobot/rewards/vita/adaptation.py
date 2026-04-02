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

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class VitaAdaptationState:
    """Fast adaptation state tracked across inference steps."""

    fast_weights: Tensor
    num_updates: Tensor

    @classmethod
    def initialize(cls, base_weight: Tensor, batch_size: int, device: torch.device) -> "VitaAdaptationState":
        weights = base_weight.detach().to(device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        updates = torch.zeros(batch_size, dtype=torch.long, device=device)
        return cls(fast_weights=weights, num_updates=updates)

    def reset_indices(self, base_weight: Tensor, mask: Tensor) -> None:
        if mask.ndim != 1:
            raise ValueError(f"Expected mask of shape (B,), got shape {tuple(mask.shape)}")
        if not torch.any(mask):
            return
        self.fast_weights[mask] = base_weight.detach().to(self.fast_weights.device)
        self.num_updates[mask] = 0


class VitaAdaptationModule(nn.Module):
    """Single-layer adaptation module with external fast weights."""

    def __init__(self, adaptation_dim: int):
        super().__init__()
        self.adapter = nn.Linear(adaptation_dim, adaptation_dim, bias=False)

    @property
    def base_weight(self) -> Tensor:
        return self.adapter.weight

    def forward(self, inputs: Tensor, fast_weights: Tensor | None = None) -> Tensor:
        if fast_weights is None:
            return self.adapter(inputs)
        if fast_weights.ndim != 3:
            raise ValueError(f"Expected fast_weights of shape (B, D, D), got {tuple(fast_weights.shape)}")
        if inputs.ndim != 2:
            raise ValueError(f"Expected inputs of shape (B, D), got {tuple(inputs.shape)}")
        return torch.einsum("bij,bj->bi", fast_weights, inputs)

    def adaptation_step(
        self,
        keys: Tensor,
        values: Tensor,
        fast_weights: Tensor,
        adaptation_lr: float,
    ) -> tuple[Tensor, Tensor]:
        predictions = self.forward(keys, fast_weights=fast_weights)
        errors = predictions - values
        adaptation_dim = keys.shape[-1]
        grad = (2.0 / adaptation_dim) * torch.einsum("bi,bj->bij", errors, keys)
        updated_fast_weights = fast_weights - adaptation_lr * grad
        losses = torch.mean((predictions - values) ** 2, dim=-1)
        return updated_fast_weights, losses

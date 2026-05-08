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
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


@dataclass
class VitaFastWeights:
    """Per-sample fast weights for the residual adaptation MLP."""

    w1: Tensor
    b1: Tensor
    w2: Tensor
    b2: Tensor


@dataclass
class VitaAdaptationState:
    """Fast adaptation state tracked across inference steps."""

    fast_weights: VitaFastWeights
    num_updates: Tensor

    @classmethod
    def initialize(cls, module: "VitaAdaptationModule", batch_size: int, device: torch.device) -> "VitaAdaptationState":
        base_weights = module.base_fast_weights()
        weights = VitaFastWeights(
            w1=base_weights.w1.detach().to(device).unsqueeze(0).expand(batch_size, -1, -1).clone(),
            b1=base_weights.b1.detach().to(device).unsqueeze(0).expand(batch_size, -1).clone(),
            w2=base_weights.w2.detach().to(device).unsqueeze(0).expand(batch_size, -1, -1).clone(),
            b2=base_weights.b2.detach().to(device).unsqueeze(0).expand(batch_size, -1).clone(),
        )
        updates = torch.zeros(batch_size, dtype=torch.long, device=device)
        return cls(fast_weights=weights, num_updates=updates)

    def reset_indices(self, module: "VitaAdaptationModule", mask: Tensor) -> None:
        if mask.ndim != 1:
            raise ValueError(f"Expected mask of shape (B,), got shape {tuple(mask.shape)}")
        if not torch.any(mask):
            return
        base_weights = module.base_fast_weights()
        self.fast_weights.w1[mask] = base_weights.w1.detach().to(self.fast_weights.w1.device)
        self.fast_weights.b1[mask] = base_weights.b1.detach().to(self.fast_weights.b1.device)
        self.fast_weights.w2[mask] = base_weights.w2.detach().to(self.fast_weights.w2.device)
        self.fast_weights.b2[mask] = base_weights.b2.detach().to(self.fast_weights.b2.device)
        self.num_updates[mask] = 0


class VitaAdaptationModule(nn.Module):
    """Two-layer residual MLP adaptation module with external fast weights."""

    def __init__(self, adaptation_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(adaptation_dim, adaptation_dim)
        self.fc2 = nn.Linear(adaptation_dim, adaptation_dim)

    @property
    def base_weight(self) -> Tensor:
        return self.fc1.weight

    def base_fast_weights(self) -> VitaFastWeights:
        return VitaFastWeights(
            w1=self.fc1.weight,
            b1=self.fc1.bias,
            w2=self.fc2.weight,
            b2=self.fc2.bias,
        )

    def _forward_with_weights(self, inputs: Tensor, weights: VitaFastWeights) -> Tensor:
        hidden = F.gelu(torch.einsum("bij,bj->bi", weights.w1, inputs) + weights.b1)
        return inputs + torch.einsum("bij,bj->bi", weights.w2, hidden) + weights.b2

    def forward(self, inputs: Tensor, fast_weights: VitaFastWeights | None = None) -> Tensor:
        if fast_weights is None:
            return inputs + self.fc2(F.gelu(self.fc1(inputs)))
        if inputs.ndim != 2:
            raise ValueError(f"Expected inputs of shape (B, D), got {tuple(inputs.shape)}")
        return self._forward_with_weights(inputs, fast_weights)

    def adaptation_step(
        self,
        keys: Tensor,
        values: Tensor,
        fast_weights: VitaFastWeights,
        adaptation_lr: float,
        first_order: bool = True,
    ) -> tuple[VitaFastWeights, Tensor]:
        losses: list[Tensor] = []
        updated_w1: list[Tensor] = []
        updated_b1: list[Tensor] = []
        updated_w2: list[Tensor] = []
        updated_b2: list[Tensor] = []

        for idx in range(keys.shape[0]):
            key_i = keys[idx : idx + 1]
            value_i = values[idx : idx + 1]

            w1_i = fast_weights.w1[idx : idx + 1].clone().requires_grad_(True)
            b1_i = fast_weights.b1[idx : idx + 1].clone().requires_grad_(True)
            w2_i = fast_weights.w2[idx : idx + 1].clone().requires_grad_(True)
            b2_i = fast_weights.b2[idx : idx + 1].clone().requires_grad_(True)
            if first_order:
                w1_i = w1_i.detach().requires_grad_(True)
                b1_i = b1_i.detach().requires_grad_(True)
                w2_i = w2_i.detach().requires_grad_(True)
                b2_i = b2_i.detach().requires_grad_(True)
            sample_weights = VitaFastWeights(w1=w1_i, b1=b1_i, w2=w2_i, b2=b2_i)

            prediction_i = self._forward_with_weights(key_i, sample_weights)
            loss_i = torch.mean((prediction_i - value_i) ** 2)
            grads = torch.autograd.grad(
                loss_i,
                [w1_i, b1_i, w2_i, b2_i],
                create_graph=not first_order,
            )
            if first_order:
                grads = [grad.detach() for grad in grads]

            updated_w1.append(w1_i - adaptation_lr * grads[0])
            updated_b1.append(b1_i - adaptation_lr * grads[1])
            updated_w2.append(w2_i - adaptation_lr * grads[2])
            updated_b2.append(b2_i - adaptation_lr * grads[3])
            losses.append(loss_i)

        updated_fast_weights = VitaFastWeights(
            w1=torch.cat(updated_w1, dim=0),
            b1=torch.cat(updated_b1, dim=0),
            w2=torch.cat(updated_w2, dim=0),
            b2=torch.cat(updated_b2, dim=0),
        )
        return updated_fast_weights, torch.stack(losses, dim=0)

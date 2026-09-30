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

import math

import torch
from torch import nn


def default_init():
    def _init(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)

    return _init


class LinearValueHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, device=None, dtype=None) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, device=device, dtype=dtype)
        self.apply(default_init())

    @classmethod
    def from_config(cls, config, input_dim: int, output_dim: int, **kwargs):
        return cls(input_dim=input_dim, output_dim=output_dim, **kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dims: int, activation=None, device=None, dtype=None) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dims, hidden_dims, device=device, dtype=dtype)
        self.ln1 = nn.LayerNorm(hidden_dims, device=device, dtype=dtype)
        self.act = activation if activation is not None else nn.ReLU()
        self.fc2 = nn.Linear(hidden_dims, hidden_dims, device=device, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_dims, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ln2(self.fc2(self.act(self.ln1(self.fc1(x)))))


class BroNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: int,
        depth: int,
        activation=None,
        output_dim: int = 1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"Unsupported depth: {depth}. Expected depth >= 1.")
        self.activation = activation if activation is not None else nn.ReLU()
        self.input_layer = nn.Linear(input_dim, hidden_dims, device=device, dtype=dtype)
        self.input_norm = nn.LayerNorm(hidden_dims, device=device, dtype=dtype)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dims, self._make_activation(), device=device, dtype=dtype)
                for _ in range(depth)
            ]
        )
        self.final_layer = nn.Linear(hidden_dims, output_dim, device=device, dtype=dtype)
        self.apply(default_init())

    def _make_activation(self) -> nn.Module:
        if isinstance(self.activation, nn.LeakyReLU):
            return nn.LeakyReLU(self.activation.negative_slope)
        for cls in (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh):
            if isinstance(self.activation, cls):
                return cls()
        return self.activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_norm(self.input_layer(x)))
        for block in self.blocks:
            x = block(x)
        return self.final_layer(x)


class BroValueHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: int = 1024,
        depth: int = 2,
        activation: str = "relu",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.proj = BroNet(
            input_dim, hidden_dims, depth, self._build_activation(activation), output_dim, device, dtype
        )

    @classmethod
    def from_config(cls, config, input_dim: int, output_dim: int, **kwargs):
        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=getattr(config, "hidden_dims", 1024),
            depth=getattr(config, "depth", 2),
            activation=getattr(config, "activation", "relu"),
            **kwargs,
        )

    @staticmethod
    def _build_activation(name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }
        try:
            return activations[name.lower()]()
        except KeyError as e:
            raise ValueError(f"Unsupported activation: {name}") from e

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


def build_value_head(config, input_dim: int, output_dim: int, **kwargs):
    head_type = getattr(config, "head_type", "linear")
    if head_type == "linear":
        return LinearValueHead.from_config(config, input_dim=input_dim, output_dim=output_dim, **kwargs)
    if head_type == "bro":
        return BroValueHead.from_config(config, input_dim=input_dim, output_dim=output_dim, **kwargs)
    raise ValueError(f"Unsupported value head type: {head_type}")

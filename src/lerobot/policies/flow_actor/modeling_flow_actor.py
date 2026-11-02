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

"""State-only flow actor for QF3.

The velocity MLP, sinusoidal time embedding, and Euler integrator follow the
from-scratch appendix of arXiv:2610.08789v1. The last linear layer starts at
zero, so the initial flow is the identity transport ``x_1 = epsilon``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from lerobot.optim.optimizers import OptimizerParams
from lerobot.utils.constants import ACTION, OBS_STATE

from ..pretrained import PreTrainedPolicy
from .configuration_flow_actor import FlowActorConfig


class SinusoidalTimeEncoding(nn.Module):
    """Fixed embedding phi(tau) from the QF3 appendix."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 2 or dim % 2 != 0:
            raise ValueError(f"time embedding dim must be even and at least 2, got {dim}")
        half = dim // 2
        index = torch.arange(half, dtype=torch.float32)
        omega = torch.exp(-index * math.log(10000.0) / float(half - 1))
        self.register_buffer("omega", omega)

    def forward(self, flow_time: Tensor) -> Tensor:
        angles = flow_time * self.omega.view(1, -1)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class FlowVelocityMLP(nn.Module):
    """MLP velocity ``v(s, x_tau, tau)``. The last layer is initialized at zero."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int | list[int],
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        hidden_dims = [hidden, hidden] if isinstance(hidden, int) else list(hidden)
        if not hidden_dims:
            raise ValueError("hidden width list is empty")
        self.time_enc = SinusoidalTimeEncoding(time_dim)
        layers: list[nn.Module] = []
        in_dim = state_dim + action_dim + time_dim
        for width in hidden_dims:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        output = nn.Linear(in_dim, action_dim)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    @property
    def output(self) -> nn.Linear:
        layer = self.net[-1]
        if not isinstance(layer, nn.Linear):
            raise TypeError("flow actor output layer is not Linear.")
        return layer

    def forward(self, state: Tensor, action_sample: Tensor, flow_time: Tensor) -> Tensor:
        encoded_time = self.time_enc(flow_time)
        return self.net(torch.cat((state, action_sample, encoded_time), dim=-1))


def euler_integrate(actor: nn.Module, state: Tensor, x0: Tensor, n_steps: int) -> Tensor:
    """Explicit Euler from tau=0 to tau=1. Query times are ``k/N`` for ``k = 0 .. N-1``."""
    if n_steps < 1:
        raise ValueError(f"euler steps must be positive, got {n_steps}")
    step = 1.0 / n_steps
    sample = x0
    batch = state.shape[0]
    for k in range(n_steps):
        flow_time = torch.full((batch, 1), k * step, dtype=state.dtype, device=state.device)
        velocity = actor(state, sample, flow_time)
        sample = sample + step * velocity
    return sample


class FlowActorPolicy(PreTrainedPolicy):
    """Flow policy whose ``select_action`` is an Euler rollout of the velocity field.

    QF3 owns the training loss. This module does not implement behavior cloning.
    """

    config_class = FlowActorConfig
    name = "flow_actor"

    def __init__(self, config: FlowActorConfig | None = None) -> None:
        if config is None:
            raise ValueError(
                "FlowActorPolicy requires a FlowActorConfig. "
                "Build one with input_features and output_features set."
            )
        super().__init__(config)
        if not isinstance(config, FlowActorConfig):
            raise TypeError(f"Expected FlowActorConfig, got {type(config).__name__}.")
        config.validate_features()
        self.config: FlowActorConfig = config
        if config.input_features is None or config.output_features is None:
            raise ValueError("flow_actor features must include observation.state and action.")
        state_feature = config.input_features[OBS_STATE]
        action_feature = config.output_features[ACTION]
        state_dim = state_feature.shape[0]
        action_dim = action_feature.shape[0]
        self.velocity = FlowVelocityMLP(state_dim, action_dim, config.hidden_dims, config.time_dim)
        self.to(config.device or "cpu")

    def get_optim_params(self) -> OptimizerParams:
        return list(self.velocity.parameters())

    def reset(self) -> None:
        """No observation queue. One call is one action."""
        return None

    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        """Action chunks are not implemented.

        ``noise`` is accepted so this override matches ``PreTrainedPolicy.predict_action_chunk``.
        One action is ``select_action``.
        """
        del batch, noise
        raise NotImplementedError("flow_actor returns one action per select_action call.")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Integrate the velocity field for ``euler_steps`` and return the action.

        ``noise`` is the initial sample ``x_0``. When it is omitted, it is drawn from
        a standard normal, which is the training-time sampler in the paper. Passing
        zeros matches the locomotion deployment choice ``epsilon = 0``. At zero
        initialization the field is zero, so the returned action equals ``noise``.
        """
        state = _state_from_batch(batch)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        action_dim = self.velocity.output.out_features
        if noise is None:
            noise = torch.randn(state.shape[0], action_dim, device=state.device, dtype=state.dtype)
        else:
            noise = noise.to(device=state.device, dtype=state.dtype)
        return euler_integrate(self.velocity, state, noise, self.config.euler_steps)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Return a detached zero. Supervised ``lerobot-train`` does not train this policy.

        The training loss is ``QF3Algorithm.update`` on a replay batch.
        """
        action = self.select_action(batch)
        zero = action.sum() * 0.0
        return zero, {"action_mean": float(action.mean())}


def _state_from_batch(batch: dict[str, Tensor]) -> Tensor:
    if OBS_STATE not in batch:
        raise KeyError("flow_actor select_action expects batch['observation.state'].")
    state = batch[OBS_STATE]
    if not isinstance(state, Tensor):
        raise TypeError(f"observation.state must be a tensor, got {type(state).__name__}.")
    return state

#!/usr/bin/env python

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

"""SuperMimic actor, world model and critic.

Architectures and parameter names match Uni-SuperMimic's ``networks.py``
(https://github.com/Motphys/Uni-SuperMimic, Apache-2.0), so its checkpoints load with
``load_state_dict`` unchanged. Mixed precision is a per-module ``amp_dtype`` instead of
the upstream module-level global.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .geometry import State, invariant_features, so3_exp, so3_log, yaw_mat


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    act: type[nn.Module] = nn.ELU,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for hidden_dim in hidden:
        layers.append(nn.Linear(dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act())
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean and variance, frozen once ``max_count`` samples have been seen."""

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = max_count
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        batch_count = x.shape[0]
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.var = (
            self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        ) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps**2)


def run_mlp(net: nn.Module, x: torch.Tensor, amp_dtype: torch.dtype | None) -> torch.Tensor:
    """Run ``net`` under autocast on CUDA/MPS when ``amp_dtype`` is set; always return fp32."""
    if amp_dtype is None or x.device.type not in ("cuda", "mps"):
        return net(x)
    with torch.autocast(x.device.type, dtype=amp_dtype):
        return net(x).float()


class SuperMimicActor(nn.Module):
    """Goal-conditioned actor: ``tanh(enc(goal))`` and the observation history feed ``dec``.

    Inputs are ``obs_hist`` (B, hist, obs_dim), oldest first, and ``goal`` (B, goal_dim); the
    output is the raw normalized joint offset (B, num_joints). Normalizers are part of the
    module, so callers pass unscaled features.
    """

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hist: int = 10,
        z_dim: int = 32,
        num_joints: int = 29,
        hidden: Sequence[int] = (2048, 1024, 512, 256),
        amp_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hist = hist
        self.amp_dtype = amp_dtype
        self.enc = mlp(goal_dim, hidden, z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, hidden, num_joints)
        output = self.dec[-1]
        assert isinstance(output, nn.Linear)
        with torch.no_grad():
            output.weight.mul_(0.01)
            output.bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        latent = torch.tanh(run_mlp(self.enc, self.goal_norm(goal), self.amp_dtype))
        obs = self.obs_norm(obs_hist).flatten(-2)
        return run_mlp(self.dec, torch.cat((latent, obs), -1), self.amp_dtype)


class WorldModel(nn.Module):
    """Predict frame-local state increments over one control step and integrate them in fp32."""

    def __init__(
        self,
        hidden: Sequence[int] = (2048, 2048, 2048),
        num_joints: int = 29,
        context_dim: int = 33,
        amp_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.context_dim = context_dim
        self.amp_dtype = amp_dtype
        self.in_dim = 13 + 3 * num_joints + context_dim
        self.out_dim = 12 + 2 * num_joints
        self.net = mlp(self.in_dim, hidden, self.out_dim, act=nn.SiLU, layer_norm=True)
        output = self.net[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)

    def inputs(self, state: State, action: torch.Tensor, normalized_context: torch.Tensor) -> torch.Tensor:
        return torch.cat((invariant_features(state), action, normalized_context), -1)

    @staticmethod
    def targets(state: State, next_state: State) -> torch.Tensor:
        """Frame-local increments from ``state`` to ``next_state``."""
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        next_position, next_rotation, next_lin_vel, next_ang_vel, next_joint_pos, next_joint_vel = next_state
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                (heading.transpose(-1, -2) @ (next_position - position).unsqueeze(-1)).squeeze(-1),
                so3_log(rotation_t @ next_rotation),
                (rotation_t @ (next_lin_vel - lin_vel).unsqueeze(-1)).squeeze(-1),
                (rotation_t @ (next_ang_vel - ang_vel).unsqueeze(-1)).squeeze(-1),
                next_joint_pos - joint_pos,
                next_joint_vel - joint_vel,
            ),
            -1,
        )

    def step(self, state: State, action: torch.Tensor, normalized_context: torch.Tensor) -> State:
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        prediction = run_mlp(
            self.net, self.in_norm(self.inputs(state, action, normalized_context)), self.amp_dtype
        )
        increment = prediction * self.out_norm.std() + self.out_norm.mean
        heading = yaw_mat(rotation)
        d_position, d_rotation, d_lin_vel, d_ang_vel, d_joint_pos, d_joint_vel = increment.split(
            (3, 3, 3, 3, self.num_joints, self.num_joints), -1
        )
        return (
            position + (heading @ d_position.unsqueeze(-1)).squeeze(-1),
            rotation @ so3_exp(d_rotation),
            lin_vel + (rotation @ d_lin_vel.unsqueeze(-1)).squeeze(-1),
            ang_vel + (rotation @ d_ang_vel.unsqueeze(-1)).squeeze(-1),
            joint_pos + d_joint_pos,
            joint_vel + d_joint_vel,
        )

    @staticmethod
    def state_error(predicted_state: State, state: State) -> torch.Tensor:
        """Frame-local prediction error, laid out like :meth:`targets`."""
        return WorldModel.targets(state, predicted_state)


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int] = (256, 128), amp_dtype: torch.dtype | None = None):
        super().__init__()
        self.amp_dtype = amp_dtype
        self.net = mlp(in_dim, hidden, 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, self.norm(x), self.amp_dtype).squeeze(-1)

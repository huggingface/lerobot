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

"""The 99-D SuperMimic proprioceptive observation and its 10-frame history."""

from __future__ import annotations

import torch

from .geometry import mat_to_6d

OBS_DIM = 29 + 29 + 3 + 3 + 29 + 6


def proprio_observation(
    rotation: torch.Tensor,
    body_angular_velocity: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    previous_action: torch.Tensor,
    reference_rotation: torch.Tensor,
    default_q: torch.Tensor,
    joint_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """``[q - q0 - offset, qd, ω_body, gravity_body, a_prev, 6D(Rᵀ R_ref)]``, no scaling or noise.

    ``rotation`` and ``reference_rotation`` must share one world heading convention.
    ``body_angular_velocity`` is the gyro reading (``Rᵀ ω_world`` in simulation).
    """
    offset = default_q if joint_offset is None else default_q + joint_offset
    rotation_t = rotation.transpose(-1, -2)
    return torch.cat(
        (
            joint_pos - offset,
            joint_vel,
            body_angular_velocity,
            -rotation[..., 2, :],
            previous_action,
            mat_to_6d(rotation_t @ reference_rotation),
        ),
        -1,
    )


class ObservationHistory:
    """Rolling ``(hist, obs_dim)`` window, oldest first, seeded by repeating the first frame."""

    def __init__(self, hist: int, obs_dim: int = OBS_DIM):
        self.hist = hist
        self.obs_dim = obs_dim
        self._buffer: torch.Tensor | None = None

    def reset(self) -> None:
        self._buffer = None

    def push(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Observation has {obs.shape[-1]} features, expected {self.obs_dim}")
        if self._buffer is None:
            self._buffer = obs.unsqueeze(-2).repeat_interleave(self.hist, dim=-2).clone()
        else:
            self._buffer = torch.cat((self._buffer[..., 1:, :], obs.unsqueeze(-2)), dim=-2)
        return self._buffer

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

"""Action interpolation for smoother robot control.

Provides configurable Nx control rate by interpolating between consecutive actions.
Useful with RTC and action-chunking policies to reduce jerkiness.
"""

import numpy as np
import torch
from torch import Tensor

from lerobot.utils.rotation import Rotation as _Rotation


def _slerp_rotvec(r0: np.ndarray, r1: np.ndarray, t: float) -> np.ndarray:
    """Geodesic (SLERP) interpolation between two rotation vectors.

    Fixes the antipodal-twin ambiguity: the same physical rotation can be
    encoded as two rotvecs lying ~2π apart in vector space. Linear interpolation
    between them sweeps through near-zero rotation, producing a non-physical path
    and large IK joint deltas. This function picks the shortest arc instead.
    """
    rot0 = _Rotation.from_rotvec(r0)
    rot1 = _Rotation.from_rotvec(r1)
    q0, q1 = rot0.as_quat(), rot1.as_quat()
    if np.dot(q0, q1) < 0:  # antipodal — negate to pick the shorter arc
        rot1 = _Rotation.from_quat(-q1)
    rel = rot0.inv() * rot1
    return (rot0 * _Rotation.from_rotvec(rel.as_rotvec() * t)).as_rotvec()


class ActionInterpolator:
    """Interpolates between consecutive actions for smoother control.

    When enabled with multiplier N, produces N actions per policy action
    by linearly interpolating between the previous and current action.
    For dimensions listed in ``rotvec_indices``, geodesic SLERP is used instead
    of linear interpolation to avoid non-physical sweeps when the policy emits
    antipodal-twin rotation vectors for the same physical orientation.

    Example with multiplier=3:
        prev_action -> [1/3 interpolated, 2/3 interpolated, current_action]

    This effectively multiplies the control rate for smoother motion.

    Usage:
        interpolator = ActionInterpolator(multiplier=2)  # 2x control rate

        # In control loop:
        if interpolator.needs_new_action():
            new_action = queue.get()
            if new_action:
                interpolator.add(new_action.cpu())

        action = interpolator.get()
        if action:
            robot.send_action(action)
    """

    def __init__(self, multiplier: int = 1, rotvec_indices: list[int] | None = None):
        """Initialize the interpolator.

        Args:
            multiplier: Control rate multiplier (1 = no interpolation, 2 = 2x, 3 = 3x, etc.)
            rotvec_indices: Start indices of rotation-vector triples in the flat action tensor.
                Each value ``i`` means ``action[i:i+3]`` is a ``(wx, wy, wz)`` rotation vector
                that should be interpolated via SLERP rather than linear interpolation.
                When ``None`` (default), all dimensions are linearly interpolated.
        """
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        self.multiplier = multiplier
        self.rotvec_indices: list[int] = rotvec_indices or []
        self._prev: Tensor | None = None
        self._buffer: list[Tensor] = []
        self._idx = 0

    @property
    def enabled(self) -> bool:
        """Whether interpolation is active (multiplier > 1)."""
        return self.multiplier > 1

    def reset(self):
        """Reset interpolation state (call between episodes)."""
        self._prev = None
        self._buffer = []
        self._idx = 0

    def needs_new_action(self) -> bool:
        """Check if a new action is needed from the queue."""
        return self._idx >= len(self._buffer)

    def add(self, action: Tensor) -> None:
        """Add a new action and compute interpolated sequence.

        Args:
            action: New action tensor from policy/queue (already on CPU).
        """
        if self.multiplier > 1 and self._prev is not None:
            self._buffer = []
            if self.rotvec_indices:
                prev_np = self._prev.detach().numpy()
                action_np = action.detach().numpy()
                for i in range(1, self.multiplier + 1):
                    t = i / self.multiplier
                    interp_np = prev_np + t * (action_np - prev_np)
                    for start in self.rotvec_indices:
                        interp_np[start : start + 3] = _slerp_rotvec(
                            prev_np[start : start + 3], action_np[start : start + 3], t
                        )
                    self._buffer.append(torch.tensor(interp_np, dtype=action.dtype, device=action.device))
            else:
                for i in range(1, self.multiplier + 1):
                    t = i / self.multiplier
                    interp = self._prev + t * (action - self._prev)
                    self._buffer.append(interp)
        else:
            # First step: no previous action yet, so run at base FPS without interpolation.
            self._buffer = [action.clone()]
        self._prev = action.clone()
        self._idx = 0

    def get(self) -> Tensor | None:
        """Get the next interpolated action.

        Returns:
            Next action tensor, or None if buffer is exhausted.
        """
        if self._idx >= len(self._buffer):
            return None
        action = self._buffer[self._idx]
        self._idx += 1
        return action

    def get_control_interval(self, fps: float) -> float:
        """Get the control interval based on interpolation multiplier.

        Args:
            fps: Base frames per second.

        Returns:
            Control interval in seconds (divided by multiplier).
        """
        return 1.0 / (fps * self.multiplier)

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

import math

import torch
from torch import Tensor

_ROTVEC_EPS = 1e-8


class ActionInterpolator:
    """Interpolates between consecutive actions for smoother control.

    When enabled with multiplier N, produces N actions per policy action
    by linearly interpolating between the previous and current action.

    Example with multiplier=3:
        prev_action -> [1/3 interpolated, 2/3 interpolated, current_action]

    This effectively multiplies the control rate for smoother motion.

    Action dimensions holding rotation vectors (axis-angle) cannot be
    linearly interpolated directly: two rotvecs can encode (nearly) the same
    rotation while lying ~2*pi apart in vector space (`r` and its antipodal
    twin `(|r| - 2*pi) * r/|r|`), and the straight line between them sweeps
    through the identity rotation. Pass ``rotation_dims`` to canonicalize such
    dimensions to the twin nearest the previous action before interpolating.

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

    def __init__(self, multiplier: int = 1, rotation_dims: list[int] | None = None):
        """Initialize the interpolator.

        Args:
            multiplier: Control rate multiplier (1 = no interpolation, 2 = 2x, 3 = 3x, etc.)
            rotation_dims: Optional flat list of action indices that hold rotation
                vectors, grouped in consecutive (wx, wy, wz) triplets, e.g.
                ``[3, 4, 5]`` for a single end-effector or ``[3, 4, 5, 10, 11, 12]``
                for a bimanual setup. When None (default), all dimensions are
                interpolated linearly as before.
        """
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        if rotation_dims is not None and len(rotation_dims) % 3 != 0:
            raise ValueError(
                f"rotation_dims must contain (wx, wy, wz) triplets, got {len(rotation_dims)} indices"
            )
        self.multiplier = multiplier
        self._rotation_triplets = (
            [rotation_dims[i : i + 3] for i in range(0, len(rotation_dims), 3)] if rotation_dims else []
        )
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

    def _canonicalize_rotvecs(self, action: Tensor) -> Tensor:
        """Replace each rotation-vector triplet with its antipodal twin when closer to the previous action.

        A rotation vector ``r`` and its twin ``(|r| - 2*pi) * r/|r|`` encode the
        same rotation. Interpolating toward whichever lies closer to the previous
        rotvec keeps the interpolated path on the short arc instead of sweeping
        through the identity rotation.
        """
        if not self._rotation_triplets or self._prev is None:
            return action
        action = action.clone()
        for triplet in self._rotation_triplets:
            r = action[..., triplet]
            norm = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
            twin = (norm - 2 * math.pi) * r / norm.clamp_min(_ROTVEC_EPS)
            prev_r = self._prev[..., triplet]
            twin_is_closer = torch.linalg.vector_norm(twin - prev_r, dim=-1, keepdim=True) < (
                torch.linalg.vector_norm(r - prev_r, dim=-1, keepdim=True)
            )
            use_twin = (norm > _ROTVEC_EPS) & twin_is_closer
            action[..., triplet] = torch.where(use_twin, twin, r)
        return action

    def add(self, action: Tensor) -> None:
        """Add a new action and compute interpolated sequence.

        Args:
            action: New action tensor from policy/queue (already on CPU).
        """
        if self.multiplier > 1 and self._prev is not None:
            action = self._canonicalize_rotvecs(action)
            self._buffer = []
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

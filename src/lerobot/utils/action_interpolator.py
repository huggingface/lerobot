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

import torch
from torch import Tensor


def _rotvec_to_quaternion(rotvec: Tensor) -> Tensor:
    """Convert one ``(wx, wy, wz)`` rotation vector to an ``(x, y, z, w)`` quaternion."""
    angle = torch.linalg.vector_norm(rotvec)
    epsilon = torch.finfo(rotvec.dtype).eps
    scale = torch.where(
        angle > epsilon,
        torch.sin(angle / 2) / angle.clamp_min(epsilon),
        0.5 - angle.square() / 48,
    )
    return torch.cat((rotvec * scale, torch.cos(angle / 2).reshape(1)))


def _quaternion_to_rotvec(quaternion: Tensor) -> Tensor:
    """Convert one ``(x, y, z, w)`` quaternion to its principal rotation vector."""
    epsilon = torch.finfo(quaternion.dtype).eps
    quaternion = quaternion / torch.linalg.vector_norm(quaternion).clamp_min(epsilon)
    quaternion = torch.where(quaternion[-1] < 0, -quaternion, quaternion)
    sin_half_angle = torch.linalg.vector_norm(quaternion[:3])
    angle = 2 * torch.atan2(sin_half_angle, quaternion[-1])
    scale = torch.where(
        sin_half_angle > epsilon,
        angle / sin_half_angle.clamp_min(epsilon),
        torch.tensor(2.0, dtype=quaternion.dtype, device=quaternion.device),
    )
    return quaternion[:3] * scale


def _slerp_rotvec(start: Tensor, end: Tensor, fraction: float) -> Tensor:
    """Interpolate two rotation vectors along the shortest SO(3) geodesic."""
    start_quaternion = _rotvec_to_quaternion(start)
    end_quaternion = _rotvec_to_quaternion(end)

    dot = torch.sum(start_quaternion * end_quaternion)
    end_quaternion = torch.where(dot < 0, -end_quaternion, end_quaternion)
    dot = dot.abs().clamp(max=1.0)

    angle = torch.arccos(dot)
    sin_angle = torch.sin(angle)
    epsilon = torch.finfo(start.dtype).eps
    spherical = (
        torch.sin((1 - fraction) * angle) / sin_angle.clamp_min(epsilon) * start_quaternion
        + torch.sin(fraction * angle) / sin_angle.clamp_min(epsilon) * end_quaternion
    )
    linear = torch.lerp(start_quaternion, end_quaternion, fraction)
    quaternion = torch.where(sin_angle > epsilon, spherical, linear)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion).clamp_min(epsilon)
    return _quaternion_to_rotvec(quaternion)


class ActionInterpolator:
    """Interpolates between consecutive actions for smoother control.

    When enabled with multiplier N, produces N actions per policy action
    by linearly interpolating between the previous and current action.
    For dimensions listed in ``rotation_dims``, geodesic SLERP is used instead
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

        # Recording stays at the base FPS: only the tick that emits the
        # policy's own action contributes a dataset frame.
        if interpolator.emitted_policy_action:
            dataset.add_frame(...)
    """

    def __init__(self, multiplier: int = 1, rotation_dims: list[int] | None = None):
        """Initialize the interpolator.

        Args:
            multiplier: Control rate multiplier (1 = no interpolation, 2 = 2x, 3 = 3x, etc.)
            rotation_dims: Flat action dimensions grouped as consecutive ``(wx, wy, wz)``
                triplets. Each triplet is interpolated via SLERP rather than linearly.
                For example, ``[3, 4, 5, 10, 11, 12]`` configures two rotations.
                When ``None`` (default), all dimensions are linearly interpolated.
        """
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        if rotation_dims is not None and len(rotation_dims) % 3 != 0:
            raise ValueError("rotation_dims must contain complete (wx, wy, wz) triplets")
        self.multiplier = multiplier
        self.rotation_dims: list[int] = rotation_dims or []
        self._rotation_triplets = [
            self.rotation_dims[i : i + 3] for i in range(0, len(self.rotation_dims), 3)
        ]
        self._prev: Tensor | None = None
        self._buffer: list[Tensor] = []
        self._idx = 0
        self._emitted_policy_action = False

    @property
    def enabled(self) -> bool:
        """Whether interpolation is active (multiplier > 1)."""
        return self.multiplier > 1

    @property
    def emitted_policy_action(self) -> bool:
        """Whether the action last returned by :meth:`get` was the policy's own output.

        :meth:`add` stores the policy action last in the interpolated buffer, so this
        is ``True`` exactly on the tick that emits ``buffer[-1]`` and ``False`` on the
        intermediate ticks leading up to it.  Strategies gate dataset recording on it:
        frames then land at ``fps`` regardless of ``multiplier``, and each one stores a
        genuine policy action paired with the observation that produced it.

        Read this *after* :meth:`get` (or after ``send_next_action``) — it describes
        the action already handed out, not the one the next call will return.
        :meth:`needs_new_action` is the question to ask *before* dispatching; reading
        that one instead would record ``buffer[0]``, the least-advanced intermediate.
        """
        return self._emitted_policy_action

    def reset(self):
        """Reset interpolation state (call between episodes)."""
        self._prev = None
        self._buffer = []
        self._idx = 0
        self._emitted_policy_action = False

    def needs_new_action(self) -> bool:
        """Check if a new action is needed from the queue."""
        return self._idx >= len(self._buffer)

    def add(self, action: Tensor) -> None:
        """Add a new action and compute interpolated sequence.

        Args:
            action: New action tensor from policy/queue (already on CPU).
        """
        invalid_dims = [index for index in self.rotation_dims if index < 0 or index >= action.shape[-1]]
        if invalid_dims:
            raise ValueError(
                f"rotation_dims contains indices {invalid_dims} out of range for an action with "
                f"{action.shape[-1]} dimensions"
            )

        if self.multiplier > 1 and self._prev is not None:
            self._buffer = []
            for i in range(1, self.multiplier):
                fraction = i / self.multiplier
                interpolated = self._prev + fraction * (action - self._prev)
                for triplet in self._rotation_triplets:
                    interpolated[triplet] = _slerp_rotvec(self._prev[triplet], action[triplet], fraction)
                self._buffer.append(interpolated)
            # The end point is the policy's action itself, appended verbatim rather
            # than computed as ``prev + 1.0 * (action - prev)``, which can land an ULP
            # away.  ``emitted_policy_action`` promises the recorded frame carries the
            # policy's own output, so make that exact.
            self._buffer.append(action.clone())
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
            self._emitted_policy_action = False
            return None
        action = self._buffer[self._idx]
        self._idx += 1
        self._emitted_policy_action = self._idx == len(self._buffer)
        return action

#!/usr/bin/env python

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

"""Custom rotation utilities to replace scipy.spatial.transform.Rotation.

The `Rotation` class is NumPy and follows scipy's `[x, y, z, w]` quaternion order. The
functions below it are torch: they run inside processor steps on batched GPU tensors, which
must keep their device and dtype. The torch quaternions use `[w, x, y, z]`.
"""

import numpy as np
import torch
from torch import Tensor


class Rotation:
    """
    Custom rotation class that provides a subset of scipy.spatial.transform.Rotation functionality.

    Supports conversions between rotation vectors, rotation matrices, and quaternions.
    """

    def __init__(self, quat: np.ndarray) -> None:
        """Initialize rotation from quaternion [x, y, z, w]."""
        self._quat = np.asarray(quat, dtype=float)
        if self._quat.shape != (4,):
            raise ValueError(f"Quaternion must have shape (4,), got {self._quat.shape}")
        # Normalize quaternion. Reject the zero vector — it has no orientation.
        norm = np.linalg.norm(self._quat)
        if norm <= 0.0 or not np.isfinite(norm):
            raise ValueError(f"Quaternion must be a non-zero finite vector; got {self._quat} (norm={norm})")
        self._quat = self._quat / norm

    @classmethod
    def from_rotvec(cls, rotvec: np.ndarray) -> "Rotation":
        """
        Create rotation from rotation vector using Rodrigues' formula.

        Args:
            rotvec: Rotation vector [x, y, z] where magnitude is angle in radians

        Returns:
            Rotation instance
        """
        rotvec = np.asarray(rotvec, dtype=float)
        angle = np.linalg.norm(rotvec)

        if angle < 1e-8:
            # For very small angles, use identity quaternion
            quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            axis = rotvec / angle
            half_angle = angle / 2.0
            sin_half = np.sin(half_angle)
            cos_half = np.cos(half_angle)

            # Quaternion [x, y, z, w]
            quat = np.array([axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, cos_half])

        return cls(quat)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "Rotation":
        """
        Create rotation from 3x3 rotation matrix.

        Args:
            matrix: 3x3 rotation matrix

        Returns:
            Rotation instance
        """
        matrix = np.asarray(matrix, dtype=float)

        # Shepherd's method for converting rotation matrix to quaternion
        trace = np.trace(matrix)

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (matrix[2, 1] - matrix[1, 2]) / s
            qy = (matrix[0, 2] - matrix[2, 0]) / s
            qz = (matrix[1, 0] - matrix[0, 1]) / s
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2  # s = 4 * qx
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2  # s = 4 * qy
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2  # s = 4 * qz
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s

        quat = np.array([qx, qy, qz, qw])
        return cls(quat)

    @classmethod
    def from_quat(cls, quat: np.ndarray) -> "Rotation":
        """
        Create rotation from quaternion.

        Args:
            quat: Quaternion [x, y, z, w] or [w, x, y, z] (specify convention in docstring)
                  This implementation expects [x, y, z, w] format

        Returns:
            Rotation instance
        """
        return cls(quat)

    def as_matrix(self) -> np.ndarray:
        """
        Convert rotation to 3x3 rotation matrix.

        Returns:
            3x3 rotation matrix
        """
        qx, qy, qz, qw = self._quat

        # Compute rotation matrix from quaternion
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=float,
        )

    def as_rotvec(self) -> np.ndarray:
        """
        Convert rotation to rotation vector.

        Returns:
            Rotation vector [x, y, z] where magnitude is angle in radians
        """
        qx, qy, qz, qw = self._quat

        # Ensure qw is positive for unique representation
        if qw < 0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw

        # Compute angle and axis
        angle = 2.0 * np.arccos(np.clip(abs(qw), 0.0, 1.0))
        sin_half_angle = np.sqrt(1.0 - qw * qw)

        if sin_half_angle < 1e-8:
            # For very small angles, use linearization: rotvec ≈ 2 * [qx, qy, qz]
            return 2.0 * np.array([qx, qy, qz])

        # Extract axis and scale by angle
        axis = np.array([qx, qy, qz]) / sin_half_angle
        return angle * axis

    def as_quat(self) -> np.ndarray:
        """
        Get quaternion representation.

        Returns:
            Quaternion [x, y, z, w]
        """
        return self._quat.copy()

    def apply(self, vectors: np.ndarray, inverse: bool = False) -> np.ndarray:
        """
        Apply this rotation to a set of vectors.

        This is equivalent to applying the rotation matrix to the vectors:
        self.as_matrix() @ vectors (or self.as_matrix().T @ vectors if inverse=True).

        Args:
            vectors: Array of shape (3,) or (N, 3) representing vectors in 3D space
            inverse: If True, apply the inverse of the rotation. Default is False.

        Returns:
            Rotated vectors with shape:
            - (3,) if input was single vector with shape (3,)
            - (N, 3) in all other cases
        """
        vectors = np.asarray(vectors, dtype=float)
        original_shape = vectors.shape

        # Handle single vector case - ensure it's 2D for matrix multiplication
        if vectors.ndim == 1:
            if len(vectors) != 3:
                raise ValueError("Single vector must have length 3")
            vectors = vectors.reshape(1, 3)
            single_vector = True
        elif vectors.ndim == 2:
            if vectors.shape[1] != 3:
                raise ValueError("Vectors must have shape (N, 3)")
            single_vector = False
        else:
            raise ValueError("Vectors must be 1D or 2D array")

        # Get rotation matrix
        rotation_matrix = self.as_matrix()

        # Apply inverse if requested (transpose for orthogonal rotation matrices)
        if inverse:
            rotation_matrix = rotation_matrix.T

        # Apply rotation: (N, 3) @ (3, 3).T -> (N, 3)
        rotated_vectors = vectors @ rotation_matrix.T

        # Return original shape for single vector case
        if single_vector and original_shape == (3,):
            return rotated_vectors.flatten()

        return rotated_vectors

    def inv(self) -> "Rotation":
        """
        Invert this rotation.

        Composition of a rotation with its inverse results in an identity transformation.

        Returns:
            Rotation instance containing the inverse of this rotation
        """
        qx, qy, qz, qw = self._quat

        # For a unit quaternion, the inverse is the conjugate: [-x, -y, -z, w]
        inverse_quat = np.array([-qx, -qy, -qz, qw])

        return Rotation(inverse_quat)

    def __mul__(self, other: "Rotation") -> "Rotation":
        """
        Compose this rotation with another rotation using the * operator.

        The composition `r2 * r1` means "apply r1 first, then r2".
        This is equivalent to applying rotation matrices: r2.as_matrix() @ r1.as_matrix()

        Args:
            other: Another Rotation instance to compose with

        Returns:
            Rotation instance representing the composition of rotations
        """
        if not isinstance(other, Rotation):
            return NotImplemented

        # Get quaternions [x, y, z, w]
        x1, y1, z1, w1 = other._quat  # Apply first
        x2, y2, z2, w2 = self._quat  # Apply second

        # Quaternion multiplication: q2 * q1 (apply q1 first, then q2)
        composed_quat = np.array(
            [
                w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,  # x component
                w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,  # y component
                w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,  # z component
                w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,  # w component
            ]
        )

        return Rotation(composed_quat)


def rotvec_to_quaternion(rotvec: Tensor) -> Tensor:
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    angle_sq = angle.square()
    small_scale = 0.5 - angle_sq / 48.0 + angle_sq.square() / 3840.0
    scale = torch.where(angle > 1e-6, torch.sin(angle / 2.0) / angle.clamp_min(1e-12), small_scale)
    return torch.cat((torch.cos(angle / 2.0), rotvec * scale), dim=-1)


def quaternion_to_rotvec(quaternion: Tensor) -> Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)
    quaternion = quaternion * torch.where(quaternion[..., :1] < 0, -1.0, 1.0)
    vector = quaternion[..., 1:]
    sin_half_angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half_angle, quaternion[..., :1].clamp_min(0.0))
    small_scale = 2.0 + sin_half_angle.square() / 3.0
    scale = torch.where(
        sin_half_angle > 1e-6,
        angle / sin_half_angle.clamp_min(1e-12),
        small_scale,
    )
    return vector * scale


def quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    left_w, left_xyz = left[..., :1], left[..., 1:]
    right_w, right_xyz = right[..., :1], right[..., 1:]
    return torch.cat(
        (
            left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True),
            left_w * right_xyz + right_w * left_xyz + torch.linalg.cross(left_xyz, right_xyz, dim=-1),
        ),
        dim=-1,
    )


def quaternion_conjugate(quaternion: Tensor) -> Tensor:
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def quaternion_rotate(quaternion: Tensor, vector: Tensor) -> Tensor:
    quaternion_xyz = quaternion[..., 1:]
    uv = torch.linalg.cross(quaternion_xyz, vector, dim=-1)
    uuv = torch.linalg.cross(quaternion_xyz, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * uv + uuv)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Build rotation matrices from the continuous 6-D representation (its first two rows).

    Gram-Schmidt orthonormalization, as in Zhou et al., "On the Continuity of Rotation
    Representations in Neural Networks". A predicted pair of 3-vectors is generally neither unit
    length nor orthogonal; this is what makes the representation usable as a network output.
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"A 6-D rotation has six values, got shape {tuple(rotation_6d.shape)}")
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:]
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    second = second - (first / first_norm.clamp_min(1e-12) * second).sum(dim=-1, keepdim=True) * (
        first / first_norm.clamp_min(1e-12)
    )
    second_norm = torch.linalg.vector_norm(second, dim=-1, keepdim=True)
    if bool(torch.any(first_norm <= 1e-8)) or bool(torch.any(second_norm <= 1e-8)):
        raise ValueError("Cannot build a rotation from a degenerate 6-D representation")
    row1 = first / first_norm
    row2 = second / second_norm
    return torch.stack((row1, row2, torch.linalg.cross(row1, row2, dim=-1)), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Drop a rotation matrix down to its first two rows, the inverse of `rotation_6d_to_matrix`."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrices have shape (..., 3, 3), got {tuple(matrix.shape)}")
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))

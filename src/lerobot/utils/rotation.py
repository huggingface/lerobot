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

"""Custom rotation utilities to replace scipy.spatial.transform.Rotation."""

import numpy as np


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

    # -----------------------------------------------------------------------
    # Batched conversions with scipy's bits (xyzw quaternions, extrinsic x-y-z Euler angles)
    # -----------------------------------------------------------------------
    # These static methods take arrays of rotations (a leading batch axis) and repeat the operations
    # of ``scipy.spatial.transform.Rotation`` (``from_euler``, ``as_matrix``, ``from_matrix``,
    # ``as_quat``) in the same order, so their results have the bits of scipy's without the
    # dependency. Datasets converted with scipy (DROID's pose columns, for one) and the policies
    # trained on them reproduce exactly through these.

    @staticmethod
    def quat_mul_xyzw(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Return the products ``p * q`` of ``(N, 4)`` xyzw quaternions (scipy's order of operations)."""
        c0 = p[:, 1] * q[:, 2] - p[:, 2] * q[:, 1]
        c1 = p[:, 2] * q[:, 0] - p[:, 0] * q[:, 2]
        c2 = p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]
        out = np.empty_like(p)
        out[:, 0] = p[:, 3] * q[:, 0] + q[:, 3] * p[:, 0] + c0
        out[:, 1] = p[:, 3] * q[:, 1] + q[:, 3] * p[:, 1] + c1
        out[:, 2] = p[:, 3] * q[:, 2] + q[:, 3] * p[:, 2] + c2
        out[:, 3] = p[:, 3] * q[:, 3] - p[:, 0] * q[:, 0] - p[:, 1] * q[:, 1]
        out[:, 3] -= p[:, 2] * q[:, 2]
        return out

    @staticmethod
    def euler_xyz_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray:
        """Convert ``(N, 3)`` extrinsic x-y-z Euler angles to ``(N, 4)`` xyzw quaternions.

        The rotation is ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``. The operations are those of
        ``Rotation.from_euler("xyz", rpy)``, so the result has the same bits.
        """
        e = np.asarray(rpy, dtype=np.float64).reshape(-1, 3)
        q = np.zeros((e.shape[0], 4))
        for axis in range(3):
            half = e[:, axis] / 2
            elem = np.zeros((e.shape[0], 4))
            elem[:, axis] = np.sin(half)
            elem[:, 3] = np.cos(half)
            q = elem if axis == 0 else Rotation.quat_mul_xyzw(elem, q)
        return q

    @staticmethod
    def _unit_quat_xyzw(q: np.ndarray) -> np.ndarray:
        """Normalize ``(..., 4)`` xyzw quaternions in float64 (scipy's order of operations)."""
        q = np.array(q, dtype=np.float64, copy=True)
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return q / np.sqrt(x * x + y * y + z * z + w * w)[..., None]

    @staticmethod
    def _quat_xyzw_matrix(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        x2, y2, z2, w2 = x * x, y * y, z * z, w * w
        xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
        m = np.empty((*q.shape[:-1], 3, 3))
        m[..., 0, 0] = x2 - y2 - z2 + w2
        m[..., 1, 0] = 2 * (xy + zw)
        m[..., 2, 0] = 2 * (xz - yw)
        m[..., 0, 1] = 2 * (xy - zw)
        m[..., 1, 1] = -x2 + y2 - z2 + w2
        m[..., 2, 1] = 2 * (yz + xw)
        m[..., 0, 2] = 2 * (xz + yw)
        m[..., 1, 2] = 2 * (yz - xw)
        m[..., 2, 2] = -x2 - y2 + z2 + w2
        return m

    @staticmethod
    def quat_xyzw_to_matrix_raw(q: np.ndarray) -> np.ndarray:
        """Convert ``(N, 4)`` xyzw quaternions to ``(N, 3, 3)`` matrices without normalizing them.

        The operations are those of ``Rotation.as_matrix`` on a rotation from ``from_euler`` (scipy
        keeps that quaternion as it is).
        """
        return Rotation._quat_xyzw_matrix(np.asarray(q, dtype=np.float64))

    @staticmethod
    def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
        """Convert ``(..., 4)`` xyzw quaternions to ``(..., 3, 3)`` float64 matrices.

        The quaternion is normalized first. The operations are those of
        ``Rotation.from_quat(q).as_matrix()``, so the result has the same bits.
        """
        return Rotation._quat_xyzw_matrix(Rotation._unit_quat_xyzw(q))

    @staticmethod
    def matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
        """Convert ``(..., 3, 3)`` rotation matrices to ``(..., 4)`` xyzw quaternions.

        The operations are those of ``Rotation.from_matrix(m).as_quat()`` (the largest-component
        branch, then an exact normalization), so the result has the same bits. The sign is not made
        canonical.
        """
        m = np.asarray(m, dtype=np.float64)
        flat = m.reshape(-1, 3, 3)
        out = np.empty((flat.shape[0], 4))
        for n, c in enumerate(flat):
            dec = [c[0, 0], c[1, 1], c[2, 2], c[0, 0] + c[1, 1] + c[2, 2]]
            choice = int(np.argmax(dec))
            q = np.empty(4)
            if choice != 3:
                i = choice
                j = (i + 1) % 3
                k = (j + 1) % 3
                q[i] = 1 - dec[3] + 2 * c[i, i]
                q[j] = c[j, i] + c[i, j]
                q[k] = c[k, i] + c[i, k]
                q[3] = c[k, j] - c[j, k]
            else:
                q[0] = c[2, 1] - c[1, 2]
                q[1] = c[0, 2] - c[2, 0]
                q[2] = c[1, 0] - c[0, 1]
                q[3] = 1 + dec[3]
            out[n] = q
        return Rotation._unit_quat_xyzw(out).reshape(*m.shape[:-2], 4)

    @staticmethod
    def euler_xyz_to_matrix(rpy: np.ndarray) -> np.ndarray:
        """Convert ``(N, 3)`` extrinsic x-y-z Euler angles to ``(N, 3, 3)`` rotation matrices (scipy's bits)."""
        return Rotation.quat_xyzw_to_matrix_raw(Rotation.euler_xyz_to_quat_xyzw(rpy))

    @staticmethod
    def matrix_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
        """Convert ``(N, 3, 3)`` rotation matrices to ``(N, 3)`` extrinsic x-y-z Euler angles.

        The inverse of :meth:`euler_xyz_to_matrix`: ``roll = atan2(R21, R22)``, ``pitch = asin(-R20)``,
        ``yaw = atan2(R10, R00)``. Within 1e-6 rad of a right angle of pitch the roll is set to 0 and
        the yaw absorbs the turn.
        """
        r = np.asarray(rot, dtype=np.float64).reshape(-1, 3, 3)
        pitch = np.arcsin(np.clip(-r[:, 2, 0], -1.0, 1.0))
        cos_pitch = np.cos(pitch)
        regular = np.abs(cos_pitch) > 1e-6
        roll = np.where(regular, np.arctan2(r[:, 2, 1], r[:, 2, 2]), 0.0)
        yaw = np.where(regular, np.arctan2(r[:, 1, 0], r[:, 0, 0]), np.arctan2(-r[:, 0, 1], r[:, 1, 1]))
        return np.stack([roll, pitch, yaw], axis=-1)

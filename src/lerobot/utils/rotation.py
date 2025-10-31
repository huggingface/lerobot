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
        # Normalize quaternion
        norm = np.linalg.norm(self._quat)
        if norm > 0:
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

    def as_euler(self, seq: str, degrees: bool = False) -> np.ndarray:
        """
        Convert the rotation to Euler angles.

        Args:
            seq: Axis sequence, e.g., "xyz", "zyx", "xzy", "yxz", "yzx", "zxy".
                 Only proper Tait-Bryan sequences with all distinct axes are supported.
            degrees: If True, return angles in degrees; otherwise radians.

        Returns:
            ndarray of shape (3,) with angles [a1, a2, a3] for the given sequence.
        """
        seq = seq.lower()
        valid = {"xyz", "xzy", "yxz", "yzx", "zxy", "zyx"}
        if seq not in valid:
            raise ValueError(
                f"Unsupported euler sequence '{seq}'. Supported: {sorted(valid)}"
            )

        R = self.as_matrix()
        r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
        r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
        r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]

        # For Tait-Bryan sequences (all axes different), formulas:
        # xyz:
        #   sy = -r20
        #   y = asin(sy)
        #   x = atan2(r21, r22)
        #   z = atan2(r10, r00)
        # xzy:
        #   sz = r10
        #   z = asin(sz)
        #   x = atan2(-r12, r11)
        #   y = atan2(-r20, r00)
        # yxz:
        #   sx = r21
        #   x = asin(sx)
        #   y = atan2(-r20, r22)
        #   z = atan2(-r01, r00)
        # yzx:
        #   sz = -r01
        #   z = asin(sz)
        #   y = atan2(r02, r00)
        #   x = atan2(r21, r11)
        # zxy:
        #   sx = -r12
        #   x = asin(sx)
        #   z = atan2(r10, r11)
        #   y = atan2(r02, r22)
        # zyx:
        #   sy = -r02
        #   y = asin(sy)
        #   z = atan2(r01, r00)
        #   x = atan2(r12, r22)
        #
        # Handle gimbal lock when |sin(mid)| ~ 1.

        eps = 1e-8

        if seq == "xyz":
            sy = -r20
            y = np.arcsin(np.clip(sy, -1.0, 1.0))
            if abs(sy) < 1 - eps:
                x = np.arctan2(r21, r22)
                z = np.arctan2(r10, r00)
            else:
                # Gimbal lock: z set to 0, solve x from r01/r02
                x = np.arctan2(-r12, r11)
                z = 0.0

        elif seq == "xzy":
            sz = r10
            z = np.arcsin(np.clip(sz, -1.0, 1.0))
            if abs(sz) < 1 - eps:
                x = np.arctan2(-r12, r11)
                y = np.arctan2(-r20, r00)
            else:
                x = np.arctan2(r21, r22)
                y = 0.0

        elif seq == "yxz":
            sx = r21
            x = np.arcsin(np.clip(sx, -1.0, 1.0))
            if abs(sx) < 1 - eps:
                y = np.arctan2(-r20, r22)
                z = np.arctan2(-r01, r00)
            else:
                y = np.arctan2(r02, r00)
                z = 0.0

        elif seq == "yzx":
            sz = -r01
            z = np.arcsin(np.clip(sz, -1.0, 1.0))
            if abs(sz) < 1 - eps:
                y = np.arctan2(r02, r00)
                x = np.arctan2(r21, r11)
            else:
                y = np.arctan2(-r20, r22)
                x = 0.0

        elif seq == "zxy":
            sx = -r12
            x = np.arcsin(np.clip(sx, -1.0, 1.0))
            if abs(sx) < 1 - eps:
                z = np.arctan2(r10, r11)
                y = np.arctan2(r02, r22)
            else:
                z = np.arctan2(-r01, r00)
                y = 0.0

        elif seq == "zyx":
            sy = -r02
            y = np.arcsin(np.clip(sy, -1.0, 1.0))
            if abs(sy) < 1 - eps:
                z = np.arctan2(r01, r00)
                x = np.arctan2(r12, r22)
            else:
                z = np.arctan2(-r10, r11)
                x = 0.0

        angles = {
            "xyz": np.array([x, y, z]),
            "xzy": np.array([x, y, z]),
            "yxz": np.array([y, x, z]),
            "yzx": np.array([y, z, x]),
            "zxy": np.array([z, x, y]),
            "zyx": np.array([z, y, x]),
        }[seq]

        if degrees:
            angles = np.degrees(angles)
        return angles

    @classmethod
    def from_euler(cls, seq: str, angles, degrees: bool = False) -> "Rotation":
        """
        Create rotation from Euler angles.

        Args:
            seq: Axis sequence, e.g., "xyz", "zyx", "xzy", "yxz", "yzx", "zxy".
            angles: Iterable of 3 angles [a1, a2, a3].
            degrees: If True, input angles are in degrees.

        Returns:
            Rotation instance.
        """
        seq = seq.lower()
        valid = {"xyz", "xzy", "yxz", "yzx", "zxy", "zyx"}
        if seq not in valid:
            raise ValueError(
                f"Unsupported euler sequence '{seq}'. Supported: {sorted(valid)}"
            )

        angles = np.asarray(angles, dtype=float).reshape(3)
        if degrees:
            angles = np.radians(angles)

        a1, a2, a3 = angles

        def Rx(a):
            ca, sa = np.cos(a), np.sin(a)
            return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

        def Ry(a):
            ca, sa = np.cos(a), np.sin(a)
            return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)

        def Rz(a):
            ca, sa = np.cos(a), np.sin(a)
            return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)

        axis_map = {"x": Rx, "y": Ry, "z": Rz}
        R1 = axis_map[seq[0]](a1)
        R2 = axis_map[seq[1]](a2)
        R3 = axis_map[seq[2]](a3)

        # Active rotations applied in order: first a1 about seq[0], then a2, then a3
        R = R3 @ R2 @ R1
        return cls.from_matrix(R)


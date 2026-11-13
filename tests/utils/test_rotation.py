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

import numpy as np
import pytest

from lerobot.utils.rotation import Rotation


def test_zero_quaternion_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        Rotation(np.zeros(4))


def test_non_finite_quaternion_rejected():
    with pytest.raises(ValueError, match="non-zero|finite"):
        Rotation(np.array([np.nan, 0.0, 0.0, 1.0]))


def test_wrong_shape_rejected():
    with pytest.raises(ValueError, match="shape"):
        Rotation(np.array([1.0, 0.0, 0.0]))


def test_identity_roundtrip():
    r = Rotation.from_rotvec(np.zeros(3))
    assert np.allclose(r.as_rotvec(), 0.0)
    assert np.allclose(r.as_matrix(), np.eye(3))


def test_rotvec_roundtrip():
    rotvec = np.array([0.1, -0.2, 0.3])
    r = Rotation.from_rotvec(rotvec)
    assert np.allclose(r.as_rotvec(), rotvec, atol=1e-6)


def test_batched_conversions_have_scipy_bits():
    """The batched static methods repeat scipy's operations, so the results are bitwise scipy's."""
    scipy_rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    rng = np.random.default_rng(0)
    rpy = rng.uniform(-np.pi, np.pi, size=(256, 3))
    rpy[:4, 1] = [np.pi / 2, -np.pi / 2, 0.0, 1e-9]  # gimbal lock and a tiny angle
    reference = scipy_rotation.from_euler("xyz", rpy)
    np.testing.assert_array_equal(Rotation.euler_xyz_to_quat_xyzw(rpy), reference.as_quat(canonical=False))
    matrices = Rotation.euler_xyz_to_matrix(rpy)
    np.testing.assert_array_equal(matrices, reference.as_matrix())
    np.testing.assert_array_equal(
        Rotation.matrix_to_quat_xyzw(matrices),
        scipy_rotation.from_matrix(matrices).as_quat(canonical=False),
    )
    quats = rng.normal(size=(256, 4))
    np.testing.assert_array_equal(
        Rotation.quat_xyzw_to_matrix(quats), scipy_rotation.from_quat(quats).as_matrix()
    )
    back = Rotation.matrix_to_euler_xyz(matrices[4:])
    np.testing.assert_allclose(Rotation.euler_xyz_to_matrix(back), matrices[4:], atol=1e-12)

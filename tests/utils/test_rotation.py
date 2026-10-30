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


@pytest.mark.parametrize("angle", [1e-12, 1e-9, 3e-8, 1e-6])
def test_small_rotations_preserve_angle_and_axis(angle):
    axis = np.array([1.0, -2.0, 3.0]) / np.sqrt(14.0)
    rotvec = angle * axis
    rotation = Rotation.from_rotvec(rotvec)

    np.testing.assert_allclose(rotation.as_rotvec(), rotvec, rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(rotation.as_quat()[:3], axis * np.sin(angle / 2), rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose((rotation * rotation).as_rotvec(), 2 * rotvec, rtol=1e-12, atol=1e-24)


@pytest.mark.parametrize("sign", [-1, 1])
def test_small_quaternion_sign_does_not_change_rotation(sign):
    angle = 3e-8
    quat = sign * np.array([0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)])
    np.testing.assert_allclose(
        Rotation.from_quat(quat).as_rotvec(), [0.0, angle, 0.0], rtol=1e-12, atol=1e-24
    )

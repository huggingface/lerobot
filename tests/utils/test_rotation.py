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

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
import torch

from lerobot.utils.utils import unwrap_scalar


def test_python_scalar_passthrough():
    assert unwrap_scalar(3) == 3
    assert unwrap_scalar(1.5) == 1.5


def test_single_element_list():
    assert unwrap_scalar([7]) == 7
    assert unwrap_scalar([[2.0]]) == 2.0


def test_empty_or_multi_list_raises():
    with pytest.raises(ValueError, match="list of length"):
        unwrap_scalar([])
    with pytest.raises(ValueError, match="list of length"):
        unwrap_scalar([1, 2])


def test_numpy_scalar_and_size1():
    assert unwrap_scalar(np.array(4.0)) == 4.0
    assert unwrap_scalar(np.array([5])) == 5


def test_numpy_multielement_raises():
    with pytest.raises(ValueError, match="scalar array"):
        unwrap_scalar(np.array([1, 2, 3]))


def test_torch_scalar():
    assert unwrap_scalar(torch.tensor(9)) == 9


def test_torch_multielement_raises():
    with pytest.raises(ValueError, match="scalar"):
        unwrap_scalar(torch.tensor([1, 2]))

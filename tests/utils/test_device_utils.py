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

from unittest.mock import patch

import pytest
import torch

from lerobot.utils.device_utils import (
    get_safe_dtype,
    get_safe_torch_device,
    is_amp_available,
    is_torch_device_available,
)


def test_cpu_always_available():
    assert get_safe_torch_device("cpu") == torch.device("cpu")
    assert is_torch_device_available("cpu")


def test_missing_cuda_raises_valueerror():
    with patch("torch.cuda.is_available", return_value=False), pytest.raises(ValueError, match="CUDA"):
        get_safe_torch_device("cuda")


def test_missing_mps_raises_valueerror():
    with patch("torch.backends.mps.is_available", return_value=False), pytest.raises(ValueError, match="MPS"):
        get_safe_torch_device("mps")


def test_missing_indexed_mps_and_xpu_raise_valueerror():
    with patch("torch.backends.mps.is_available", return_value=False), pytest.raises(ValueError, match="MPS"):
        get_safe_torch_device("mps:0")
    with patch("torch.xpu.is_available", return_value=False), pytest.raises(ValueError, match="XPU"):
        get_safe_torch_device("xpu:0")


def test_indexed_devices_are_available():
    with patch("torch.cuda.is_available", return_value=True):
        assert is_torch_device_available("cuda:1")
    with patch("torch.backends.mps.is_available", return_value=True):
        assert is_torch_device_available("mps:0")
    with patch("torch.xpu.is_available", return_value=True):
        assert is_torch_device_available("xpu:0")


def test_amp_availability_ignores_device_index():
    assert is_amp_available("cuda:1")
    assert is_amp_available("xpu:0")
    assert not is_amp_available("mps:0")


def test_amp_availability_accepts_torch_device():
    assert is_amp_available(torch.device("cuda:1"))
    assert not is_amp_available(torch.device("mps"))


def test_safe_dtype_ignores_device_index():
    assert get_safe_dtype(torch.float64, "mps:0") == torch.float32
    assert get_safe_dtype(torch.float64, torch.device("mps:0")) == torch.float32

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

import logging
import re
from unittest.mock import patch

import pytest
import torch

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.rewards.classifier.configuration_classifier import RewardClassifierConfig
from lerobot.utils import device_utils
from lerobot.utils.device_utils import get_safe_torch_device, is_torch_device_available


def test_cpu_always_available():
    assert get_safe_torch_device("cpu") == torch.device("cpu")
    assert is_torch_device_available("cpu")


def test_missing_cuda_raises_valueerror():
    with patch("torch.cuda.is_available", return_value=False), pytest.raises(ValueError, match="CUDA"):
        get_safe_torch_device("cuda")


def test_missing_mps_raises_valueerror():
    with patch("torch.backends.mps.is_available", return_value=False), pytest.raises(ValueError, match="MPS"):
        get_safe_torch_device("mps")


@pytest.mark.parametrize(
    ("cuda", "kernel_module", "named_driver"),
    [
        ("13.0", "Kernel Module  570.86.15  Thu Jan 15 2026", "570.86.15"),
        ("13.2", "Open Kernel Module for x86_64  580.126.09  Release Build", None),
        ("12.8", "Kernel Module  570.86.15  Thu Jan 15 2026", None),
        (None, "Kernel Module  570.86.15  Thu Jan 15 2026", None),
        ("13.2", "Open Kernel Module for aarch64  TempVersion  Release Build  (r595_00)", None),
        ("13.0", None, None),
    ],
)
def test_cpu_fallback_warning_names_an_nvidia_driver_too_old_for_cuda_13(
    monkeypatch, tmp_path, caplog, cuda, kernel_module, named_driver
):
    version_file = tmp_path / "version"
    if kernel_module is not None:
        version_file.write_text(
            f"NVRM version: NVIDIA UNIX {kernel_module}\nGCC version:  gcc version 13.2.0\n"
        )
    monkeypatch.setattr(device_utils, "NVIDIA_DRIVER_VERSION_FILE", version_file)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for make_config in (lambda: ACTConfig(device="cuda"), lambda: RewardClassifierConfig(device="cuda")):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            make_config()
        assert "Device 'cuda' is not available" in caplog.text
        hint = re.search(r"needs NVIDIA driver 580\.65 or newer, and this host has (\S+)\.", caplog.text)
        assert (hint[1] if hint else None) == named_driver


def test_driver_hint_is_empty_when_cuda_works_on_an_older_driver(monkeypatch, tmp_path):
    """CUDA forward compatibility can run a CUDA 13 build on a driver older than 580.65."""
    version_file = tmp_path / "version"
    version_file.write_text("NVRM version: NVIDIA UNIX Kernel Module  570.86.15  Thu Jan 15 2026\n")
    monkeypatch.setattr(device_utils, "NVIDIA_DRIVER_VERSION_FILE", version_file)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert device_utils.cuda_driver_hint() == ""

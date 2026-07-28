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

import pytest
import torch

pytest.importorskip("transformers")

import lerobot.policies.pi052.modeling_pi052 as modeling_pi052  # noqa: E402
from lerobot.policies.pi052.configuration_pi052 import PI052Config  # noqa: E402


def test_flex_backend_skips_non_cuda_without_initializing(monkeypatch):
    monkeypatch.setattr(modeling_pi052, "_flex_fns", None)
    monkeypatch.setattr(torch, "compile", lambda *args, **kwargs: pytest.fail("torch.compile was called"))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *args, **kwargs: pytest.fail("CUDA properties were queried"),
    )

    assert modeling_pi052._get_flex_fns(torch.device("cpu")) is None
    assert modeling_pi052._get_flex_kernel_options(torch.device("cpu")) is None
    assert modeling_pi052._flex_fns is None


def test_flex_initialization_failure_falls_back(monkeypatch, caplog):
    monkeypatch.setattr(modeling_pi052, "_flex_fns", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_compile(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(torch, "compile", fail_compile)

    with caplog.at_level(logging.WARNING, logger=modeling_pi052.__name__):
        assert modeling_pi052._get_flex_fns(torch.device("cuda", 0)) is None

    assert modeling_pi052._flex_fns is False
    assert "FlexAttention unavailable" in caplog.text


def test_flex_rejects_single_repeat_configuration():
    with pytest.raises(ValueError, match="use_flex_attention requires flow_num_repeats > 1"):
        PI052Config(use_flex_attention=True, flow_num_repeats=1)


def test_flex_accepts_amortized_repeat_configuration():
    config = PI052Config(use_flex_attention=True, flow_num_repeats=5)
    assert config.use_flex_attention

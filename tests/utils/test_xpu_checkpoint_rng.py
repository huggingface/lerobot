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

from functools import partial

import pytest
import torch
from safetensors.torch import save_file

from lerobot.utils.constants import RNG_STATE
from lerobot.utils.random_utils import (
    deserialize_torch_rng_state,
    load_rng_state,
    save_rng_state,
    serialize_rng_state,
    serialize_torch_rng_state,
)


@pytest.fixture
def xpu_rng(monkeypatch):
    # A real CPU generator supplies the same state API without requiring XPU hardware.
    generator = torch.Generator(device="cpu").manual_seed(12345)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "get_rng_state", generator.get_state)
    monkeypatch.setattr(torch.xpu, "set_rng_state", generator.set_state)
    return generator


@pytest.mark.parametrize("via_file", [False, True])
def test_restored_xpu_rng_reproduces_following_samples(xpu_rng, tmp_path, via_file):
    if via_file:
        save_rng_state(tmp_path)
        restore = partial(load_rng_state, tmp_path)
    else:
        restore = partial(deserialize_torch_rng_state, serialize_torch_rng_state())

    expected = torch.rand(8, generator=xpu_rng, device="cpu")
    xpu_rng.manual_seed(9876)
    restore()
    torch.testing.assert_close(torch.rand(8, generator=xpu_rng, device="cpu"), expected, rtol=0, atol=0)


def test_legacy_checkpoint_without_xpu_state_loads_on_xpu(xpu_rng, tmp_path, monkeypatch):
    legacy = serialize_rng_state()
    legacy.pop("torch_xpu_rng_state", None)
    save_file(legacy, tmp_path / RNG_STATE)
    xpu_rng.manual_seed(555)
    original = xpu_rng.get_state()

    def unexpected_restore(*args):
        pytest.fail("A legacy checkpoint has no XPU state to restore")

    monkeypatch.setattr(torch.xpu, "is_available", unexpected_restore)
    monkeypatch.setattr(torch.xpu, "set_rng_state", unexpected_restore)
    load_rng_state(tmp_path)
    torch.testing.assert_close(xpu_rng.get_state(), original)


def test_xpu_checkpoint_loads_without_xpu_hardware(xpu_rng, tmp_path, monkeypatch):
    save_rng_state(tmp_path)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)

    def unexpected_restore(*args):
        pytest.fail("An unavailable XPU must not be initialized")

    monkeypatch.setattr(torch.xpu, "set_rng_state", unexpected_restore)
    load_rng_state(tmp_path)


def test_cpu_checkpoint_does_not_read_unavailable_xpu_state(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)

    def unexpected_read(*args):
        pytest.fail("An unavailable XPU must not be initialized")

    monkeypatch.setattr(torch.xpu, "get_rng_state", unexpected_read)
    save_rng_state(tmp_path)

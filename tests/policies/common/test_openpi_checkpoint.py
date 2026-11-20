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

"""Building parameters on the meta device and loading a checkpoint into them."""

import json
import logging
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from lerobot.policies.common import openpi_checkpoint
from lerobot.policies.common.openpi_checkpoint import (
    _load_state_dict_into_meta_model,
    _parameter_to_meta,
    _parameters_on_meta,
    load_complete_checkpoint,
)
from tests.utils import require_cuda


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        self.norm = nn.LayerNorm(4)
        self.frozen = nn.Linear(4, 4).to(torch.bfloat16).requires_grad_(False)
        self.fixed = nn.Parameter(torch.ones(4), requires_grad=False)
        self.register_buffer("table", torch.arange(4.0) * 2, persistent=False)
        self.register_buffer("scale", torch.ones(4))


class ToyPolicy(nn.Module):
    _supports_meta_load = True

    def __init__(self, config):
        super().__init__()
        # Not called "model": the loader must work without that name.
        self.toy = Toy()

    def _convert_openpi_state_dict(self, state_dict):
        return {f"toy.{name}": tensor for name, tensor in state_dict.items()}


class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        weight = nn.Parameter(torch.ones(4))
        self.a, self.b = nn.Module(), nn.Module()
        self.a.weight = weight
        self.b.weight = weight


def tensors_of(model):
    return {**dict(model.named_parameters()), **dict(model.named_buffers())}


def assert_same(actual, expected):
    actual_tensors, expected_tensors = tensors_of(actual), tensors_of(expected)
    assert actual_tensors.keys() == expected_tensors.keys()
    for name, tensor in expected_tensors.items():
        other = actual_tensors[name]
        assert other.dtype == tensor.dtype, name
        assert other.device == tensor.device, name
        assert other.requires_grad == tensor.requires_grad, name
        assert torch.equal(other, tensor), name


@pytest.fixture
def checkpoint():
    torch.manual_seed(0)
    return {name: tensor.float() for name, tensor in Toy().state_dict().items()}


def load_on_meta(state_dict):
    with _parameters_on_meta():
        model = Toy()
    _load_state_dict_into_meta_model(model, state_dict, "cpu")
    return model


def load_toy_policy(policy_cls, path):
    return load_complete_checkpoint(policy_cls, lambda: str(path), SimpleNamespace(device="cpu"))


def test_other_threads_build_real_parameters():
    built = []
    with _parameters_on_meta():
        worker = threading.Thread(target=lambda: built.append(nn.Linear(2, 2)))
        worker.start()
        worker.join()
    assert not built[0].weight.is_meta


def test_importing_the_loader_registers_no_hook():
    code = (
        "from torch.nn.modules import module\n"
        "hooks, register = [], module.register_module_parameter_registration_hook\n"
        "module.register_module_parameter_registration_hook = lambda hook: hooks.append(hook) or register(hook)\n"
        "import lerobot.policies, lerobot.policies.common.openpi_checkpoint as loader\n"
        "assert loader._parameter_to_meta not in hooks, hooks"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_hook_is_registered_once(monkeypatch):
    registered = []
    monkeypatch.setattr(openpi_checkpoint, "_hook_registered", False)
    monkeypatch.setattr(openpi_checkpoint, "register_module_parameter_registration_hook", registered.append)
    with _parameters_on_meta():
        pass
    with _parameters_on_meta(), _parameters_on_meta():
        pass
    assert registered == [_parameter_to_meta]


def test_leaving_a_nested_block_keeps_the_outer_one():
    with _parameters_on_meta():
        with _parameters_on_meta():
            pass
        assert nn.Linear(2, 2).weight.is_meta
    assert not nn.Linear(2, 2).weight.is_meta


def test_shared_parameter_stays_shared():
    with _parameters_on_meta():
        model = Shared()
        model.c = nn.Module()
        model.c.weight = model.a.weight
    assert model.a.weight is model.b.weight is model.c.weight


@require_cuda
def test_tensors_move_to_the_device_as_they_load(checkpoint):
    with _parameters_on_meta():
        model = Toy()
    _load_state_dict_into_meta_model(model, checkpoint, "cuda")
    for name, tensor in model.state_dict().items():
        assert tensor.is_cuda and torch.equal(tensor.cpu(), checkpoint[name].to(tensor.dtype)), name


def test_each_name_gets_its_own_contiguous_storage(checkpoint):
    checkpoint["norm.weight"] = checkpoint["norm.bias"]
    # A conversion may return a strided view, such as a transpose.
    checkpoint["frozen.weight"] = checkpoint["frozen.weight"].t().contiguous().t()
    model = load_on_meta(checkpoint)
    pointers = {model.norm.weight.data_ptr(), model.norm.bias.data_ptr(), checkpoint["norm.bias"].data_ptr()}
    assert len(pointers) == 3
    assert model.frozen.weight.is_contiguous()


def test_incomplete_checkpoint_raises(checkpoint):
    checkpoint.pop("norm.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_on_meta(checkpoint)


def test_each_tensor_is_copied_on_its_device_then_moved_before_the_next(checkpoint, monkeypatch):
    # Moving every tensor only after all are copied would hold the whole checkpoint in host memory; moving one
    # before its CPU copy would copy it straight from the file, which the loader avoids.
    events = []
    to = torch.Tensor.to

    def record(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args and isinstance(args[0], (str, torch.device)) else None)
        events.append("copy" if device is None else f"to {device}")
        return to(tensor, *args, **kwargs)

    with _parameters_on_meta():
        model = Toy()
    monkeypatch.setattr(torch.Tensor, "to", record)
    _load_state_dict_into_meta_model(model, checkpoint, "cpu")
    assert events == ["copy", "to cpu"] * len(checkpoint)


def test_complete_checkpoint_streams_the_tensors_read_from_the_file(checkpoint, tmp_path, monkeypatch):
    # Those tensors still point into the file; copying them before the stream would hold a second full copy.
    save_file(checkpoint, tmp_path / "model.safetensors")
    read, streamed = [], []
    load_into_meta = openpi_checkpoint._load_state_dict_into_meta_model

    def read_file(*args, **kwargs):
        tensors = load_file(*args, **kwargs)
        read.extend(tensors.values())
        return tensors

    def stream(model, state_dict, device):
        streamed.extend(state_dict.values())
        load_into_meta(model, state_dict, device)

    monkeypatch.setattr(openpi_checkpoint, "load_file", read_file)
    monkeypatch.setattr(openpi_checkpoint, "_load_state_dict_into_meta_model", stream)
    policy = load_toy_policy(ToyPolicy, tmp_path / "model.safetensors")
    expected = Toy()
    expected.load_state_dict(checkpoint)
    assert_same(policy.toy, expected)
    assert len(streamed) == len(checkpoint)
    assert {tensor.data_ptr() for tensor in streamed} <= {tensor.data_ptr() for tensor in read}


def test_complete_checkpoint_with_an_unreadable_tensor_loads_the_regular_way(tmp_path, caplog):
    # The header is valid, but torch has no 6-bit float, so the file opens and only reading the tensor fails.
    header = json.dumps({"weight": {"dtype": "F6_E2M3", "shape": [4], "data_offsets": [0, 3]}}).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(3))
    with caplog.at_level(logging.INFO, logger=openpi_checkpoint.__name__):
        assert load_toy_policy(ToyPolicy, path) is None
    assert "Loading the weights the regular way: cannot read" in caplog.text


def test_complete_checkpoint_the_conversion_rejects_loads_the_regular_way(checkpoint, tmp_path, caplog):
    # The regular load catches the same error, so a failing conversion ends as it does there.
    class Rejecting(ToyPolicy):
        _supports_meta_load = True

        def _convert_openpi_state_dict(self, state_dict):
            raise ValueError("unknown layout")

    save_file(checkpoint, tmp_path / "model.safetensors")
    with caplog.at_level(logging.INFO, logger=openpi_checkpoint.__name__):
        assert load_toy_policy(Rejecting, tmp_path / "model.safetensors") is None
    assert "regular way: cannot convert the checkpoint (unknown layout)" in caplog.text


def test_policy_that_cannot_be_built_on_meta_loads_the_regular_way(caplog):
    class MetaUnsupported(ToyPolicy):
        _supports_meta_load = True

        def __init__(self, config):
            super().__init__(config)
            if openpi_checkpoint.building_on_meta():
                raise NotImplementedError("no meta kernel")

    class NoTokenizer(ToyPolicy):
        _supports_meta_load = True

        def __init__(self, config):
            raise RuntimeError("tokenizer not found")

    def download():
        raise AssertionError("the file was requested before the policy was built")

    with caplog.at_level(logging.WARNING, logger=openpi_checkpoint.__name__):
        assert load_complete_checkpoint(MetaUnsupported, download, SimpleNamespace(device="cpu")) is None
        # Any other constructor error is raised by this first build, before the download, so the regular
        # build never repeats it.
        with pytest.raises(RuntimeError, match="tokenizer not found"):
            load_complete_checkpoint(NoTokenizer, download, SimpleNamespace(device="cpu"))
    assert "regular way: cannot build MetaUnsupported on the meta device (no meta kernel)" in caplog.text

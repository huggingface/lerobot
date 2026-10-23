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

import threading

import pytest
import torch
from torch import nn
from torch.nn.modules import module as torch_module

from lerobot.policies.pretrained import _load_state_dict_into_meta_model, _parameters_on_meta
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


class Shared(nn.Module):
    def __init__(self):
        super().__init__()
        weight = nn.Parameter(torch.ones(4))
        self.a, self.b = nn.Module(), nn.Module()
        self.a.weight = weight
        self.b.weight = weight


class Derived(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))
        self.register_buffer("doubled", self.weight.detach() * 2, persistent=False)


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
    _load_state_dict_into_meta_model(model, state_dict.items(), "cpu")
    return model


def test_only_parameters_go_to_meta():
    with _parameters_on_meta():
        model = Toy()
    assert all(param.is_meta for param in model.parameters())
    assert not model.fixed.requires_grad
    assert torch.equal(model.table, torch.arange(4.0) * 2)
    assert not Toy().embed.weight.is_meta


def test_other_threads_build_real_parameters():
    built = []
    with _parameters_on_meta():
        worker = threading.Thread(target=lambda: built.append(nn.Linear(2, 2)))
        worker.start()
        worker.join()
    assert not built[0].weight.is_meta


def test_hook_registry_never_changes():
    hooks = dict(torch_module._global_parameter_registration_hooks)
    with _parameters_on_meta(), _parameters_on_meta():
        Toy()
        assert torch_module._global_parameter_registration_hooks == hooks
    assert torch_module._global_parameter_registration_hooks == hooks


def test_concurrent_blocks_do_not_interfere():
    both_inside, first_left = threading.Barrier(2), threading.Event()
    built = []

    def first():
        with _parameters_on_meta():
            both_inside.wait()
            built.append(Toy())
        first_left.set()

    def second():
        with _parameters_on_meta():
            both_inside.wait()
            first_left.wait()
            built.append(Toy())

    workers = [threading.Thread(target=first), threading.Thread(target=second)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert len(built) == 2 and all(param.is_meta for model in built for param in model.parameters())


def test_shared_parameter_stays_shared():
    with _parameters_on_meta():
        model = Shared()
        model.c = nn.Module()
        model.c.weight = model.a.weight
    assert model.a.weight is model.b.weight is model.c.weight
    with pytest.raises(ValueError, match="shares parameters"):
        _load_state_dict_into_meta_model(model, {"a.weight": torch.ones(4)}.items(), "cpu")


def test_buffer_computed_from_a_parameter_raises():
    with _parameters_on_meta():
        model = Derived()
    with pytest.raises(RuntimeError, match=r"computed these buffers from parameters.*doubled"):
        _load_state_dict_into_meta_model(model, {"weight": torch.ones(4)}.items(), "cpu")


def test_matches_regular_load(checkpoint):
    expected = Toy()
    expected.load_state_dict(checkpoint)
    model = load_on_meta(checkpoint)
    assert_same(model, expected)
    assert model.frozen.weight.dtype == torch.bfloat16


@require_cuda
def test_tensors_move_to_the_device_as_they_load(checkpoint):
    with _parameters_on_meta():
        model = Toy()
    _load_state_dict_into_meta_model(model, checkpoint.items(), "cuda")
    for name, tensor in model.state_dict().items():
        assert tensor.is_cuda and torch.equal(tensor.cpu(), checkpoint[name].to(tensor.dtype)), name


def test_each_name_gets_its_own_storage(checkpoint):
    checkpoint["norm.weight"] = checkpoint["norm.bias"]
    model = load_on_meta(checkpoint)
    pointers = {model.norm.weight.data_ptr(), model.norm.bias.data_ptr(), checkpoint["norm.bias"].data_ptr()}
    assert len(pointers) == 3


@pytest.mark.parametrize(
    "edit, error",
    [
        (lambda sd: sd.pop("norm.weight"), "Missing key"),
        (lambda sd: sd.update(extra=torch.zeros(1)), "Unexpected key"),
        (lambda sd: sd.update({"norm.weight": torch.ones(5)}), "size mismatch"),
    ],
)
def test_incomplete_checkpoint_raises(checkpoint, edit, error):
    edit(checkpoint)
    with pytest.raises(RuntimeError, match=error):
        load_on_meta(checkpoint)

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""CPU regression tests for the shared Gaussian readout cache.

The tiny fixture follows upstream test_gaussian_readout.py (Apache-2.0,
Copyright 2025 The HuggingFace Inc. team). No policy weights or rollouts are used.
"""

import os
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.common import gaussian_readout as gar
from tests.policies.common.test_gaussian_readout import _sample_tiny, tiny_flow  # noqa: F401


@pytest.fixture
def bound_teacher(tmp_path):
    torch.manual_seed(20261101)
    teacher = nn.Module()
    teacher.action_out_proj = nn.Linear(8, 8)
    teacher.register_buffer("measured_buffer", torch.ones(1))
    teacher.config = SimpleNamespace(chunk_size=4, max_action_dim=8)
    gar.bind_processor_sha256(teacher, "processor-hash")
    path = str(tmp_path / "readout.safetensors")
    readout = gar.GaussianReadout(
        8,
        chunk_size=4,
        motion_dim=6,
        gripper_index=6,
        max_action_dim=8,
        action_dim=7,
        open_normalized=-1.0,
        close_normalized=1.0,
        action_mean=torch.zeros(7),
        action_std=torch.ones(7),
    )
    gar.save_readout(
        readout,
        path,
        teacher_sha256=gar.state_dict_sha256(teacher),
        processor_sha256="processor-hash",
    )
    return teacher, path


def test_unchanged_teacher_reuses_without_rehash(bound_teacher, monkeypatch):
    teacher, path = bound_teacher
    loaded = gar._cached_readout(teacher, path)
    monkeypatch.setattr(gar, "state_dict_sha256", lambda *_: pytest.fail("unneeded full hash"))
    assert gar._cached_readout(teacher, path) is loaded


@pytest.mark.parametrize(
    "operation", ["load_state_dict", "optimizer_step", "replace_parameter", "buffer_update"]
)
def test_changed_teacher_warm_and_cold_agree(bound_teacher, operation):
    teacher, path = bound_teacher
    gar._cached_readout(teacher, path)
    if operation == "load_state_dict":
        changed = {key: value.clone() for key, value in teacher.state_dict().items()}
        changed["action_out_proj.weight"].add_(1)
        teacher.load_state_dict(changed)
    elif operation == "optimizer_step":
        optimizer = torch.optim.SGD(teacher.parameters(), lr=0.1)
        teacher.action_out_proj.weight.grad = torch.ones_like(teacher.action_out_proj.weight)
        optimizer.step()
    elif operation == "replace_parameter":
        teacher.action_out_proj.weight = nn.Parameter(teacher.action_out_proj.weight.detach().clone() + 1)
    else:
        teacher.measured_buffer.add_(1)
    # The unchanged native cold loader supplies the expected compatibility verdict.
    with pytest.raises(ValueError, match="teacher hash"):
        gar.load_gaussian_readout(path, teacher, "processor-hash")
    with pytest.raises(ValueError, match="teacher hash"):
        gar._cached_readout(teacher, path)


def test_equal_value_reload_remains_accepted(bound_teacher):
    teacher, path = bound_teacher
    gar._cached_readout(teacher, path)
    teacher.load_state_dict({key: value.clone() for key, value in teacher.state_dict().items()})
    cold = gar.load_gaussian_readout(path, teacher, "processor-hash")
    warm = gar._cached_readout(teacher, path)
    for key, value in cold.state_dict().items():
        assert torch.equal(warm.state_dict()[key], value), key


def test_processor_change_remains_rejected(bound_teacher):
    teacher, path = bound_teacher
    gar._cached_readout(teacher, path)
    gar.bind_processor_sha256(teacher, "different-processor")
    with pytest.raises(ValueError, match="processor hash"):
        gar._cached_readout(teacher, path)


@pytest.mark.parametrize("field", ["chunk_size", "max_action_dim", "in_features"])
def test_changed_loader_shape_contract_rechecked(bound_teacher, field):
    teacher, path = bound_teacher
    gar._cached_readout(teacher, path)
    subject = teacher.action_out_proj if field == "in_features" else teacher.config
    setattr(subject, field, getattr(subject, field) + 1)
    with pytest.raises(ValueError, match="does not match"):
        gar.load_gaussian_readout(path, teacher, "processor-hash")
    with pytest.raises(ValueError, match="does not match"):
        gar._cached_readout(teacher, path)


def test_valid_reload_revalidates_then_reuses(bound_teacher, monkeypatch):
    teacher, path = bound_teacher
    gar._cached_readout(teacher, path)
    teacher.load_state_dict({key: value.clone() for key, value in teacher.state_dict().items()})
    calls = []
    original = gar.state_dict_sha256

    def count_hash(model):
        calls.append(1)
        return original(model)

    monkeypatch.setattr(gar, "state_dict_sha256", count_hash)
    loaded = gar._cached_readout(teacher, path)
    assert gar._cached_readout(teacher, path) is loaded
    assert len(calls) == 1


def test_inference_tensor_requires_explicit_preparation(bound_teacher):
    teacher, path = bound_teacher
    with torch.inference_mode():
        teacher.action_out_proj.weight = nn.Parameter(teacher.action_out_proj.weight.clone())
        with pytest.raises(ValueError, match="prepare_gaussian_readout_teacher"):
            gar._cached_readout(teacher, path)
        gar.prepare_gaussian_readout_teacher(teacher)
        gar._cached_readout(teacher, path)
        gar._cached_readout(teacher, path)
        teacher.action_out_proj.weight.add_(1)
        with pytest.raises(ValueError, match="teacher hash"):
            gar._cached_readout(teacher, path)


def test_restore_compatible_weights_after_refusal(bound_teacher):
    teacher, path = bound_teacher
    original = {key: value.clone() for key, value in teacher.state_dict().items()}
    gar._cached_readout(teacher, path)
    teacher.measured_buffer.add_(1)
    with pytest.raises(ValueError, match="teacher hash"):
        gar._cached_readout(teacher, path)
    teacher.load_state_dict(original)
    assert isinstance(gar._cached_readout(teacher, path), gar.GaussianReadout)


@pytest.mark.parametrize("operation", ["load_state_dict", "optimizer_step"])
def test_native_sampler_rejects_teacher_update(tiny_flow, bound_teacher, operation):  # noqa: F811
    model, backbone = tiny_flow
    teacher, path = bound_teacher
    readout = gar.load_gaussian_readout(path, teacher, "processor-hash")
    gar.save_readout(
        readout, path, teacher_sha256=gar.state_dict_sha256(model), processor_sha256="processor-hash"
    )
    gar.bind_processor_sha256(model, "processor-hash")
    model.config.gaussian_readout_path = path
    tokens = torch.zeros(1, 2, dtype=torch.long)
    masks = torch.ones_like(tokens, dtype=torch.bool)
    state = torch.zeros(1, 8)
    with torch.no_grad():
        assert _sample_tiny(model, tokens, masks, state).shape == (1, 4, 8)
    if operation == "load_state_dict":
        changed = {key: value.clone() for key, value in model.state_dict().items()}
        changed["action_out_proj.weight"].add_(1)
        model.load_state_dict(changed)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model.action_out_proj.weight.grad = torch.ones_like(model.action_out_proj.weight)
        optimizer.step()
    with pytest.raises(ValueError, match="teacher hash"):
        gar.load_gaussian_readout(path, model, "processor-hash")
    calls_before = backbone.suffix_calls
    rng_before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="teacher hash"):
        _sample_tiny(model, tokens, masks, state)
    assert backbone.suffix_calls == calls_before
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_cache_does_not_retain_replaced_teacher_weights(bound_teacher):
    teacher, path = bound_teacher
    gar._cached_readout(teacher, path)
    old = weakref.ref(teacher.action_out_proj.weight)
    teacher.action_out_proj.weight = nn.Parameter(teacher.action_out_proj.weight.detach().clone() + 1)
    assert old() is None
    with pytest.raises(ValueError, match="teacher hash"):
        gar._cached_readout(teacher, path)


def test_untracked_write_requires_explicit_cache_clear(bound_teacher):
    teacher, path = bound_teacher
    gar._cached_readout(teacher, path)
    # Record the fast path's declared boundary instead of claiming a universal guard.
    teacher.action_out_proj.weight.data.add_(1)
    gar._cached_readout(teacher, path)
    teacher._gar_cache = None
    with pytest.raises(ValueError, match="teacher hash"):
        gar._cached_readout(teacher, path)


def test_data_storage_replacement_revalidates(bound_teacher):
    teacher, path = bound_teacher
    gar._cached_readout(teacher, path)
    saved = teacher.action_out_proj.weight.data
    teacher.action_out_proj.weight.data = saved + 1
    with pytest.raises(ValueError, match="teacher hash"):
        gar._cached_readout(teacher, path)


def test_nonpersistent_buffer_changes_teacher_hash(bound_teacher):
    teacher, path = bound_teacher
    loaded = gar._cached_readout(teacher, path)
    teacher.register_buffer("rotary_frequency", torch.ones(8), persistent=False)
    before = gar.state_dict_sha256(teacher)
    gar.save_readout(loaded, path, teacher_sha256=before, processor_sha256="processor-hash")
    gar._cached_readout(teacher, path)
    teacher.rotary_frequency.mul_(0)
    assert gar.state_dict_sha256(teacher) != before
    with pytest.raises(ValueError, match="teacher hash"):
        gar._cached_readout(teacher, path)


def test_artifact_replacement_and_cached_readout_mutation(bound_teacher):
    teacher, path = bound_teacher
    original = gar._cached_readout(teacher, path)
    with torch.no_grad():
        original.affine_bias.fill_(5)
    restored = gar._cached_readout(teacher, path)
    assert restored is not original
    assert (restored.affine_bias == 0).all()
    with torch.no_grad():
        restored.affine_bias.fill_(7)
    stamp = os.stat(path).st_mtime_ns + 1_000_000_000
    gar.save_readout(
        restored, path, teacher_sha256=gar.state_dict_sha256(teacher), processor_sha256="processor-hash"
    )
    os.utime(path, ns=(stamp, stamp))
    loaded = gar._cached_readout(teacher, path)
    assert loaded is not restored
    assert (loaded.affine_bias == 7).all()


def test_prepared_inference_teacher_reuses_without_rehash(bound_teacher, monkeypatch):
    teacher, path = bound_teacher
    with torch.inference_mode():
        teacher.action_out_proj.weight = nn.Parameter(teacher.action_out_proj.weight.clone())
    old = teacher.action_out_proj.weight
    gar.prepare_gaussian_readout_teacher(teacher)
    assert teacher.action_out_proj.weight is not old
    assert not torch.is_inference(teacher.action_out_proj.weight)
    loaded = gar._cached_readout(teacher, path)
    monkeypatch.setattr(gar, "state_dict_sha256", lambda *_: pytest.fail("unneeded full hash"))
    with torch.inference_mode():
        assert gar._cached_readout(teacher, path) is loaded


@pytest.mark.parametrize("field", ["min_period", "max_period", "use_typed_attention_masks"])
def test_inference_settings_in_cache_token(bound_teacher, field):
    teacher, _ = bound_teacher
    owner = teacher if field == "use_typed_attention_masks" else teacher.config
    setattr(owner, field, 1)
    before = gar._teacher_cache_token(teacher)
    setattr(owner, field, 2)
    assert gar._teacher_cache_token(teacher) != before

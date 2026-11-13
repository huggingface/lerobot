#!/usr/bin/env python

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

import random

import numpy as np
import pytest
import torch

from lerobot.utils.random_utils import (
    deserialize_numpy_rng_state,
    deserialize_python_rng_state,
    deserialize_rng_state,
    deserialize_torch_rng_state,
    get_rng_state,
    load_rng_state,
    preserve_rng_state,
    save_rng_state,
    seeded_context,
    serialize_numpy_rng_state,
    serialize_python_rng_state,
    serialize_rng_state,
    serialize_torch_rng_state,
    set_rng_state,
    set_seed,
)


@pytest.fixture
def fixed_seed():
    """Fixture to set a consistent initial seed for each test."""
    set_seed(12345)
    yield


def test_serialize_deserialize_python_rng(fixed_seed):
    # Save state after generating val1
    _ = random.random()
    st = serialize_python_rng_state()
    # Next random is val2
    val2 = random.random()
    # Restore the state, so the next random should match val2
    deserialize_python_rng_state(st)
    val3 = random.random()
    assert val2 == val3


@pytest.mark.parametrize("draws_before_save", [0, 1, 2, 3])
def test_python_gaussian_checkpoint_continuation(fixed_seed, tmp_path, draws_before_save):
    for _ in range(draws_before_save):
        random.gauss(0, 1)
    save_rng_state(tmp_path)
    expected = [random.gauss(0, 1) for _ in range(6)]

    random.seed(999)
    load_rng_state(tmp_path)
    assert [random.gauss(0, 1) for _ in range(6)] == expected


@pytest.mark.parametrize("cached_gaussian", [None, 0.0, 0.12345678901234568])
def test_python_rng_cached_gaussian_roundtrip(fixed_seed, cached_gaussian):
    original = (*random.getstate()[:2], cached_gaussian)
    random.setstate(original)
    state = serialize_python_rng_state()

    random.gauss(0, 1)
    deserialize_python_rng_state(state)
    assert random.getstate() == original


def test_deserialize_python_rng_legacy_state(fixed_seed):
    state = serialize_python_rng_state()
    state.pop("py_rng_cached_gaussian", None)
    expected = random.random()
    random.gauss(0, 1)

    deserialize_python_rng_state(state)
    assert random.getstate()[2] is None
    assert random.random() == expected


def test_serialize_deserialize_numpy_rng(fixed_seed):
    _ = np.random.rand()
    st = serialize_numpy_rng_state()
    val2 = np.random.rand()
    deserialize_numpy_rng_state(st)
    val3 = np.random.rand()
    assert val2 == val3


@pytest.mark.parametrize("draws_before_save", [0, 1, 2, 3])
def test_numpy_gaussian_checkpoint_continuation(fixed_seed, tmp_path, draws_before_save):
    np.random.standard_normal(draws_before_save)
    save_rng_state(tmp_path)
    expected = np.random.standard_normal(6)

    np.random.seed(999)
    load_rng_state(tmp_path)
    np.testing.assert_array_equal(np.random.standard_normal(6), expected)


def test_deserialize_numpy_rng_legacy_float32_cache(fixed_seed):
    np.random.standard_normal()
    state = serialize_numpy_rng_state()
    state["np_rng_cached_gaussian"] = state["np_rng_cached_gaussian"].float()
    expected = state["np_rng_cached_gaussian"].item()

    np.random.seed(999)
    deserialize_numpy_rng_state(state)
    assert np.random.standard_normal() == expected


def test_serialize_deserialize_torch_rng(fixed_seed):
    _ = torch.rand(1).item()
    st = serialize_torch_rng_state()
    val2 = torch.rand(1).item()
    deserialize_torch_rng_state(st)
    val3 = torch.rand(1).item()
    assert val2 == val3


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_serialize_deserialize_torch_rng_mps(fixed_seed):
    _ = torch.rand(1, device="mps").item()
    st = serialize_torch_rng_state()
    assert "torch_mps_rng_state" in st
    val2 = torch.rand(1, device="mps").item()
    deserialize_torch_rng_state(st)
    val3 = torch.rand(1, device="mps").item()
    assert val2 == val3


def test_serialize_deserialize_rng(fixed_seed):
    # Generate one from each library
    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(1).item()
    # Serialize
    st = serialize_rng_state()
    # Generate second set
    val_py2 = random.random()
    val_np2 = np.random.rand()
    val_th2 = torch.rand(1).item()
    # Restore, so the next draws should match val_py2, val_np2, val_th2
    deserialize_rng_state(st)
    assert random.random() == val_py2
    assert np.random.rand() == val_np2
    assert torch.rand(1).item() == val_th2


def test_get_set_rng_state(fixed_seed):
    st = get_rng_state()
    val1 = (random.random(), np.random.rand(), torch.rand(1).item())
    # Change states
    random.random()
    np.random.rand()
    torch.rand(1)
    # Restore
    set_rng_state(st)
    val2 = (random.random(), np.random.rand(), torch.rand(1).item())
    assert val1 == val2


def test_set_seed():
    set_seed(1337)
    val1 = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(1337)
    val2 = (random.random(), np.random.rand(), torch.rand(1).item())
    assert val1 == val2


def test_seeded_context(fixed_seed):
    val1 = (random.random(), np.random.rand(), torch.rand(1).item())
    with seeded_context(1337):
        seeded_val1 = (random.random(), np.random.rand(), torch.rand(1).item())
    val2 = (random.random(), np.random.rand(), torch.rand(1).item())
    with seeded_context(1337):
        seeded_val2 = (random.random(), np.random.rand(), torch.rand(1).item())

    assert seeded_val1 == seeded_val2
    assert all(a != b for a, b in zip(val1, seeded_val1, strict=True))  # changed inside the context
    assert all(a != b for a, b in zip(val2, seeded_val2, strict=True))  # changed again after exiting


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_seeded_context_restores_rng_after_exception(fixed_seed, error_type):
    original = get_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    set_rng_state(original)
    error = error_type("context failed")

    with pytest.raises(error_type) as exc_info, seeded_context(1337):
        random.random()
        np.random.rand()
        torch.rand(3)
        raise error

    assert exc_info.value is error
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seeded_context_restores_rng_after_invalid_seed(fixed_seed, seed):
    original = get_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    set_rng_state(original)

    with pytest.raises(ValueError), seeded_context(seed):
        pytest.fail("An invalid NumPy seed must fail before entering the context")

    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


@pytest.mark.parametrize("inner_seed, error_type", [(456, RuntimeError), (-1, ValueError)])
def test_nested_seeded_context_restores_outer_rng_after_exception(fixed_seed, inner_seed, error_type):
    with seeded_context(123):
        original = get_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        set_rng_state(original)

        with pytest.raises(error_type), seeded_context(inner_seed):
            raise error_type("inner context failed")

        assert random.random() == expected[0]
        assert np.random.rand() == expected[1]
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)


@pytest.mark.parametrize("fail", [False, True])
def test_seeded_context_preserves_gaussian_caches(fixed_seed, fail):
    random.gauss(0, 1)
    np.random.standard_normal()
    original = get_rng_state()
    expected_python = [random.gauss(0, 1) for _ in range(6)]
    expected_numpy = np.random.standard_normal(6)
    set_rng_state(original)

    try:
        with seeded_context(1337):
            random.gauss(0, 1)
            np.random.standard_normal()
            if fail:
                raise RuntimeError("context failed")
    except RuntimeError:
        if not fail:
            raise

    assert [random.gauss(0, 1) for _ in range(6)] == expected_python
    np.testing.assert_array_equal(np.random.standard_normal(6), expected_numpy)


def _cpu_rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def _assert_cpu_rng_state(expected):
    assert random.getstate() == expected[0]
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == expected[1][0]
    np.testing.assert_array_equal(actual_numpy[1], expected[1][1])
    assert actual_numpy[2:] == expected[1][2:]
    assert torch.equal(torch.get_rng_state(), expected[2])


def _draw_cpu():
    return random.gauss(0, 1), np.random.standard_normal(), torch.rand(3)


@pytest.mark.parametrize("error_type", [None, RuntimeError, KeyboardInterrupt])
def test_preserve_rng_state_cpu_caches_and_nested_scopes(fixed_seed, error_type):
    random.gauss(0, 1)
    np.random.standard_normal()
    original = _cpu_rng_state()
    assert original[0][2] is not None
    assert original[1][3] == 1
    error = error_type("context failed") if error_type else None

    def run():
        with preserve_rng_state(torch.device("cpu")):
            first = _draw_cpu()
            assert first[:2] == (original[0][2], original[1][4])
            outer = _cpu_rng_state()
            with pytest.raises(RuntimeError), preserve_rng_state(torch.device("cpu")):
                _draw_cpu()
                raise RuntimeError("inner failure")
            _assert_cpu_rng_state(outer)
            with preserve_rng_state(torch.device("cpu")):
                nested = _draw_cpu()
            _assert_cpu_rng_state(outer)
            continuation = _draw_cpu()
            assert nested[:2] == continuation[:2]
            assert torch.equal(nested[2], continuation[2])
            if error is not None:
                raise error
            return first

    if error is not None:
        with pytest.raises(error_type) as exc_info:
            run()
        assert exc_info.value is error
    else:
        first = run()
        _assert_cpu_rng_state(original)
        continuation = _draw_cpu()
        assert first[:2] == continuation[:2]
        assert torch.equal(first[2], continuation[2])
        random.setstate(original[0])
        np.random.set_state(original[1])
        torch.set_rng_state(original[2])
    _assert_cpu_rng_state(original)


def test_preserve_cpu_never_reseeds_or_accesses_accelerators(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU preservation must not reseed or access an accelerator")

    for module, names in (
        (random, ["seed"]),
        (np.random, ["seed"]),
        (torch, ["seed", "manual_seed"]),
        (torch.backends.mps, ["is_available"]),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    for backend in (torch.cuda, torch.mps, torch.xpu):
        for name in (
            "is_available",
            "is_initialized",
            "device_count",
            "current_device",
            "get_rng_state",
            "set_rng_state",
            "manual_seed",
            "manual_seed_all",
            "seed",
            "seed_all",
        ):
            if hasattr(backend, name):
                monkeypatch.setattr(backend, name, forbidden)
    original = _cpu_rng_state()
    with preserve_rng_state(torch.device("cpu")):
        _draw_cpu()
    _assert_cpu_rng_state(original)


@pytest.mark.parametrize(
    "device_name, selected", [("cuda:2", 2), ("cuda", 1), ("xpu:2", 2), ("xpu", 1), ("mps", 0)]
)
@pytest.mark.parametrize("error_type", [None, RuntimeError, KeyboardInterrupt])
def test_preserve_selected_backend_routing(monkeypatch, device_name, selected, error_type):
    # Real fork_rng, fake device state APIs: this checks routing, not native GPU behavior.
    device = torch.device(device_name)
    backend = getattr(torch, device.type)
    states = {0: torch.tensor([10]), 1: torch.tensor([11]), 2: torch.tensor([12])}
    current = 1
    current_calls = []
    calls = []

    def get_state(index):
        calls.append(("get", index))
        return states[index].clone()

    def set_state(state, index):
        calls.append(("set", index))
        states[index] = state.clone()

    def forbidden(*args, **kwargs):
        pytest.fail("Must not enumerate or probe unrelated devices")

    def current_device():
        current_calls.append(current)
        return current

    for name in ("cuda", "mps", "xpu"):
        module = getattr(torch, name)
        for method in ("is_available", "device_count", "get_rng_state", "set_rng_state"):
            monkeypatch.setattr(module, method, forbidden)
    monkeypatch.setattr(backend, "get_rng_state", get_state)
    monkeypatch.setattr(backend, "set_rng_state", set_state)
    if device.type != "mps":
        monkeypatch.setattr(backend, "current_device", current_device)
    original = _cpu_rng_state()
    error = error_type("device context failed") if error_type else None

    def run():
        nonlocal current
        with preserve_rng_state(device):
            states[selected] += 20
            current = 0
            _draw_cpu()
            if error is not None:
                raise error

    if error is None:
        run()
    else:
        with pytest.raises(error_type) as exc_info:
            run()
        assert exc_info.value is error
    assert calls == [("get", selected), ("set", selected)]
    assert current_calls == ([1] if device.index is None and device.type != "mps" else [])
    assert [states[i].item() for i in range(3)] == [10, 11, 12]
    _assert_cpu_rng_state(original)


@pytest.mark.gpu
@pytest.mark.parametrize("device_type", ["cuda", "mps", "xpu"])
def test_preserve_rng_state_native_device(device_type):
    backend = getattr(torch, device_type)
    if not backend.is_available():
        pytest.skip(f"Native {device_type} unavailable")
    device = torch.device(device_type, 0)
    original = backend.get_rng_state(device)
    with preserve_rng_state(device):
        expected = torch.rand(8, device=device)
        assert not torch.equal(backend.get_rng_state(device), original)
    assert torch.equal(backend.get_rng_state(device), original)
    assert torch.equal(torch.rand(8, device=device), expected)

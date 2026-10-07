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


def test_serialize_deserialize_numpy_rng(fixed_seed):
    _ = np.random.rand()
    st = serialize_numpy_rng_state()
    val2 = np.random.rand()
    deserialize_numpy_rng_state(st)
    val3 = np.random.rand()
    assert val2 == val3


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

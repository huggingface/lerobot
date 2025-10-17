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

import numpy as np
import pytest
import torch

from lerobot.processor import TransitionKey
from lerobot.processor.converters import (
    batch_to_transition,
    create_transition,
    to_tensor,
    transition_to_batch,
)
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, OBS_STR, REWARD


# Tests for the unified to_tensor function
def test_to_tensor_numpy_arrays():
    """Test to_tensor with various numpy arrays."""
    # Regular numpy array
    arr = np.array([1.0, 2.0, 3.0])
    result = to_tensor(arr)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    # Different numpy dtypes should convert to float32 by default
    int_arr = np.array([1, 2, 3], dtype=np.int64)
    result = to_tensor(int_arr)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    # uint8 arrays (previously "preserved") should now convert
    uint8_arr = np.array([100, 150, 200], dtype=np.uint8)
    result = to_tensor(uint8_arr)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([100.0, 150.0, 200.0]))


def test_to_tensor_numpy_scalars():
    """Test to_tensor with numpy scalars (0-dimensional arrays)."""
    # numpy float32 scalar
    scalar = np.float32(3.14)
    result = to_tensor(scalar)
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0  # Should be 0-dimensional tensor
    assert result.dtype == torch.float32
    assert result.item() == pytest.approx(3.14)

    # numpy int32 scalar
    int_scalar = np.int32(42)
    result = to_tensor(int_scalar)
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 0
    assert result.dtype == torch.float32
    assert result.item() == pytest.approx(42.0)


def test_to_tensor_python_scalars():
    """Test to_tensor with Python scalars."""
    # Python int
    result = to_tensor(42)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.item() == pytest.approx(42.0)

    # Python float
    result = to_tensor(3.14)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.item() == pytest.approx(3.14)


def test_to_tensor_sequences():
    """Test to_tensor with lists and tuples."""
    # List
    result = to_tensor([1, 2, 3])
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    # Tuple
    result = to_tensor((4.5, 5.5, 6.5))
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([4.5, 5.5, 6.5]))


def test_to_tensor_existing_tensors():
    """Test to_tensor with existing PyTorch tensors."""
    # Tensor with same dtype should pass through with potential device change
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    result = to_tensor(tensor)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, tensor)

    # Tensor with different dtype should convert
    int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
    result = to_tensor(int_tensor)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))


def test_to_tensor_dictionaries():
    """Test to_tensor with nested dictionaries."""
    # Simple dictionary
    data = {"mean": [0.1, 0.2], "std": np.array([1.0, 2.0]), "count": 42}
    result = to_tensor(data)
    assert isinstance(result, dict)
    assert isinstance(result["mean"], torch.Tensor)
    assert isinstance(result["std"], torch.Tensor)
    assert isinstance(result["count"], torch.Tensor)
    assert torch.allclose(result["mean"], torch.tensor([0.1, 0.2]))
    assert torch.allclose(result["std"], torch.tensor([1.0, 2.0]))
    assert result["count"].item() == pytest.approx(42.0)

    # Nested dictionary
    nested = {
        ACTION: {"mean": [0.1, 0.2], "std": [1.0, 2.0]},
        OBS_STR: {"mean": np.array([0.5, 0.6]), "count": 10},
    }
    result = to_tensor(nested)
    assert isinstance(result, dict)
    assert isinstance(result[ACTION], dict)
    assert isinstance(result[OBS_STR], dict)
    assert isinstance(result[ACTION]["mean"], torch.Tensor)
    assert isinstance(result[OBS_STR]["mean"], torch.Tensor)
    assert torch.allclose(result[ACTION]["mean"], torch.tensor([0.1, 0.2]))
    assert torch.allclose(result[OBS_STR]["mean"], torch.tensor([0.5, 0.6]))


def test_to_tensor_none_filtering():
    """Test that None values are filtered out from dictionaries."""
    data = {"valid": [1, 2, 3], "none_value": None, "nested": {"valid": [4, 5], "also_none": None}}
    result = to_tensor(data)
    assert "none_value" not in result
    assert "also_none" not in result["nested"]
    assert "valid" in result
    assert "valid" in result["nested"]
    assert torch.allclose(result["valid"], torch.tensor([1.0, 2.0, 3.0]))


def test_to_tensor_dtype_parameter():
    """Test to_tensor with different dtype parameters."""
    arr = np.array([1, 2, 3])

    # Default dtype (float32)
    result = to_tensor(arr)
    assert result.dtype == torch.float32

    # Explicit float32
    result = to_tensor(arr, dtype=torch.float32)
    assert result.dtype == torch.float32

    # Float64
    result = to_tensor(arr, dtype=torch.float64)
    assert result.dtype == torch.float64

    # Preserve original dtype
    float64_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = to_tensor(float64_arr, dtype=None)
    assert result.dtype == torch.float64


def test_to_tensor_device_parameter():
    """Test to_tensor with device parameter."""
    arr = np.array([1.0, 2.0, 3.0])

    # CPU device (default)
    result = to_tensor(arr, device="cpu")
    assert result.device.type == "cpu"

    # CUDA device (if available)
    if torch.cuda.is_available():
        result = to_tensor(arr, device="cuda")
        assert result.device.type == "cuda"


def test_to_tensor_empty_dict():
    """Test to_tensor with empty dictionary."""
    result = to_tensor({})
    assert isinstance(result, dict)
    assert len(result) == 0


def test_to_tensor_unsupported_type():
    """Test to_tensor with unsupported types raises TypeError."""
    with pytest.raises(TypeError, match="Unsupported type for tensor conversion"):
        to_tensor("unsupported_string")

    with pytest.raises(TypeError, match="Unsupported type for tensor conversion"):
        to_tensor(object())


def test_batch_to_transition_with_index_fields():
    """Test that batch_to_transition handles index and task_index fields correctly."""

    # Create batch with index and task_index fields
    batch = {
        OBS_STATE: torch.randn(1, 7),
        ACTION: torch.randn(1, 4),
        REWARD: 1.5,
        DONE: False,
        "task": ["pick_cube"],
        "index": torch.tensor([42], dtype=torch.int64),
        "task_index": torch.tensor([3], dtype=torch.int64),
    }

    transition = batch_to_transition(batch)

    # Check basic transition structure
    assert TransitionKey.OBSERVATION in transition
    assert TransitionKey.ACTION in transition
    assert TransitionKey.COMPLEMENTARY_DATA in transition

    # Check that index and task_index are in complementary_data
    comp_data = transition[TransitionKey.COMPLEMENTARY_DATA]
    assert "index" in comp_data
    assert "task_index" in comp_data
    assert "task" in comp_data

    # Verify values
    assert torch.equal(comp_data["index"], batch["index"])
    assert torch.equal(comp_data["task_index"], batch["task_index"])
    assert comp_data["task"] == batch["task"]


def testtransition_to_batch_with_index_fields():
    """Test that transition_to_batch handles index and task_index fields correctly."""

    # Create transition with index and task_index in complementary_data
    transition = create_transition(
        observation={OBS_STATE: torch.randn(1, 7)},
        action=torch.randn(1, 4),
        reward=1.5,
        done=False,
        complementary_data={
            "task": ["navigate"],
            "index": torch.tensor([100], dtype=torch.int64),
            "task_index": torch.tensor([5], dtype=torch.int64),
        },
    )

    batch = transition_to_batch(transition)

    # Check that index and task_index are in the batch
    assert "index" in batch
    assert "task_index" in batch
    assert "task" in batch

    # Verify values
    assert torch.equal(batch["index"], transition[TransitionKey.COMPLEMENTARY_DATA]["index"])
    assert torch.equal(batch["task_index"], transition[TransitionKey.COMPLEMENTARY_DATA]["task_index"])
    assert batch["task"] == transition[TransitionKey.COMPLEMENTARY_DATA]["task"]


def test_batch_to_transition_without_index_fields():
    """Test that conversion works without index and task_index fields."""

    # Batch without index/task_index
    batch = {
        OBS_STATE: torch.randn(1, 7),
        ACTION: torch.randn(1, 4),
        "task": ["pick_cube"],
    }

    transition = batch_to_transition(batch)
    comp_data = transition[TransitionKey.COMPLEMENTARY_DATA]

    # Should have task but not index/task_index
    assert "task" in comp_data
    assert "index" not in comp_data
    assert "task_index" not in comp_data


def test_transition_to_batch_without_index_fields():
    """Test that conversion works without index and task_index fields."""

    # Transition without index/task_index
    transition = create_transition(
        observation={OBS_STATE: torch.randn(1, 7)},
        action=torch.randn(1, 4),
        complementary_data={"task": ["navigate"]},
    )

    batch = transition_to_batch(transition)

    # Should have task but not index/task_index
    assert "task" in batch
    assert "index" not in batch
    assert "task_index" not in batch

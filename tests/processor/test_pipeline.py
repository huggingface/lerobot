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

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
import torch.nn as nn

from lerobot.processor import EnvTransition, ProcessorStepRegistry, RobotProcessor


@dataclass
class MockStep:
    """Mock pipeline step for testing - demonstrates best practices.

    This example shows the proper separation:
    - JSON-serializable attributes (name, counter) go in get_config()
    - Only torch tensors go in state_dict()

    Note: The counter is part of the configuration, so it will be restored
    when the step is recreated from config during loading.
    """

    name: str = "mock_step"
    counter: int = 0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Add a counter to the complementary_data."""
        obs, action, reward, done, truncated, info, comp_data = transition

        comp_data = {} if comp_data is None else dict(comp_data)  # Make a copy

        comp_data[f"{self.name}_counter"] = self.counter
        self.counter += 1

        return (obs, action, reward, done, truncated, info, comp_data)

    def get_config(self) -> dict[str, Any]:
        # Return all JSON-serializable attributes that should be persisted
        # These will be passed to __init__ when loading
        return {"name": self.name, "counter": self.counter}

    def state_dict(self) -> dict[str, torch.Tensor]:
        # Only return torch tensors (empty in this case since we have no tensor state)
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        # No tensor state to load
        pass

    def reset(self) -> None:
        self.counter = 0


@dataclass
class MockStepWithoutOptionalMethods:
    """Mock step that only implements the required __call__ method."""

    multiplier: float = 2.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Multiply reward by multiplier."""
        obs, action, reward, done, truncated, info, comp_data = transition

        if reward is not None:
            reward = reward * self.multiplier

        return (obs, action, reward, done, truncated, info, comp_data)


@dataclass
class MockStepWithTensorState:
    """Mock step demonstrating mixed JSON attributes and tensor state."""

    name: str = "tensor_step"
    learning_rate: float = 0.01
    window_size: int = 10

    def __init__(self, name: str = "tensor_step", learning_rate: float = 0.01, window_size: int = 10):
        self.name = name
        self.learning_rate = learning_rate
        self.window_size = window_size
        # Tensor state
        self.running_mean = torch.zeros(window_size)
        self.running_count = torch.tensor(0)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Update running statistics."""
        obs, action, reward, done, truncated, info, comp_data = transition

        if reward is not None:
            # Update running mean
            idx = self.running_count % self.window_size
            self.running_mean[idx] = reward
            self.running_count += 1

        return transition

    def get_config(self) -> dict[str, Any]:
        # Only JSON-serializable attributes
        return {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "window_size": self.window_size,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        # Only tensor state
        return {
            "running_mean": self.running_mean,
            "running_count": self.running_count,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.running_mean = state["running_mean"]
        self.running_count = state["running_count"]

    def reset(self) -> None:
        self.running_mean.zero_()
        self.running_count.zero_()


def test_empty_pipeline():
    """Test pipeline with no steps."""
    pipeline = RobotProcessor()

    transition = (None, None, 0.0, False, False, {}, {})
    result = pipeline(transition)

    assert result == transition
    assert len(pipeline) == 0


def test_single_step_pipeline():
    """Test pipeline with a single step."""
    step = MockStep("test_step")
    pipeline = RobotProcessor([step])

    transition = (None, None, 0.0, False, False, {}, {})
    result = pipeline(transition)

    assert len(pipeline) == 1
    assert result[6]["test_step_counter"] == 0  # complementary_data

    # Call again to test counter increment
    result = pipeline(transition)
    assert result[6]["test_step_counter"] == 1


def test_multiple_steps_pipeline():
    """Test pipeline with multiple steps."""
    step1 = MockStep("step1")
    step2 = MockStep("step2")
    pipeline = RobotProcessor([step1, step2])

    transition = (None, None, 0.0, False, False, {}, {})
    result = pipeline(transition)

    assert len(pipeline) == 2
    assert result[6]["step1_counter"] == 0
    assert result[6]["step2_counter"] == 0


def test_invalid_transition_format():
    """Test pipeline with invalid transition format."""
    pipeline = RobotProcessor([MockStep()])

    # Test with wrong number of elements
    with pytest.raises(ValueError, match="EnvTransition must be a 7-tuple"):
        pipeline((None, None, 0.0))  # Only 3 elements

    # Test with wrong type
    with pytest.raises(ValueError, match="EnvTransition must be a 7-tuple"):
        pipeline("not a tuple")


def test_step_through():
    """Test step_through method with tuple input."""
    step1 = MockStep("step1")
    step2 = MockStep("step2")
    pipeline = RobotProcessor([step1, step2])

    transition = (None, None, 0.0, False, False, {}, {})

    results = list(pipeline.step_through(transition))

    assert len(results) == 3  # Original + 2 steps
    assert results[0] == transition  # Original
    assert "step1_counter" in results[1][6]  # After step1
    assert "step2_counter" in results[2][6]  # After step2

    # Ensure all results are tuples (same format as input)
    for result in results:
        assert isinstance(result, tuple)
        assert len(result) == 7


def test_step_through_with_dict():
    """Test step_through method with dict input."""
    step1 = MockStep("step1")
    step2 = MockStep("step2")
    pipeline = RobotProcessor([step1, step2])

    batch = {
        "observation.image": None,
        "action": None,
        "next.reward": 0.0,
        "next.done": False,
        "next.truncated": False,
        "info": {},
    }

    results = list(pipeline.step_through(batch))

    assert len(results) == 3  # Original + 2 steps

    # Ensure all results are dicts (same format as input)
    for result in results:
        assert isinstance(result, dict)

    # Check that the processing worked - the complementary data from steps
    # should show up in the info or complementary_data fields when converted back to dict
    # Note: This depends on how _default_transition_to_batch handles complementary_data
    # For now, just check that we get dict outputs


def test_indexing():
    """Test pipeline indexing."""
    step1 = MockStep("step1")
    step2 = MockStep("step2")
    pipeline = RobotProcessor([step1, step2])

    # Test integer indexing
    assert pipeline[0] is step1
    assert pipeline[1] is step2

    # Test slice indexing
    sub_pipeline = pipeline[0:1]
    assert isinstance(sub_pipeline, RobotProcessor)
    assert len(sub_pipeline) == 1
    assert sub_pipeline[0] is step1


def test_hooks():
    """Test before/after step hooks."""
    step = MockStep("test_step")
    pipeline = RobotProcessor([step])

    before_calls = []
    after_calls = []

    def before_hook(idx: int, transition: EnvTransition):
        before_calls.append(idx)
        return transition

    def after_hook(idx: int, transition: EnvTransition):
        after_calls.append(idx)
        return transition

    pipeline.register_before_step_hook(before_hook)
    pipeline.register_after_step_hook(after_hook)

    transition = (None, None, 0.0, False, False, {}, {})
    pipeline(transition)

    assert before_calls == [0]
    assert after_calls == [0]


def test_hook_modification():
    """Test that hooks can modify transitions."""
    step = MockStep("test_step")
    pipeline = RobotProcessor([step])

    def modify_reward_hook(idx: int, transition: EnvTransition):
        obs, action, reward, done, truncated, info, comp_data = transition
        return (obs, action, 42.0, done, truncated, info, comp_data)

    pipeline.register_before_step_hook(modify_reward_hook)

    transition = (None, None, 0.0, False, False, {}, {})
    result = pipeline(transition)

    assert result[2] == 42.0  # reward modified by hook


def test_reset():
    """Test pipeline reset functionality."""
    step = MockStep("test_step")
    pipeline = RobotProcessor([step])

    reset_called = []

    def reset_hook():
        reset_called.append(True)

    pipeline.register_reset_hook(reset_hook)

    # Make some calls to increment counter
    transition = (None, None, 0.0, False, False, {}, {})
    pipeline(transition)
    pipeline(transition)

    assert step.counter == 2

    # Reset should reset step and call hook
    pipeline.reset()

    assert step.counter == 0
    assert len(reset_called) == 1


def test_profile_steps():
    """Test step profiling functionality."""
    step1 = MockStep("step1")
    step2 = MockStep("step2")
    pipeline = RobotProcessor([step1, step2])

    transition = (None, None, 0.0, False, False, {}, {})

    profile_results = pipeline.profile_steps(transition, num_runs=10)

    assert len(profile_results) == 2
    assert "step_0_MockStep" in profile_results
    assert "step_1_MockStep" in profile_results
    assert all(isinstance(time, float) and time >= 0 for time in profile_results.values())


def test_save_and_load_pretrained():
    """Test saving and loading pipeline.

    This test demonstrates that JSON-serializable attributes (like counter)
    are saved in the config and restored when the step is recreated.
    """
    step1 = MockStep("step1")
    step2 = MockStep("step2")

    # Increment counters to have some state
    step1.counter = 5
    step2.counter = 10

    pipeline = RobotProcessor([step1, step2], name="TestPipeline", seed=42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save pipeline
        pipeline.save_pretrained(tmp_dir)

        # Check files were created
        config_path = Path(tmp_dir) / "processor.json"
        assert config_path.exists()

        # Check config content
        with open(config_path) as f:
            config = json.load(f)

        assert config["name"] == "TestPipeline"
        assert config["seed"] == 42
        assert len(config["steps"]) == 2

        # Verify counters are saved in config, not in separate state files
        assert config["steps"][0]["config"]["counter"] == 5
        assert config["steps"][1]["config"]["counter"] == 10

        # Load pipeline
        loaded_pipeline = RobotProcessor.from_pretrained(tmp_dir)

        assert loaded_pipeline.name == "TestPipeline"
        assert loaded_pipeline.seed == 42
        assert len(loaded_pipeline) == 2

        # Check that counter was restored from config
        assert loaded_pipeline.steps[0].counter == 5
        assert loaded_pipeline.steps[1].counter == 10


def test_step_without_optional_methods():
    """Test pipeline with steps that don't implement optional methods."""
    step = MockStepWithoutOptionalMethods(multiplier=3.0)
    pipeline = RobotProcessor([step])

    transition = (None, None, 2.0, False, False, {}, {})
    result = pipeline(transition)

    assert result[2] == 6.0  # 2.0 * 3.0

    # Reset should work even if step doesn't implement reset
    pipeline.reset()

    # Save/load should work even without optional methods
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)
        loaded_pipeline = RobotProcessor.from_pretrained(tmp_dir)
        assert len(loaded_pipeline) == 1


def test_mixed_json_and_tensor_state():
    """Test step with both JSON attributes and tensor state."""
    step = MockStepWithTensorState(name="stats", learning_rate=0.05, window_size=5)
    pipeline = RobotProcessor([step])

    # Process some transitions with rewards
    for i in range(10):
        transition = (None, None, float(i), False, False, {}, {})
        pipeline(transition)

    # Check state
    assert step.running_count.item() == 10
    assert step.learning_rate == 0.05

    # Save and load
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Check that both config and state files were created
        config_path = Path(tmp_dir) / "processor.json"
        state_path = Path(tmp_dir) / "step_0.safetensors"
        assert config_path.exists()
        assert state_path.exists()

        # Load and verify
        loaded_pipeline = RobotProcessor.from_pretrained(tmp_dir)
        loaded_step = loaded_pipeline.steps[0]

        # Check JSON attributes were restored
        assert loaded_step.name == "stats"
        assert loaded_step.learning_rate == 0.05
        assert loaded_step.window_size == 5

        # Check tensor state was restored
        assert loaded_step.running_count.item() == 10
        assert torch.allclose(loaded_step.running_mean, step.running_mean)


class MockModuleStep(nn.Module):
    """Mock step that inherits from nn.Module to test state_dict handling of module parameters."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.running_mean = nn.Parameter(torch.zeros(hidden_dim), requires_grad=False)
        self.counter = 0  # Non-tensor state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Process transition and update running mean."""
        obs, action, reward, done, truncated, info, comp_data = transition

        if obs is not None and isinstance(obs, torch.Tensor):
            # Process observation through linear layer
            processed = self.forward(obs[:, : self.input_dim])

            # Update running mean in-place (don't reassign the parameter)
            with torch.no_grad():
                self.running_mean.mul_(0.9).add_(processed.mean(dim=0), alpha=0.1)

            self.counter += 1

        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "counter": self.counter,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Override to return all module parameters and buffers."""
        # Get the module's state dict (includes all parameters and buffers)
        return super().state_dict()

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Override to load all module parameters and buffers."""
        # Use the module's load_state_dict
        super().load_state_dict(state)

    def reset(self) -> None:
        self.running_mean.zero_()
        self.counter = 0


def test_to_device_with_state_dict():
    """Test moving pipeline to device for steps with state_dict."""
    step = MockStepWithTensorState(name="device_test", window_size=5)
    pipeline = RobotProcessor([step])

    # Process some transitions to populate state
    for i in range(10):
        transition = (None, None, float(i), False, False, {}, {})
        pipeline(transition)

    # Check initial device (should be CPU)
    assert step.running_mean.device.type == "cpu"
    assert step.running_count.device.type == "cpu"

    # Move to same device (CPU)
    result = pipeline.to("cpu")
    assert result is pipeline  # Check it returns self
    assert step.running_mean.device.type == "cpu"
    assert step.running_count.device.type == "cpu"

    # Test with torch.device object
    result = pipeline.to(torch.device("cpu"))
    assert result is pipeline
    assert step.running_mean.device.type == "cpu"

    # If CUDA is available, test GPU transfer
    if torch.cuda.is_available():
        result = pipeline.to("cuda")
        assert result is pipeline
        assert step.running_mean.device.type == "cuda"
        assert step.running_count.device.type == "cuda"

        # Move back to CPU
        pipeline.to("cpu")
        assert step.running_mean.device.type == "cpu"
        assert step.running_count.device.type == "cpu"


def test_to_device_with_module():
    """Test moving pipeline to device for steps that inherit from nn.Module.

    Even though the step inherits from nn.Module, the pipeline will use the
    state_dict/load_state_dict approach to move tensors to the device.
    """
    module_step = MockModuleStep(input_dim=5, hidden_dim=3)
    pipeline = RobotProcessor([module_step])

    # Process some data
    obs = torch.randn(2, 5)
    transition = (obs, None, 1.0, False, False, {}, {})
    pipeline(transition)

    # Check initial device
    assert module_step.linear.weight.device.type == "cpu"
    assert module_step.running_mean.device.type == "cpu"

    # Move to same device
    result = pipeline.to("cpu")
    assert result is pipeline
    assert module_step.linear.weight.device.type == "cpu"
    assert module_step.running_mean.device.type == "cpu"

    # If CUDA is available, test GPU transfer
    if torch.cuda.is_available():
        result = pipeline.to("cuda:0")
        assert result is pipeline
        assert module_step.linear.weight.device.type == "cuda"
        assert module_step.linear.weight.device.index == 0
        assert module_step.running_mean.device.type == "cuda"
        assert module_step.running_mean.device.index == 0

        # Verify the module still works after transfer
        obs_cuda = torch.randn(2, 5, device="cuda:0")
        transition = (obs_cuda, None, 1.0, False, False, {}, {})
        pipeline(transition)  # Should not raise an error


def test_to_device_mixed_steps():
    """Test moving pipeline with various types of steps, all using state_dict approach."""
    module_step = MockModuleStep()
    state_dict_step = MockStepWithTensorState()
    simple_step = MockStepWithoutOptionalMethods()  # No tensor state

    pipeline = RobotProcessor([module_step, state_dict_step, simple_step])

    # Process some data
    for i in range(5):
        transition = (torch.randn(2, 10), None, float(i), False, False, {}, {})
        pipeline(transition)

    # Check initial state
    assert module_step.linear.weight.device.type == "cpu"
    assert state_dict_step.running_mean.device.type == "cpu"

    # Move to device
    result = pipeline.to("cpu")
    assert result is pipeline

    if torch.cuda.is_available():
        pipeline.to("cuda")
        assert module_step.linear.weight.device.type == "cuda"
        assert module_step.running_mean.device.type == "cuda"
        assert state_dict_step.running_mean.device.type == "cuda"
        assert state_dict_step.running_count.device.type == "cuda"


def test_to_device_empty_state():
    """Test moving pipeline with steps that have empty state_dict."""
    step = MockStep("empty_state")  # This step has empty state_dict
    pipeline = RobotProcessor([step])

    # Should not raise an error even with empty state
    result = pipeline.to("cpu")
    assert result is pipeline

    if torch.cuda.is_available():
        result = pipeline.to("cuda")
        assert result is pipeline


def test_to_device_preserves_functionality():
    """Test that pipeline functionality is preserved after device transfer."""
    step = MockStepWithTensorState(window_size=3)
    pipeline = RobotProcessor([step])

    # Process initial data
    rewards = [1.0, 2.0, 3.0]
    for r in rewards:
        transition = (None, None, r, False, False, {}, {})
        pipeline(transition)

    # Check state before transfer
    initial_mean = step.running_mean.clone()
    initial_count = step.running_count.clone()

    # Move to device (CPU to CPU in this case, but tests the mechanism)
    pipeline.to("cpu")

    # Verify state is preserved
    assert torch.allclose(step.running_mean, initial_mean)
    assert step.running_count == initial_count

    # Process more data to ensure functionality
    transition = (None, None, 4.0, False, False, {}, {})
    _ = pipeline(transition)

    assert step.running_count == 4
    assert step.running_mean[0] == 4.0  # First slot should have been overwritten with 4.0


def test_to_device_invalid_device():
    """Test error handling for invalid devices."""
    pipeline = RobotProcessor([MockStep()])

    # Invalid device names should raise an error from PyTorch
    with pytest.raises(RuntimeError):
        pipeline.to("invalid_device")


def test_to_device_chaining():
    """Test that to() returns self for method chaining."""
    step1 = MockStepWithTensorState()
    step2 = MockModuleStep()
    pipeline = RobotProcessor([step1, step2])

    # Test chaining
    result = pipeline.to("cpu").reset()
    assert result is None  # reset() returns None

    # Can chain multiple to() calls
    result1 = pipeline.to("cpu")
    result2 = result1.to("cpu")
    assert result1 is pipeline
    assert result2 is pipeline


class MockNonModuleStepWithState:
    """Mock step that explicitly does NOT inherit from nn.Module but has tensor state.

    This tests the state_dict/load_state_dict path for regular classes.
    """

    def __init__(self, name: str = "non_module_step", feature_dim: int = 10):
        self.name = name
        self.feature_dim = feature_dim

        # Initialize tensor state - these are regular tensors, not nn.Parameters
        self.weights = torch.randn(feature_dim, feature_dim)
        self.bias = torch.zeros(feature_dim)
        self.running_stats = torch.zeros(feature_dim)
        self.step_count = torch.tensor(0)

        # Non-tensor state
        self.config_value = 42
        self.history = []

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Process transition using tensor operations."""
        obs, action, reward, done, truncated, info, comp_data = transition

        if obs is not None and isinstance(obs, torch.Tensor) and obs.numel() >= self.feature_dim:
            # Perform some tensor operations
            flat_obs = obs.flatten()[: self.feature_dim]

            # Simple linear transformation (ensure dimensions match for matmul)
            output = torch.matmul(self.weights.T, flat_obs) + self.bias

            # Update running stats
            self.running_stats = 0.9 * self.running_stats + 0.1 * output
            self.step_count += 1

            # Add to complementary data
            comp_data = {} if comp_data is None else dict(comp_data)
            comp_data[f"{self.name}_mean_output"] = output.mean().item()
            comp_data[f"{self.name}_steps"] = self.step_count.item()

        return (obs, action, reward, done, truncated, info, comp_data)

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_dim": self.feature_dim,
            "config_value": self.config_value,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return only tensor state."""
        return {
            "weights": self.weights,
            "bias": self.bias,
            "running_stats": self.running_stats,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load tensor state."""
        self.weights = state["weights"]
        self.bias = state["bias"]
        self.running_stats = state["running_stats"]
        self.step_count = state["step_count"]

    def reset(self) -> None:
        """Reset statistics but keep learned parameters."""
        self.running_stats.zero_()
        self.step_count.zero_()
        self.history.clear()


def test_to_device_non_module_class():
    """Test moving pipeline to device for regular classes (non nn.Module) with tensor state.

    This ensures the state_dict/load_state_dict approach works for classes that
    don't inherit from nn.Module but still have tensor state to manage.
    """
    # Create a non-module step with tensor state
    non_module_step = MockNonModuleStepWithState(name="device_test", feature_dim=5)
    pipeline = RobotProcessor([non_module_step])

    # Process some data to populate state
    for i in range(3):
        obs = torch.randn(2, 5)
        transition = (obs, None, float(i), False, False, {}, {})
        result = pipeline(transition)
        comp_data = result[6]
        assert f"{non_module_step.name}_steps" in comp_data

    # Verify all tensors are on CPU initially
    assert non_module_step.weights.device.type == "cpu"
    assert non_module_step.bias.device.type == "cpu"
    assert non_module_step.running_stats.device.type == "cpu"
    assert non_module_step.step_count.device.type == "cpu"

    # Verify step count
    assert non_module_step.step_count.item() == 3

    # Store initial values for comparison
    initial_weights = non_module_step.weights.clone()
    initial_bias = non_module_step.bias.clone()
    initial_stats = non_module_step.running_stats.clone()

    # Move to same device (CPU)
    result = pipeline.to("cpu")
    assert result is pipeline

    # Verify tensors are still on CPU and values unchanged
    assert non_module_step.weights.device.type == "cpu"
    assert torch.allclose(non_module_step.weights, initial_weights)
    assert torch.allclose(non_module_step.bias, initial_bias)
    assert torch.allclose(non_module_step.running_stats, initial_stats)

    # If CUDA is available, test GPU transfer
    if torch.cuda.is_available():
        # Move to GPU
        pipeline.to("cuda")

        # Verify all tensors moved to GPU
        assert non_module_step.weights.device.type == "cuda"
        assert non_module_step.bias.device.type == "cuda"
        assert non_module_step.running_stats.device.type == "cuda"
        assert non_module_step.step_count.device.type == "cuda"

        # Verify values are preserved
        assert torch.allclose(non_module_step.weights.cpu(), initial_weights)
        assert torch.allclose(non_module_step.bias.cpu(), initial_bias)
        assert torch.allclose(non_module_step.running_stats.cpu(), initial_stats)
        assert non_module_step.step_count.item() == 3

        # Test that step still works on GPU
        obs_gpu = torch.randn(2, 5, device="cuda")
        transition = (obs_gpu, None, 1.0, False, False, {}, {})
        result = pipeline(transition)
        comp_data = result[6]

        # Verify processing worked
        assert comp_data[f"{non_module_step.name}_steps"] == 4

        # Move back to CPU
        pipeline.to("cpu")
        assert non_module_step.weights.device.type == "cpu"
        assert non_module_step.step_count.item() == 4


def test_to_device_module_vs_non_module():
    """Test that both nn.Module and non-Module steps work with the same state_dict approach."""
    # Create both types of steps
    module_step = MockModuleStep(input_dim=5, hidden_dim=3)
    non_module_step = MockNonModuleStepWithState(name="non_module", feature_dim=5)

    # Create pipeline with both
    pipeline = RobotProcessor([module_step, non_module_step])

    # Process some data
    obs = torch.randn(2, 5)
    transition = (obs, None, 1.0, False, False, {}, {})
    _ = pipeline(transition)

    # Check initial devices
    assert module_step.linear.weight.device.type == "cpu"
    assert module_step.running_mean.device.type == "cpu"
    assert non_module_step.weights.device.type == "cpu"
    assert non_module_step.running_stats.device.type == "cpu"

    # Both should have been called
    assert module_step.counter == 1
    assert non_module_step.step_count.item() == 1

    if torch.cuda.is_available():
        # Move to GPU
        pipeline.to("cuda")

        # Verify both types of steps moved correctly
        assert module_step.linear.weight.device.type == "cuda"
        assert module_step.running_mean.device.type == "cuda"
        assert non_module_step.weights.device.type == "cuda"
        assert non_module_step.running_stats.device.type == "cuda"

        # Process data on GPU
        obs_gpu = torch.randn(2, 5, device="cuda")
        transition = (obs_gpu, None, 2.0, False, False, {}, {})
        _ = pipeline(transition)

        # Verify both steps processed the data
        assert module_step.counter == 2
        assert non_module_step.step_count.item() == 2

        # Move back to CPU and verify
        pipeline.to("cpu")
        assert module_step.linear.weight.device.type == "cpu"
        assert non_module_step.weights.device.type == "cpu"


# Tests for overrides functionality
@dataclass
class MockStepWithNonSerializableParam:
    """Mock step that requires a non-serializable parameter."""

    def __init__(self, name: str = "mock_env_step", multiplier: float = 1.0, env: Any = None):
        self.name = name
        # Add type validation for multiplier
        if isinstance(multiplier, str):
            raise ValueError(f"multiplier must be a number, got string '{multiplier}'")
        if not isinstance(multiplier, (int, float)):
            raise TypeError(f"multiplier must be a number, got {type(multiplier).__name__}")
        self.multiplier = float(multiplier)
        self.env = env  # Non-serializable parameter (like gym.Env)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs, action, reward, done, truncated, info, comp_data = transition

        # Use the env parameter if provided
        if self.env is not None:
            comp_data = {} if comp_data is None else dict(comp_data)
            comp_data[f"{self.name}_env_info"] = str(self.env)

        # Apply multiplier to reward
        if reward is not None:
            reward = reward * self.multiplier

        return (obs, action, reward, done, truncated, info, comp_data)

    def get_config(self) -> Dict[str, Any]:
        # Note: env is intentionally NOT included here as it's not serializable
        return {
            "name": self.name,
            "multiplier": self.multiplier,
        }

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass


@ProcessorStepRegistry.register("registered_mock_step")
@dataclass
class RegisteredMockStep:
    """Mock step registered in the registry."""

    value: int = 42
    device: str = "cpu"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs, action, reward, done, truncated, info, comp_data = transition

        comp_data = {} if comp_data is None else dict(comp_data)
        comp_data["registered_step_value"] = self.value
        comp_data["registered_step_device"] = self.device

        return (obs, action, reward, done, truncated, info, comp_data)

    def get_config(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "device": self.device,
        }

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        pass

    def reset(self) -> None:
        pass


class MockEnvironment:
    """Mock environment for testing non-serializable parameters."""

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"MockEnvironment({self.name})"


def test_from_pretrained_with_overrides():
    """Test loading processor with parameter overrides."""
    # Create a processor with steps that need overrides
    env_step = MockStepWithNonSerializableParam(name="env_step", multiplier=2.0)
    registered_step = RegisteredMockStep(value=100, device="cpu")

    pipeline = RobotProcessor([env_step, registered_step], name="TestOverrides")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save the pipeline
        pipeline.save_pretrained(tmp_dir)

        # Create a mock environment for override
        mock_env = MockEnvironment("test_env")

        # Load with overrides
        overrides = {
            "MockStepWithNonSerializableParam": {
                "env": mock_env,
                "multiplier": 3.0,  # Override the multiplier too
            },
            "registered_mock_step": {"device": "cuda", "value": 200},
        }

        loaded_pipeline = RobotProcessor.from_pretrained(tmp_dir, overrides=overrides)

        # Verify the pipeline was loaded correctly
        assert len(loaded_pipeline) == 2
        assert loaded_pipeline.name == "TestOverrides"

        # Test the loaded steps
        transition = (None, None, 1.0, False, False, {}, {})
        result = loaded_pipeline(transition)

        # Check that overrides were applied
        comp_data = result[6]
        assert "env_step_env_info" in comp_data
        assert comp_data["env_step_env_info"] == "MockEnvironment(test_env)"
        assert comp_data["registered_step_value"] == 200
        assert comp_data["registered_step_device"] == "cuda"

        # Check that multiplier override was applied
        assert result[2] == 3.0  # 1.0 * 3.0 (overridden multiplier)


def test_from_pretrained_with_partial_overrides():
    """Test loading processor with overrides for only some steps."""
    step1 = MockStepWithNonSerializableParam(name="step1", multiplier=1.0)
    step2 = MockStepWithNonSerializableParam(name="step2", multiplier=2.0)

    pipeline = RobotProcessor([step1, step2])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Override only one step
        overrides = {"MockStepWithNonSerializableParam": {"multiplier": 5.0}}

        # The current implementation applies overrides to ALL steps with the same class name
        # Both steps will get the override
        loaded_pipeline = RobotProcessor.from_pretrained(tmp_dir, overrides=overrides)

        transition = (None, None, 1.0, False, False, {}, {})
        result = loaded_pipeline(transition)

        # The reward should be affected by both steps, both getting the override
        # First step: 1.0 * 5.0 = 5.0 (overridden)
        # Second step: 5.0 * 5.0 = 25.0 (also overridden)
        assert result[2] == 25.0


def test_from_pretrained_invalid_override_key():
    """Test that invalid override keys raise KeyError."""
    step = MockStepWithNonSerializableParam()
    pipeline = RobotProcessor([step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Try to override a non-existent step
        overrides = {"NonExistentStep": {"param": "value"}}

        with pytest.raises(KeyError, match="Override keys.*do not match any step"):
            RobotProcessor.from_pretrained(tmp_dir, overrides=overrides)


def test_from_pretrained_multiple_invalid_override_keys():
    """Test that multiple invalid override keys are reported."""
    step = MockStepWithNonSerializableParam()
    pipeline = RobotProcessor([step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Try to override multiple non-existent steps
        overrides = {"NonExistentStep1": {"param": "value1"}, "NonExistentStep2": {"param": "value2"}}

        with pytest.raises(KeyError) as exc_info:
            RobotProcessor.from_pretrained(tmp_dir, overrides=overrides)

        error_msg = str(exc_info.value)
        assert "NonExistentStep1" in error_msg
        assert "NonExistentStep2" in error_msg
        assert "Available step keys" in error_msg


def test_from_pretrained_registered_step_override():
    """Test overriding registered steps using registry names."""
    registered_step = RegisteredMockStep(value=50, device="cpu")
    pipeline = RobotProcessor([registered_step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Override using registry name
        overrides = {"registered_mock_step": {"value": 999, "device": "cuda"}}

        loaded_pipeline = RobotProcessor.from_pretrained(tmp_dir, overrides=overrides)

        # Test that overrides were applied
        transition = (None, None, 0.0, False, False, {}, {})
        result = loaded_pipeline(transition)

        comp_data = result[6]
        assert comp_data["registered_step_value"] == 999
        assert comp_data["registered_step_device"] == "cuda"


def test_from_pretrained_mixed_registered_and_unregistered():
    """Test overriding both registered and unregistered steps."""
    unregistered_step = MockStepWithNonSerializableParam(name="unregistered", multiplier=1.0)
    registered_step = RegisteredMockStep(value=10, device="cpu")

    pipeline = RobotProcessor([unregistered_step, registered_step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        mock_env = MockEnvironment("mixed_test")

        overrides = {
            "MockStepWithNonSerializableParam": {"env": mock_env, "multiplier": 4.0},
            "registered_mock_step": {"value": 777},
        }

        loaded_pipeline = RobotProcessor.from_pretrained(tmp_dir, overrides=overrides)

        # Test both steps
        transition = (None, None, 2.0, False, False, {}, {})
        result = loaded_pipeline(transition)

        comp_data = result[6]
        assert comp_data["unregistered_env_info"] == "MockEnvironment(mixed_test)"
        assert comp_data["registered_step_value"] == 777
        assert result[2] == 8.0  # 2.0 * 4.0


def test_from_pretrained_no_overrides():
    """Test that from_pretrained works without overrides (backward compatibility)."""
    step = MockStepWithNonSerializableParam(name="no_override", multiplier=3.0)
    pipeline = RobotProcessor([step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Load without overrides
        loaded_pipeline = RobotProcessor.from_pretrained(tmp_dir)

        assert len(loaded_pipeline) == 1

        # Test that the step works (env will be None)
        transition = (None, None, 1.0, False, False, {}, {})
        result = loaded_pipeline(transition)

        assert result[2] == 3.0  # 1.0 * 3.0


def test_from_pretrained_empty_overrides():
    """Test that from_pretrained works with empty overrides dict."""
    step = MockStepWithNonSerializableParam(multiplier=2.0)
    pipeline = RobotProcessor([step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Load with empty overrides
        loaded_pipeline = RobotProcessor.from_pretrained(tmp_dir, overrides={})

        assert len(loaded_pipeline) == 1

        # Test that the step works normally
        transition = (None, None, 1.0, False, False, {}, {})
        result = loaded_pipeline(transition)

        assert result[2] == 2.0


def test_from_pretrained_override_instantiation_error():
    """Test that instantiation errors with overrides are properly reported."""
    step = MockStepWithNonSerializableParam(multiplier=1.0)
    pipeline = RobotProcessor([step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Try to override with invalid parameter type
        overrides = {
            "MockStepWithNonSerializableParam": {
                "multiplier": "invalid_type"  # Should be float, not string
            }
        }

        with pytest.raises(ValueError, match="Failed to instantiate processor step"):
            RobotProcessor.from_pretrained(tmp_dir, overrides=overrides)


def test_from_pretrained_with_state_and_overrides():
    """Test that overrides work correctly with steps that have tensor state."""
    step = MockStepWithTensorState(name="tensor_step", learning_rate=0.01, window_size=5)
    pipeline = RobotProcessor([step])

    # Process some data to create state
    for i in range(10):
        transition = (None, None, float(i), False, False, {}, {})
        pipeline(transition)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Load with overrides
        overrides = {
            "MockStepWithTensorState": {
                "learning_rate": 0.05,  # Override learning rate
                "window_size": 3,  # Override window size
            }
        }

        loaded_pipeline = RobotProcessor.from_pretrained(tmp_dir, overrides=overrides)
        loaded_step = loaded_pipeline.steps[0]

        # Check that config overrides were applied
        assert loaded_step.learning_rate == 0.05
        assert loaded_step.window_size == 3

        # Check that tensor state was preserved
        assert loaded_step.running_count.item() == 10

        # The running_mean should still have the original window_size (5) from saved state
        # but the new step will use window_size=3 for future operations
        assert loaded_step.running_mean.shape[0] == 5  # From saved state


def test_from_pretrained_override_error_messages():
    """Test that error messages for override failures are helpful."""
    step1 = MockStepWithNonSerializableParam(name="step1")
    step2 = RegisteredMockStep()
    pipeline = RobotProcessor([step1, step2])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Test with invalid override key
        overrides = {"WrongStepName": {"param": "value"}}

        with pytest.raises(KeyError) as exc_info:
            RobotProcessor.from_pretrained(tmp_dir, overrides=overrides)

        error_msg = str(exc_info.value)
        assert "WrongStepName" in error_msg
        assert "Available step keys" in error_msg
        assert "MockStepWithNonSerializableParam" in error_msg
        assert "registered_mock_step" in error_msg


class MockStepWithMixedState:
    """Mock step demonstrating proper separation of tensor and non-tensor state.

    Non-tensor state should go in get_config(), only tensors in state_dict().
    """

    def __init__(self, name: str = "mixed_state"):
        self.name = name
        self.tensor_data = torch.randn(5)
        self.numpy_data = np.array([1, 2, 3, 4, 5])  # Goes in config
        self.scalar_value = 42  # Goes in config
        self.list_value = [1, 2, 3]  # Goes in config

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Simple pass-through
        return transition

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return ONLY tensor state as per the type contract."""
        return {
            "tensor_data": self.tensor_data,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load tensor state only."""
        self.tensor_data = state["tensor_data"]

    def get_config(self) -> dict[str, Any]:
        """Non-tensor state goes here."""
        return {
            "name": self.name,
            "numpy_data": self.numpy_data.tolist(),  # Convert to list for JSON serialization
            "scalar_value": self.scalar_value,
            "list_value": self.list_value,
        }


def test_to_device_with_mixed_state_types():
    """Test that to() only moves tensor state, while non-tensor state remains in config."""
    step = MockStepWithMixedState()
    pipeline = RobotProcessor([step])

    # Store initial values
    initial_numpy = step.numpy_data.copy()
    initial_scalar = step.scalar_value
    initial_list = step.list_value.copy()

    # Check initial state
    assert step.tensor_data.device.type == "cpu"
    assert isinstance(step.numpy_data, np.ndarray)
    assert isinstance(step.scalar_value, int)
    assert isinstance(step.list_value, list)

    # Verify state_dict only contains tensors
    state = step.state_dict()
    assert all(isinstance(v, torch.Tensor) for v in state.values())
    assert "tensor_data" in state
    assert "numpy_data" not in state

    # Move to same device
    pipeline.to("cpu")

    # Verify tensor moved and non-tensor attributes unchanged
    assert step.tensor_data.device.type == "cpu"
    assert np.array_equal(step.numpy_data, initial_numpy)
    assert step.scalar_value == initial_scalar
    assert step.list_value == initial_list

    if torch.cuda.is_available():
        # Move to GPU
        pipeline.to("cuda")

        # Only tensor should move to GPU
        assert step.tensor_data.device.type == "cuda"

        # Non-tensor values should remain unchanged
        assert isinstance(step.numpy_data, np.ndarray)
        assert np.array_equal(step.numpy_data, initial_numpy)
        assert step.scalar_value == initial_scalar
        assert step.list_value == initial_list

        # Move back to CPU
        pipeline.to("cpu")
        assert step.tensor_data.device.type == "cpu"

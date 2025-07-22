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
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from lerobot.processor import EnvTransition, ProcessorStepRegistry, RobotProcessor
from lerobot.processor.pipeline import TransitionKey


def create_transition(
    observation=None, action=None, reward=0.0, done=False, truncated=False, info=None, complementary_data=None
):
    """Helper to create an EnvTransition dictionary."""
    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: reward,
        TransitionKey.DONE: done,
        TransitionKey.TRUNCATED: truncated,
        TransitionKey.INFO: info if info is not None else {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data if complementary_data is not None else {},
    }


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
        comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        comp_data = {} if comp_data is None else dict(comp_data)  # Make a copy

        comp_data[f"{self.name}_counter"] = self.counter
        self.counter += 1

        # Create a new transition with updated complementary_data
        new_transition = transition.copy()
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp_data
        return new_transition

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
        reward = transition.get(TransitionKey.REWARD)

        if reward is not None:
            new_transition = transition.copy()
            new_transition[TransitionKey.REWARD] = reward * self.multiplier
            return new_transition

        return transition


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
        reward = transition.get(TransitionKey.REWARD)

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

    transition = create_transition()
    result = pipeline(transition)

    assert result == transition
    assert len(pipeline) == 0


def test_single_step_pipeline():
    """Test pipeline with a single step."""
    step = MockStep("test_step")
    pipeline = RobotProcessor([step])

    transition = create_transition()
    result = pipeline(transition)

    assert len(pipeline) == 1
    assert result[TransitionKey.COMPLEMENTARY_DATA]["test_step_counter"] == 0

    # Call again to test counter increment
    result = pipeline(transition)
    assert result[TransitionKey.COMPLEMENTARY_DATA]["test_step_counter"] == 1


def test_multiple_steps_pipeline():
    """Test pipeline with multiple steps."""
    step1 = MockStep("step1")
    step2 = MockStep("step2")
    pipeline = RobotProcessor([step1, step2])

    transition = create_transition()
    result = pipeline(transition)

    assert len(pipeline) == 2
    assert result[TransitionKey.COMPLEMENTARY_DATA]["step1_counter"] == 0
    assert result[TransitionKey.COMPLEMENTARY_DATA]["step2_counter"] == 0


def test_invalid_transition_format():
    """Test pipeline with invalid transition format."""
    pipeline = RobotProcessor([MockStep()])

    # Test with wrong type (tuple instead of dict)
    with pytest.raises(ValueError, match="EnvTransition must be a dictionary"):
        pipeline((None, None, 0.0, False, False, {}, {}))  # Tuple instead of dict

    # Test with wrong type (string)
    with pytest.raises(ValueError, match="EnvTransition must be a dictionary"):
        pipeline("not a dict")


def test_step_through():
    """Test step_through method with dict input."""
    step1 = MockStep("step1")
    step2 = MockStep("step2")
    pipeline = RobotProcessor([step1, step2])

    transition = create_transition()

    results = list(pipeline.step_through(transition))

    assert len(results) == 3  # Original + 2 steps
    assert results[0] == transition  # Original
    assert "step1_counter" in results[1][TransitionKey.COMPLEMENTARY_DATA]  # After step1
    assert "step2_counter" in results[2][TransitionKey.COMPLEMENTARY_DATA]  # After step2

    # Ensure all results are dicts (same format as input)
    for result in results:
        assert isinstance(result, dict)
        assert all(isinstance(k, TransitionKey) for k in result.keys())


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


def test_step_through_no_hooks():
    """Test that step_through doesn't execute hooks."""
    step = MockStep("test_step")
    pipeline = RobotProcessor([step])

    hook_calls = []

    def tracking_hook(idx: int, transition: EnvTransition):
        hook_calls.append(f"hook_called_step_{idx}")

    # Register hooks
    pipeline.register_before_step_hook(tracking_hook)
    pipeline.register_after_step_hook(tracking_hook)

    # Use step_through
    transition = create_transition()
    results = list(pipeline.step_through(transition))

    # Verify step was executed (counter should increment)
    assert len(results) == 2  # Initial + 1 step
    assert results[1][TransitionKey.COMPLEMENTARY_DATA]["test_step_counter"] == 0

    # Verify hooks were NOT called
    assert len(hook_calls) == 0

    # Now use __call__ to verify hooks ARE called there
    hook_calls.clear()
    pipeline(transition)

    # Verify hooks were called (before and after for 1 step = 2 calls)
    assert len(hook_calls) == 2
    assert hook_calls == ["hook_called_step_0", "hook_called_step_0"]


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

    def after_hook(idx: int, transition: EnvTransition):
        after_calls.append(idx)

    pipeline.register_before_step_hook(before_hook)
    pipeline.register_after_step_hook(after_hook)

    transition = create_transition()
    pipeline(transition)

    assert before_calls == [0]
    assert after_calls == [0]


def test_reset():
    """Test pipeline reset functionality."""
    step = MockStep("test_step")
    pipeline = RobotProcessor([step])

    reset_called = []

    def reset_hook():
        reset_called.append(True)

    pipeline.register_reset_hook(reset_hook)

    # Make some calls to increment counter
    transition = create_transition()
    pipeline(transition)
    pipeline(transition)

    assert step.counter == 2

    # Reset should reset step and call hook
    pipeline.reset()

    assert step.counter == 0
    assert len(reset_called) == 1


def test_unregister_hooks():
    """Test unregistering hooks from the pipeline."""
    step = MockStep("test_step")
    pipeline = RobotProcessor([step])

    # Test before_step_hook
    before_calls = []

    def before_hook(idx: int, transition: EnvTransition):
        before_calls.append(idx)

    pipeline.register_before_step_hook(before_hook)

    # Verify hook is registered
    transition = create_transition()
    pipeline(transition)
    assert len(before_calls) == 1

    # Unregister and verify it's no longer called
    pipeline.unregister_before_step_hook(before_hook)
    before_calls.clear()
    pipeline(transition)
    assert len(before_calls) == 0

    # Test after_step_hook
    after_calls = []

    def after_hook(idx: int, transition: EnvTransition):
        after_calls.append(idx)

    pipeline.register_after_step_hook(after_hook)
    pipeline(transition)
    assert len(after_calls) == 1

    pipeline.unregister_after_step_hook(after_hook)
    after_calls.clear()
    pipeline(transition)
    assert len(after_calls) == 0

    # Test reset_hook
    reset_calls = []

    def reset_hook():
        reset_calls.append(True)

    pipeline.register_reset_hook(reset_hook)
    pipeline.reset()
    assert len(reset_calls) == 1

    pipeline.unregister_reset_hook(reset_hook)
    reset_calls.clear()
    pipeline.reset()
    assert len(reset_calls) == 0


def test_unregister_nonexistent_hook():
    """Test error handling when unregistering hooks that don't exist."""
    pipeline = RobotProcessor([MockStep()])

    def some_hook(idx: int, transition: EnvTransition):
        pass

    def reset_hook():
        pass

    # Test unregistering hooks that were never registered
    with pytest.raises(ValueError, match="not found in before_step_hooks"):
        pipeline.unregister_before_step_hook(some_hook)

    with pytest.raises(ValueError, match="not found in after_step_hooks"):
        pipeline.unregister_after_step_hook(some_hook)

    with pytest.raises(ValueError, match="not found in reset_hooks"):
        pipeline.unregister_reset_hook(reset_hook)


def test_multiple_hooks_and_selective_unregister():
    """Test registering multiple hooks and selectively unregistering them."""
    pipeline = RobotProcessor([MockStep("step1"), MockStep("step2")])

    calls_1 = []
    calls_2 = []
    calls_3 = []

    def hook1(idx: int, transition: EnvTransition):
        calls_1.append(f"hook1_step{idx}")

    def hook2(idx: int, transition: EnvTransition):
        calls_2.append(f"hook2_step{idx}")

    def hook3(idx: int, transition: EnvTransition):
        calls_3.append(f"hook3_step{idx}")

    # Register multiple hooks
    pipeline.register_before_step_hook(hook1)
    pipeline.register_before_step_hook(hook2)
    pipeline.register_before_step_hook(hook3)

    # Run pipeline - all hooks should be called for both steps
    transition = create_transition()
    pipeline(transition)

    assert calls_1 == ["hook1_step0", "hook1_step1"]
    assert calls_2 == ["hook2_step0", "hook2_step1"]
    assert calls_3 == ["hook3_step0", "hook3_step1"]

    # Clear calls
    calls_1.clear()
    calls_2.clear()
    calls_3.clear()

    # Unregister middle hook
    pipeline.unregister_before_step_hook(hook2)

    # Run again - only hook1 and hook3 should be called
    pipeline(transition)

    assert calls_1 == ["hook1_step0", "hook1_step1"]
    assert calls_2 == []  # hook2 was unregistered
    assert calls_3 == ["hook3_step0", "hook3_step1"]


def test_hook_execution_order_documentation():
    """Test and document that hooks are executed sequentially in registration order."""
    pipeline = RobotProcessor([MockStep("step")])

    execution_order = []

    def hook_a(idx: int, transition: EnvTransition):
        execution_order.append("A")

    def hook_b(idx: int, transition: EnvTransition):
        execution_order.append("B")

    def hook_c(idx: int, transition: EnvTransition):
        execution_order.append("C")

    # Register in specific order: A, B, C
    pipeline.register_before_step_hook(hook_a)
    pipeline.register_before_step_hook(hook_b)
    pipeline.register_before_step_hook(hook_c)

    transition = create_transition()
    pipeline(transition)

    # Verify execution order matches registration order
    assert execution_order == ["A", "B", "C"]

    # Test that after unregistering B and re-registering it, it goes to the end
    pipeline.unregister_before_step_hook(hook_b)
    execution_order.clear()

    pipeline(transition)
    assert execution_order == ["A", "C"]  # B is gone

    # Re-register B - it should now be at the end
    pipeline.register_before_step_hook(hook_b)
    execution_order.clear()

    pipeline(transition)
    assert execution_order == ["A", "C", "B"]  # B is now last


def test_profile_steps():
    """Test step profiling functionality."""
    step1 = MockStep("step1")
    step2 = MockStep("step2")
    pipeline = RobotProcessor([step1, step2])

    transition = create_transition()

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
        config_path = Path(tmp_dir) / "testpipeline.json"  # Based on name="TestPipeline"
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

    transition = create_transition(reward=2.0)
    result = pipeline(transition)

    assert result[TransitionKey.REWARD] == 6.0  # 2.0 * 3.0

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
        transition = create_transition(reward=float(i))
        pipeline(transition)

    # Check state
    assert step.running_count.item() == 10
    assert step.learning_rate == 0.05

    # Save and load
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Check that both config and state files were created
        config_path = Path(tmp_dir) / "robotprocessor.json"  # Default name is "RobotProcessor"
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
        obs = transition.get(TransitionKey.OBSERVATION)

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
        transition = create_transition(reward=float(i))
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
    transition = create_transition(observation=obs, reward=1.0)
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
        transition = create_transition(observation=obs_cuda, reward=1.0)
        pipeline(transition)  # Should not raise an error


def test_to_device_mixed_steps():
    """Test moving pipeline with various types of steps, all using state_dict approach."""
    module_step = MockModuleStep()
    state_dict_step = MockStepWithTensorState()
    simple_step = MockStepWithoutOptionalMethods()  # No tensor state

    pipeline = RobotProcessor([module_step, state_dict_step, simple_step])

    # Process some data
    for i in range(5):
        transition = create_transition(observation=torch.randn(2, 10), reward=float(i))
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
        transition = create_transition(reward=r)
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
    transition = create_transition(reward=4.0)
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
        obs = transition.get(TransitionKey.OBSERVATION)
        comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

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

            # Return updated transition
            new_transition = transition.copy()
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp_data
            return new_transition

        return transition

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
        transition = create_transition(observation=obs, reward=float(i))
        result = pipeline(transition)
        comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
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
        transition = create_transition(observation=obs_gpu, reward=1.0)
        result = pipeline(transition)
        comp_data = result[TransitionKey.COMPLEMENTARY_DATA]

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
    transition = create_transition(observation=obs, reward=1.0)
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
        transition = create_transition(observation=obs_gpu, reward=2.0)
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
        reward = transition.get(TransitionKey.REWARD)
        comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        # Use the env parameter if provided
        if self.env is not None:
            comp_data = {} if comp_data is None else dict(comp_data)
            comp_data[f"{self.name}_env_info"] = str(self.env)

        # Apply multiplier to reward
        new_transition = transition.copy()
        if reward is not None:
            new_transition[TransitionKey.REWARD] = reward * self.multiplier

        if comp_data:
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        # Note: env is intentionally NOT included here as it's not serializable
        return {
            "name": self.name,
            "multiplier": self.multiplier,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
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
        comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        comp_data = {} if comp_data is None else dict(comp_data)
        comp_data["registered_step_value"] = self.value
        comp_data["registered_step_device"] = self.device

        new_transition = transition.copy()
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp_data
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "device": self.device,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
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
        transition = create_transition(reward=1.0)
        result = loaded_pipeline(transition)

        # Check that overrides were applied
        comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
        assert "env_step_env_info" in comp_data
        assert comp_data["env_step_env_info"] == "MockEnvironment(test_env)"
        assert comp_data["registered_step_value"] == 200
        assert comp_data["registered_step_device"] == "cuda"

        # Check that multiplier override was applied
        assert result[TransitionKey.REWARD] == 3.0  # 1.0 * 3.0 (overridden multiplier)


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

        transition = create_transition(reward=1.0)
        result = loaded_pipeline(transition)

        # The reward should be affected by both steps, both getting the override
        # First step: 1.0 * 5.0 = 5.0 (overridden)
        # Second step: 5.0 * 5.0 = 25.0 (also overridden)
        assert result[TransitionKey.REWARD] == 25.0


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
        transition = create_transition()
        result = loaded_pipeline(transition)

        comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
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
        transition = create_transition(reward=2.0)
        result = loaded_pipeline(transition)

        comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
        assert comp_data["unregistered_env_info"] == "MockEnvironment(mixed_test)"
        assert comp_data["registered_step_value"] == 777
        assert result[TransitionKey.REWARD] == 8.0  # 2.0 * 4.0


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
        transition = create_transition(reward=1.0)
        result = loaded_pipeline(transition)

        assert result[TransitionKey.REWARD] == 3.0  # 1.0 * 3.0


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
        transition = create_transition(reward=1.0)
        result = loaded_pipeline(transition)

        assert result[TransitionKey.REWARD] == 2.0


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
        transition = create_transition(reward=float(i))
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


def test_repr_empty_processor():
    """Test __repr__ with empty processor."""
    pipeline = RobotProcessor()
    repr_str = repr(pipeline)

    expected = "RobotProcessor(name='RobotProcessor', steps=0: [])"
    assert repr_str == expected


def test_repr_single_step():
    """Test __repr__ with single step."""
    step = MockStep("test_step")
    pipeline = RobotProcessor([step])
    repr_str = repr(pipeline)

    expected = "RobotProcessor(name='RobotProcessor', steps=1: [MockStep])"
    assert repr_str == expected


def test_repr_multiple_steps_under_limit():
    """Test __repr__ with 2-3 steps (all shown)."""
    step1 = MockStep("step1")
    step2 = MockStepWithoutOptionalMethods()
    pipeline = RobotProcessor([step1, step2])
    repr_str = repr(pipeline)

    expected = "RobotProcessor(name='RobotProcessor', steps=2: [MockStep, MockStepWithoutOptionalMethods])"
    assert repr_str == expected

    # Test with 3 steps (boundary case)
    step3 = MockStepWithTensorState()
    pipeline = RobotProcessor([step1, step2, step3])
    repr_str = repr(pipeline)

    expected = "RobotProcessor(name='RobotProcessor', steps=3: [MockStep, MockStepWithoutOptionalMethods, MockStepWithTensorState])"
    assert repr_str == expected


def test_repr_many_steps_truncated():
    """Test __repr__ with more than 3 steps (truncated with ellipsis)."""
    step1 = MockStep("step1")
    step2 = MockStepWithoutOptionalMethods()
    step3 = MockStepWithTensorState()
    step4 = MockModuleStep()
    step5 = MockNonModuleStepWithState()

    pipeline = RobotProcessor([step1, step2, step3, step4, step5])
    repr_str = repr(pipeline)

    expected = "RobotProcessor(name='RobotProcessor', steps=5: [MockStep, MockStepWithoutOptionalMethods, ..., MockNonModuleStepWithState])"
    assert repr_str == expected


def test_repr_with_custom_name():
    """Test __repr__ with custom processor name."""
    step = MockStep("test_step")
    pipeline = RobotProcessor([step], name="CustomProcessor")
    repr_str = repr(pipeline)

    expected = "RobotProcessor(name='CustomProcessor', steps=1: [MockStep])"
    assert repr_str == expected


def test_repr_with_seed():
    """Test __repr__ with seed parameter."""
    step = MockStep("test_step")
    pipeline = RobotProcessor([step], seed=42)
    repr_str = repr(pipeline)

    expected = "RobotProcessor(name='RobotProcessor', steps=1: [MockStep], seed=42)"
    assert repr_str == expected


def test_repr_with_custom_name_and_seed():
    """Test __repr__ with both custom name and seed."""
    step1 = MockStep("step1")
    step2 = MockStepWithoutOptionalMethods()
    pipeline = RobotProcessor([step1, step2], name="MyProcessor", seed=123)
    repr_str = repr(pipeline)

    expected = (
        "RobotProcessor(name='MyProcessor', steps=2: [MockStep, MockStepWithoutOptionalMethods], seed=123)"
    )
    assert repr_str == expected


def test_repr_without_seed():
    """Test __repr__ when seed is explicitly None (should not show seed)."""
    step = MockStep("test_step")
    pipeline = RobotProcessor([step], name="TestProcessor", seed=None)
    repr_str = repr(pipeline)

    expected = "RobotProcessor(name='TestProcessor', steps=1: [MockStep])"
    assert repr_str == expected


def test_repr_various_step_types():
    """Test __repr__ with different types of steps to verify class name extraction."""
    step1 = MockStep()
    step2 = MockStepWithTensorState()
    step3 = MockModuleStep()
    step4 = MockNonModuleStepWithState()

    pipeline = RobotProcessor([step1, step2, step3, step4], name="MixedSteps")
    repr_str = repr(pipeline)

    expected = "RobotProcessor(name='MixedSteps', steps=4: [MockStep, MockStepWithTensorState, ..., MockNonModuleStepWithState])"
    assert repr_str == expected


def test_repr_edge_case_long_names():
    """Test __repr__ handles steps with long class names properly."""
    step1 = MockStepWithNonSerializableParam()
    step2 = MockStepWithoutOptionalMethods()
    step3 = MockStepWithTensorState()
    step4 = MockNonModuleStepWithState()

    pipeline = RobotProcessor([step1, step2, step3, step4], name="LongNames", seed=999)
    repr_str = repr(pipeline)

    expected = "RobotProcessor(name='LongNames', steps=4: [MockStepWithNonSerializableParam, MockStepWithoutOptionalMethods, ..., MockNonModuleStepWithState], seed=999)"
    assert repr_str == expected


# Tests for config filename features and multiple processors
def test_save_with_custom_config_filename():
    """Test saving processor with custom config filename."""
    step = MockStep("test")
    pipeline = RobotProcessor([step], name="TestProcessor")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save with custom filename
        pipeline.save_pretrained(tmp_dir, config_filename="my_custom_config.json")

        # Check file exists
        config_path = Path(tmp_dir) / "my_custom_config.json"
        assert config_path.exists()

        # Check content
        with open(config_path) as f:
            config = json.load(f)
        assert config["name"] == "TestProcessor"

        # Load with specific filename
        loaded = RobotProcessor.from_pretrained(tmp_dir, config_filename="my_custom_config.json")
        assert loaded.name == "TestProcessor"


def test_multiple_processors_same_directory():
    """Test saving multiple processors to the same directory with different config files."""
    # Create different processors
    preprocessor = RobotProcessor([MockStep("pre1"), MockStep("pre2")], name="preprocessor")

    postprocessor = RobotProcessor([MockStepWithoutOptionalMethods(multiplier=0.5)], name="postprocessor")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save both to same directory
        preprocessor.save_pretrained(tmp_dir)
        postprocessor.save_pretrained(tmp_dir)

        # Check both config files exist
        assert (Path(tmp_dir) / "preprocessor.json").exists()
        assert (Path(tmp_dir) / "postprocessor.json").exists()

        # Load them back
        loaded_pre = RobotProcessor.from_pretrained(tmp_dir, config_filename="preprocessor.json")
        loaded_post = RobotProcessor.from_pretrained(tmp_dir, config_filename="postprocessor.json")

        assert loaded_pre.name == "preprocessor"
        assert loaded_post.name == "postprocessor"
        assert len(loaded_pre) == 2
        assert len(loaded_post) == 1


def test_auto_detect_single_config():
    """Test automatic config detection when there's only one JSON file."""
    step = MockStepWithTensorState()
    pipeline = RobotProcessor([step], name="SingleConfig")

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Load without specifying config_filename
        loaded = RobotProcessor.from_pretrained(tmp_dir)
        assert loaded.name == "SingleConfig"


def test_error_multiple_configs_no_filename():
    """Test error when multiple configs exist and no filename specified."""
    proc1 = RobotProcessor([MockStep()], name="processor1")
    proc2 = RobotProcessor([MockStep()], name="processor2")

    with tempfile.TemporaryDirectory() as tmp_dir:
        proc1.save_pretrained(tmp_dir)
        proc2.save_pretrained(tmp_dir)

        # Should raise error
        with pytest.raises(ValueError, match="Multiple .json files found"):
            RobotProcessor.from_pretrained(tmp_dir)


def test_state_file_naming_with_indices():
    """Test that state files include step indices to avoid conflicts."""
    # Create multiple steps of same type with state
    step1 = MockStepWithTensorState(name="norm1", window_size=5)
    step2 = MockStepWithTensorState(name="norm2", window_size=10)
    step3 = MockModuleStep(input_dim=5)

    pipeline = RobotProcessor([step1, step2, step3])

    # Process some data to create state
    for i in range(5):
        transition = create_transition(observation=torch.randn(2, 5), reward=float(i))
        pipeline(transition)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Check state files have indices
        state_files = sorted(Path(tmp_dir).glob("*.safetensors"))
        assert len(state_files) == 3

        # Files should be named with indices
        expected_names = ["step_0.safetensors", "step_1.safetensors", "step_2.safetensors"]
        actual_names = [f.name for f in state_files]
        assert actual_names == expected_names


def test_state_file_naming_with_registry():
    """Test state file naming for registered steps includes both index and name."""

    # Register a test step
    @ProcessorStepRegistry.register("test_stateful_step")
    @dataclass
    class TestStatefulStep:
        value: int = 0

        def __init__(self, value: int = 0):
            self.value = value
            self.state_tensor = torch.randn(3, 3)

        def __call__(self, transition: EnvTransition) -> EnvTransition:
            return transition

        def get_config(self):
            return {"value": self.value}

        def state_dict(self):
            return {"state_tensor": self.state_tensor}

        def load_state_dict(self, state):
            self.state_tensor = state["state_tensor"]

    try:
        # Create pipeline with registered steps
        step1 = TestStatefulStep(1)
        step2 = TestStatefulStep(2)
        pipeline = RobotProcessor([step1, step2])

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)

            # Check state files
            state_files = sorted(Path(tmp_dir).glob("*.safetensors"))
            assert len(state_files) == 2

            # Should include both index and registry name
            expected_names = [
                "step_0_test_stateful_step.safetensors",
                "step_1_test_stateful_step.safetensors",
            ]
            actual_names = [f.name for f in state_files]
            assert actual_names == expected_names

    finally:
        # Cleanup registry
        ProcessorStepRegistry.unregister("test_stateful_step")


# More comprehensive override tests
def test_override_with_nested_config():
    """Test overrides with nested configuration dictionaries."""

    @ProcessorStepRegistry.register("complex_config_step")
    @dataclass
    class ComplexConfigStep:
        name: str = "complex"
        simple_param: int = 42
        nested_config: dict = None

        def __post_init__(self):
            if self.nested_config is None:
                self.nested_config = {"level1": {"level2": "default"}}

        def __call__(self, transition: EnvTransition) -> EnvTransition:
            comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
            comp_data = dict(comp_data)
            comp_data["config_value"] = self.nested_config.get("level1", {}).get("level2", "missing")

            new_transition = transition.copy()
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = comp_data
            return new_transition

        def get_config(self):
            return {"name": self.name, "simple_param": self.simple_param, "nested_config": self.nested_config}

    try:
        step = ComplexConfigStep()
        pipeline = RobotProcessor([step])

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)

            # Load with nested override
            loaded = RobotProcessor.from_pretrained(
                tmp_dir,
                overrides={"complex_config_step": {"nested_config": {"level1": {"level2": "overridden"}}}},
            )

            # Test that override worked
            transition = create_transition()
            result = loaded(transition)
            assert result[TransitionKey.COMPLEMENTARY_DATA]["config_value"] == "overridden"
    finally:
        ProcessorStepRegistry.unregister("complex_config_step")


def test_override_preserves_defaults():
    """Test that overrides only affect specified parameters."""
    step = MockStepWithNonSerializableParam(name="test", multiplier=2.0)
    pipeline = RobotProcessor([step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Override only one parameter
        loaded = RobotProcessor.from_pretrained(
            tmp_dir,
            overrides={
                "MockStepWithNonSerializableParam": {
                    "multiplier": 5.0  # Only override multiplier
                }
            },
        )

        # Check that name was preserved from saved config
        loaded_step = loaded.steps[0]
        assert loaded_step.name == "test"  # Original value
        assert loaded_step.multiplier == 5.0  # Overridden value


def test_override_type_validation():
    """Test that type errors in overrides are caught properly."""
    step = MockStepWithTensorState(learning_rate=0.01)
    pipeline = RobotProcessor([step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Try to override with wrong type
        overrides = {
            "MockStepWithTensorState": {
                "window_size": "not_an_int"  # Should be int
            }
        }

        with pytest.raises(ValueError, match="Failed to instantiate"):
            RobotProcessor.from_pretrained(tmp_dir, overrides=overrides)


def test_override_with_callables():
    """Test overriding with callable objects."""

    @ProcessorStepRegistry.register("callable_step")
    @dataclass
    class CallableStep:
        name: str = "callable_step"
        transform_fn: Any = None

        def __call__(self, transition: EnvTransition) -> EnvTransition:
            obs = transition.get(TransitionKey.OBSERVATION)
            if obs is not None and self.transform_fn is not None:
                processed_obs = {}
                for k, v in obs.items():
                    processed_obs[k] = self.transform_fn(v)

                new_transition = transition.copy()
                new_transition[TransitionKey.OBSERVATION] = processed_obs
                return new_transition
            return transition

        def get_config(self):
            return {"name": self.name}

    try:
        step = CallableStep()
        pipeline = RobotProcessor([step])

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)

            # Define a transform function
            def double_values(x):
                if isinstance(x, (int, float)):
                    return x * 2
                elif isinstance(x, torch.Tensor):
                    return x * 2
                return x

            # Load with callable override
            loaded = RobotProcessor.from_pretrained(
                tmp_dir, overrides={"callable_step": {"transform_fn": double_values}}
            )

            # Test it works
            transition = create_transition(observation={"value": torch.tensor(5.0)})
            result = loaded(transition)
            assert result[TransitionKey.OBSERVATION]["value"].item() == 10.0
    finally:
        ProcessorStepRegistry.unregister("callable_step")


def test_override_multiple_same_class_warning():
    """Test behavior when multiple steps of same class exist."""
    step1 = MockStepWithNonSerializableParam(name="step1", multiplier=1.0)
    step2 = MockStepWithNonSerializableParam(name="step2", multiplier=2.0)
    pipeline = RobotProcessor([step1, step2])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Override affects all instances of the class
        loaded = RobotProcessor.from_pretrained(
            tmp_dir, overrides={"MockStepWithNonSerializableParam": {"multiplier": 10.0}}
        )

        # Both steps get the same override
        assert loaded.steps[0].multiplier == 10.0
        assert loaded.steps[1].multiplier == 10.0

        # But original names are preserved
        assert loaded.steps[0].name == "step1"
        assert loaded.steps[1].name == "step2"


def test_config_filename_special_characters():
    """Test config filenames with special characters are sanitized."""
    # Processor name with special characters
    pipeline = RobotProcessor([MockStep()], name="My/Processor\\With:Special*Chars")

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Check that filename was sanitized
        json_files = list(Path(tmp_dir).glob("*.json"))
        assert len(json_files) == 1

        # Should have replaced special chars with underscores
        expected_name = "my_processor_with_special_chars.json"
        assert json_files[0].name == expected_name


def test_override_with_device_strings():
    """Test overriding device parameters with string values."""

    @ProcessorStepRegistry.register("device_aware_step")
    @dataclass
    class DeviceAwareStep:
        device: str = "cpu"

        def __init__(self, device: str = "cpu"):
            self.device = device
            self.buffer = torch.zeros(10, device=device)

        def __call__(self, transition: EnvTransition) -> EnvTransition:
            return transition

        def get_config(self):
            return {"device": str(self.device)}

        def state_dict(self):
            return {"buffer": self.buffer}

        def load_state_dict(self, state):
            self.buffer = state["buffer"]

    try:
        step = DeviceAwareStep(device="cpu")
        pipeline = RobotProcessor([step])

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)

            # Override device
            if torch.cuda.is_available():
                loaded = RobotProcessor.from_pretrained(
                    tmp_dir, overrides={"device_aware_step": {"device": "cuda:0"}}
                )

                loaded_step = loaded.steps[0]
                assert loaded_step.device == "cuda:0"
                # Note: buffer will still be on CPU from saved state
                # until .to() is called on the processor

    finally:
        ProcessorStepRegistry.unregister("device_aware_step")


def test_from_pretrained_nonexistent_path():
    """Test error handling when loading from non-existent sources."""
    from huggingface_hub.errors import HfHubHTTPError, HFValidationError

    # Test with an invalid repo ID (too many slashes) - caught by HF validation
    with pytest.raises(HFValidationError):
        RobotProcessor.from_pretrained("/path/that/does/not/exist")

    # Test with a non-existent but valid Hub repo format
    with pytest.raises((FileNotFoundError, HfHubHTTPError)):
        RobotProcessor.from_pretrained("nonexistent-user/nonexistent-repo")

    # Test with a local directory that exists but has no config files
    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(FileNotFoundError, match="No .json configuration files found"):
            RobotProcessor.from_pretrained(tmp_dir)


def test_save_load_with_custom_converter_functions():
    """Test that custom to_transition and to_output functions are NOT saved."""

    def custom_to_transition(batch):
        # Custom conversion logic
        return {
            TransitionKey.OBSERVATION: batch.get("obs"),
            TransitionKey.ACTION: batch.get("act"),
            TransitionKey.REWARD: batch.get("rew", 0.0),
            TransitionKey.DONE: batch.get("done", False),
            TransitionKey.TRUNCATED: batch.get("truncated", False),
            TransitionKey.INFO: {},
            TransitionKey.COMPLEMENTARY_DATA: {},
        }

    def custom_to_output(transition):
        # Custom output format
        return {
            "obs": transition.get(TransitionKey.OBSERVATION),
            "act": transition.get(TransitionKey.ACTION),
            "rew": transition.get(TransitionKey.REWARD),
            "done": transition.get(TransitionKey.DONE),
            "truncated": transition.get(TransitionKey.TRUNCATED),
        }

    # Create processor with custom converters
    pipeline = RobotProcessor([MockStep()], to_transition=custom_to_transition, to_output=custom_to_output)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Load - should use default converters
        loaded = RobotProcessor.from_pretrained(tmp_dir)

        # Verify it uses default converters by checking with standard batch format
        batch = {
            "observation.image": torch.randn(1, 3, 32, 32),
            "action": torch.randn(1, 7),
            "next.reward": torch.tensor([1.0]),
            "next.done": torch.tensor([False]),
            "next.truncated": torch.tensor([False]),
            "info": {},
        }

        # Should work with standard format (wouldn't work with custom converter)
        result = loaded(batch)
        assert "observation.image" in result  # Standard format preserved

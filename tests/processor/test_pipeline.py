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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features
from lerobot.processor import (
    DataProcessorPipeline,
    EnvTransition,
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
)
from lerobot.processor.converters import create_transition, identity_transition
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, REWARD, TRUNCATED
from tests.conftest import assert_contract_is_typed


@dataclass
class MockStep(ProcessorStep):
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

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We do not test features here
        return features


@dataclass
class MockStepWithoutOptionalMethods(ProcessorStep):
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

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We do not test features here
        return features


@dataclass
class MockStepWithTensorState(ProcessorStep):
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

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We do not test features here
        return features


def test_empty_pipeline():
    """Test pipeline with no steps."""
    pipeline = DataProcessorPipeline([], to_transition=identity_transition, to_output=identity_transition)

    transition = create_transition()
    result = pipeline(transition)

    assert result == transition
    assert len(pipeline) == 0


def test_single_step_pipeline():
    """Test pipeline with a single step."""
    step = MockStep("test_step")
    pipeline = DataProcessorPipeline([step], to_transition=identity_transition, to_output=identity_transition)

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
    pipeline = DataProcessorPipeline(
        [step1, step2], to_transition=identity_transition, to_output=identity_transition
    )

    transition = create_transition()
    result = pipeline(transition)

    assert len(pipeline) == 2
    assert result[TransitionKey.COMPLEMENTARY_DATA]["step1_counter"] == 0
    assert result[TransitionKey.COMPLEMENTARY_DATA]["step2_counter"] == 0


def test_invalid_transition_format():
    """Test pipeline with invalid transition format."""
    pipeline = DataProcessorPipeline([MockStep()])

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
    pipeline = DataProcessorPipeline([step1, step2])

    transition = create_transition()

    results = list(pipeline.step_through(transition))

    assert len(results) == 3  # Original + 2 steps
    assert results[0] == transition  # Original
    assert "step1_counter" in results[1][TransitionKey.COMPLEMENTARY_DATA]  # After step1
    assert "step2_counter" in results[2][TransitionKey.COMPLEMENTARY_DATA]  # After step2

    # Ensure all results are dicts (same format as input)
    for result in results:
        assert isinstance(result, dict)
        assert all(isinstance(k, TransitionKey) for k in result)


def test_step_through_with_dict():
    """Test step_through method with dict input."""
    step1 = MockStep("step1")
    step2 = MockStep("step2")
    pipeline = DataProcessorPipeline([step1, step2])

    batch = {
        OBS_IMAGE: None,
        ACTION: None,
        REWARD: 0.0,
        DONE: False,
        TRUNCATED: False,
        "info": {},
    }

    results = list(pipeline.step_through(batch))

    assert len(results) == 3  # Original + 2 steps

    # Ensure all results are EnvTransition dicts (regardless of input format)
    for result in results:
        assert isinstance(result, dict)
        # Check that keys are TransitionKey enums or at least valid transition keys
        for key in result:
            assert key in [
                TransitionKey.OBSERVATION,
                TransitionKey.ACTION,
                TransitionKey.REWARD,
                TransitionKey.DONE,
                TransitionKey.TRUNCATED,
                TransitionKey.INFO,
                TransitionKey.COMPLEMENTARY_DATA,
            ]

    # Check that the processing worked - verify step counters in complementary_data
    assert results[1].get(TransitionKey.COMPLEMENTARY_DATA, {}).get("step1_counter") == 0
    assert results[2].get(TransitionKey.COMPLEMENTARY_DATA, {}).get("step1_counter") == 0
    assert results[2].get(TransitionKey.COMPLEMENTARY_DATA, {}).get("step2_counter") == 0


def test_step_through_no_hooks():
    """Test that step_through doesn't execute hooks."""
    step = MockStep("test_step")
    pipeline = DataProcessorPipeline([step])

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
    pipeline = DataProcessorPipeline([step1, step2])

    # Test integer indexing
    assert pipeline[0] is step1
    assert pipeline[1] is step2

    # Test slice indexing
    sub_pipeline = pipeline[0:1]
    assert isinstance(sub_pipeline, DataProcessorPipeline)
    assert len(sub_pipeline) == 1
    assert sub_pipeline[0] is step1


def test_hooks():
    """Test before/after step hooks."""
    step = MockStep("test_step")
    pipeline = DataProcessorPipeline([step])

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


def test_unregister_hooks():
    """Test unregistering hooks from the pipeline."""
    step = MockStep("test_step")
    pipeline = DataProcessorPipeline([step])

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


def test_unregister_nonexistent_hook():
    """Test error handling when unregistering hooks that don't exist."""
    pipeline = DataProcessorPipeline([MockStep()])

    def some_hook(idx: int, transition: EnvTransition):
        pass

    def reset_hook():
        pass

    # Test unregistering hooks that were never registered
    with pytest.raises(ValueError, match="not found in before_step_hooks"):
        pipeline.unregister_before_step_hook(some_hook)

    with pytest.raises(ValueError, match="not found in after_step_hooks"):
        pipeline.unregister_after_step_hook(some_hook)


def test_multiple_hooks_and_selective_unregister():
    """Test registering multiple hooks and selectively unregistering them."""
    pipeline = DataProcessorPipeline([MockStep("step1"), MockStep("step2")])

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
    pipeline = DataProcessorPipeline([MockStep("step")])

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

    pipeline = DataProcessorPipeline([step1, step2], name="TestPipeline")

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
        assert len(config["steps"]) == 2

        # Verify counters are saved in config, not in separate state files
        assert config["steps"][0]["config"]["counter"] == 5
        assert config["steps"][1]["config"]["counter"] == 10

        # Load pipeline
        loaded_pipeline = DataProcessorPipeline.from_pretrained(tmp_dir, config_filename="testpipeline.json")

        assert loaded_pipeline.name == "TestPipeline"
        assert len(loaded_pipeline) == 2

        # Check that counter was restored from config
        assert loaded_pipeline.steps[0].counter == 5
        assert loaded_pipeline.steps[1].counter == 10


def test_step_without_optional_methods():
    """Test pipeline with steps that don't implement optional methods."""
    step = MockStepWithoutOptionalMethods(multiplier=3.0)
    pipeline = DataProcessorPipeline(
        [step], to_transition=identity_transition, to_output=identity_transition
    )  # Identity for EnvTransition input/output

    transition = create_transition(reward=2.0)
    result = pipeline(transition)

    assert result[TransitionKey.REWARD] == 6.0  # 2.0 * 3.0

    # Reset should work even if step doesn't implement reset
    pipeline.reset()

    # Save/load should work even without optional methods
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)
        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir, config_filename="dataprocessorpipeline.json"
        )
        assert len(loaded_pipeline) == 1


def test_mixed_json_and_tensor_state():
    """Test step with both JSON attributes and tensor state."""
    step = MockStepWithTensorState(name="stats", learning_rate=0.05, window_size=5)
    pipeline = DataProcessorPipeline([step])

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
        config_path = Path(tmp_dir) / "dataprocessorpipeline.json"  # Default name is "RobotProcessor"
        state_path = Path(tmp_dir) / "dataprocessorpipeline_step_0.safetensors"
        assert config_path.exists()
        assert state_path.exists()

        # Load and verify
        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir, config_filename="dataprocessorpipeline.json"
        )
        loaded_step = loaded_pipeline.steps[0]

        # Check JSON attributes were restored
        assert loaded_step.name == "stats"
        assert loaded_step.learning_rate == 0.05
        assert loaded_step.window_size == 5

        # Check tensor state was restored
        assert loaded_step.running_count.item() == 10
        assert torch.allclose(loaded_step.running_mean, step.running_mean)


class MockModuleStep(ProcessorStep, nn.Module):
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
        return nn.Module.state_dict(self)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Override to load all module parameters and buffers."""
        # Use the module's load_state_dict
        nn.Module.load_state_dict(self, state)

    def reset(self) -> None:
        self.running_mean.zero_()
        self.counter = 0

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We do not test features here
        return features


class MockNonModuleStepWithState(ProcessorStep):
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

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We do not test features here
        return features


# Tests for overrides functionality
@dataclass
class MockStepWithNonSerializableParam(ProcessorStep):
    """Mock step that requires a non-serializable parameter."""

    def __init__(self, name: str = "mock_env_step", multiplier: float = 1.0, env: Any = None):
        self.name = name
        # Add type validation for multiplier
        if isinstance(multiplier, str):
            raise ValueError(f"multiplier must be a number, got string '{multiplier}'")
        if not isinstance(multiplier, (int | float)):
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

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We do not test features here
        return features


@ProcessorStepRegistry.register("registered_mock_step")
@dataclass
class RegisteredMockStep(ProcessorStep):
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

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We do not test features here
        return features


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

    pipeline = DataProcessorPipeline([env_step, registered_step], name="TestOverrides")

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

        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir,
            config_filename="testoverrides.json",
            overrides=overrides,
            to_transition=identity_transition,
            to_output=identity_transition,
        )

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

    pipeline = DataProcessorPipeline([step1, step2])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Override only one step
        overrides = {"MockStepWithNonSerializableParam": {"multiplier": 5.0}}

        # The current implementation applies overrides to ALL steps with the same class name
        # Both steps will get the override
        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir,
            config_filename="dataprocessorpipeline.json",
            overrides=overrides,
            to_transition=identity_transition,
            to_output=identity_transition,
        )

        transition = create_transition(reward=1.0)
        result = loaded_pipeline(transition)

        # The reward should be affected by both steps, both getting the override
        # First step: 1.0 * 5.0 = 5.0 (overridden)
        # Second step: 5.0 * 5.0 = 25.0 (also overridden)
        assert result[TransitionKey.REWARD] == 25.0


def test_from_pretrained_invalid_override_key():
    """Test that invalid override keys raise KeyError."""
    step = MockStepWithNonSerializableParam()
    pipeline = DataProcessorPipeline([step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Try to override a non-existent step
        overrides = {"NonExistentStep": {"param": "value"}}

        with pytest.raises(KeyError, match="Override keys.*do not match any step"):
            DataProcessorPipeline.from_pretrained(
                tmp_dir, config_filename="dataprocessorpipeline.json", overrides=overrides
            )


def test_from_pretrained_multiple_invalid_override_keys():
    """Test that multiple invalid override keys are reported."""
    step = MockStepWithNonSerializableParam()
    pipeline = DataProcessorPipeline([step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Try to override multiple non-existent steps
        overrides = {"NonExistentStep1": {"param": "value1"}, "NonExistentStep2": {"param": "value2"}}

        with pytest.raises(KeyError) as exc_info:
            DataProcessorPipeline.from_pretrained(
                tmp_dir, config_filename="dataprocessorpipeline.json", overrides=overrides
            )

        error_msg = str(exc_info.value)
        assert "NonExistentStep1" in error_msg
        assert "NonExistentStep2" in error_msg
        assert "Available step keys" in error_msg


def test_from_pretrained_registered_step_override():
    """Test overriding registered steps using registry names."""
    registered_step = RegisteredMockStep(value=50, device="cpu")
    pipeline = DataProcessorPipeline([registered_step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Override using registry name
        overrides = {"registered_mock_step": {"value": 999, "device": "cuda"}}

        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir,
            config_filename="dataprocessorpipeline.json",
            overrides=overrides,
            to_transition=identity_transition,
            to_output=identity_transition,
        )

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

    pipeline = DataProcessorPipeline([unregistered_step, registered_step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        mock_env = MockEnvironment("mixed_test")

        overrides = {
            "MockStepWithNonSerializableParam": {"env": mock_env, "multiplier": 4.0},
            "registered_mock_step": {"value": 777},
        }

        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir,
            config_filename="dataprocessorpipeline.json",
            overrides=overrides,
            to_transition=identity_transition,
            to_output=identity_transition,
        )

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
    pipeline = DataProcessorPipeline([step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Load without overrides
        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir,
            config_filename="dataprocessorpipeline.json",
            to_transition=identity_transition,
            to_output=identity_transition,
        )

        assert len(loaded_pipeline) == 1

        # Test that the step works (env will be None)
        transition = create_transition(reward=1.0)
        result = loaded_pipeline(transition)

        assert result[TransitionKey.REWARD] == 3.0  # 1.0 * 3.0


def test_from_pretrained_empty_overrides():
    """Test that from_pretrained works with empty overrides dict."""
    step = MockStepWithNonSerializableParam(multiplier=2.0)
    pipeline = DataProcessorPipeline([step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Load with empty overrides
        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir,
            config_filename="dataprocessorpipeline.json",
            overrides={},
            to_transition=identity_transition,
            to_output=identity_transition,
        )

        assert len(loaded_pipeline) == 1

        # Test that the step works normally
        transition = create_transition(reward=1.0)
        result = loaded_pipeline(transition)

        assert result[TransitionKey.REWARD] == 2.0


def test_from_pretrained_override_instantiation_error():
    """Test that instantiation errors with overrides are properly reported."""
    step = MockStepWithNonSerializableParam(multiplier=1.0)
    pipeline = DataProcessorPipeline([step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Try to override with invalid parameter type
        overrides = {
            "MockStepWithNonSerializableParam": {
                "multiplier": "invalid_type"  # Should be float, not string
            }
        }

        with pytest.raises(ValueError, match="Failed to instantiate processor step"):
            DataProcessorPipeline.from_pretrained(
                tmp_dir, config_filename="dataprocessorpipeline.json", overrides=overrides
            )


def test_from_pretrained_with_state_and_overrides():
    """Test that overrides work correctly with steps that have tensor state."""
    step = MockStepWithTensorState(name="tensor_step", learning_rate=0.01, window_size=5)
    pipeline = DataProcessorPipeline([step])

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

        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir, config_filename="dataprocessorpipeline.json", overrides=overrides
        )
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
    pipeline = DataProcessorPipeline([step1, step2])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Test with invalid override key
        overrides = {"WrongStepName": {"param": "value"}}

        with pytest.raises(KeyError) as exc_info:
            DataProcessorPipeline.from_pretrained(
                tmp_dir, config_filename="dataprocessorpipeline.json", overrides=overrides
            )

        error_msg = str(exc_info.value)
        assert "WrongStepName" in error_msg
        assert "Available step keys" in error_msg
        assert "MockStepWithNonSerializableParam" in error_msg
        assert "registered_mock_step" in error_msg


def test_repr_empty_processor():
    """Test __repr__ with empty processor."""
    pipeline = DataProcessorPipeline()
    repr_str = repr(pipeline)

    expected = "DataProcessorPipeline(name='DataProcessorPipeline', steps=0: [])"
    assert repr_str == expected


def test_repr_single_step():
    """Test __repr__ with single step."""
    step = MockStep("test_step")
    pipeline = DataProcessorPipeline([step])
    repr_str = repr(pipeline)

    expected = "DataProcessorPipeline(name='DataProcessorPipeline', steps=1: [MockStep])"
    assert repr_str == expected


def test_repr_multiple_steps_under_limit():
    """Test __repr__ with 2-3 steps (all shown)."""
    step1 = MockStep("step1")
    step2 = MockStepWithoutOptionalMethods()
    pipeline = DataProcessorPipeline([step1, step2])
    repr_str = repr(pipeline)

    expected = "DataProcessorPipeline(name='DataProcessorPipeline', steps=2: [MockStep, MockStepWithoutOptionalMethods])"
    assert repr_str == expected

    # Test with 3 steps (boundary case)
    step3 = MockStepWithTensorState()
    pipeline = DataProcessorPipeline([step1, step2, step3])
    repr_str = repr(pipeline)

    expected = "DataProcessorPipeline(name='DataProcessorPipeline', steps=3: [MockStep, MockStepWithoutOptionalMethods, MockStepWithTensorState])"
    assert repr_str == expected


def test_repr_many_steps_truncated():
    """Test __repr__ with more than 3 steps (truncated with ellipsis)."""
    step1 = MockStep("step1")
    step2 = MockStepWithoutOptionalMethods()
    step3 = MockStepWithTensorState()
    step4 = MockModuleStep()
    step5 = MockNonModuleStepWithState()

    pipeline = DataProcessorPipeline([step1, step2, step3, step4, step5])
    repr_str = repr(pipeline)

    expected = "DataProcessorPipeline(name='DataProcessorPipeline', steps=5: [MockStep, MockStepWithoutOptionalMethods, ..., MockNonModuleStepWithState])"
    assert repr_str == expected


def test_repr_with_custom_name():
    """Test __repr__ with custom processor name."""
    step = MockStep("test_step")
    pipeline = DataProcessorPipeline([step], name="CustomProcessor")
    repr_str = repr(pipeline)

    expected = "DataProcessorPipeline(name='CustomProcessor', steps=1: [MockStep])"
    assert repr_str == expected


def test_repr_with_seed():
    """Test __repr__ with seed parameter."""
    step = MockStep("test_step")
    pipeline = DataProcessorPipeline([step])
    repr_str = repr(pipeline)

    expected = "DataProcessorPipeline(name='DataProcessorPipeline', steps=1: [MockStep])"
    assert repr_str == expected


def test_repr_with_custom_name_and_seed():
    """Test __repr__ with both custom name and seed."""
    step1 = MockStep("step1")
    step2 = MockStepWithoutOptionalMethods()
    pipeline = DataProcessorPipeline([step1, step2], name="MyProcessor")
    repr_str = repr(pipeline)

    expected = (
        "DataProcessorPipeline(name='MyProcessor', steps=2: [MockStep, MockStepWithoutOptionalMethods])"
    )
    assert repr_str == expected


def test_repr_without_seed():
    """Test __repr__ when seed is explicitly None (should not show seed)."""
    step = MockStep("test_step")
    pipeline = DataProcessorPipeline([step], name="TestProcessor")
    repr_str = repr(pipeline)

    expected = "DataProcessorPipeline(name='TestProcessor', steps=1: [MockStep])"
    assert repr_str == expected


def test_repr_various_step_types():
    """Test __repr__ with different types of steps to verify class name extraction."""
    step1 = MockStep()
    step2 = MockStepWithTensorState()
    step3 = MockModuleStep()
    step4 = MockNonModuleStepWithState()

    pipeline = DataProcessorPipeline([step1, step2, step3, step4], name="MixedSteps")
    repr_str = repr(pipeline)

    expected = "DataProcessorPipeline(name='MixedSteps', steps=4: [MockStep, MockStepWithTensorState, ..., MockNonModuleStepWithState])"
    assert repr_str == expected


def test_repr_edge_case_long_names():
    """Test __repr__ handles steps with long class names properly."""
    step1 = MockStepWithNonSerializableParam()
    step2 = MockStepWithoutOptionalMethods()
    step3 = MockStepWithTensorState()
    step4 = MockNonModuleStepWithState()

    pipeline = DataProcessorPipeline([step1, step2, step3, step4], name="LongNames")
    repr_str = repr(pipeline)

    expected = "DataProcessorPipeline(name='LongNames', steps=4: [MockStepWithNonSerializableParam, MockStepWithoutOptionalMethods, ..., MockNonModuleStepWithState])"
    assert repr_str == expected


# Tests for config filename features and multiple processors
def test_save_with_custom_config_filename():
    """Test saving processor with custom config filename."""
    step = MockStep("test")
    pipeline = DataProcessorPipeline([step], name="TestProcessor")

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
        loaded = DataProcessorPipeline.from_pretrained(tmp_dir, config_filename="my_custom_config.json")
        assert loaded.name == "TestProcessor"


def test_multiple_processors_same_directory():
    """Test saving multiple processors to the same directory with different config files."""
    # Create different processors
    preprocessor = DataProcessorPipeline([MockStep("pre1"), MockStep("pre2")], name="preprocessor")

    postprocessor = DataProcessorPipeline(
        [MockStepWithoutOptionalMethods(multiplier=0.5)], name="postprocessor"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save both to same directory
        preprocessor.save_pretrained(tmp_dir)
        postprocessor.save_pretrained(tmp_dir)

        # Check both config files exist
        assert (Path(tmp_dir) / "preprocessor.json").exists()
        assert (Path(tmp_dir) / "postprocessor.json").exists()

        # Load them back
        loaded_pre = DataProcessorPipeline.from_pretrained(tmp_dir, config_filename="preprocessor.json")
        loaded_post = DataProcessorPipeline.from_pretrained(tmp_dir, config_filename="postprocessor.json")

        assert loaded_pre.name == "preprocessor"
        assert loaded_post.name == "postprocessor"
        assert len(loaded_pre) == 2
        assert len(loaded_post) == 1


def test_explicit_config_filename_loading():
    """Test explicit config filename loading (no more auto-detection)."""
    step = MockStepWithTensorState()
    pipeline = DataProcessorPipeline([step], name="SingleConfig")

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Load with explicit config_filename (now required)
        loaded = DataProcessorPipeline.from_pretrained(tmp_dir, config_filename="singleconfig.json")
        assert loaded.name == "SingleConfig"


def test_explicit_config_selection_with_multiple_configs():
    """Test explicit config selection when multiple configs exist."""
    proc1 = DataProcessorPipeline([MockStep()], name="processor1")
    proc2 = DataProcessorPipeline([MockStep()], name="processor2")

    with tempfile.TemporaryDirectory() as tmp_dir:
        proc1.save_pretrained(tmp_dir)
        proc2.save_pretrained(tmp_dir)

        # Can load specific configs explicitly
        loaded1 = DataProcessorPipeline.from_pretrained(tmp_dir, config_filename="processor1.json")
        loaded2 = DataProcessorPipeline.from_pretrained(tmp_dir, config_filename="processor2.json")

        assert loaded1.name == "processor1"
        assert loaded2.name == "processor2"


def test_state_file_naming_with_indices():
    """Test that state files include pipeline name and step indices to avoid conflicts."""
    # Create multiple steps of same type with state
    step1 = MockStepWithTensorState(name="norm1", window_size=5)
    step2 = MockStepWithTensorState(name="norm2", window_size=10)
    step3 = MockModuleStep(input_dim=5)

    pipeline = DataProcessorPipeline([step1, step2, step3])

    # Process some data to create state
    for i in range(5):
        transition = create_transition(observation=torch.randn(2, 5), reward=float(i))
        pipeline(transition)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Check state files have indices
        state_files = sorted(Path(tmp_dir).glob("*.safetensors"))
        assert len(state_files) == 3

        # Files should be named with pipeline name prefix and indices
        expected_names = [
            "dataprocessorpipeline_step_0.safetensors",
            "dataprocessorpipeline_step_1.safetensors",
            "dataprocessorpipeline_step_2.safetensors",
        ]
        actual_names = [f.name for f in state_files]
        assert actual_names == expected_names


def test_state_file_naming_with_registry():
    """Test state file naming for registered steps includes pipeline name, index and registry name."""

    # Register a test step
    @ProcessorStepRegistry.register("test_stateful_step")
    @dataclass
    class TestStatefulStep(ProcessorStep):
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

        def transform_features(
            self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
        ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
            # We do not test features here
            return features

    try:
        # Create pipeline with registered steps
        step1 = TestStatefulStep(1)
        step2 = TestStatefulStep(2)
        pipeline = DataProcessorPipeline([step1, step2])

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)

            # Check state files
            state_files = sorted(Path(tmp_dir).glob("*.safetensors"))
            assert len(state_files) == 2

            # Should include pipeline name, index and registry name
            expected_names = [
                "dataprocessorpipeline_step_0_test_stateful_step.safetensors",
                "dataprocessorpipeline_step_1_test_stateful_step.safetensors",
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
    class ComplexConfigStep(ProcessorStep):
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

        def transform_features(
            self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
        ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
            # We do not test features here
            return features

    try:
        step = ComplexConfigStep()
        pipeline = DataProcessorPipeline([step])

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)

            # Load with nested override
            loaded = DataProcessorPipeline.from_pretrained(
                tmp_dir,
                config_filename="dataprocessorpipeline.json",
                overrides={"complex_config_step": {"nested_config": {"level1": {"level2": "overridden"}}}},
                to_transition=identity_transition,
                to_output=identity_transition,
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
    pipeline = DataProcessorPipeline([step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Override only one parameter
        loaded = DataProcessorPipeline.from_pretrained(
            tmp_dir,
            config_filename="dataprocessorpipeline.json",
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
    pipeline = DataProcessorPipeline([step])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Try to override with wrong type
        overrides = {
            "MockStepWithTensorState": {
                "window_size": "not_an_int"  # Should be int
            }
        }

        with pytest.raises(ValueError, match="Failed to instantiate"):
            DataProcessorPipeline.from_pretrained(
                tmp_dir, config_filename="dataprocessorpipeline.json", overrides=overrides
            )


def test_override_with_callables():
    """Test overriding with callable objects."""

    @ProcessorStepRegistry.register("callable_step")
    @dataclass
    class CallableStep(ProcessorStep):
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

        def transform_features(
            self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
        ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
            # We do not test features here
            return features

    try:
        step = CallableStep()
        pipeline = DataProcessorPipeline([step])

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)

            # Define a transform function
            def double_values(x):
                if isinstance(x, (int | float | torch.Tensor)):
                    return x * 2
                return x

            # Load with callable override
            loaded = DataProcessorPipeline.from_pretrained(
                tmp_dir,
                config_filename="dataprocessorpipeline.json",
                overrides={"callable_step": {"transform_fn": double_values}},
                to_transition=identity_transition,
                to_output=identity_transition,
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
    pipeline = DataProcessorPipeline([step1, step2])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Override affects all instances of the class
        loaded = DataProcessorPipeline.from_pretrained(
            tmp_dir,
            config_filename="dataprocessorpipeline.json",
            overrides={"MockStepWithNonSerializableParam": {"multiplier": 10.0}},
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
    pipeline = DataProcessorPipeline([MockStep()], name="My/Processor\\With:Special*Chars")

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Check that filename was sanitized
        json_files = list(Path(tmp_dir).glob("*.json"))
        assert len(json_files) == 1

        # Should have replaced special chars with underscores
        expected_name = "my_processor_with_special_chars.json"
        assert json_files[0].name == expected_name


def test_state_file_naming_with_multiple_processors():
    """Test that state files are properly prefixed with pipeline names to avoid conflicts."""
    # Create two processors with state
    step1 = MockStepWithTensorState(name="norm", window_size=5)
    preprocessor = DataProcessorPipeline([step1], name="PreProcessor")

    step2 = MockStepWithTensorState(name="norm", window_size=10)
    postprocessor = DataProcessorPipeline([step2], name="PostProcessor")

    # Process some data to create state
    for i in range(3):
        transition = create_transition(reward=float(i))
        preprocessor(transition)
        postprocessor(transition)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save both processors to the same directory
        preprocessor.save_pretrained(tmp_dir)
        postprocessor.save_pretrained(tmp_dir)

        # Check that all files exist and are distinct
        assert (Path(tmp_dir) / "preprocessor.json").exists()
        assert (Path(tmp_dir) / "postprocessor.json").exists()
        assert (Path(tmp_dir) / "preprocessor_step_0.safetensors").exists()
        assert (Path(tmp_dir) / "postprocessor_step_0.safetensors").exists()

        # Load both back and verify they work correctly
        loaded_pre = DataProcessorPipeline.from_pretrained(tmp_dir, config_filename="preprocessor.json")
        loaded_post = DataProcessorPipeline.from_pretrained(tmp_dir, config_filename="postprocessor.json")

        assert loaded_pre.name == "PreProcessor"
        assert loaded_post.name == "PostProcessor"
        assert loaded_pre.steps[0].window_size == 5
        assert loaded_post.steps[0].window_size == 10


def test_override_with_device_strings():
    """Test overriding device parameters with string values."""

    @ProcessorStepRegistry.register("device_aware_step")
    @dataclass
    class DeviceAwareStep(ProcessorStep):
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

        def transform_features(
            self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
        ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
            # We do not test features here
            return features

    try:
        step = DeviceAwareStep(device="cpu")
        pipeline = DataProcessorPipeline([step])

        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline.save_pretrained(tmp_dir)

            # Override device
            if torch.cuda.is_available():
                loaded = DataProcessorPipeline.from_pretrained(
                    tmp_dir,
                    config_filename="dataprocessorpipeline.json",
                    overrides={"device_aware_step": {"device": "cuda:0"}},
                )

                loaded_step = loaded.steps[0]
                assert loaded_step.device == "cuda:0"
                # Note: buffer will still be on CPU from saved state
                # until .to() is called on the processor

    finally:
        ProcessorStepRegistry.unregister("device_aware_step")


def test_from_pretrained_nonexistent_path():
    """Test error handling when loading from non-existent sources."""
    from huggingface_hub.errors import HfHubHTTPError

    # Test with an invalid local path - should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        DataProcessorPipeline.from_pretrained("/path/that/does/not/exist", config_filename="processor.json")

    # Test with a path that doesn't exist as a directory
    with pytest.raises(FileNotFoundError):
        DataProcessorPipeline.from_pretrained("user/repo/extra/path", config_filename="processor.json")

    # Test with a non-existent Hub repo
    with pytest.raises((FileNotFoundError, HfHubHTTPError)):
        DataProcessorPipeline.from_pretrained(
            "nonexistent-user/nonexistent-repo", config_filename="processor.json"
        )

    # Test with a local directory that exists but has no config files
    with tempfile.TemporaryDirectory() as tmp_dir, pytest.raises(FileNotFoundError):
        # Since the directory exists but has no config, it will raise FileNotFoundError
        DataProcessorPipeline.from_pretrained(tmp_dir, config_filename="processor.json")


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
    pipeline = DataProcessorPipeline(
        [MockStep()], to_transition=custom_to_transition, to_output=custom_to_output
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)

        # Load - should use default converters
        loaded = DataProcessorPipeline.from_pretrained(tmp_dir, config_filename="dataprocessorpipeline.json")

        # Verify it uses default converters by checking with standard batch format
        batch = {
            OBS_IMAGE: torch.randn(1, 3, 32, 32),
            ACTION: torch.randn(1, 7),
            REWARD: torch.tensor([1.0]),
            DONE: torch.tensor([False]),
            TRUNCATED: torch.tensor([False]),
            "info": {},
        }

        # Should work with standard format (wouldn't work with custom converter)
        result = loaded(batch)
        # With new behavior, default to_output is _default_transition_to_batch, so result is batch dict
        assert OBS_IMAGE in result


class NonCompliantStep:
    """Intentionally non-compliant: missing features."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return transition


class NonCallableStep(ProcessorStep):
    """Intentionally non-compliant: missing __call__."""

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def test_construction_rejects_step_without_call():
    """Test that DataProcessorPipeline rejects steps that don't inherit from ProcessorStep."""
    with pytest.raises(
        TypeError, match=r"Can't instantiate abstract class NonCallableStep with abstract method __call_"
    ):
        DataProcessorPipeline([NonCallableStep()])

    with pytest.raises(TypeError, match=r"must inherit from ProcessorStep"):
        DataProcessorPipeline([NonCompliantStep()])


@dataclass
class FeatureContractAddStep(ProcessorStep):
    """Adds a PolicyFeature"""

    key: str = "a"
    value: PolicyFeature = PolicyFeature(type=FeatureType.STATE, shape=(1,))

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.OBSERVATION][self.key] = self.value
        return features


@dataclass
class FeatureContractMutateStep(ProcessorStep):
    """Mutates a PolicyFeature"""

    key: str = "a"
    fn: Callable[[PolicyFeature | None], PolicyFeature] = identity_transition  # noqa: E731

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.OBSERVATION][self.key] = self.fn(
            features[PipelineFeatureType.OBSERVATION].get(self.key)
        )
        return features


@dataclass
class FeatureContractBadReturnStep(ProcessorStep):
    """Returns a non-dict"""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return ["not-a-dict"]


@dataclass
class FeatureContractRemoveStep(ProcessorStep):
    """Removes a PolicyFeature"""

    key: str

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.OBSERVATION].pop(self.key, None)
        return features


def test_features_orders_and_merges(policy_feature_factory):
    p = DataProcessorPipeline(
        [
            FeatureContractAddStep("a", policy_feature_factory(FeatureType.STATE, (1,))),
            FeatureContractMutateStep("a", lambda v: PolicyFeature(type=v.type, shape=(3,))),
            FeatureContractAddStep("b", policy_feature_factory(FeatureType.ENV, (2,))),
        ]
    )
    out = p.transform_features({PipelineFeatureType.OBSERVATION: {}})
    assert out[PipelineFeatureType.OBSERVATION]["a"].type == FeatureType.STATE and out[
        PipelineFeatureType.OBSERVATION
    ]["a"].shape == (3,)
    assert out[PipelineFeatureType.OBSERVATION]["b"].type == FeatureType.ENV and out[
        PipelineFeatureType.OBSERVATION
    ]["b"].shape == (2,)
    assert_contract_is_typed(out)


def test_features_respects_initial_without_mutation(policy_feature_factory):
    initial = {
        PipelineFeatureType.OBSERVATION: {
            "seed": policy_feature_factory(FeatureType.STATE, (7,)),
            "nested": policy_feature_factory(FeatureType.ENV, (0,)),
        }
    }
    p = DataProcessorPipeline(
        [
            FeatureContractMutateStep("seed", lambda v: PolicyFeature(type=v.type, shape=(v.shape[0] + 1,))),
            FeatureContractMutateStep(
                "nested", lambda v: PolicyFeature(type=v.type, shape=(v.shape[0] + 5,))
            ),
        ]
    )
    out = p.transform_features(initial_features=initial)

    assert out[PipelineFeatureType.OBSERVATION]["seed"].shape == (8,)
    assert out[PipelineFeatureType.OBSERVATION]["nested"].shape == (5,)
    # Initial dict must be preserved
    assert initial[PipelineFeatureType.OBSERVATION]["seed"].shape == (7,)
    assert initial[PipelineFeatureType.OBSERVATION]["nested"].shape == (0,)

    assert_contract_is_typed(out)


def test_features_execution_order_tracking():
    class Track(ProcessorStep):
        def __init__(self, label):
            self.label = label

        def __call__(self, transition: EnvTransition) -> EnvTransition:
            return transition

        def transform_features(
            self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
        ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
            code = {"A": 1, "B": 2, "C": 3}[self.label]
            pf = features[PipelineFeatureType.OBSERVATION].get(
                "order", PolicyFeature(type=FeatureType.ENV, shape=())
            )
            features[PipelineFeatureType.OBSERVATION]["order"] = PolicyFeature(
                type=pf.type, shape=pf.shape + (code,)
            )
            return features

    out = DataProcessorPipeline([Track("A"), Track("B"), Track("C")]).transform_features(
        initial_features={PipelineFeatureType.OBSERVATION: {}}
    )
    assert out[PipelineFeatureType.OBSERVATION]["order"].shape == (1, 2, 3)


def test_features_remove_key(policy_feature_factory):
    p = DataProcessorPipeline(
        [
            FeatureContractAddStep("a", policy_feature_factory(FeatureType.STATE, (1,))),
            FeatureContractRemoveStep("a"),
        ]
    )
    out = p.transform_features({PipelineFeatureType.OBSERVATION: {}})
    assert "a" not in out[PipelineFeatureType.OBSERVATION]


def test_features_remove_from_initial(policy_feature_factory):
    initial = {
        PipelineFeatureType.OBSERVATION: {
            "keep": policy_feature_factory(FeatureType.STATE, (1,)),
            "drop": policy_feature_factory(FeatureType.STATE, (1,)),
        },
    }
    p = DataProcessorPipeline([FeatureContractRemoveStep("drop")])
    out = p.transform_features(initial_features=initial)
    assert (
        "drop" not in out[PipelineFeatureType.OBSERVATION]
        and out[PipelineFeatureType.OBSERVATION]["keep"] == initial[PipelineFeatureType.OBSERVATION]["keep"]
    )


@dataclass
class AddActionEEAndJointFeatures(ProcessorStep):
    """Adds both EE and JOINT action features."""

    def __call__(self, tr):
        return tr

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # EE features
        features[PipelineFeatureType.ACTION]["action.ee.x"] = float
        features[PipelineFeatureType.ACTION]["action.ee.y"] = float
        # JOINT features
        features[PipelineFeatureType.ACTION]["action.j1.pos"] = float
        features[PipelineFeatureType.ACTION]["action.j2.pos"] = float
        return features


@dataclass
class AddObservationStateFeatures(ProcessorStep):
    """Adds state features (and optionally an image spec to test precedence)."""

    add_front_image: bool = False
    front_image_shape: tuple = (240, 320, 3)

    def __call__(self, tr):
        return tr

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # State features (mix EE and a joint state)
        features[PipelineFeatureType.OBSERVATION][f"{OBS_STATE}.ee.x"] = float
        features[PipelineFeatureType.OBSERVATION][f"{OBS_STATE}.j1.pos"] = float
        if self.add_front_image:
            features[PipelineFeatureType.OBSERVATION][f"{OBS_IMAGES}.front"] = self.front_image_shape
        return features


def test_aggregate_joint_action_only():
    rp = DataProcessorPipeline([AddActionEEAndJointFeatures()])
    initial = {PipelineFeatureType.OBSERVATION: {"front": (480, 640, 3)}, PipelineFeatureType.ACTION: {}}

    out = aggregate_pipeline_dataset_features(
        pipeline=rp,
        initial_features=initial,
        use_videos=True,
        patterns=["action.j1.pos", "action.j2.pos"],
    )

    # Expect only ACTION with joint names
    assert ACTION in out and OBS_STATE not in out
    assert out[ACTION]["dtype"] == "float32"
    assert set(out[ACTION]["names"]) == {"j1.pos", "j2.pos"}
    assert out[ACTION]["shape"] == (len(out[ACTION]["names"]),)


def test_aggregate_ee_action_and_observation_with_videos():
    rp = DataProcessorPipeline([AddActionEEAndJointFeatures(), AddObservationStateFeatures()])
    initial = {"front": (480, 640, 3), "side": (720, 1280, 3)}

    out = aggregate_pipeline_dataset_features(
        pipeline=rp,
        initial_features={PipelineFeatureType.OBSERVATION: initial, PipelineFeatureType.ACTION: {}},
        use_videos=True,
        patterns=["action.ee", OBS_STATE],
    )

    # Action should pack only EE names
    assert ACTION in out
    assert set(out[ACTION]["names"]) == {"ee.x", "ee.y"}
    assert out[ACTION]["dtype"] == "float32"

    # Observation state should pack both ee.x and j1.pos as a vector
    assert OBS_STATE in out
    assert set(out[OBS_STATE]["names"]) == {"ee.x", "j1.pos"}
    assert out[OBS_STATE]["dtype"] == "float32"

    # Cameras from initial_features appear as videos
    for cam in ("front", "side"):
        key = f"{OBS_IMAGES}.{cam}"
        assert key in out
        assert out[key]["dtype"] == "video"
        assert out[key]["shape"] == initial[cam]
        assert out[key]["names"] == ["height", "width", "channels"]


def test_aggregate_both_action_types():
    rp = DataProcessorPipeline([AddActionEEAndJointFeatures()])
    out = aggregate_pipeline_dataset_features(
        pipeline=rp,
        initial_features={PipelineFeatureType.ACTION: {}, PipelineFeatureType.OBSERVATION: {}},
        use_videos=True,
        patterns=["action.ee", "action.j1", "action.j2.pos"],
    )

    assert ACTION in out
    expected = {"ee.x", "ee.y", "j1.pos", "j2.pos"}
    assert set(out[ACTION]["names"]) == expected
    assert out[ACTION]["shape"] == (len(expected),)


def test_aggregate_images_when_use_videos_false():
    rp = DataProcessorPipeline([AddObservationStateFeatures(add_front_image=True)])
    initial = {"back": (480, 640, 3)}

    out = aggregate_pipeline_dataset_features(
        pipeline=rp,
        initial_features={PipelineFeatureType.ACTION: {}, PipelineFeatureType.OBSERVATION: initial},
        use_videos=False,  # expect "image" dtype
        patterns=None,
    )

    key = f"{OBS_IMAGES}.back"
    key_front = f"{OBS_IMAGES}.front"
    assert key not in out
    assert key_front not in out


def test_aggregate_images_when_use_videos_true():
    rp = DataProcessorPipeline([AddObservationStateFeatures(add_front_image=True)])
    initial = {"back": (480, 640, 3)}

    out = aggregate_pipeline_dataset_features(
        pipeline=rp,
        initial_features={PipelineFeatureType.OBSERVATION: initial, PipelineFeatureType.ACTION: {}},
        use_videos=True,
        patterns=None,
    )

    key = f"{OBS_IMAGES}.front"
    key_back = f"{OBS_IMAGES}.back"
    assert key in out
    assert key_back in out
    assert out[key]["dtype"] == "video"
    assert out[key_back]["dtype"] == "video"
    assert out[key_back]["shape"] == initial["back"]


def test_initial_camera_not_overridden_by_step_image():
    # Step explicitly sets a different front image shape; initial has another shape.
    # aggregate_pipeline_dataset_features should keep the step's value (setdefault behavior on initial cams).
    rp = DataProcessorPipeline(
        [AddObservationStateFeatures(add_front_image=True, front_image_shape=(240, 320, 3))]
    )
    initial = {"front": (480, 640, 3)}  # should NOT override the step-provided (240, 320, 3)

    out = aggregate_pipeline_dataset_features(
        pipeline=rp,
        initial_features={PipelineFeatureType.ACTION: {}, PipelineFeatureType.OBSERVATION: initial},
        use_videos=True,
        patterns=[f"{OBS_IMAGES}.front"],
    )

    key = f"{OBS_IMAGES}.front"
    assert key in out
    assert out[key]["shape"] == (240, 320, 3)  # from the step, not from initial

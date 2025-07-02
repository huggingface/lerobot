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
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from lerobot.processor.pipeline import RobotPipeline, EnvTransition, PipelineStep


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
        
        if comp_data is None:
            comp_data = {}
        else:
            comp_data = dict(comp_data)  # Make a copy
        
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
    pipeline = RobotPipeline()
    
    transition = (None, None, 0.0, False, False, {}, {})
    result = pipeline(transition)
    
    assert result == transition
    assert len(pipeline) == 0

def test_single_step_pipeline():
    """Test pipeline with a single step."""
    step = MockStep("test_step")
    pipeline = RobotPipeline([step])
    
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
    pipeline = RobotPipeline([step1, step2])
    
    transition = (None, None, 0.0, False, False, {}, {})
    result = pipeline(transition)
    
    assert len(pipeline) == 2
    assert result[6]["step1_counter"] == 0
    assert result[6]["step2_counter"] == 0

def test_invalid_transition_format():
    """Test pipeline with invalid transition format."""
    pipeline = RobotPipeline([MockStep()])
    
    # Test with wrong number of elements
    with pytest.raises(ValueError, match="EnvTransition must be a 7-tuple"):
        pipeline((None, None, 0.0))  # Only 3 elements
    
    # Test with wrong type
    with pytest.raises(ValueError, match="EnvTransition must be a 7-tuple"):
        pipeline("not a tuple")

def test_step_through():
    """Test step_through method."""
    step1 = MockStep("step1")
    step2 = MockStep("step2")
    pipeline = RobotPipeline([step1, step2])
    
    transition = (None, None, 0.0, False, False, {}, {})
    
    results = list(pipeline.step_through(transition))
    
    assert len(results) == 3  # Original + 2 steps
    assert results[0] == transition  # Original
    assert "step1_counter" in results[1][6]  # After step1
    assert "step2_counter" in results[2][6]  # After step2

def test_indexing():
    """Test pipeline indexing."""
    step1 = MockStep("step1")
    step2 = MockStep("step2")
    pipeline = RobotPipeline([step1, step2])
    
    # Test integer indexing
    assert pipeline[0] is step1
    assert pipeline[1] is step2
    
    # Test slice indexing
    sub_pipeline = pipeline[0:1]
    assert isinstance(sub_pipeline, RobotPipeline)
    assert len(sub_pipeline) == 1
    assert sub_pipeline[0] is step1

def test_hooks():
    """Test before/after step hooks."""
    step = MockStep("test_step")
    pipeline = RobotPipeline([step])
    
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
    pipeline = RobotPipeline([step])
    
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
    pipeline = RobotPipeline([step])
    
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
    pipeline = RobotPipeline([step1, step2])
    
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
    
    pipeline = RobotPipeline([step1, step2], name="TestPipeline", seed=42)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save pipeline
        pipeline.save_pretrained(tmp_dir)
        
        # Check files were created
        config_path = Path(tmp_dir) / "pipeline.json"
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
        loaded_pipeline = RobotPipeline.from_pretrained(tmp_dir)
        
        assert loaded_pipeline.name == "TestPipeline"
        assert loaded_pipeline.seed == 42
        assert len(loaded_pipeline) == 2
        
        # Check that counter was restored from config
        assert loaded_pipeline.steps[0].counter == 5
        assert loaded_pipeline.steps[1].counter == 10

def test_step_without_optional_methods():
    """Test pipeline with steps that don't implement optional methods."""
    step = MockStepWithoutOptionalMethods(multiplier=3.0)
    pipeline = RobotPipeline([step])
    
    transition = (None, None, 2.0, False, False, {}, {})
    result = pipeline(transition)
    
    assert result[2] == 6.0  # 2.0 * 3.0
    
    # Reset should work even if step doesn't implement reset
    pipeline.reset()
    
    # Save/load should work even without optional methods
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)
        loaded_pipeline = RobotPipeline.from_pretrained(tmp_dir)
        assert len(loaded_pipeline) == 1

def test_mixed_json_and_tensor_state():
    """Test step with both JSON attributes and tensor state."""
    step = MockStepWithTensorState(name="stats", learning_rate=0.05, window_size=5)
    pipeline = RobotPipeline([step])
    
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
        config_path = Path(tmp_dir) / "pipeline.json" 
        state_path = Path(tmp_dir) / "step_0.safetensors"
        assert config_path.exists()
        assert state_path.exists()
        
        # Load and verify
        loaded_pipeline = RobotPipeline.from_pretrained(tmp_dir)
        loaded_step = loaded_pipeline.steps[0]
        
        # Check JSON attributes were restored
        assert loaded_step.name == "stats"
        assert loaded_step.learning_rate == 0.05
        assert loaded_step.window_size == 5
        
        # Check tensor state was restored
        assert loaded_step.running_count.item() == 10
        assert torch.allclose(loaded_step.running_mean, step.running_mean)

 
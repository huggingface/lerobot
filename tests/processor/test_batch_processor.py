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

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DataProcessorPipeline,
    ProcessorStepRegistry,
    TransitionKey,
)
from lerobot.processor.converters import create_transition, identity_transition
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE


def test_state_1d_to_2d():
    """Test that 1D state tensors get unsqueezed to 2D."""
    processor = AddBatchDimensionProcessorStep()

    # Test observation.state
    state_1d = torch.randn(7)
    observation = {OBS_STATE: state_1d}
    transition = create_transition(observation=observation, action=torch.empty(0))

    result = processor(transition)

    processed_state = result[TransitionKey.OBSERVATION][OBS_STATE]
    assert processed_state.shape == (1, 7)
    assert torch.allclose(processed_state.squeeze(0), state_1d)


def test_env_state_1d_to_2d():
    """Test that 1D environment state tensors get unsqueezed to 2D."""
    processor = AddBatchDimensionProcessorStep()

    # Test observation.environment_state
    env_state_1d = torch.randn(10)
    observation = {OBS_ENV_STATE: env_state_1d}
    transition = create_transition(observation=observation, action=torch.empty(0))

    result = processor(transition)

    processed_env_state = result[TransitionKey.OBSERVATION][OBS_ENV_STATE]
    assert processed_env_state.shape == (1, 10)
    assert torch.allclose(processed_env_state.squeeze(0), env_state_1d)


def test_image_3d_to_4d():
    """Test that 3D image tensors get unsqueezed to 4D."""
    processor = AddBatchDimensionProcessorStep()

    # Test observation.image
    image_3d = torch.randn(224, 224, 3)
    observation = {OBS_IMAGE: image_3d}
    transition = create_transition(observation=observation, action=torch.empty(0))

    result = processor(transition)

    processed_image = result[TransitionKey.OBSERVATION][OBS_IMAGE]
    assert processed_image.shape == (1, 224, 224, 3)
    assert torch.allclose(processed_image.squeeze(0), image_3d)


def test_multiple_images_3d_to_4d():
    """Test that 3D image tensors in observation.images.* get unsqueezed to 4D."""
    processor = AddBatchDimensionProcessorStep()

    # Test observation.images.camera1 and observation.images.camera2
    image1_3d = torch.randn(64, 64, 3)
    image2_3d = torch.randn(128, 128, 3)
    observation = {
        f"{OBS_IMAGES}.camera1": image1_3d,
        f"{OBS_IMAGES}.camera2": image2_3d,
    }
    transition = create_transition(observation=observation, action=torch.empty(0))

    result = processor(transition)

    processed_obs = result[TransitionKey.OBSERVATION]
    processed_image1 = processed_obs[f"{OBS_IMAGES}.camera1"]
    processed_image2 = processed_obs[f"{OBS_IMAGES}.camera2"]

    assert processed_image1.shape == (1, 64, 64, 3)
    assert processed_image2.shape == (1, 128, 128, 3)
    assert torch.allclose(processed_image1.squeeze(0), image1_3d)
    assert torch.allclose(processed_image2.squeeze(0), image2_3d)


def test_already_batched_tensors_unchanged():
    """Test that already batched tensors remain unchanged."""
    processor = AddBatchDimensionProcessorStep()

    # Create already batched tensors
    state_2d = torch.randn(1, 7)
    env_state_2d = torch.randn(1, 10)
    image_4d = torch.randn(1, 224, 224, 3)

    observation = {
        OBS_STATE: state_2d,
        OBS_ENV_STATE: env_state_2d,
        OBS_IMAGE: image_4d,
    }
    transition = create_transition(observation=observation, action=torch.empty(0))

    result = processor(transition)

    processed_obs = result[TransitionKey.OBSERVATION]

    # Should remain unchanged
    assert torch.allclose(processed_obs[OBS_STATE], state_2d)
    assert torch.allclose(processed_obs[OBS_ENV_STATE], env_state_2d)
    assert torch.allclose(processed_obs[OBS_IMAGE], image_4d)


def test_higher_dimensional_tensors_unchanged():
    """Test that tensors with more dimensions than expected remain unchanged."""
    processor = AddBatchDimensionProcessorStep()

    # Create tensors with more dimensions
    state_3d = torch.randn(2, 7, 5)  # More than 1D
    image_5d = torch.randn(2, 3, 224, 224, 1)  # More than 3D

    observation = {
        OBS_STATE: state_3d,
        OBS_IMAGE: image_5d,
    }
    transition = create_transition(observation=observation, action=torch.empty(0))

    result = processor(transition)

    processed_obs = result[TransitionKey.OBSERVATION]

    # Should remain unchanged
    assert torch.allclose(processed_obs[OBS_STATE], state_3d)
    assert torch.allclose(processed_obs[OBS_IMAGE], image_5d)


def test_non_tensor_values_unchanged():
    """Test that non-tensor values in observations remain unchanged."""
    processor = AddBatchDimensionProcessorStep()

    observation = {
        OBS_STATE: [1, 2, 3],  # List, not tensor
        OBS_IMAGE: "not_a_tensor",  # String
        "custom_key": 42,  # Integer
        "another_key": {"nested": "dict"},  # Dict
    }
    transition = create_transition(observation=observation, action=torch.empty(0))

    result = processor(transition)

    processed_obs = result[TransitionKey.OBSERVATION]

    # Should remain unchanged
    assert processed_obs[OBS_STATE] == [1, 2, 3]
    assert processed_obs[OBS_IMAGE] == "not_a_tensor"
    assert processed_obs["custom_key"] == 42
    assert processed_obs["another_key"] == {"nested": "dict"}


def test_none_observation():
    """Test processor handles None observation gracefully."""
    processor = AddBatchDimensionProcessorStep()

    transition = create_transition(observation={}, action=torch.empty(0))
    result = processor(transition)

    assert result[TransitionKey.OBSERVATION] == {}


def test_empty_observation():
    """Test processor handles empty observation dict."""
    processor = AddBatchDimensionProcessorStep()

    observation = {}
    transition = create_transition(observation=observation, action=torch.empty(0))

    result = processor(transition)

    assert result[TransitionKey.OBSERVATION] == {}


def test_mixed_observation():
    """Test processor with mixed observation containing various types and dimensions."""
    processor = AddBatchDimensionProcessorStep()

    state_1d = torch.randn(5)
    env_state_2d = torch.randn(1, 8)  # Already batched
    image_3d = torch.randn(32, 32, 3)
    other_tensor = torch.randn(3, 3, 3, 3)  # 4D, should be unchanged

    observation = {
        OBS_STATE: state_1d,
        OBS_ENV_STATE: env_state_2d,
        OBS_IMAGE: image_3d,
        f"{OBS_IMAGES}.front": torch.randn(64, 64, 3),  # 3D, should be batched
        f"{OBS_IMAGES}.back": torch.randn(1, 64, 64, 3),  # 4D, should be unchanged
        "other_tensor": other_tensor,
        "non_tensor": "string_value",
    }
    transition = create_transition(observation=observation, action=torch.empty(0))

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check transformations
    assert processed_obs[OBS_STATE].shape == (1, 5)
    assert processed_obs[OBS_ENV_STATE].shape == (1, 8)  # Unchanged
    assert processed_obs[OBS_IMAGE].shape == (1, 32, 32, 3)
    assert processed_obs[f"{OBS_IMAGES}.front"].shape == (1, 64, 64, 3)
    assert processed_obs[f"{OBS_IMAGES}.back"].shape == (1, 64, 64, 3)  # Unchanged
    assert processed_obs["other_tensor"].shape == (3, 3, 3, 3)  # Unchanged
    assert processed_obs["non_tensor"] == "string_value"  # Unchanged


def test_integration_with_robot_processor():
    """Test AddBatchDimensionProcessorStep integration with RobotProcessor."""
    to_batch_processor = AddBatchDimensionProcessorStep()
    pipeline = DataProcessorPipeline(
        [to_batch_processor], to_transition=identity_transition, to_output=identity_transition
    )

    # Create unbatched observation
    observation = {
        OBS_STATE: torch.randn(7),
        OBS_IMAGE: torch.randn(224, 224, 3),
    }
    transition = create_transition(observation=observation, action=torch.empty(0))

    result = pipeline(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    assert processed_obs[OBS_STATE].shape == (1, 7)
    assert processed_obs[OBS_IMAGE].shape == (1, 224, 224, 3)


def test_serialization_methods():
    """Test get_config, state_dict, load_state_dict, and reset methods."""
    processor = AddBatchDimensionProcessorStep()

    # Test get_config
    config = processor.get_config()
    assert isinstance(config, dict)
    assert config == {}

    # Test state_dict
    state = processor.state_dict()
    assert isinstance(state, dict)
    assert state == {}

    # Test load_state_dict (should not raise an error)
    processor.load_state_dict({})

    # Test reset (should not raise an error)
    processor.reset()


def test_save_and_load_pretrained():
    """Test saving and loading AddBatchDimensionProcessorStep with RobotProcessor."""
    processor = AddBatchDimensionProcessorStep()
    pipeline = DataProcessorPipeline(
        [processor], name="BatchPipeline", to_transition=identity_transition, to_output=identity_transition
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save pipeline
        pipeline.save_pretrained(tmp_dir)

        # Check config file exists
        config_path = Path(tmp_dir) / "batchpipeline.json"
        assert config_path.exists()

        # Load pipeline
        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir,
            config_filename="batchpipeline.json",
            to_transition=identity_transition,
            to_output=identity_transition,
        )

        assert loaded_pipeline.name == "BatchPipeline"
        assert len(loaded_pipeline) == 1
        assert isinstance(loaded_pipeline.steps[0], AddBatchDimensionProcessorStep)

        # Test functionality of loaded processor
        observation = {OBS_STATE: torch.randn(5)}
        transition = create_transition(observation=observation, action=torch.empty(0))

        result = loaded_pipeline(transition)
        assert result[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 5)


def test_registry_functionality():
    """Test that AddBatchDimensionProcessorStep is properly registered."""
    # Check that the processor is registered
    registered_class = ProcessorStepRegistry.get("to_batch_processor")
    assert registered_class is AddBatchDimensionProcessorStep

    # Check that it's in the list of registered processors
    assert "to_batch_processor" in ProcessorStepRegistry.list()


def test_registry_based_save_load():
    """Test saving and loading using registry name."""
    processor = AddBatchDimensionProcessorStep()
    pipeline = DataProcessorPipeline(
        [processor], to_transition=identity_transition, to_output=identity_transition
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)
        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            tmp_dir,
            config_filename="dataprocessorpipeline.json",
            to_transition=identity_transition,
            to_output=identity_transition,
        )

        # Verify the loaded processor works
        observation = {
            OBS_STATE: torch.randn(3),
            OBS_IMAGE: torch.randn(100, 100, 3),
        }
        transition = create_transition(observation=observation, action=torch.empty(0))

        result = loaded_pipeline(transition)
        processed_obs = result[TransitionKey.OBSERVATION]

        assert processed_obs[OBS_STATE].shape == (1, 3)
        assert processed_obs[OBS_IMAGE].shape == (1, 100, 100, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_compatibility():
    """Test processor works with tensors on different devices."""
    processor = AddBatchDimensionProcessorStep()

    # Create tensors on GPU
    state_1d = torch.randn(7, device="cuda")
    image_3d = torch.randn(64, 64, 3, device="cuda")

    observation = {
        OBS_STATE: state_1d,
        OBS_IMAGE: image_3d,
    }
    transition = create_transition(observation=observation, action=torch.empty(0))

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check shapes and that tensors stayed on GPU
    assert processed_obs[OBS_STATE].shape == (1, 7)
    assert processed_obs[OBS_IMAGE].shape == (1, 64, 64, 3)
    assert processed_obs[OBS_STATE].device.type == "cuda"
    assert processed_obs[OBS_IMAGE].device.type == "cuda"


def test_processor_preserves_other_transition_keys():
    """Test that processor only modifies observation and preserves other transition keys."""
    processor = AddBatchDimensionProcessorStep()

    action = torch.randn(5)
    reward = 1.5
    done = True
    truncated = False
    info = {"step": 10}
    comp_data = {"extra": "data"}

    observation = {OBS_STATE: torch.randn(7)}

    transition = create_transition(
        observation=observation,
        action=action,
        reward=reward,
        done=done,
        truncated=truncated,
        info=info,
        complementary_data=comp_data,
    )

    result = processor(transition)

    # Check that non-observation keys are preserved
    assert torch.allclose(result[TransitionKey.ACTION], action)
    assert result[TransitionKey.REWARD] == reward
    assert result[TransitionKey.DONE] == done
    assert result[TransitionKey.TRUNCATED] == truncated
    assert result[TransitionKey.INFO] == info
    assert result[TransitionKey.COMPLEMENTARY_DATA] == comp_data

    # Check that observation was processed
    assert result[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 7)


def test_edge_case_zero_dimensional_tensors():
    """Test processor handles 0D tensors (scalars) correctly."""
    processor = AddBatchDimensionProcessorStep()

    # 0D tensors should not be modified
    scalar_tensor = torch.tensor(42.0)

    observation = {
        OBS_STATE: scalar_tensor,
        "scalar_value": scalar_tensor,
    }
    transition = create_transition(observation=observation, action=torch.empty(0))

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # 0D tensors should remain unchanged
    assert torch.allclose(processed_obs[OBS_STATE], scalar_tensor)
    assert torch.allclose(processed_obs["scalar_value"], scalar_tensor)


# Action-specific tests
def test_action_1d_to_2d():
    """Test that 1D action tensors get batch dimension added."""
    processor = AddBatchDimensionProcessorStep()

    # Create 1D action tensor
    action_1d = torch.randn(4)
    transition = create_transition(observation={}, action=action_1d)

    result = processor(transition)

    # Should add batch dimension
    assert result[TransitionKey.ACTION].shape == (1, 4)
    assert torch.equal(result[TransitionKey.ACTION][0], action_1d)


def test_action_already_batched():
    """Test that already batched action tensors remain unchanged."""
    processor = AddBatchDimensionProcessorStep()

    # Test various batch sizes
    action_batched_1 = torch.randn(1, 4)
    action_batched_5 = torch.randn(5, 4)

    # Single batch
    transition = create_transition(action=action_batched_1, observation={})
    result = processor(transition)
    assert torch.equal(result[TransitionKey.ACTION], action_batched_1)

    # Multiple batch
    transition = create_transition(action=action_batched_5, observation={})
    result = processor(transition)
    assert torch.equal(result[TransitionKey.ACTION], action_batched_5)


def test_action_higher_dimensional():
    """Test that higher dimensional action tensors remain unchanged."""
    processor = AddBatchDimensionProcessorStep()

    # 3D action tensor (e.g., sequence of actions)
    action_3d = torch.randn(2, 4, 3)
    transition = create_transition(action=action_3d, observation={})
    result = processor(transition)
    assert torch.equal(result[TransitionKey.ACTION], action_3d)

    # 4D action tensor
    action_4d = torch.randn(2, 10, 4, 3)
    transition = create_transition(action=action_4d, observation={})
    result = processor(transition)
    assert torch.equal(result[TransitionKey.ACTION], action_4d)


def test_action_scalar_tensor():
    """Test that scalar (0D) action tensors remain unchanged."""
    processor = AddBatchDimensionProcessorStep()

    action_scalar = torch.tensor(1.5)
    transition = create_transition(action=action_scalar, observation={})
    result = processor(transition)

    # Should remain scalar
    assert result[TransitionKey.ACTION].dim() == 0
    assert torch.equal(result[TransitionKey.ACTION], action_scalar)


def test_action_non_tensor_raises_error():
    """Test that non-tensor actions raise ValueError for PolicyAction processors."""
    processor = AddBatchDimensionProcessorStep()

    # List action should raise error
    action_list = [0.1, 0.2, 0.3, 0.4]
    transition = create_transition(action=action_list)
    with pytest.raises(ValueError, match="Action should be a PolicyAction type"):
        processor(transition)

    # Numpy array action should raise error
    action_numpy = np.array([1, 2, 3, 4])
    transition = create_transition(action=action_numpy)
    with pytest.raises(ValueError, match="Action should be a PolicyAction type"):
        processor(transition)

    # String action should raise error
    action_string = "forward"
    transition = create_transition(action=action_string)
    with pytest.raises(ValueError, match="Action should be a PolicyAction type"):
        processor(transition)

    # Dict action should raise error
    action_dict = {"linear": [0.5, 0.0], "angular": 0.2}
    transition = create_transition(action=action_dict)
    with pytest.raises(ValueError, match="Action should be a PolicyAction type"):
        processor(transition)


def test_action_none():
    """Test that empty action tensor is handled correctly."""
    processor = AddBatchDimensionProcessorStep()

    transition = create_transition(action=torch.empty(0), observation={})
    result = processor(transition)
    # Empty 1D tensor becomes empty 2D tensor with batch dimension
    assert result[TransitionKey.ACTION].shape == (1, 0)


def test_action_with_observation():
    """Test action processing together with observation processing."""
    processor = AddBatchDimensionProcessorStep()

    # Both need batching
    observation = {
        OBS_STATE: torch.randn(7),
        OBS_IMAGE: torch.randn(64, 64, 3),
    }
    action = torch.randn(4)

    transition = create_transition(observation=observation, action=action)
    result = processor(transition)

    # Both should be batched
    assert result[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 7)
    assert result[TransitionKey.OBSERVATION][OBS_IMAGE].shape == (1, 64, 64, 3)
    assert result[TransitionKey.ACTION].shape == (1, 4)


def test_action_different_sizes():
    """Test action processing with various action dimensions."""
    processor = AddBatchDimensionProcessorStep()

    # Different action sizes (robot with different DOF)
    action_sizes = [1, 2, 4, 7, 10, 20]

    for size in action_sizes:
        action = torch.randn(size)
        transition = create_transition(action=action, observation={})
        result = processor(transition)

        assert result[TransitionKey.ACTION].shape == (1, size)
        assert torch.equal(result[TransitionKey.ACTION][0], action)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_action_device_compatibility():
    """Test action processing on different devices."""
    processor = AddBatchDimensionProcessorStep()

    # CUDA action
    action_cuda = torch.randn(4, device="cuda")
    transition = create_transition(action=action_cuda, observation={})
    result = processor(transition)

    assert result[TransitionKey.ACTION].shape == (1, 4)
    assert result[TransitionKey.ACTION].device.type == "cuda"

    # CPU action
    action_cpu = torch.randn(4, device="cpu")
    transition = create_transition(action=action_cpu, observation={})
    result = processor(transition)

    assert result[TransitionKey.ACTION].shape == (1, 4)
    assert result[TransitionKey.ACTION].device.type == "cpu"


def test_action_dtype_preservation():
    """Test that action dtype is preserved during processing."""
    processor = AddBatchDimensionProcessorStep()

    # Different dtypes
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]

    for dtype in dtypes:
        action = torch.randn(4).to(dtype)
        transition = create_transition(action=action, observation={})
        result = processor(transition)

        assert result[TransitionKey.ACTION].dtype == dtype
        assert result[TransitionKey.ACTION].shape == (1, 4)


def test_empty_action_tensor():
    """Test handling of empty action tensors."""
    processor = AddBatchDimensionProcessorStep()

    # Empty 1D tensor
    action_empty = torch.tensor([])
    transition = create_transition(action=action_empty, observation={})
    result = processor(transition)

    # Should add batch dimension even to empty tensor
    assert result[TransitionKey.ACTION].shape == (1, 0)

    # Empty 2D tensor (already batched)
    action_empty_2d = torch.randn(1, 0)
    transition = create_transition(action=action_empty_2d, observation={})
    result = processor(transition)

    # Should remain unchanged
    assert result[TransitionKey.ACTION].shape == (1, 0)


# Task-specific tests
def test_task_string_to_list():
    """Test that string tasks get wrapped in lists to add batch dimension."""
    processor = AddBatchDimensionProcessorStep()

    # Create complementary data with string task
    complementary_data = {"task": "pick_cube"}
    transition = create_transition(
        action=torch.empty(0), observation={}, complementary_data=complementary_data
    )

    result = processor(transition)

    # String task should be wrapped in list
    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
    assert processed_comp_data["task"] == ["pick_cube"]
    assert isinstance(processed_comp_data["task"], list)
    assert len(processed_comp_data["task"]) == 1


def test_task_string_validation():
    """Test that only string and list of strings are valid task values."""
    processor = AddBatchDimensionProcessorStep()

    # Valid string task - should be converted to list
    complementary_data = {"task": "valid_task"}
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )
    result = processor(transition)
    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
    assert processed_comp_data["task"] == ["valid_task"]

    # Valid list of strings - should remain unchanged
    complementary_data = {"task": ["task1", "task2"]}
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )
    result = processor(transition)
    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
    assert processed_comp_data["task"] == ["task1", "task2"]


def test_task_list_of_strings():
    """Test that lists of strings remain unchanged (already batched)."""
    processor = AddBatchDimensionProcessorStep()

    # Test various list of strings
    test_lists = [
        ["pick_cube"],  # Single string in list
        ["pick_cube", "place_cube"],  # Multiple strings
        ["task1", "task2", "task3"],  # Three strings
        [],  # Empty list
        [""],  # List with empty string
        ["task with spaces", "task_with_underscores"],  # Mixed formats
    ]

    for task_list in test_lists:
        complementary_data = {"task": task_list}
        transition = create_transition(
            complementary_data=complementary_data, observation={}, action=torch.empty(0)
        )

        result = processor(transition)

        # Should remain unchanged since it's already a list
        processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
        assert processed_comp_data["task"] == task_list
        assert isinstance(processed_comp_data["task"], list)


def test_complementary_data_none():
    """Test processor handles None complementary_data gracefully."""
    processor = AddBatchDimensionProcessorStep()

    transition = create_transition(complementary_data=None, action=torch.empty(0), observation={})
    result = processor(transition)

    assert result[TransitionKey.COMPLEMENTARY_DATA] == {}


def test_complementary_data_empty():
    """Test processor handles empty complementary_data dict."""
    processor = AddBatchDimensionProcessorStep()

    complementary_data = {}
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )

    result = processor(transition)

    assert result[TransitionKey.COMPLEMENTARY_DATA] == {}


def test_complementary_data_no_task():
    """Test processor handles complementary_data without task field."""
    processor = AddBatchDimensionProcessorStep()

    complementary_data = {
        "episode_id": 123,
        "timestamp": 1234567890.0,
        "extra_info": "some data",
    }
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )

    result = processor(transition)

    # Should remain unchanged
    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
    assert processed_comp_data == complementary_data


def test_complementary_data_mixed():
    """Test processor with mixed complementary_data containing task and other fields."""
    processor = AddBatchDimensionProcessorStep()

    complementary_data = {
        "task": "stack_blocks",
        "episode_id": 456,
        "difficulty": "hard",
        "metadata": {"scene": "kitchen"},
    }
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )

    result = processor(transition)

    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]

    # Task should be batched
    assert processed_comp_data["task"] == ["stack_blocks"]

    # Other fields should remain unchanged
    assert processed_comp_data["episode_id"] == 456
    assert processed_comp_data["difficulty"] == "hard"
    assert processed_comp_data["metadata"] == {"scene": "kitchen"}


def test_task_with_observation_and_action():
    """Test task processing together with observation and action processing."""
    processor = AddBatchDimensionProcessorStep()

    # All components need batching
    observation = {
        OBS_STATE: torch.randn(5),
        OBS_IMAGE: torch.randn(32, 32, 3),
    }
    action = torch.randn(4)
    complementary_data = {"task": "navigate_to_goal"}

    transition = create_transition(
        observation=observation, action=action, complementary_data=complementary_data
    )

    result = processor(transition)

    # All should be batched
    assert result[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 5)
    assert result[TransitionKey.OBSERVATION][OBS_IMAGE].shape == (1, 32, 32, 3)
    assert result[TransitionKey.ACTION].shape == (1, 4)
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == ["navigate_to_goal"]


def test_task_comprehensive_string_cases():
    """Test task processing with comprehensive string cases and edge cases."""
    processor = AddBatchDimensionProcessorStep()

    # Test various string formats
    string_tasks = [
        "pick_and_place",
        "navigate",
        "open_drawer",
        "",  # Empty string (valid but edge case)
        "task with spaces",
        "task_with_underscores",
        "task-with-dashes",
        "UPPERCASE_TASK",
        "MixedCaseTask",
        "task123",
        "数字任务",  # Unicode task
        "🤖 robot task",  # Emoji in task
        "task\nwith\nnewlines",  # Special characters
        "task\twith\ttabs",
        "task with 'quotes'",
        'task with "double quotes"',
    ]

    # Test that all string tasks get properly batched
    for task in string_tasks:
        complementary_data = {"task": task}
        transition = create_transition(
            complementary_data=complementary_data, observation={}, action=torch.empty(0)
        )

        result = processor(transition)

        processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
        assert processed_comp_data["task"] == [task]
        assert isinstance(processed_comp_data["task"], list)
        assert len(processed_comp_data["task"]) == 1

    # Test various list of strings (should remain unchanged)
    list_tasks = [
        ["single_task"],
        ["task1", "task2"],
        ["pick", "place", "navigate"],
        [],  # Empty list
        [""],  # List with empty string
        ["task with spaces", "task_with_underscores", "UPPERCASE"],
        ["🤖 task", "数字任务", "normal_task"],  # Mixed formats
    ]

    for task_list in list_tasks:
        complementary_data = {"task": task_list}
        transition = create_transition(
            complementary_data=complementary_data, observation={}, action=torch.empty(0)
        )

        result = processor(transition)

        processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
        assert processed_comp_data["task"] == task_list
        assert isinstance(processed_comp_data["task"], list)


def test_task_preserves_other_keys():
    """Test that task processing preserves other keys in complementary_data."""
    processor = AddBatchDimensionProcessorStep()

    complementary_data = {
        "task": "clean_table",
        "robot_id": "robot_123",
        "motor_id": "motor_456",
        "config": {"speed": "slow", "precision": "high"},
        "metrics": [1.0, 2.0, 3.0],
    }
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )

    result = processor(transition)

    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]

    # Task should be processed
    assert processed_comp_data["task"] == ["clean_table"]

    # All other keys should be preserved exactly
    assert processed_comp_data["robot_id"] == "robot_123"
    assert processed_comp_data["motor_id"] == "motor_456"
    assert processed_comp_data["config"] == {"speed": "slow", "precision": "high"}
    assert processed_comp_data["metrics"] == [1.0, 2.0, 3.0]


# Index and task_index specific tests
def test_index_scalar_to_1d():
    """Test that 0D index tensor gets unsqueezed to 1D."""
    processor = AddBatchDimensionProcessorStep()

    # Create 0D index tensor (scalar)
    index_0d = torch.tensor(42, dtype=torch.int64)
    complementary_data = {"index": index_0d}
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )

    result = processor(transition)

    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
    assert processed_comp_data["index"].shape == (1,)
    assert processed_comp_data["index"].dtype == torch.int64
    assert processed_comp_data["index"][0] == 42


def test_task_index_scalar_to_1d():
    """Test that 0D task_index tensor gets unsqueezed to 1D."""
    processor = AddBatchDimensionProcessorStep()

    # Create 0D task_index tensor (scalar)
    task_index_0d = torch.tensor(7, dtype=torch.int64)
    complementary_data = {"task_index": task_index_0d}
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )

    result = processor(transition)

    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
    assert processed_comp_data["task_index"].shape == (1,)
    assert processed_comp_data["task_index"].dtype == torch.int64
    assert processed_comp_data["task_index"][0] == 7


def test_index_and_task_index_together():
    """Test processing both index and task_index together."""
    processor = AddBatchDimensionProcessorStep()

    # Create 0D tensors for both
    index_0d = torch.tensor(100, dtype=torch.int64)
    task_index_0d = torch.tensor(3, dtype=torch.int64)
    complementary_data = {
        "index": index_0d,
        "task_index": task_index_0d,
        "task": "pick_object",
    }
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )

    result = processor(transition)

    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]

    # Check index
    assert processed_comp_data["index"].shape == (1,)
    assert processed_comp_data["index"][0] == 100

    # Check task_index
    assert processed_comp_data["task_index"].shape == (1,)
    assert processed_comp_data["task_index"][0] == 3

    # Check task is also processed
    assert processed_comp_data["task"] == ["pick_object"]


def test_index_already_batched():
    """Test that already batched index tensors remain unchanged."""
    processor = AddBatchDimensionProcessorStep()

    # Create already batched tensors
    index_1d = torch.tensor([42], dtype=torch.int64)
    index_2d = torch.tensor([[42, 43]], dtype=torch.int64)

    # Test 1D (already batched)
    complementary_data = {"index": index_1d}
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )
    result = processor(transition)
    assert torch.equal(result[TransitionKey.COMPLEMENTARY_DATA]["index"], index_1d)

    # Test 2D
    complementary_data = {"index": index_2d}
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )
    result = processor(transition)
    assert torch.equal(result[TransitionKey.COMPLEMENTARY_DATA]["index"], index_2d)


def test_task_index_already_batched():
    """Test that already batched task_index tensors remain unchanged."""
    processor = AddBatchDimensionProcessorStep()

    # Create already batched tensors
    task_index_1d = torch.tensor([7], dtype=torch.int64)
    task_index_2d = torch.tensor([[7, 8]], dtype=torch.int64)

    # Test 1D (already batched)
    complementary_data = {"task_index": task_index_1d}
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )
    result = processor(transition)
    assert torch.equal(result[TransitionKey.COMPLEMENTARY_DATA]["task_index"], task_index_1d)

    # Test 2D
    complementary_data = {"task_index": task_index_2d}
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )
    result = processor(transition)
    assert torch.equal(result[TransitionKey.COMPLEMENTARY_DATA]["task_index"], task_index_2d)


def test_index_non_tensor_unchanged():
    """Test that non-tensor index values remain unchanged."""
    processor = AddBatchDimensionProcessorStep()

    complementary_data = {
        "index": 42,  # Plain int, not tensor
        "task_index": [1, 2, 3],  # List, not tensor
    }
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )

    result = processor(transition)

    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
    assert processed_comp_data["index"] == 42
    assert processed_comp_data["task_index"] == [1, 2, 3]


def test_index_dtype_preservation():
    """Test that index and task_index dtype is preserved during processing."""
    processor = AddBatchDimensionProcessorStep()

    # Test different dtypes
    dtypes = [torch.int32, torch.int64, torch.long]

    for dtype in dtypes:
        index_0d = torch.tensor(42, dtype=dtype)
        task_index_0d = torch.tensor(7, dtype=dtype)
        complementary_data = {
            "index": index_0d,
            "task_index": task_index_0d,
        }
        transition = create_transition(
            complementary_data=complementary_data, observation={}, action=torch.empty(0)
        )

        result = processor(transition)

        processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
        assert processed_comp_data["index"].dtype == dtype
        assert processed_comp_data["task_index"].dtype == dtype


def test_index_with_full_transition():
    """Test index/task_index processing with full transition data."""
    processor = AddBatchDimensionProcessorStep()

    # Create full transition with all components
    observation = {
        OBS_STATE: torch.randn(7),
        OBS_IMAGE: torch.randn(64, 64, 3),
    }
    action = torch.randn(4)
    complementary_data = {
        "task": "navigate_to_goal",
        "index": torch.tensor(1000, dtype=torch.int64),
        "task_index": torch.tensor(5, dtype=torch.int64),
        "episode_id": 123,
    }

    transition = create_transition(
        observation=observation,
        action=action,
        reward=0.5,
        done=False,
        complementary_data=complementary_data,
    )

    result = processor(transition)

    # Check all components are processed correctly
    assert result[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 7)
    assert result[TransitionKey.OBSERVATION][OBS_IMAGE].shape == (1, 64, 64, 3)
    assert result[TransitionKey.ACTION].shape == (1, 4)

    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]
    assert processed_comp_data["task"] == ["navigate_to_goal"]
    assert processed_comp_data["index"].shape == (1,)
    assert processed_comp_data["index"][0] == 1000
    assert processed_comp_data["task_index"].shape == (1,)
    assert processed_comp_data["task_index"][0] == 5
    assert processed_comp_data["episode_id"] == 123  # Non-tensor field unchanged


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_index_device_compatibility():
    """Test processor works with index/task_index tensors on different devices."""
    processor = AddBatchDimensionProcessorStep()

    # Create tensors on GPU
    index_0d = torch.tensor(42, dtype=torch.int64, device="cuda")
    task_index_0d = torch.tensor(7, dtype=torch.int64, device="cuda")

    complementary_data = {
        "index": index_0d,
        "task_index": task_index_0d,
    }
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )

    result = processor(transition)
    processed_comp_data = result[TransitionKey.COMPLEMENTARY_DATA]

    # Check shapes and that tensors stayed on GPU
    assert processed_comp_data["index"].shape == (1,)
    assert processed_comp_data["task_index"].shape == (1,)
    assert processed_comp_data["index"].device.type == "cuda"
    assert processed_comp_data["task_index"].device.type == "cuda"


def test_empty_index_tensor():
    """Test handling of empty index tensors."""
    processor = AddBatchDimensionProcessorStep()

    # Empty 0D tensor doesn't make sense, but test empty 1D
    index_empty = torch.tensor([], dtype=torch.int64)
    complementary_data = {"index": index_empty}
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )

    result = processor(transition)

    # Should remain unchanged (already 1D)
    assert result[TransitionKey.COMPLEMENTARY_DATA]["index"].shape == (0,)


def test_action_processing_creates_new_transition():
    """Test that the processor creates a new transition object with correctly processed action."""
    processor = AddBatchDimensionProcessorStep()

    action = torch.randn(4)
    transition = create_transition(action=action, observation={})

    # Store reference to original transition
    original_transition = transition

    # Process
    result = processor(transition)

    # Should be a different object (functional design, not in-place mutation)
    assert result is not original_transition
    # Original transition should remain unchanged
    assert original_transition[TransitionKey.ACTION].shape == (4,)
    # Result should have correctly processed action with batch dimension
    assert result[TransitionKey.ACTION].shape == (1, 4)
    assert torch.equal(result[TransitionKey.ACTION][0], action)


def test_task_processing_creates_new_transition():
    """Test that the processor creates a new transition object with correctly processed task."""
    processor = AddBatchDimensionProcessorStep()

    complementary_data = {"task": "sort_objects"}
    transition = create_transition(
        complementary_data=complementary_data, observation={}, action=torch.empty(0)
    )

    # Store reference to original transition and complementary_data
    original_transition = transition
    original_comp_data = complementary_data

    # Process
    result = processor(transition)

    # Should be different transition object (functional design)
    assert result is not original_transition
    # The task should be processed correctly (wrapped in list)
    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == ["sort_objects"]
    # Original complementary data is also modified (current behavior)
    assert original_comp_data["task"] == "sort_objects"

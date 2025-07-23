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

import pytest
import torch

from lerobot.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot.processor import ProcessorStepRegistry, RobotProcessor, ToBatchProcessor, TransitionKey


def create_transition(
    observation=None, action=None, reward=None, done=None, truncated=None, info=None, complementary_data=None
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


def test_state_1d_to_2d():
    """Test that 1D state tensors get unsqueezed to 2D."""
    processor = ToBatchProcessor()

    # Test observation.state
    state_1d = torch.randn(7)
    observation = {OBS_STATE: state_1d}
    transition = create_transition(observation=observation)

    result = processor(transition)

    processed_state = result[TransitionKey.OBSERVATION][OBS_STATE]
    assert processed_state.shape == (1, 7)
    assert torch.allclose(processed_state.squeeze(0), state_1d)


def test_env_state_1d_to_2d():
    """Test that 1D environment state tensors get unsqueezed to 2D."""
    processor = ToBatchProcessor()

    # Test observation.environment_state
    env_state_1d = torch.randn(10)
    observation = {OBS_ENV_STATE: env_state_1d}
    transition = create_transition(observation=observation)

    result = processor(transition)

    processed_env_state = result[TransitionKey.OBSERVATION][OBS_ENV_STATE]
    assert processed_env_state.shape == (1, 10)
    assert torch.allclose(processed_env_state.squeeze(0), env_state_1d)


def test_image_3d_to_4d():
    """Test that 3D image tensors get unsqueezed to 4D."""
    processor = ToBatchProcessor()

    # Test observation.image
    image_3d = torch.randn(224, 224, 3)
    observation = {OBS_IMAGE: image_3d}
    transition = create_transition(observation=observation)

    result = processor(transition)

    processed_image = result[TransitionKey.OBSERVATION][OBS_IMAGE]
    assert processed_image.shape == (1, 224, 224, 3)
    assert torch.allclose(processed_image.squeeze(0), image_3d)


def test_multiple_images_3d_to_4d():
    """Test that 3D image tensors in observation.images.* get unsqueezed to 4D."""
    processor = ToBatchProcessor()

    # Test observation.images.camera1 and observation.images.camera2
    image1_3d = torch.randn(64, 64, 3)
    image2_3d = torch.randn(128, 128, 3)
    observation = {
        f"{OBS_IMAGES}.camera1": image1_3d,
        f"{OBS_IMAGES}.camera2": image2_3d,
    }
    transition = create_transition(observation=observation)

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
    processor = ToBatchProcessor()

    # Create already batched tensors
    state_2d = torch.randn(1, 7)
    env_state_2d = torch.randn(1, 10)
    image_4d = torch.randn(1, 224, 224, 3)

    observation = {
        OBS_STATE: state_2d,
        OBS_ENV_STATE: env_state_2d,
        OBS_IMAGE: image_4d,
    }
    transition = create_transition(observation=observation)

    result = processor(transition)

    processed_obs = result[TransitionKey.OBSERVATION]

    # Should remain unchanged
    assert torch.allclose(processed_obs[OBS_STATE], state_2d)
    assert torch.allclose(processed_obs[OBS_ENV_STATE], env_state_2d)
    assert torch.allclose(processed_obs[OBS_IMAGE], image_4d)


def test_higher_dimensional_tensors_unchanged():
    """Test that tensors with more dimensions than expected remain unchanged."""
    processor = ToBatchProcessor()

    # Create tensors with more dimensions
    state_3d = torch.randn(2, 7, 5)  # More than 1D
    image_5d = torch.randn(2, 3, 224, 224, 1)  # More than 3D

    observation = {
        OBS_STATE: state_3d,
        OBS_IMAGE: image_5d,
    }
    transition = create_transition(observation=observation)

    result = processor(transition)

    processed_obs = result[TransitionKey.OBSERVATION]

    # Should remain unchanged
    assert torch.allclose(processed_obs[OBS_STATE], state_3d)
    assert torch.allclose(processed_obs[OBS_IMAGE], image_5d)


def test_non_tensor_values_unchanged():
    """Test that non-tensor values in observations remain unchanged."""
    processor = ToBatchProcessor()

    observation = {
        OBS_STATE: [1, 2, 3],  # List, not tensor
        OBS_IMAGE: "not_a_tensor",  # String
        "custom_key": 42,  # Integer
        "another_key": {"nested": "dict"},  # Dict
    }
    transition = create_transition(observation=observation)

    result = processor(transition)

    processed_obs = result[TransitionKey.OBSERVATION]

    # Should remain unchanged
    assert processed_obs[OBS_STATE] == [1, 2, 3]
    assert processed_obs[OBS_IMAGE] == "not_a_tensor"
    assert processed_obs["custom_key"] == 42
    assert processed_obs["another_key"] == {"nested": "dict"}


def test_none_observation():
    """Test processor handles None observation gracefully."""
    processor = ToBatchProcessor()

    transition = create_transition(observation=None)
    result = processor(transition)

    assert result[TransitionKey.OBSERVATION] is None


def test_empty_observation():
    """Test processor handles empty observation dict."""
    processor = ToBatchProcessor()

    observation = {}
    transition = create_transition(observation=observation)

    result = processor(transition)

    assert result[TransitionKey.OBSERVATION] == {}


def test_mixed_observation():
    """Test processor with mixed observation containing various types and dimensions."""
    processor = ToBatchProcessor()

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
    transition = create_transition(observation=observation)

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
    """Test ToBatchProcessor integration with RobotProcessor."""
    to_batch_processor = ToBatchProcessor()
    pipeline = RobotProcessor([to_batch_processor])

    # Create unbatched observation
    observation = {
        OBS_STATE: torch.randn(7),
        OBS_IMAGE: torch.randn(224, 224, 3),
    }
    transition = create_transition(observation=observation)

    result = pipeline(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    assert processed_obs[OBS_STATE].shape == (1, 7)
    assert processed_obs[OBS_IMAGE].shape == (1, 224, 224, 3)


def test_serialization_methods():
    """Test get_config, state_dict, load_state_dict, and reset methods."""
    processor = ToBatchProcessor()

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
    """Test saving and loading ToBatchProcessor with RobotProcessor."""
    processor = ToBatchProcessor()
    pipeline = RobotProcessor([processor], name="BatchPipeline")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save pipeline
        pipeline.save_pretrained(tmp_dir)

        # Check config file exists
        config_path = Path(tmp_dir) / "batchpipeline.json"
        assert config_path.exists()

        # Load pipeline
        loaded_pipeline = RobotProcessor.from_pretrained(tmp_dir)

        assert loaded_pipeline.name == "BatchPipeline"
        assert len(loaded_pipeline) == 1
        assert isinstance(loaded_pipeline.steps[0], ToBatchProcessor)

        # Test functionality of loaded processor
        observation = {OBS_STATE: torch.randn(5)}
        transition = create_transition(observation=observation)

        result = loaded_pipeline(transition)
        assert result[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 5)


def test_registry_functionality():
    """Test that ToBatchProcessor is properly registered."""
    # Check that the processor is registered
    registered_class = ProcessorStepRegistry.get("to_batch_processor")
    assert registered_class is ToBatchProcessor

    # Check that it's in the list of registered processors
    assert "to_batch_processor" in ProcessorStepRegistry.list()


def test_registry_based_save_load():
    """Test saving and loading using registry name."""
    processor = ToBatchProcessor()
    pipeline = RobotProcessor([processor])

    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline.save_pretrained(tmp_dir)
        loaded_pipeline = RobotProcessor.from_pretrained(tmp_dir)

        # Verify the loaded processor works
        observation = {
            OBS_STATE: torch.randn(3),
            OBS_IMAGE: torch.randn(100, 100, 3),
        }
        transition = create_transition(observation=observation)

        result = loaded_pipeline(transition)
        processed_obs = result[TransitionKey.OBSERVATION]

        assert processed_obs[OBS_STATE].shape == (1, 3)
        assert processed_obs[OBS_IMAGE].shape == (1, 100, 100, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_compatibility():
    """Test processor works with tensors on different devices."""
    processor = ToBatchProcessor()

    # Create tensors on GPU
    state_1d = torch.randn(7, device="cuda")
    image_3d = torch.randn(64, 64, 3, device="cuda")

    observation = {
        OBS_STATE: state_1d,
        OBS_IMAGE: image_3d,
    }
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check shapes and that tensors stayed on GPU
    assert processed_obs[OBS_STATE].shape == (1, 7)
    assert processed_obs[OBS_IMAGE].shape == (1, 64, 64, 3)
    assert processed_obs[OBS_STATE].device.type == "cuda"
    assert processed_obs[OBS_IMAGE].device.type == "cuda"


def test_processor_preserves_other_transition_keys():
    """Test that processor only modifies observation and preserves other transition keys."""
    processor = ToBatchProcessor()

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
    processor = ToBatchProcessor()

    # 0D tensors should not be modified
    scalar_tensor = torch.tensor(42.0)

    observation = {
        OBS_STATE: scalar_tensor,
        "scalar_value": scalar_tensor,
    }
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # 0D tensors should remain unchanged
    assert torch.allclose(processed_obs[OBS_STATE], scalar_tensor)
    assert torch.allclose(processed_obs["scalar_value"], scalar_tensor)

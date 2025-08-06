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

from lerobot.configs.types import FeatureType
from lerobot.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot.processor import VanillaObservationProcessor
from lerobot.processor.pipeline import TransitionKey
from tests.conftest import assert_contract_is_typed


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
        TransitionKey.INFO: info,
        TransitionKey.COMPLEMENTARY_DATA: complementary_data,
    }


def test_process_single_image():
    """Test processing a single image."""
    processor = VanillaObservationProcessor()

    # Create a mock image (H, W, C) format, uint8
    image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)

    observation = {"pixels": image}
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check that the image was processed correctly
    assert "observation.image" in processed_obs
    processed_img = processed_obs["observation.image"]

    # Check shape: should be (1, 3, 64, 64) - batch, channels, height, width
    assert processed_img.shape == (1, 3, 64, 64)

    # Check dtype and range
    assert processed_img.dtype == torch.float32
    assert processed_img.min() >= 0.0
    assert processed_img.max() <= 1.0


def test_process_image_dict():
    """Test processing multiple images in a dictionary."""
    processor = VanillaObservationProcessor()

    # Create mock images
    image1 = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    image2 = np.random.randint(0, 256, size=(48, 48, 3), dtype=np.uint8)

    observation = {"pixels": {"camera1": image1, "camera2": image2}}
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check that both images were processed
    assert "observation.images.camera1" in processed_obs
    assert "observation.images.camera2" in processed_obs

    # Check shapes
    assert processed_obs["observation.images.camera1"].shape == (1, 3, 32, 32)
    assert processed_obs["observation.images.camera2"].shape == (1, 3, 48, 48)


def test_process_batched_image():
    """Test processing already batched images."""
    processor = VanillaObservationProcessor()

    # Create a batched image (B, H, W, C)
    image = np.random.randint(0, 256, size=(2, 64, 64, 3), dtype=np.uint8)

    observation = {"pixels": image}
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check that batch dimension is preserved
    assert processed_obs["observation.image"].shape == (2, 3, 64, 64)


def test_invalid_image_format():
    """Test error handling for invalid image formats."""
    processor = VanillaObservationProcessor()

    # Test wrong channel order (channels first)
    image = np.random.randint(0, 256, size=(3, 64, 64), dtype=np.uint8)
    observation = {"pixels": image}
    transition = create_transition(observation=observation)

    with pytest.raises(ValueError, match="Expected channel-last images"):
        processor(transition)


def test_invalid_image_dtype():
    """Test error handling for invalid image dtype."""
    processor = VanillaObservationProcessor()

    # Test wrong dtype
    image = np.random.rand(64, 64, 3).astype(np.float32)
    observation = {"pixels": image}
    transition = create_transition(observation=observation)

    with pytest.raises(ValueError, match="Expected torch.uint8 images"):
        processor(transition)


def test_no_pixels_in_observation():
    """Test processor when no pixels are in observation."""
    processor = VanillaObservationProcessor()

    observation = {"other_data": np.array([1, 2, 3])}
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Should preserve other data unchanged
    assert "other_data" in processed_obs
    np.testing.assert_array_equal(processed_obs["other_data"], np.array([1, 2, 3]))


def test_none_observation():
    """Test processor with None observation."""
    processor = VanillaObservationProcessor()

    transition = create_transition()
    result = processor(transition)

    assert result == transition


def test_serialization_methods():
    """Test serialization methods."""
    processor = VanillaObservationProcessor()

    # Test get_config
    config = processor.get_config()
    assert isinstance(config, dict)

    # Test state_dict
    state = processor.state_dict()
    assert isinstance(state, dict)

    # Test load_state_dict (should not raise)
    processor.load_state_dict(state)

    # Test reset (should not raise)
    processor.reset()


def test_process_environment_state():
    """Test processing environment_state."""
    processor = VanillaObservationProcessor()

    env_state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    observation = {"environment_state": env_state}
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check that environment_state was renamed and processed
    assert "observation.environment_state" in processed_obs
    assert "environment_state" not in processed_obs

    processed_state = processed_obs["observation.environment_state"]
    assert processed_state.shape == (1, 3)  # Batch dimension added
    assert processed_state.dtype == torch.float32
    torch.testing.assert_close(processed_state, torch.tensor([[1.0, 2.0, 3.0]]))


def test_process_agent_pos():
    """Test processing agent_pos."""
    processor = VanillaObservationProcessor()

    agent_pos = np.array([0.5, -0.5, 1.0], dtype=np.float32)
    observation = {"agent_pos": agent_pos}
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check that agent_pos was renamed and processed
    assert "observation.state" in processed_obs
    assert "agent_pos" not in processed_obs

    processed_state = processed_obs["observation.state"]
    assert processed_state.shape == (1, 3)  # Batch dimension added
    assert processed_state.dtype == torch.float32
    torch.testing.assert_close(processed_state, torch.tensor([[0.5, -0.5, 1.0]]))


def test_process_batched_states():
    """Test processing already batched states."""
    processor = VanillaObservationProcessor()

    env_state = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    agent_pos = np.array([[0.5, -0.5], [1.0, -1.0]], dtype=np.float32)

    observation = {"environment_state": env_state, "agent_pos": agent_pos}
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check that batch dimensions are preserved
    assert processed_obs["observation.environment_state"].shape == (2, 2)
    assert processed_obs["observation.state"].shape == (2, 2)


def test_process_both_states():
    """Test processing both environment_state and agent_pos."""
    processor = VanillaObservationProcessor()

    env_state = np.array([1.0, 2.0], dtype=np.float32)
    agent_pos = np.array([0.5, -0.5], dtype=np.float32)

    observation = {"environment_state": env_state, "agent_pos": agent_pos, "other_data": "keep_me"}
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check that both states were processed
    assert "observation.environment_state" in processed_obs
    assert "observation.state" in processed_obs

    # Check that original keys were removed
    assert "environment_state" not in processed_obs
    assert "agent_pos" not in processed_obs

    # Check that other data was preserved
    assert processed_obs["other_data"] == "keep_me"


def test_no_states_in_observation():
    """Test processor when no states are in observation."""
    processor = VanillaObservationProcessor()

    observation = {"other_data": np.array([1, 2, 3])}
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Should preserve data unchanged
    np.testing.assert_array_equal(processed_obs, observation)


def test_complete_observation_processing():
    """Test processing a complete observation with both images and states."""
    processor = VanillaObservationProcessor()

    # Create mock data
    image = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    env_state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    agent_pos = np.array([0.5, -0.5, 1.0], dtype=np.float32)

    observation = {
        "pixels": image,
        "environment_state": env_state,
        "agent_pos": agent_pos,
        "other_data": "preserve_me",
    }
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    # Check that image was processed
    assert "observation.image" in processed_obs
    assert processed_obs["observation.image"].shape == (1, 3, 32, 32)

    # Check that states were processed
    assert "observation.environment_state" in processed_obs
    assert "observation.state" in processed_obs

    # Check that original keys were removed
    assert "pixels" not in processed_obs
    assert "environment_state" not in processed_obs
    assert "agent_pos" not in processed_obs

    # Check that other data was preserved
    assert processed_obs["other_data"] == "preserve_me"


def test_image_only_processing():
    """Test processing observation with only images."""
    processor = VanillaObservationProcessor()

    image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    observation = {"pixels": image}
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    assert "observation.image" in processed_obs
    assert len(processed_obs) == 1


def test_state_only_processing():
    """Test processing observation with only states."""
    processor = VanillaObservationProcessor()

    agent_pos = np.array([1.0, 2.0], dtype=np.float32)
    observation = {"agent_pos": agent_pos}
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    assert "observation.state" in processed_obs
    assert "agent_pos" not in processed_obs


def test_empty_observation():
    """Test processing empty observation."""
    processor = VanillaObservationProcessor()

    observation = {}
    transition = create_transition(observation=observation)

    result = processor(transition)
    processed_obs = result[TransitionKey.OBSERVATION]

    assert processed_obs == {}


def test_equivalent_to_original_function():
    """Test that ObservationProcessor produces equivalent results to preprocess_observation."""
    # Import the original function for comparison
    from lerobot.envs.utils import preprocess_observation

    processor = VanillaObservationProcessor()

    # Create test data similar to what the original function expects
    image = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    env_state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    agent_pos = np.array([0.5, -0.5, 1.0], dtype=np.float32)

    observation = {"pixels": image, "environment_state": env_state, "agent_pos": agent_pos}

    # Process with original function
    original_result = preprocess_observation(observation)

    # Process with new processor
    transition = create_transition(observation=observation)
    processor_result = processor(transition)[TransitionKey.OBSERVATION]

    # Compare results
    assert set(original_result.keys()) == set(processor_result.keys())

    for key in original_result:
        torch.testing.assert_close(original_result[key], processor_result[key])


def test_equivalent_with_image_dict():
    """Test equivalence with dictionary of images."""
    from lerobot.envs.utils import preprocess_observation

    processor = VanillaObservationProcessor()

    # Create test data with multiple cameras
    image1 = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    image2 = np.random.randint(0, 256, size=(48, 48, 3), dtype=np.uint8)
    agent_pos = np.array([1.0, 2.0], dtype=np.float32)

    observation = {"pixels": {"cam1": image1, "cam2": image2}, "agent_pos": agent_pos}

    # Process with original function
    original_result = preprocess_observation(observation)

    # Process with new processor
    transition = create_transition(observation=observation)
    processor_result = processor(transition)[TransitionKey.OBSERVATION]

    # Compare results
    assert set(original_result.keys()) == set(processor_result.keys())

    for key in original_result:
        torch.testing.assert_close(original_result[key], processor_result[key])


def test_image_processor_feature_contract_pixels_to_image(policy_feature_factory):
    processor = VanillaObservationProcessor()
    features = {
        "pixels": policy_feature_factory(FeatureType.VISUAL, (3, 64, 64)),
        "keep": policy_feature_factory(FeatureType.ENV, (1,)),
    }
    out = processor.feature_contract(features.copy())

    assert OBS_IMAGE in out and out[OBS_IMAGE] == features["pixels"]
    assert "pixels" not in out
    assert out["keep"] == features["keep"]
    assert_contract_is_typed(out)


def test_image_processor_feature_contract_observation_pixels_to_image(policy_feature_factory):
    processor = VanillaObservationProcessor()
    features = {
        "observation.pixels": policy_feature_factory(FeatureType.VISUAL, (3, 64, 64)),
        "keep": policy_feature_factory(FeatureType.ENV, (1,)),
    }
    out = processor.feature_contract(features.copy())

    assert OBS_IMAGE in out and out[OBS_IMAGE] == features["observation.pixels"]
    assert "observation.pixels" not in out
    assert out["keep"] == features["keep"]
    assert_contract_is_typed(out)


def test_image_processor_feature_contract_multi_camera_and_prefixed(policy_feature_factory):
    processor = VanillaObservationProcessor()
    features = {
        "pixels.front": policy_feature_factory(FeatureType.VISUAL, (3, 64, 64)),
        "pixels.wrist": policy_feature_factory(FeatureType.VISUAL, (3, 64, 64)),
        "observation.pixels.rear": policy_feature_factory(FeatureType.VISUAL, (3, 64, 64)),
        "keep": policy_feature_factory(FeatureType.ENV, (7,)),
    }
    out = processor.feature_contract(features.copy())

    assert f"{OBS_IMAGES}.front" in out and out[f"{OBS_IMAGES}.front"] == features["pixels.front"]
    assert f"{OBS_IMAGES}.wrist" in out and out[f"{OBS_IMAGES}.wrist"] == features["pixels.wrist"]
    assert f"{OBS_IMAGES}.rear" in out and out[f"{OBS_IMAGES}.rear"] == features["observation.pixels.rear"]
    assert "pixels.front" not in out and "pixels.wrist" not in out and "observation.pixels.rear" not in out
    assert out["keep"] == features["keep"]
    assert_contract_is_typed(out)


def test_state_processor_feature_contract_environment_and_agent_pos(policy_feature_factory):
    processor = VanillaObservationProcessor()
    features = {
        "environment_state": policy_feature_factory(FeatureType.STATE, (3,)),
        "agent_pos": policy_feature_factory(FeatureType.STATE, (7,)),
        "keep": policy_feature_factory(FeatureType.ENV, (1,)),
    }
    out = processor.feature_contract(features.copy())

    assert OBS_ENV_STATE in out and out[OBS_ENV_STATE] == features["environment_state"]
    assert OBS_STATE in out and out[OBS_STATE] == features["agent_pos"]
    assert "environment_state" not in out and "agent_pos" not in out
    assert out["keep"] == features["keep"]
    assert_contract_is_typed(out)


def test_state_processor_feature_contract_prefixed_inputs(policy_feature_factory):
    proc = VanillaObservationProcessor()
    features = {
        "observation.environment_state": policy_feature_factory(FeatureType.STATE, (2,)),
        "observation.agent_pos": policy_feature_factory(FeatureType.STATE, (4,)),
    }
    out = proc.feature_contract(features.copy())

    assert OBS_ENV_STATE in out and out[OBS_ENV_STATE] == features["observation.environment_state"]
    assert OBS_STATE in out and out[OBS_STATE] == features["observation.agent_pos"]
    assert "environment_state" not in out and "agent_pos" not in out
    assert_contract_is_typed(out)

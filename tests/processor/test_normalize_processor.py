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
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor.normalize_processor import (
    NormalizerProcessor,
    UnnormalizerProcessor,
    _convert_stats_to_tensors,
    hotswap_stats,
)
from lerobot.processor.pipeline import IdentityProcessor, RobotProcessor, TransitionKey


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


def test_numpy_conversion():
    stats = {
        "observation.image": {
            "mean": np.array([0.5, 0.5, 0.5]),
            "std": np.array([0.2, 0.2, 0.2]),
        }
    }
    tensor_stats = _convert_stats_to_tensors(stats)

    assert isinstance(tensor_stats["observation.image"]["mean"], torch.Tensor)
    assert isinstance(tensor_stats["observation.image"]["std"], torch.Tensor)
    assert torch.allclose(tensor_stats["observation.image"]["mean"], torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(tensor_stats["observation.image"]["std"], torch.tensor([0.2, 0.2, 0.2]))


def test_tensor_conversion():
    stats = {
        "action": {
            "mean": torch.tensor([0.0, 0.0]),
            "std": torch.tensor([1.0, 1.0]),
        }
    }
    tensor_stats = _convert_stats_to_tensors(stats)

    assert tensor_stats["action"]["mean"].dtype == torch.float32
    assert tensor_stats["action"]["std"].dtype == torch.float32


def test_scalar_conversion():
    stats = {
        "reward": {
            "mean": 0.5,
            "std": 0.1,
        }
    }
    tensor_stats = _convert_stats_to_tensors(stats)

    assert torch.allclose(tensor_stats["reward"]["mean"], torch.tensor(0.5))
    assert torch.allclose(tensor_stats["reward"]["std"], torch.tensor(0.1))


def test_list_conversion():
    stats = {
        "observation.state": {
            "min": [0.0, -1.0, -2.0],
            "max": [1.0, 1.0, 2.0],
        }
    }
    tensor_stats = _convert_stats_to_tensors(stats)

    assert torch.allclose(tensor_stats["observation.state"]["min"], torch.tensor([0.0, -1.0, -2.0]))
    assert torch.allclose(tensor_stats["observation.state"]["max"], torch.tensor([1.0, 1.0, 2.0]))


def test_unsupported_type():
    stats = {
        "bad_key": {
            "mean": "string_value",
        }
    }
    with pytest.raises(TypeError, match="Unsupported type"):
        _convert_stats_to_tensors(stats)


# Helper functions to create feature maps and norm maps
def _create_observation_features():
    return {
        "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
    }


def _create_observation_norm_map():
    return {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.STATE: NormalizationMode.MIN_MAX,
    }


# Fixtures for observation normalisation tests using NormalizerProcessor
@pytest.fixture
def observation_stats():
    return {
        "observation.image": {
            "mean": np.array([0.5, 0.5, 0.5]),
            "std": np.array([0.2, 0.2, 0.2]),
        },
        "observation.state": {
            "min": np.array([0.0, -1.0]),
            "max": np.array([1.0, 1.0]),
        },
    }


@pytest.fixture
def observation_normalizer(observation_stats):
    """Return a NormalizerProcessor that only has observation stats (no action)."""
    features = _create_observation_features()
    norm_map = _create_observation_norm_map()
    return NormalizerProcessor(features=features, norm_map=norm_map, stats=observation_stats)


def test_mean_std_normalization(observation_normalizer):
    observation = {
        "observation.image": torch.tensor([0.7, 0.5, 0.3]),
        "observation.state": torch.tensor([0.5, 0.0]),
    }
    transition = create_transition(observation=observation)

    normalized_transition = observation_normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    # Check mean/std normalization
    expected_image = (torch.tensor([0.7, 0.5, 0.3]) - 0.5) / 0.2
    assert torch.allclose(normalized_obs["observation.image"], expected_image)


def test_min_max_normalization(observation_normalizer):
    observation = {
        "observation.state": torch.tensor([0.5, 0.0]),
    }
    transition = create_transition(observation=observation)

    normalized_transition = observation_normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    # Check min/max normalization to [-1, 1]
    # For state[0]: 2 * (0.5 - 0.0) / (1.0 - 0.0) - 1 = 0.0
    # For state[1]: 2 * (0.0 - (-1.0)) / (1.0 - (-1.0)) - 1 = 0.0
    expected_state = torch.tensor([0.0, 0.0])
    assert torch.allclose(normalized_obs["observation.state"], expected_state, atol=1e-6)


def test_selective_normalization(observation_stats):
    features = _create_observation_features()
    norm_map = _create_observation_norm_map()
    normalizer = NormalizerProcessor(
        features=features,
        norm_map=norm_map,
        stats=observation_stats,
        normalize_observation_keys={"observation.image"},
    )

    observation = {
        "observation.image": torch.tensor([0.7, 0.5, 0.3]),
        "observation.state": torch.tensor([0.5, 0.0]),
    }
    transition = create_transition(observation=observation)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    # Only image should be normalized
    assert torch.allclose(normalized_obs["observation.image"], (torch.tensor([0.7, 0.5, 0.3]) - 0.5) / 0.2)
    # State should remain unchanged
    assert torch.allclose(normalized_obs["observation.state"], observation["observation.state"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_compatibility(observation_stats):
    features = _create_observation_features()
    norm_map = _create_observation_norm_map()
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=observation_stats)
    observation = {
        "observation.image": torch.tensor([0.7, 0.5, 0.3]).cuda(),
    }
    transition = create_transition(observation=observation)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    assert normalized_obs["observation.image"].device.type == "cuda"


def test_from_lerobot_dataset():
    # Mock dataset
    mock_dataset = Mock()
    mock_dataset.meta.stats = {
        "observation.image": {"mean": [0.5], "std": [0.2]},
        "action": {"mean": [0.0], "std": [1.0]},
    }

    features = {
        "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        "action": PolicyFeature(FeatureType.ACTION, (1,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }

    normalizer = NormalizerProcessor.from_lerobot_dataset(mock_dataset, features, norm_map)

    # Both observation and action statistics should be present in tensor stats
    assert "observation.image" in normalizer._tensor_stats
    assert "action" in normalizer._tensor_stats


def test_state_dict_save_load(observation_normalizer):
    # Save state
    state_dict = observation_normalizer.state_dict()
    print("State dict:", state_dict)

    # Create new normalizer and load state
    features = _create_observation_features()
    norm_map = _create_observation_norm_map()
    new_normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats={})
    new_normalizer.load_state_dict(state_dict)

    # Test that it works the same
    observation = {"observation.image": torch.tensor([0.7, 0.5, 0.3])}
    transition = create_transition(observation=observation)

    result1 = observation_normalizer(transition)[TransitionKey.OBSERVATION]
    result2 = new_normalizer(transition)[TransitionKey.OBSERVATION]

    assert torch.allclose(result1["observation.image"], result2["observation.image"])


# Fixtures for ActionUnnormalizer tests
@pytest.fixture
def action_stats_mean_std():
    return {
        "mean": np.array([0.0, 0.0, 0.0]),
        "std": np.array([1.0, 2.0, 0.5]),
    }


@pytest.fixture
def action_stats_min_max():
    return {
        "min": np.array([-1.0, -2.0, 0.0]),
        "max": np.array([1.0, 2.0, 1.0]),
    }


def _create_action_features():
    return {
        "action": PolicyFeature(FeatureType.ACTION, (3,)),
    }


def _create_action_norm_map_mean_std():
    return {
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }


def _create_action_norm_map_min_max():
    return {
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }


def test_mean_std_unnormalization(action_stats_mean_std):
    features = _create_action_features()
    norm_map = _create_action_norm_map_mean_std()
    unnormalizer = UnnormalizerProcessor(
        features=features, norm_map=norm_map, stats={"action": action_stats_mean_std}
    )

    normalized_action = torch.tensor([1.0, -0.5, 2.0])
    transition = create_transition(action=normalized_action)

    unnormalized_transition = unnormalizer(transition)
    unnormalized_action = unnormalized_transition[TransitionKey.ACTION]

    # action * std + mean
    expected = torch.tensor([1.0 * 1.0 + 0.0, -0.5 * 2.0 + 0.0, 2.0 * 0.5 + 0.0])
    assert torch.allclose(unnormalized_action, expected)


def test_min_max_unnormalization(action_stats_min_max):
    features = _create_action_features()
    norm_map = _create_action_norm_map_min_max()
    unnormalizer = UnnormalizerProcessor(
        features=features, norm_map=norm_map, stats={"action": action_stats_min_max}
    )

    # Actions in [-1, 1]
    normalized_action = torch.tensor([0.0, -1.0, 1.0])
    transition = create_transition(action=normalized_action)

    unnormalized_transition = unnormalizer(transition)
    unnormalized_action = unnormalized_transition[TransitionKey.ACTION]

    # Map from [-1, 1] to [min, max]
    # (action + 1) / 2 * (max - min) + min
    expected = torch.tensor(
        [
            (0.0 + 1) / 2 * (1.0 - (-1.0)) + (-1.0),  # 0.0
            (-1.0 + 1) / 2 * (2.0 - (-2.0)) + (-2.0),  # -2.0
            (1.0 + 1) / 2 * (1.0 - 0.0) + 0.0,  # 1.0
        ]
    )
    assert torch.allclose(unnormalized_action, expected)


def test_numpy_action_input(action_stats_mean_std):
    features = _create_action_features()
    norm_map = _create_action_norm_map_mean_std()
    unnormalizer = UnnormalizerProcessor(
        features=features, norm_map=norm_map, stats={"action": action_stats_mean_std}
    )

    normalized_action = np.array([1.0, -0.5, 2.0], dtype=np.float32)
    transition = create_transition(action=normalized_action)

    unnormalized_transition = unnormalizer(transition)
    unnormalized_action = unnormalized_transition[TransitionKey.ACTION]

    assert isinstance(unnormalized_action, torch.Tensor)
    expected = torch.tensor([1.0, -1.0, 1.0])
    assert torch.allclose(unnormalized_action, expected)


def test_none_action(action_stats_mean_std):
    features = _create_action_features()
    norm_map = _create_action_norm_map_mean_std()
    unnormalizer = UnnormalizerProcessor(
        features=features, norm_map=norm_map, stats={"action": action_stats_mean_std}
    )

    transition = create_transition()
    result = unnormalizer(transition)

    # Should return transition unchanged
    assert result == transition


def test_action_from_lerobot_dataset():
    mock_dataset = Mock()
    mock_dataset.meta.stats = {"action": {"mean": [0.0], "std": [1.0]}}
    features = {"action": PolicyFeature(FeatureType.ACTION, (1,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.MEAN_STD}
    unnormalizer = UnnormalizerProcessor.from_lerobot_dataset(mock_dataset, features, norm_map)
    assert "mean" in unnormalizer._tensor_stats["action"]


# Fixtures for NormalizerProcessor tests
@pytest.fixture
def full_stats():
    return {
        "observation.image": {
            "mean": np.array([0.5, 0.5, 0.5]),
            "std": np.array([0.2, 0.2, 0.2]),
        },
        "observation.state": {
            "min": np.array([0.0, -1.0]),
            "max": np.array([1.0, 1.0]),
        },
        "action": {
            "mean": np.array([0.0, 0.0]),
            "std": np.array([1.0, 2.0]),
        },
    }


def _create_full_features():
    return {
        "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
        "action": PolicyFeature(FeatureType.ACTION, (2,)),
    }


def _create_full_norm_map():
    return {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.STATE: NormalizationMode.MIN_MAX,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }


@pytest.fixture
def normalizer_processor(full_stats):
    features = _create_full_features()
    norm_map = _create_full_norm_map()
    return NormalizerProcessor(features=features, norm_map=norm_map, stats=full_stats)


def test_combined_normalization(normalizer_processor):
    observation = {
        "observation.image": torch.tensor([0.7, 0.5, 0.3]),
        "observation.state": torch.tensor([0.5, 0.0]),
    }
    action = torch.tensor([1.0, -0.5])
    transition = create_transition(
        observation=observation,
        action=action,
        reward=1.0,
        done=False,
        truncated=False,
        info={},
        complementary_data={},
    )

    processed_transition = normalizer_processor(transition)

    # Check normalized observations
    processed_obs = processed_transition[TransitionKey.OBSERVATION]
    expected_image = (torch.tensor([0.7, 0.5, 0.3]) - 0.5) / 0.2
    assert torch.allclose(processed_obs["observation.image"], expected_image)

    # Check normalized action
    processed_action = processed_transition[TransitionKey.ACTION]
    expected_action = torch.tensor([(1.0 - 0.0) / 1.0, (-0.5 - 0.0) / 2.0])
    assert torch.allclose(processed_action, expected_action)

    # Check other fields remain unchanged
    assert processed_transition[TransitionKey.REWARD] == 1.0
    assert not processed_transition[TransitionKey.DONE]


def test_processor_from_lerobot_dataset(full_stats):
    # Mock dataset
    mock_dataset = Mock()
    mock_dataset.meta.stats = full_stats

    features = _create_full_features()
    norm_map = _create_full_norm_map()

    processor = NormalizerProcessor.from_lerobot_dataset(
        mock_dataset, features, norm_map, normalize_observation_keys={"observation.image"}
    )

    assert processor.normalize_observation_keys == {"observation.image"}
    assert "observation.image" in processor._tensor_stats
    assert "action" in processor._tensor_stats


def test_get_config(full_stats):
    features = _create_full_features()
    norm_map = _create_full_norm_map()
    processor = NormalizerProcessor(
        features=features,
        norm_map=norm_map,
        stats=full_stats,
        normalize_observation_keys={"observation.image"},
        eps=1e-6,
    )

    config = processor.get_config()
    expected_config = {
        "normalize_observation_keys": ["observation.image"],
        "eps": 1e-6,
        "features": {
            "observation.image": {"type": "VISUAL", "shape": (3, 96, 96)},
            "observation.state": {"type": "STATE", "shape": (2,)},
            "action": {"type": "ACTION", "shape": (2,)},
        },
        "norm_map": {
            "VISUAL": "MEAN_STD",
            "STATE": "MIN_MAX",
            "ACTION": "MEAN_STD",
        },
    }
    assert config == expected_config


def test_integration_with_robot_processor(normalizer_processor):
    """Test integration with RobotProcessor pipeline"""
    robot_processor = RobotProcessor([normalizer_processor], to_transition=lambda x: x, to_output=lambda x: x)

    observation = {
        "observation.image": torch.tensor([0.7, 0.5, 0.3]),
        "observation.state": torch.tensor([0.5, 0.0]),
    }
    action = torch.tensor([1.0, -0.5])
    transition = create_transition(
        observation=observation,
        action=action,
        reward=1.0,
        done=False,
        truncated=False,
        info={},
        complementary_data={},
    )

    processed_transition = robot_processor(transition)

    # Verify the processing worked
    assert isinstance(processed_transition[TransitionKey.OBSERVATION], dict)
    assert isinstance(processed_transition[TransitionKey.ACTION], torch.Tensor)


# Edge case tests
def test_empty_observation():
    stats = {"observation.image": {"mean": [0.5], "std": [0.2]}}
    features = {"observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96))}
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    transition = create_transition()
    result = normalizer(transition)

    assert result == transition


def test_empty_stats():
    features = {"observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96))}
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats={})
    observation = {"observation.image": torch.tensor([0.5])}
    transition = create_transition(observation=observation)

    result = normalizer(transition)
    # Should return observation unchanged since no stats are available
    assert torch.allclose(
        result[TransitionKey.OBSERVATION]["observation.image"], observation["observation.image"]
    )


def test_partial_stats():
    """If statistics are incomplete, the value should pass through unchanged."""
    stats = {"observation.image": {"mean": [0.5]}}  # Missing std / (min,max)
    features = {"observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96))}
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)
    observation = {"observation.image": torch.tensor([0.7])}
    transition = create_transition(observation=observation)

    processed = normalizer(transition)[TransitionKey.OBSERVATION]
    assert torch.allclose(processed["observation.image"], observation["observation.image"])


def test_missing_action_stats_no_error():
    mock_dataset = Mock()
    mock_dataset.meta.stats = {"observation.image": {"mean": [0.5], "std": [0.2]}}

    features = {"observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96))}
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}

    processor = UnnormalizerProcessor.from_lerobot_dataset(mock_dataset, features, norm_map)
    # The tensor stats should not contain the 'action' key
    assert "action" not in processor._tensor_stats


def test_serialization_roundtrip(full_stats):
    """Test that features and norm_map can be serialized and deserialized correctly."""
    features = _create_full_features()
    norm_map = _create_full_norm_map()
    original_processor = NormalizerProcessor(
        features=features,
        norm_map=norm_map,
        stats=full_stats,
        normalize_observation_keys={"observation.image"},
        eps=1e-6,
    )

    # Get config (serialization)
    config = original_processor.get_config()

    # Create a new processor from the config (deserialization)
    new_processor = NormalizerProcessor(
        features=config["features"],
        norm_map=config["norm_map"],
        stats=full_stats,
        normalize_observation_keys=set(config["normalize_observation_keys"]),
        eps=config["eps"],
    )

    # Test that both processors work the same way
    observation = {
        "observation.image": torch.tensor([0.7, 0.5, 0.3]),
        "observation.state": torch.tensor([0.5, 0.0]),
    }
    action = torch.tensor([1.0, -0.5])
    transition = create_transition(
        observation=observation,
        action=action,
        reward=1.0,
        done=False,
        truncated=False,
        info={},
        complementary_data={},
    )

    result1 = original_processor(transition)
    result2 = new_processor(transition)

    # Compare results
    assert torch.allclose(
        result1[TransitionKey.OBSERVATION]["observation.image"],
        result2[TransitionKey.OBSERVATION]["observation.image"],
    )
    assert torch.allclose(result1[TransitionKey.ACTION], result2[TransitionKey.ACTION])

    # Verify features and norm_map are correctly reconstructed
    assert (
        new_processor.transform_features(features).keys()
        == original_processor.transform_features(features).keys()
    )
    for key in new_processor.transform_features(features):
        assert (
            new_processor.transform_features(features)[key].type
            == original_processor.transform_features(features)[key].type
        )
        assert (
            new_processor.transform_features(features)[key].shape
            == original_processor.transform_features(features)[key].shape
        )

    assert new_processor.norm_map == original_processor.norm_map


# Identity normalization tests
def test_identity_normalization_observations():
    """Test that IDENTITY mode skips normalization for observations."""
    features = {
        "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,  # IDENTITY mode
        FeatureType.STATE: NormalizationMode.MEAN_STD,  # Normal mode for comparison
    }
    stats = {
        "observation.image": {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
        "observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
    }

    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    observation = {
        "observation.image": torch.tensor([0.7, 0.5, 0.3]),
        "observation.state": torch.tensor([1.0, -0.5]),
    }
    transition = create_transition(observation=observation)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    # Image should remain unchanged (IDENTITY)
    assert torch.allclose(normalized_obs["observation.image"], observation["observation.image"])

    # State should be normalized (MEAN_STD)
    expected_state = (torch.tensor([1.0, -0.5]) - torch.tensor([0.0, 0.0])) / torch.tensor([1.0, 1.0])
    assert torch.allclose(normalized_obs["observation.state"], expected_state)


def test_identity_normalization_actions():
    """Test that IDENTITY mode skips normalization for actions."""
    features = {"action": PolicyFeature(FeatureType.ACTION, (2,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.IDENTITY}
    stats = {"action": {"mean": [0.0, 0.0], "std": [1.0, 2.0]}}

    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    action = torch.tensor([1.0, -0.5])
    transition = create_transition(action=action)

    normalized_transition = normalizer(transition)

    # Action should remain unchanged
    assert torch.allclose(normalized_transition[TransitionKey.ACTION], action)


def test_identity_unnormalization_observations():
    """Test that IDENTITY mode skips unnormalization for observations."""
    features = {
        "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,  # IDENTITY mode
        FeatureType.STATE: NormalizationMode.MIN_MAX,  # Normal mode for comparison
    }
    stats = {
        "observation.image": {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
        "observation.state": {"min": [-1.0, -1.0], "max": [1.0, 1.0]},
    }

    unnormalizer = UnnormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    observation = {
        "observation.image": torch.tensor([0.7, 0.5, 0.3]),
        "observation.state": torch.tensor([0.0, -1.0]),  # Normalized values in [-1, 1]
    }
    transition = create_transition(observation=observation)

    unnormalized_transition = unnormalizer(transition)
    unnormalized_obs = unnormalized_transition[TransitionKey.OBSERVATION]

    # Image should remain unchanged (IDENTITY)
    assert torch.allclose(unnormalized_obs["observation.image"], observation["observation.image"])

    # State should be unnormalized (MIN_MAX)
    # (0.0 + 1) / 2 * (1.0 - (-1.0)) + (-1.0) = 0.0
    # (-1.0 + 1) / 2 * (1.0 - (-1.0)) + (-1.0) = -1.0
    expected_state = torch.tensor([0.0, -1.0])
    assert torch.allclose(unnormalized_obs["observation.state"], expected_state)


def test_identity_unnormalization_actions():
    """Test that IDENTITY mode skips unnormalization for actions."""
    features = {"action": PolicyFeature(FeatureType.ACTION, (2,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.IDENTITY}
    stats = {"action": {"min": [-1.0, -2.0], "max": [1.0, 2.0]}}

    unnormalizer = UnnormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    action = torch.tensor([0.5, -0.8])  # Normalized values
    transition = create_transition(action=action)

    unnormalized_transition = unnormalizer(transition)

    # Action should remain unchanged
    assert torch.allclose(unnormalized_transition[TransitionKey.ACTION], action)


def test_identity_with_missing_stats():
    """Test that IDENTITY mode works even when stats are missing."""
    features = {
        "observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        "action": PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.IDENTITY,
    }
    stats = {}  # No stats provided

    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    observation = {"observation.image": torch.tensor([0.7, 0.5, 0.3])}
    action = torch.tensor([1.0, -0.5])
    transition = create_transition(observation=observation, action=action)

    # Both should work without errors and return unchanged data
    normalized_transition = normalizer(transition)
    unnormalized_transition = unnormalizer(transition)

    assert torch.allclose(
        normalized_transition[TransitionKey.OBSERVATION]["observation.image"],
        observation["observation.image"],
    )
    assert torch.allclose(normalized_transition[TransitionKey.ACTION], action)
    assert torch.allclose(
        unnormalized_transition[TransitionKey.OBSERVATION]["observation.image"],
        observation["observation.image"],
    )
    assert torch.allclose(unnormalized_transition[TransitionKey.ACTION], action)


def test_identity_mixed_with_other_modes():
    """Test IDENTITY mode mixed with other normalization modes."""
    features = {
        "observation.image": PolicyFeature(FeatureType.VISUAL, (3,)),
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
        "action": PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    stats = {
        "observation.image": {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},  # Will be ignored
        "observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        "action": {"min": [-1.0, -1.0], "max": [1.0, 1.0]},
    }

    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    observation = {
        "observation.image": torch.tensor([0.7, 0.5, 0.3]),
        "observation.state": torch.tensor([1.0, -0.5]),
    }
    action = torch.tensor([0.5, 0.0])
    transition = create_transition(observation=observation, action=action)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]
    normalized_action = normalized_transition[TransitionKey.ACTION]

    # Image should remain unchanged (IDENTITY)
    assert torch.allclose(normalized_obs["observation.image"], observation["observation.image"])

    # State should be normalized (MEAN_STD)
    expected_state = torch.tensor([1.0, -0.5])  # (x - 0) / 1 = x
    assert torch.allclose(normalized_obs["observation.state"], expected_state)

    # Action should be normalized (MIN_MAX) to [-1, 1]
    # 2 * (0.5 - (-1)) / (1 - (-1)) - 1 = 2 * 1.5 / 2 - 1 = 0.5
    # 2 * (0.0 - (-1)) / (1 - (-1)) - 1 = 2 * 1.0 / 2 - 1 = 0.0
    expected_action = torch.tensor([0.5, 0.0])
    assert torch.allclose(normalized_action, expected_action)


def test_identity_defaults_when_not_in_norm_map():
    """Test that IDENTITY is used as default when feature type not in norm_map."""
    features = {
        "observation.image": PolicyFeature(FeatureType.VISUAL, (3,)),
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
    }
    norm_map = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        # VISUAL not specified, should default to IDENTITY
    }
    stats = {
        "observation.image": {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
        "observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
    }

    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    observation = {
        "observation.image": torch.tensor([0.7, 0.5, 0.3]),
        "observation.state": torch.tensor([1.0, -0.5]),
    }
    transition = create_transition(observation=observation)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    # Image should remain unchanged (defaults to IDENTITY)
    assert torch.allclose(normalized_obs["observation.image"], observation["observation.image"])

    # State should be normalized (explicitly MEAN_STD)
    expected_state = torch.tensor([1.0, -0.5])
    assert torch.allclose(normalized_obs["observation.state"], expected_state)


def test_identity_roundtrip():
    """Test that IDENTITY normalization and unnormalization are true inverses."""
    features = {
        "observation.image": PolicyFeature(FeatureType.VISUAL, (3,)),
        "action": PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.IDENTITY,
    }
    stats = {
        "observation.image": {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
        "action": {"min": [-1.0, -1.0], "max": [1.0, 1.0]},
    }

    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    original_observation = {"observation.image": torch.tensor([0.7, 0.5, 0.3])}
    original_action = torch.tensor([0.5, -0.2])
    original_transition = create_transition(observation=original_observation, action=original_action)

    # Normalize then unnormalize
    normalized = normalizer(original_transition)
    roundtrip = unnormalizer(normalized)

    # Should be identical to original
    assert torch.allclose(
        roundtrip[TransitionKey.OBSERVATION]["observation.image"], original_observation["observation.image"]
    )
    assert torch.allclose(roundtrip[TransitionKey.ACTION], original_action)


def test_identity_config_serialization():
    """Test that IDENTITY mode is properly saved and loaded in config."""
    features = {
        "observation.image": PolicyFeature(FeatureType.VISUAL, (3,)),
        "action": PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    stats = {
        "observation.image": {"mean": [0.5], "std": [0.2]},
        "action": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
    }

    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    # Get config
    config = normalizer.get_config()

    # Check that IDENTITY is properly serialized
    assert config["norm_map"]["VISUAL"] == "IDENTITY"
    assert config["norm_map"]["ACTION"] == "MEAN_STD"

    # Create new processor from config (simulating load)
    new_normalizer = NormalizerProcessor(
        features=config["features"],
        norm_map=config["norm_map"],
        stats=stats,
        eps=config["eps"],
    )

    # Test that both work the same way
    observation = {"observation.image": torch.tensor([0.7])}
    action = torch.tensor([1.0, -0.5])
    transition = create_transition(observation=observation, action=action)

    result1 = normalizer(transition)
    result2 = new_normalizer(transition)

    # Results should be identical
    assert torch.allclose(
        result1[TransitionKey.OBSERVATION]["observation.image"],
        result2[TransitionKey.OBSERVATION]["observation.image"],
    )
    assert torch.allclose(result1[TransitionKey.ACTION], result2[TransitionKey.ACTION])


# def test_unsupported_normalization_mode_error():
#     """Test that unsupported normalization modes raise appropriate errors."""
#     features = {"observation.state": PolicyFeature(FeatureType.STATE, (2,))}

#     # Create an invalid norm_map (this would never happen in practice, but tests error handling)
#     from enum import Enum

#     class InvalidMode(str, Enum):
#         INVALID = "INVALID"

#     # We can't actually pass an invalid enum to the processor due to type checking,
#     # but we can test the error by manipulating the norm_map after creation
#     norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD}
#     stats = {"observation.state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]}}

#     normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

#     # Manually inject an invalid mode to test error handling
#     normalizer.norm_map[FeatureType.STATE] = "INVALID_MODE"

#     observation = {"observation.state": torch.tensor([1.0, -0.5])}
#     transition = create_transition(observation=observation)

#     with pytest.raises(ValueError, match="Unsupported normalization mode"):
#         normalizer(transition)


def test_hotswap_stats_basic_functionality():
    """Test that hotswap_stats correctly updates stats in normalizer/unnormalizer steps."""
    # Create initial stats
    initial_stats = {
        "observation.image": {"mean": np.array([0.5, 0.5, 0.5]), "std": np.array([0.2, 0.2, 0.2])},
        "action": {"mean": np.array([0.0, 0.0]), "std": np.array([1.0, 1.0])},
    }

    # Create new stats for hotswapping
    new_stats = {
        "observation.image": {"mean": np.array([0.3, 0.3, 0.3]), "std": np.array([0.1, 0.1, 0.1])},
        "action": {"mean": np.array([0.1, 0.1]), "std": np.array([0.5, 0.5])},
    }

    # Create features and norm_map
    features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }

    # Create processors
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=initial_stats)
    unnormalizer = UnnormalizerProcessor(features=features, norm_map=norm_map, stats=initial_stats)
    identity = IdentityProcessor()

    # Create robot processor
    robot_processor = RobotProcessor(steps=[normalizer, unnormalizer, identity])

    # Hotswap stats
    new_processor = hotswap_stats(robot_processor, new_stats)

    # Check that normalizer and unnormalizer have new stats
    assert new_processor.steps[0].stats == new_stats
    assert new_processor.steps[1].stats == new_stats

    # Check that tensor stats are updated correctly
    expected_tensor_stats = _convert_stats_to_tensors(new_stats)
    for key in expected_tensor_stats:
        for stat_name in expected_tensor_stats[key]:
            torch.testing.assert_close(
                new_processor.steps[0]._tensor_stats[key][stat_name], expected_tensor_stats[key][stat_name]
            )
            torch.testing.assert_close(
                new_processor.steps[1]._tensor_stats[key][stat_name], expected_tensor_stats[key][stat_name]
            )


def test_hotswap_stats_deep_copy():
    """Test that hotswap_stats creates a deep copy and doesn't modify the original processor."""
    initial_stats = {
        "observation.image": {"mean": np.array([0.5, 0.5, 0.5]), "std": np.array([0.2, 0.2, 0.2])},
    }

    new_stats = {
        "observation.image": {"mean": np.array([0.3, 0.3, 0.3]), "std": np.array([0.1, 0.1, 0.1])},
    }

    features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}

    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=initial_stats)
    original_processor = RobotProcessor(steps=[normalizer])

    # Store reference to original stats
    original_stats_reference = original_processor.steps[0].stats
    original_tensor_stats_reference = original_processor.steps[0]._tensor_stats

    # Hotswap stats
    new_processor = hotswap_stats(original_processor, new_stats)

    # Original processor should be unchanged
    assert original_processor.steps[0].stats is original_stats_reference
    assert original_processor.steps[0]._tensor_stats is original_tensor_stats_reference
    assert original_processor.steps[0].stats == initial_stats

    # New processor should have new stats
    assert new_processor.steps[0].stats == new_stats
    assert new_processor.steps[0].stats is not original_stats_reference

    # Processors should be different objects
    assert new_processor is not original_processor
    assert new_processor.steps[0] is not original_processor.steps[0]


def test_hotswap_stats_only_affects_normalizer_steps():
    """Test that hotswap_stats only modifies NormalizerProcessor and UnnormalizerProcessor steps."""
    stats = {
        "observation.image": {"mean": np.array([0.5]), "std": np.array([0.2])},
    }

    new_stats = {
        "observation.image": {"mean": np.array([0.3]), "std": np.array([0.1])},
    }

    features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}

    # Create mixed steps
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessor(features=features, norm_map=norm_map, stats=stats)
    identity = IdentityProcessor()

    robot_processor = RobotProcessor(steps=[normalizer, identity, unnormalizer])

    # Hotswap stats
    new_processor = hotswap_stats(robot_processor, new_stats)

    # Check that only normalizer and unnormalizer steps are affected
    assert new_processor.steps[0].stats == new_stats  # normalizer
    assert new_processor.steps[2].stats == new_stats  # unnormalizer

    # Identity processor should remain unchanged (and it doesn't have stats attribute)
    assert not hasattr(new_processor.steps[1], "stats")


def test_hotswap_stats_empty_stats():
    """Test hotswap_stats with empty stats dictionary."""
    initial_stats = {
        "observation.image": {"mean": np.array([0.5]), "std": np.array([0.2])},
    }

    empty_stats = {}

    features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}

    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=initial_stats)
    robot_processor = RobotProcessor(steps=[normalizer])

    # Hotswap with empty stats
    new_processor = hotswap_stats(robot_processor, empty_stats)

    # Should update to empty stats
    assert new_processor.steps[0].stats == empty_stats
    assert new_processor.steps[0]._tensor_stats == {}


def test_hotswap_stats_no_normalizer_steps():
    """Test hotswap_stats with a processor that has no normalizer/unnormalizer steps."""
    stats = {
        "observation.image": {"mean": np.array([0.5]), "std": np.array([0.2])},
    }

    # Create processor with only identity steps
    robot_processor = RobotProcessor(steps=[IdentityProcessor(), IdentityProcessor()])

    # Hotswap stats - should work without error
    new_processor = hotswap_stats(robot_processor, stats)

    # Should return a different object (deep copy)
    assert new_processor is not robot_processor

    # Steps should be deep copied but unchanged
    assert len(new_processor.steps) == len(robot_processor.steps)
    for i, step in enumerate(new_processor.steps):
        assert step is not robot_processor.steps[i]  # Different objects
        assert isinstance(step, type(robot_processor.steps[i]))  # Same type


def test_hotswap_stats_preserves_other_attributes():
    """Test that hotswap_stats preserves other processor attributes like features and norm_map."""
    initial_stats = {
        "observation.image": {"mean": np.array([0.5]), "std": np.array([0.2])},
    }

    new_stats = {
        "observation.image": {"mean": np.array([0.3]), "std": np.array([0.1])},
    }

    features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}
    normalize_observation_keys = {"observation.image"}
    eps = 1e-6

    normalizer = NormalizerProcessor(
        features=features,
        norm_map=norm_map,
        stats=initial_stats,
        normalize_observation_keys=normalize_observation_keys,
        eps=eps,
    )
    robot_processor = RobotProcessor(steps=[normalizer])

    # Hotswap stats
    new_processor = hotswap_stats(robot_processor, new_stats)

    # Check that other attributes are preserved
    new_normalizer = new_processor.steps[0]
    assert new_normalizer.features == features
    assert new_normalizer.norm_map == norm_map
    assert new_normalizer.normalize_observation_keys == normalize_observation_keys
    assert new_normalizer.eps == eps

    # But stats should be updated
    assert new_normalizer.stats == new_stats


def test_hotswap_stats_multiple_normalizer_types():
    """Test hotswap_stats with multiple normalizer and unnormalizer steps."""
    initial_stats = {
        "observation.image": {"mean": np.array([0.5]), "std": np.array([0.2])},
        "action": {"min": np.array([-1.0]), "max": np.array([1.0])},
    }

    new_stats = {
        "observation.image": {"mean": np.array([0.3]), "std": np.array([0.1])},
        "action": {"min": np.array([-2.0]), "max": np.array([2.0])},
    }

    features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }

    # Create multiple normalizers and unnormalizers
    normalizer1 = NormalizerProcessor(features=features, norm_map=norm_map, stats=initial_stats)
    normalizer2 = NormalizerProcessor(features=features, norm_map=norm_map, stats=initial_stats)
    unnormalizer1 = UnnormalizerProcessor(features=features, norm_map=norm_map, stats=initial_stats)
    unnormalizer2 = UnnormalizerProcessor(features=features, norm_map=norm_map, stats=initial_stats)

    robot_processor = RobotProcessor(steps=[normalizer1, unnormalizer1, normalizer2, unnormalizer2])

    # Hotswap stats
    new_processor = hotswap_stats(robot_processor, new_stats)

    # All normalizer/unnormalizer steps should be updated
    for step in new_processor.steps:
        assert step.stats == new_stats

        # Check tensor stats conversion
        expected_tensor_stats = _convert_stats_to_tensors(new_stats)
        for key in expected_tensor_stats:
            for stat_name in expected_tensor_stats[key]:
                torch.testing.assert_close(
                    step._tensor_stats[key][stat_name], expected_tensor_stats[key][stat_name]
                )


def test_hotswap_stats_with_different_data_types():
    """Test hotswap_stats with various data types in stats."""
    initial_stats = {
        "observation.image": {"mean": np.array([0.5]), "std": np.array([0.2])},
    }

    # New stats with different data types (int, float, list, tuple)
    new_stats = {
        "observation.image": {
            "mean": [0.3, 0.4, 0.5],  # list
            "std": (0.1, 0.2, 0.3),  # tuple
            "min": 0,  # int
            "max": 1.0,  # float
        },
        "action": {
            "mean": np.array([0.1, 0.2]),  # numpy array
            "std": torch.tensor([0.5, 0.6]),  # torch tensor
        },
    }

    features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }

    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=initial_stats)
    robot_processor = RobotProcessor(steps=[normalizer])

    # Hotswap stats
    new_processor = hotswap_stats(robot_processor, new_stats)

    # Check that stats are updated
    assert new_processor.steps[0].stats == new_stats

    # Check that tensor conversion worked correctly
    tensor_stats = new_processor.steps[0]._tensor_stats
    assert isinstance(tensor_stats["observation.image"]["mean"], torch.Tensor)
    assert isinstance(tensor_stats["observation.image"]["std"], torch.Tensor)
    assert isinstance(tensor_stats["observation.image"]["min"], torch.Tensor)
    assert isinstance(tensor_stats["observation.image"]["max"], torch.Tensor)
    assert isinstance(tensor_stats["action"]["mean"], torch.Tensor)
    assert isinstance(tensor_stats["action"]["std"], torch.Tensor)

    # Check values
    torch.testing.assert_close(tensor_stats["observation.image"]["mean"], torch.tensor([0.3, 0.4, 0.5]))
    torch.testing.assert_close(tensor_stats["observation.image"]["std"], torch.tensor([0.1, 0.2, 0.3]))
    torch.testing.assert_close(tensor_stats["observation.image"]["min"], torch.tensor(0.0))
    torch.testing.assert_close(tensor_stats["observation.image"]["max"], torch.tensor(1.0))


def test_hotswap_stats_functional_test():
    """Test that hotswapped processor actually works functionally."""
    # Create test data
    observation = {
        "observation.image": torch.tensor([[[0.6, 0.7], [0.8, 0.9]], [[0.5, 0.6], [0.7, 0.8]]]),
    }
    action = torch.tensor([0.5, -0.5])
    transition = create_transition(observation=observation, action=action)

    # Initial stats
    initial_stats = {
        "observation.image": {"mean": np.array([0.5, 0.4]), "std": np.array([0.2, 0.3])},
        "action": {"mean": np.array([0.0, 0.0]), "std": np.array([1.0, 1.0])},
    }

    # New stats
    new_stats = {
        "observation.image": {"mean": np.array([0.3, 0.2]), "std": np.array([0.1, 0.2])},
        "action": {"mean": np.array([0.1, -0.1]), "std": np.array([0.5, 0.5])},
    }

    features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(2, 2, 2)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }

    # Create original processor
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=initial_stats)
    original_processor = RobotProcessor(steps=[normalizer], to_transition=lambda x: x, to_output=lambda x: x)

    # Process with original stats
    original_result = original_processor(transition)

    # Hotswap stats
    new_processor = hotswap_stats(original_processor, new_stats)

    # Process with new stats
    new_result = new_processor(transition)

    # Results should be different since normalization changed
    assert not torch.allclose(
        original_result["observation"]["observation.image"],
        new_result["observation"]["observation.image"],
        rtol=1e-3,
        atol=1e-3,
    )
    assert not torch.allclose(original_result["action"], new_result["action"], rtol=1e-3, atol=1e-3)

    # Verify that the new processor is actually using the new stats by checking internal state
    assert new_processor.steps[0].stats == new_stats
    assert torch.allclose(
        new_processor.steps[0]._tensor_stats["observation.image"]["mean"], torch.tensor([0.3, 0.2])
    )
    assert torch.allclose(
        new_processor.steps[0]._tensor_stats["observation.image"]["std"], torch.tensor([0.1, 0.2])
    )
    assert torch.allclose(new_processor.steps[0]._tensor_stats["action"]["mean"], torch.tensor([0.1, -0.1]))
    assert torch.allclose(new_processor.steps[0]._tensor_stats["action"]["std"], torch.tensor([0.5, 0.5]))

    # Test that normalization actually happens (output should not equal input)
    assert not torch.allclose(
        new_result["observation"]["observation.image"], observation["observation.image"]
    )
    assert not torch.allclose(new_result["action"], action)


def test_zero_std_uses_eps():
    """When std == 0, (x-mean)/(std+eps) is well-defined; x==mean should map to 0."""
    features = {"observation.state": PolicyFeature(FeatureType.STATE, (1,))}
    norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD}
    stats = {"observation.state": {"mean": np.array([0.5]), "std": np.array([0.0])}}
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats, eps=1e-6)

    observation = {"observation.state": torch.tensor([0.5])}  # equals mean
    out = normalizer(create_transition(observation=observation))
    assert torch.allclose(out[TransitionKey.OBSERVATION]["observation.state"], torch.tensor([0.0]))


def test_min_equals_max_maps_to_minus_one():
    """When min == max, MIN_MAX path maps to -1 after [-1,1] scaling for x==min."""
    features = {"observation.state": PolicyFeature(FeatureType.STATE, (1,))}
    norm_map = {FeatureType.STATE: NormalizationMode.MIN_MAX}
    stats = {"observation.state": {"min": np.array([2.0]), "max": np.array([2.0])}}
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats, eps=1e-6)

    observation = {"observation.state": torch.tensor([2.0])}
    out = normalizer(create_transition(observation=observation))
    assert torch.allclose(out[TransitionKey.OBSERVATION]["observation.state"], torch.tensor([-1.0]))


def test_action_normalized_despite_normalize_observation_keys():
    """Action normalization is independent of normalize_observation_keys filter for observations."""
    features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (1,)),
        "action": PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {FeatureType.STATE: NormalizationMode.IDENTITY, FeatureType.ACTION: NormalizationMode.MEAN_STD}
    stats = {"action": {"mean": np.array([1.0, -1.0]), "std": np.array([2.0, 4.0])}}
    normalizer = NormalizerProcessor(
        features=features, norm_map=norm_map, stats=stats, normalize_observation_keys={"observation.state"}
    )

    transition = create_transition(
        observation={"observation.state": torch.tensor([3.0])}, action=torch.tensor([3.0, 3.0])
    )
    out = normalizer(transition)
    # (3-1)/2 = 1.0 ; (3-(-1))/4 = 1.0
    assert torch.allclose(out[TransitionKey.ACTION], torch.tensor([1.0, 1.0]))


def test_unnormalize_observations_mean_std_and_min_max():
    features = {
        "observation.ms": PolicyFeature(FeatureType.STATE, (2,)),
        "observation.mm": PolicyFeature(FeatureType.STATE, (2,)),
    }
    # Build two processors: one mean/std and one min/max
    unnorm_ms = UnnormalizerProcessor(
        features={"observation.ms": features["observation.ms"]},
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={"observation.ms": {"mean": np.array([1.0, -1.0]), "std": np.array([2.0, 4.0])}},
    )
    unnorm_mm = UnnormalizerProcessor(
        features={"observation.mm": features["observation.mm"]},
        norm_map={FeatureType.STATE: NormalizationMode.MIN_MAX},
        stats={"observation.mm": {"min": np.array([0.0, -2.0]), "max": np.array([2.0, 2.0])}},
    )

    tr = create_transition(
        observation={
            "observation.ms": torch.tensor([0.0, 0.0]),  # → mean
            "observation.mm": torch.tensor([0.0, 0.0]),  # → mid-point
        }
    )
    out_ms = unnorm_ms(tr)[TransitionKey.OBSERVATION]["observation.ms"]
    out_mm = unnorm_mm(tr)[TransitionKey.OBSERVATION]["observation.mm"]
    assert torch.allclose(out_ms, torch.tensor([1.0, -1.0]))
    assert torch.allclose(out_mm, torch.tensor([1.0, 0.0]))  # mid of [0,2] and [-2,2]


def test_unknown_observation_keys_ignored():
    features = {"observation.state": PolicyFeature(FeatureType.STATE, (1,))}
    norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD}
    stats = {"observation.state": {"mean": np.array([0.0]), "std": np.array([1.0])}}
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    obs = {"observation.state": torch.tensor([1.0]), "observation.unknown": torch.tensor([5.0])}
    tr = create_transition(observation=obs)
    out = normalizer(tr)

    # Unknown key should pass through unchanged and not be tracked
    assert torch.allclose(out[TransitionKey.OBSERVATION]["observation.unknown"], obs["observation.unknown"])


def test_batched_action_normalization():
    features = {"action": PolicyFeature(FeatureType.ACTION, (2,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.MEAN_STD}
    stats = {"action": {"mean": np.array([1.0, -1.0]), "std": np.array([2.0, 4.0])}}
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    actions = torch.tensor([[1.0, -1.0], [3.0, 3.0]])  # first equals mean → zeros; second → [1, 1]
    out = normalizer(create_transition(action=actions))[TransitionKey.ACTION]
    expected = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    assert torch.allclose(out, expected)


def test_complementary_data_preservation():
    features = {"observation.state": PolicyFeature(FeatureType.STATE, (1,))}
    norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD}
    stats = {"observation.state": {"mean": np.array([0.0]), "std": np.array([1.0])}}
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    comp = {"existing": 123}
    tr = create_transition(observation={"observation.state": torch.tensor([1.0])}, complementary_data=comp)
    out = normalizer(tr)
    new_comp = out[TransitionKey.COMPLEMENTARY_DATA]
    assert new_comp["existing"] == 123


def test_roundtrip_normalize_unnormalize_non_identity():
    features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
        "action": PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MIN_MAX}
    stats = {
        "observation.state": {"mean": np.array([1.0, -1.0]), "std": np.array([2.0, 4.0])},
        "action": {"min": np.array([-2.0, 0.0]), "max": np.array([2.0, 4.0])},
    }
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    # Add a time dimension in action for broadcasting check (B,T,D)
    obs = {"observation.state": torch.tensor([[3.0, 3.0], [1.0, -1.0]])}
    act = torch.tensor([[[0.0, -1.0], [1.0, 1.0]]])  # shape (1,2,2) already in [-1,1]

    tr = create_transition(observation=obs, action=act)
    out = unnormalizer(normalizer(tr))

    assert torch.allclose(
        out[TransitionKey.OBSERVATION]["observation.state"], obs["observation.state"], atol=1e-5
    )
    assert torch.allclose(out[TransitionKey.ACTION], act, atol=1e-5)

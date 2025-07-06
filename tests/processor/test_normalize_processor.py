from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor.normalize_processor import (
    NormalizerProcessor,
    UnnormalizerProcessor,
    _convert_stats_to_tensors,
)
from lerobot.processor.pipeline import RobotProcessor, TransitionIndex


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
    transition = (observation, None, None, None, None, None, None)

    normalized_transition = observation_normalizer(transition)
    normalized_obs = normalized_transition[TransitionIndex.OBSERVATION]

    # Check mean/std normalization
    expected_image = (torch.tensor([0.7, 0.5, 0.3]) - 0.5) / 0.2
    assert torch.allclose(normalized_obs["observation.image"], expected_image)


def test_min_max_normalization(observation_normalizer):
    observation = {
        "observation.state": torch.tensor([0.5, 0.0]),
    }
    transition = (observation, None, None, None, None, None, None)

    normalized_transition = observation_normalizer(transition)
    normalized_obs = normalized_transition[TransitionIndex.OBSERVATION]

    # Check min/max normalization to [-1, 1]
    # For state[0]: 2 * (0.5 - 0.0) / (1.0 - 0.0) - 1 = 0.0
    # For state[1]: 2 * (0.0 - (-1.0)) / (1.0 - (-1.0)) - 1 = 0.0
    expected_state = torch.tensor([0.0, 0.0])
    assert torch.allclose(normalized_obs["observation.state"], expected_state, atol=1e-6)


def test_selective_normalization(observation_stats):
    features = _create_observation_features()
    norm_map = _create_observation_norm_map()
    normalizer = NormalizerProcessor(
        features=features, norm_map=norm_map, stats=observation_stats, normalize_keys={"observation.image"}
    )

    observation = {
        "observation.image": torch.tensor([0.7, 0.5, 0.3]),
        "observation.state": torch.tensor([0.5, 0.0]),
    }
    transition = (observation, None, None, None, None, None, None)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionIndex.OBSERVATION]

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
    transition = (observation, None, None, None, None, None, None)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionIndex.OBSERVATION]

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

    # Create new normalizer and load state
    features = _create_observation_features()
    norm_map = _create_observation_norm_map()
    new_normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats={})
    new_normalizer.load_state_dict(state_dict)

    # Test that it works the same
    observation = {"observation.image": torch.tensor([0.7, 0.5, 0.3])}
    transition = (observation, None, None, None, None, None, None)

    result1 = observation_normalizer(transition)[0]
    result2 = new_normalizer(transition)[0]

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
    transition = (None, normalized_action, None, None, None, None, None)

    unnormalized_transition = unnormalizer(transition)
    unnormalized_action = unnormalized_transition[TransitionIndex.ACTION]

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
    transition = (None, normalized_action, None, None, None, None, None)

    unnormalized_transition = unnormalizer(transition)
    unnormalized_action = unnormalized_transition[TransitionIndex.ACTION]

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
    transition = (None, normalized_action, None, None, None, None, None)

    unnormalized_transition = unnormalizer(transition)
    unnormalized_action = unnormalized_transition[TransitionIndex.ACTION]

    assert isinstance(unnormalized_action, torch.Tensor)
    expected = torch.tensor([1.0, -1.0, 1.0])
    assert torch.allclose(unnormalized_action, expected)


def test_none_action(action_stats_mean_std):
    features = _create_action_features()
    norm_map = _create_action_norm_map_mean_std()
    unnormalizer = UnnormalizerProcessor(
        features=features, norm_map=norm_map, stats={"action": action_stats_mean_std}
    )

    transition = (None, None, None, None, None, None, None)
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
    transition = (observation, action, 1.0, False, False, {}, {})

    processed_transition = normalizer_processor(transition)

    # Check normalized observations
    processed_obs = processed_transition[TransitionIndex.OBSERVATION]
    expected_image = (torch.tensor([0.7, 0.5, 0.3]) - 0.5) / 0.2
    assert torch.allclose(processed_obs["observation.image"], expected_image)

    # Check normalized action
    processed_action = processed_transition[TransitionIndex.ACTION]
    expected_action = torch.tensor([(1.0 - 0.0) / 1.0, (-0.5 - 0.0) / 2.0])
    assert torch.allclose(processed_action, expected_action)

    # Check other fields remain unchanged
    assert processed_transition[TransitionIndex.REWARD] == 1.0
    assert not processed_transition[TransitionIndex.DONE]


def test_processor_from_lerobot_dataset(full_stats):
    # Mock dataset
    mock_dataset = Mock()
    mock_dataset.meta.stats = full_stats

    features = _create_full_features()
    norm_map = _create_full_norm_map()

    processor = NormalizerProcessor.from_lerobot_dataset(
        mock_dataset, features, norm_map, normalize_keys={"observation.image"}
    )

    assert processor.normalize_keys == {"observation.image"}
    assert "observation.image" in processor._tensor_stats
    assert "action" in processor._tensor_stats


def test_get_config(full_stats):
    features = _create_full_features()
    norm_map = _create_full_norm_map()
    processor = NormalizerProcessor(
        features=features, norm_map=norm_map, stats=full_stats, normalize_keys={"observation.image"}, eps=1e-6
    )

    config = processor.get_config()
    assert config == {"normalize_keys": ["observation.image"], "eps": 1e-6}


def test_integration_with_robot_processor(normalizer_processor):
    """Test integration with RobotProcessor pipeline"""
    robot_processor = RobotProcessor([normalizer_processor])

    observation = {
        "observation.image": torch.tensor([0.7, 0.5, 0.3]),
        "observation.state": torch.tensor([0.5, 0.0]),
    }
    action = torch.tensor([1.0, -0.5])
    transition = (observation, action, 1.0, False, False, {}, {})

    processed_transition = robot_processor(transition)

    # Verify the processing worked
    assert isinstance(processed_transition[TransitionIndex.OBSERVATION], dict)
    assert isinstance(processed_transition[TransitionIndex.ACTION], torch.Tensor)


# Edge case tests
def test_empty_observation():
    stats = {"observation.image": {"mean": [0.5], "std": [0.2]}}
    features = {"observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96))}
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)

    transition = (None, None, None, None, None, None, None)
    result = normalizer(transition)

    assert result == transition


def test_empty_stats():
    features = {"observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96))}
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats={})
    observation = {"observation.image": torch.tensor([0.5])}
    transition = (observation, None, None, None, None, None, None)

    result = normalizer(transition)
    # Should return observation unchanged since no stats are available
    assert torch.allclose(result[0]["observation.image"], observation["observation.image"])


def test_partial_stats():
    """If statistics are incomplete, the value should pass through unchanged."""
    stats = {"observation.image": {"mean": [0.5]}}  # Missing std / (min,max)
    features = {"observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96))}
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}
    normalizer = NormalizerProcessor(features=features, norm_map=norm_map, stats=stats)
    observation = {"observation.image": torch.tensor([0.7])}
    transition = (observation, None, None, None, None, None, None)

    processed = normalizer(transition)[TransitionIndex.OBSERVATION]
    assert torch.allclose(processed["observation.image"], observation["observation.image"])


def test_missing_action_stats_no_error():
    mock_dataset = Mock()
    mock_dataset.meta.stats = {"observation.image": {"mean": [0.5], "std": [0.2]}}

    features = {"observation.image": PolicyFeature(FeatureType.VISUAL, (3, 96, 96))}
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}

    processor = UnnormalizerProcessor.from_lerobot_dataset(mock_dataset, features, norm_map)
    # The tensor stats should not contain the 'action' key
    assert "action" not in processor._tensor_stats

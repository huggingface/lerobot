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
from lerobot.processor import (
    DataProcessorPipeline,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
    hotswap_stats,
)
from lerobot.processor.converters import create_transition, identity_transition, to_tensor
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE, OBS_STR
from lerobot.utils.utils import auto_select_torch_device


def test_numpy_conversion():
    stats = {
        OBS_IMAGE: {
            "mean": np.array([0.5, 0.5, 0.5]),
            "std": np.array([0.2, 0.2, 0.2]),
        }
    }
    tensor_stats = to_tensor(stats)

    assert isinstance(tensor_stats[OBS_IMAGE]["mean"], torch.Tensor)
    assert isinstance(tensor_stats[OBS_IMAGE]["std"], torch.Tensor)
    assert torch.allclose(tensor_stats[OBS_IMAGE]["mean"], torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(tensor_stats[OBS_IMAGE]["std"], torch.tensor([0.2, 0.2, 0.2]))


def test_tensor_conversion():
    stats = {
        ACTION: {
            "mean": torch.tensor([0.0, 0.0]),
            "std": torch.tensor([1.0, 1.0]),
        }
    }
    tensor_stats = to_tensor(stats)

    assert tensor_stats[ACTION]["mean"].dtype == torch.float32
    assert tensor_stats[ACTION]["std"].dtype == torch.float32


def test_scalar_conversion():
    stats = {
        "reward": {
            "mean": 0.5,
            "std": 0.1,
        }
    }
    tensor_stats = to_tensor(stats)

    assert torch.allclose(tensor_stats["reward"]["mean"], torch.tensor(0.5))
    assert torch.allclose(tensor_stats["reward"]["std"], torch.tensor(0.1))


def test_list_conversion():
    stats = {
        OBS_STATE: {
            "min": [0.0, -1.0, -2.0],
            "max": [1.0, 1.0, 2.0],
        }
    }
    tensor_stats = to_tensor(stats)

    assert torch.allclose(tensor_stats[OBS_STATE]["min"], torch.tensor([0.0, -1.0, -2.0]))
    assert torch.allclose(tensor_stats[OBS_STATE]["max"], torch.tensor([1.0, 1.0, 2.0]))


def test_unsupported_type():
    stats = {
        "bad_key": {
            "mean": "string_value",
        }
    }
    with pytest.raises(TypeError, match="Unsupported type"):
        to_tensor(stats)


# Helper functions to create feature maps and norm maps
def _create_observation_features():
    return {
        OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
    }


def _create_observation_norm_map():
    return {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.STATE: NormalizationMode.MIN_MAX,
    }


# Fixtures for observation normalisation tests using NormalizerProcessorStep
@pytest.fixture
def observation_stats():
    return {
        OBS_IMAGE: {
            "mean": np.array([0.5, 0.5, 0.5]),
            "std": np.array([0.2, 0.2, 0.2]),
        },
        OBS_STATE: {
            "min": np.array([0.0, -1.0]),
            "max": np.array([1.0, 1.0]),
        },
    }


@pytest.fixture
def observation_normalizer(observation_stats):
    """Return a NormalizerProcessorStep that only has observation stats (no action)."""
    features = _create_observation_features()
    norm_map = _create_observation_norm_map()
    return NormalizerProcessorStep(features=features, norm_map=norm_map, stats=observation_stats)


def test_mean_std_normalization(observation_normalizer):
    observation = {
        OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3]),
        OBS_STATE: torch.tensor([0.5, 0.0]),
    }
    transition = create_transition(observation=observation)

    normalized_transition = observation_normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    # Check mean/std normalization
    expected_image = (torch.tensor([0.7, 0.5, 0.3]) - 0.5) / 0.2
    assert torch.allclose(normalized_obs[OBS_IMAGE], expected_image)


def test_min_max_normalization(observation_normalizer):
    observation = {
        OBS_STATE: torch.tensor([0.5, 0.0]),
    }
    transition = create_transition(observation=observation)

    normalized_transition = observation_normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    # Check min/max normalization to [-1, 1]
    # For state[0]: 2 * (0.5 - 0.0) / (1.0 - 0.0) - 1 = 0.0
    # For state[1]: 2 * (0.0 - (-1.0)) / (1.0 - (-1.0)) - 1 = 0.0
    expected_state = torch.tensor([0.0, 0.0])
    assert torch.allclose(normalized_obs[OBS_STATE], expected_state, atol=1e-6)


def test_quantile_normalization():
    """Test QUANTILES mode using 1st-99th percentiles."""
    features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
    }
    norm_map = {
        FeatureType.STATE: NormalizationMode.QUANTILES,
    }
    stats = {
        "observation.state": {
            "q01": np.array([0.1, -0.8]),  # 1st percentile
            "q99": np.array([0.9, 0.8]),  # 99th percentile
        },
    }

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    observation = {
        "observation.state": torch.tensor([0.5, 0.0]),
    }
    transition = create_transition(observation=observation)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    # Check quantile normalization to [-1, 1]
    # For state[0]: 2 * (0.5 - 0.1) / (0.9 - 0.1) - 1 = 2 * 0.4 / 0.8 - 1 = 0.0
    # For state[1]: 2 * (0.0 - (-0.8)) / (0.8 - (-0.8)) - 1 = 2 * 0.8 / 1.6 - 1 = 0.0
    expected_state = torch.tensor([0.0, 0.0])
    assert torch.allclose(normalized_obs["observation.state"], expected_state, atol=1e-6)


def test_quantile10_normalization():
    """Test QUANTILE10 mode using 10th-90th percentiles."""
    features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
    }
    norm_map = {
        FeatureType.STATE: NormalizationMode.QUANTILE10,
    }
    stats = {
        "observation.state": {
            "q10": np.array([0.2, -0.6]),  # 10th percentile
            "q90": np.array([0.8, 0.6]),  # 90th percentile
        },
    }

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    observation = {
        "observation.state": torch.tensor([0.5, 0.0]),
    }
    transition = create_transition(observation=observation)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    # Check quantile normalization to [-1, 1]
    # For state[0]: 2 * (0.5 - 0.2) / (0.8 - 0.2) - 1 = 2 * 0.3 / 0.6 - 1 = 0.0
    # For state[1]: 2 * (0.0 - (-0.6)) / (0.6 - (-0.6)) - 1 = 2 * 0.6 / 1.2 - 1 = 0.0
    expected_state = torch.tensor([0.0, 0.0])
    assert torch.allclose(normalized_obs["observation.state"], expected_state, atol=1e-6)


def test_quantile_unnormalization():
    """Test that quantile normalization can be reversed properly."""
    features = {
        "action": PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {
        FeatureType.ACTION: NormalizationMode.QUANTILES,
    }
    stats = {
        "action": {
            "q01": np.array([0.1, -0.8]),
            "q99": np.array([0.9, 0.8]),
        },
    }

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    # Test round-trip normalization
    original_action = torch.tensor([0.5, 0.0])
    transition = create_transition(action=original_action)

    # Normalize then unnormalize
    normalized = normalizer(transition)
    unnormalized = unnormalizer(normalized)

    # Should recover original values
    recovered_action = unnormalized[TransitionKey.ACTION]
    assert torch.allclose(recovered_action, original_action, atol=1e-6)


def test_quantile_division_by_zero():
    """Test quantile normalization handles edge case where q01 == q99."""
    features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (1,)),
    }
    norm_map = {
        FeatureType.STATE: NormalizationMode.QUANTILES,
    }
    stats = {
        "observation.state": {
            "q01": np.array([0.5]),  # Same value
            "q99": np.array([0.5]),  # Same value -> division by zero case
        },
    }

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    observation = {
        "observation.state": torch.tensor([0.5]),
    }
    transition = create_transition(observation=observation)

    # Should not crash and should handle gracefully
    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    # When quantiles are identical, should normalize to 0 (due to epsilon handling)
    assert torch.isfinite(normalized_obs["observation.state"]).all()


def test_quantile_partial_stats():
    """Test that quantile normalization handles missing quantile stats by raising."""
    features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
    }
    norm_map = {
        FeatureType.STATE: NormalizationMode.QUANTILES,
    }

    # Missing q99 - should pass through unchanged
    stats_partial = {
        "observation.state": {
            "q01": np.array([0.1, -0.8]),  # Only q01, missing q99
        },
    }

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats_partial)

    observation = {
        "observation.state": torch.tensor([0.5, 0.0]),
    }
    transition = create_transition(observation=observation)

    with pytest.raises(ValueError, match="QUANTILES normalization mode requires q01 and q99 stats"):
        _ = normalizer(transition)


def test_quantile_mixed_with_other_modes():
    """Test quantile normalization mixed with other normalization modes."""
    features = {
        "observation.image": PolicyFeature(FeatureType.VISUAL, (3,)),
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
        "action": PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,  # Standard normalization
        FeatureType.STATE: NormalizationMode.QUANTILES,  # Quantile normalization
        FeatureType.ACTION: NormalizationMode.QUANTILE10,  # Different quantile mode
    }
    stats = {
        "observation.image": {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
        "observation.state": {"q01": [0.1, -0.8], "q99": [0.9, 0.8]},
        "action": {"q10": [0.2, -0.6], "q90": [0.8, 0.6]},
    }

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    observation = {
        "observation.image": torch.tensor([0.7, 0.5, 0.3]),
        "observation.state": torch.tensor([0.5, 0.0]),  # Should use QUANTILES
    }
    action = torch.tensor([0.5, 0.0])  # Should use QUANTILE10
    transition = create_transition(observation=observation, action=action)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]
    normalized_action = normalized_transition[TransitionKey.ACTION]

    # Image should be mean/std normalized: (0.7 - 0.5) / 0.2 = 1.0, etc.
    expected_image = (torch.tensor([0.7, 0.5, 0.3]) - 0.5) / 0.2
    assert torch.allclose(normalized_obs["observation.image"], expected_image)

    # State should be quantile normalized: 2 * (0.5 - 0.1) / (0.9 - 0.1) - 1 = 0.0, etc.
    expected_state = torch.tensor([0.0, 0.0])
    assert torch.allclose(normalized_obs["observation.state"], expected_state, atol=1e-6)

    # Action should be quantile10 normalized: 2 * (0.5 - 0.2) / (0.8 - 0.2) - 1 = 0.0, etc.
    expected_action = torch.tensor([0.0, 0.0])
    assert torch.allclose(normalized_action, expected_action, atol=1e-6)


def test_quantile_with_missing_stats():
    """Test that quantile normalization handles completely missing stats gracefully."""
    features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
    }
    norm_map = {
        FeatureType.STATE: NormalizationMode.QUANTILES,
    }
    stats = {}  # No stats provided

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    observation = {
        "observation.state": torch.tensor([0.5, 0.0]),
    }
    transition = create_transition(observation=observation)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    # Should pass through unchanged when no stats available
    assert torch.allclose(normalized_obs["observation.state"], observation["observation.state"])


def test_selective_normalization(observation_stats):
    features = _create_observation_features()
    norm_map = _create_observation_norm_map()
    normalizer = NormalizerProcessorStep(
        features=features,
        norm_map=norm_map,
        stats=observation_stats,
        normalize_observation_keys={OBS_IMAGE},
    )

    observation = {
        OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3]),
        OBS_STATE: torch.tensor([0.5, 0.0]),
    }
    transition = create_transition(observation=observation)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    # Only image should be normalized
    assert torch.allclose(normalized_obs[OBS_IMAGE], (torch.tensor([0.7, 0.5, 0.3]) - 0.5) / 0.2)
    # State should remain unchanged
    assert torch.allclose(normalized_obs[OBS_STATE], observation[OBS_STATE])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_compatibility(observation_stats):
    features = _create_observation_features()
    norm_map = _create_observation_norm_map()
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=observation_stats)
    observation = {
        OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3]).cuda(),
    }
    transition = create_transition(observation=observation)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    assert normalized_obs[OBS_IMAGE].device.type == "cuda"


def test_from_lerobot_dataset():
    # Mock dataset
    mock_dataset = Mock()
    mock_dataset.meta.stats = {
        OBS_IMAGE: {"mean": [0.5], "std": [0.2]},
        ACTION: {"mean": [0.0], "std": [1.0]},
    }

    features = {
        OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        ACTION: PolicyFeature(FeatureType.ACTION, (1,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }

    normalizer = NormalizerProcessorStep.from_lerobot_dataset(mock_dataset, features, norm_map)

    # Both observation and action statistics should be present in tensor stats
    assert OBS_IMAGE in normalizer._tensor_stats
    assert ACTION in normalizer._tensor_stats


def test_state_dict_save_load(observation_normalizer):
    # Save state
    state_dict = observation_normalizer.state_dict()
    print("State dict:", state_dict)

    # Create new normalizer and load state
    features = _create_observation_features()
    norm_map = _create_observation_norm_map()
    new_normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats={})
    new_normalizer.load_state_dict(state_dict)

    # Test that it works the same
    observation = {OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3])}
    transition = create_transition(observation=observation)

    result1 = observation_normalizer(transition)[TransitionKey.OBSERVATION]
    result2 = new_normalizer(transition)[TransitionKey.OBSERVATION]

    assert torch.allclose(result1[OBS_IMAGE], result2[OBS_IMAGE])


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
        ACTION: PolicyFeature(FeatureType.ACTION, (3,)),
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
    unnormalizer = UnnormalizerProcessorStep(
        features=features, norm_map=norm_map, stats={ACTION: action_stats_mean_std}
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
    unnormalizer = UnnormalizerProcessorStep(
        features=features, norm_map=norm_map, stats={ACTION: action_stats_min_max}
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


def test_tensor_action_input(action_stats_mean_std):
    features = _create_action_features()
    norm_map = _create_action_norm_map_mean_std()
    unnormalizer = UnnormalizerProcessorStep(
        features=features, norm_map=norm_map, stats={ACTION: action_stats_mean_std}
    )

    normalized_action = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float32)
    transition = create_transition(action=normalized_action)

    unnormalized_transition = unnormalizer(transition)
    unnormalized_action = unnormalized_transition[TransitionKey.ACTION]

    assert isinstance(unnormalized_action, torch.Tensor)
    expected = torch.tensor([1.0, -1.0, 1.0])
    assert torch.allclose(unnormalized_action, expected)


def test_none_action(action_stats_mean_std):
    features = _create_action_features()
    norm_map = _create_action_norm_map_mean_std()
    unnormalizer = UnnormalizerProcessorStep(
        features=features, norm_map=norm_map, stats={ACTION: action_stats_mean_std}
    )

    transition = create_transition()
    result = unnormalizer(transition)

    # Should return transition unchanged
    assert result == transition


def test_action_from_lerobot_dataset():
    mock_dataset = Mock()
    mock_dataset.meta.stats = {ACTION: {"mean": [0.0], "std": [1.0]}}
    features = {ACTION: PolicyFeature(FeatureType.ACTION, (1,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.MEAN_STD}
    unnormalizer = UnnormalizerProcessorStep.from_lerobot_dataset(mock_dataset, features, norm_map)
    assert "mean" in unnormalizer._tensor_stats[ACTION]


# Fixtures for NormalizerProcessorStep tests
@pytest.fixture
def full_stats():
    return {
        OBS_IMAGE: {
            "mean": np.array([0.5, 0.5, 0.5]),
            "std": np.array([0.2, 0.2, 0.2]),
        },
        OBS_STATE: {
            "min": np.array([0.0, -1.0]),
            "max": np.array([1.0, 1.0]),
        },
        ACTION: {
            "mean": np.array([0.0, 0.0]),
            "std": np.array([1.0, 2.0]),
        },
    }


def _create_full_features():
    return {
        OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
        ACTION: PolicyFeature(FeatureType.ACTION, (2,)),
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
    return NormalizerProcessorStep(features=features, norm_map=norm_map, stats=full_stats)


def test_combined_normalization(normalizer_processor):
    observation = {
        OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3]),
        OBS_STATE: torch.tensor([0.5, 0.0]),
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
    assert torch.allclose(processed_obs[OBS_IMAGE], expected_image)

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

    processor = NormalizerProcessorStep.from_lerobot_dataset(
        mock_dataset, features, norm_map, normalize_observation_keys={OBS_IMAGE}
    )

    assert processor.normalize_observation_keys == {OBS_IMAGE}
    assert OBS_IMAGE in processor._tensor_stats
    assert ACTION in processor._tensor_stats


def test_get_config(full_stats):
    features = _create_full_features()
    norm_map = _create_full_norm_map()
    processor = NormalizerProcessorStep(
        features=features,
        norm_map=norm_map,
        stats=full_stats,
        normalize_observation_keys={OBS_IMAGE},
        eps=1e-6,
    )

    config = processor.get_config()
    expected_config = {
        "normalize_observation_keys": [OBS_IMAGE],
        "eps": 1e-6,
        "features": {
            OBS_IMAGE: {"type": "VISUAL", "shape": (3, 96, 96)},
            OBS_STATE: {"type": "STATE", "shape": (2,)},
            ACTION: {"type": "ACTION", "shape": (2,)},
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
    robot_processor = DataProcessorPipeline(
        [normalizer_processor], to_transition=identity_transition, to_output=identity_transition
    )

    observation = {
        OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3]),
        OBS_STATE: torch.tensor([0.5, 0.0]),
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
    stats = {OBS_IMAGE: {"mean": [0.5], "std": [0.2]}}
    features = {OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 96, 96))}
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    transition = create_transition()
    result = normalizer(transition)

    assert result == transition


def test_empty_stats():
    features = {OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 96, 96))}
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats={})
    observation = {OBS_IMAGE: torch.tensor([0.5])}
    transition = create_transition(observation=observation)

    result = normalizer(transition)
    # Should return observation unchanged since no stats are available
    assert torch.allclose(result[TransitionKey.OBSERVATION][OBS_IMAGE], observation[OBS_IMAGE])


def test_partial_stats():
    """If statistics are incomplete, we should raise."""
    stats = {OBS_IMAGE: {"mean": [0.5]}}  # Missing std / (min,max)
    features = {OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 96, 96))}
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    observation = {OBS_IMAGE: torch.tensor([0.7])}
    transition = create_transition(observation=observation)

    with pytest.raises(ValueError, match="MEAN_STD normalization mode requires mean and std stats"):
        _ = normalizer(transition)[TransitionKey.OBSERVATION]


def test_missing_action_stats_no_error():
    mock_dataset = Mock()
    mock_dataset.meta.stats = {OBS_IMAGE: {"mean": [0.5], "std": [0.2]}}

    features = {OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 96, 96))}
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}

    processor = UnnormalizerProcessorStep.from_lerobot_dataset(mock_dataset, features, norm_map)
    # The tensor stats should not contain the 'action' key
    assert ACTION not in processor._tensor_stats


def test_serialization_roundtrip(full_stats):
    """Test that features and norm_map can be serialized and deserialized correctly."""
    features = _create_full_features()
    norm_map = _create_full_norm_map()
    original_processor = NormalizerProcessorStep(
        features=features,
        norm_map=norm_map,
        stats=full_stats,
        normalize_observation_keys={OBS_IMAGE},
        eps=1e-6,
    )

    # Get config (serialization)
    config = original_processor.get_config()

    # Create a new processor from the config (deserialization)
    new_processor = NormalizerProcessorStep(
        features=config["features"],
        norm_map=config["norm_map"],
        stats=full_stats,
        normalize_observation_keys=set(config["normalize_observation_keys"]),
        eps=config["eps"],
    )

    # Test that both processors work the same way
    observation = {
        OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3]),
        OBS_STATE: torch.tensor([0.5, 0.0]),
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
        result1[TransitionKey.OBSERVATION][OBS_IMAGE],
        result2[TransitionKey.OBSERVATION][OBS_IMAGE],
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
        OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,  # IDENTITY mode
        FeatureType.STATE: NormalizationMode.MEAN_STD,  # Normal mode for comparison
    }
    stats = {
        OBS_IMAGE: {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
        OBS_STATE: {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
    }

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    observation = {
        OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3]),
        OBS_STATE: torch.tensor([1.0, -0.5]),
    }
    transition = create_transition(observation=observation)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    # Image should remain unchanged (IDENTITY)
    assert torch.allclose(normalized_obs[OBS_IMAGE], observation[OBS_IMAGE])

    # State should be normalized (MEAN_STD)
    expected_state = (torch.tensor([1.0, -0.5]) - torch.tensor([0.0, 0.0])) / torch.tensor([1.0, 1.0])
    assert torch.allclose(normalized_obs[OBS_STATE], expected_state)


def test_identity_normalization_actions():
    """Test that IDENTITY mode skips normalization for actions."""
    features = {ACTION: PolicyFeature(FeatureType.ACTION, (2,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.IDENTITY}
    stats = {ACTION: {"mean": [0.0, 0.0], "std": [1.0, 2.0]}}

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    action = torch.tensor([1.0, -0.5])
    transition = create_transition(action=action)

    normalized_transition = normalizer(transition)

    # Action should remain unchanged
    assert torch.allclose(normalized_transition[TransitionKey.ACTION], action)


def test_identity_unnormalization_observations():
    """Test that IDENTITY mode skips unnormalization for observations."""
    features = {
        OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,  # IDENTITY mode
        FeatureType.STATE: NormalizationMode.MIN_MAX,  # Normal mode for comparison
    }
    stats = {
        OBS_IMAGE: {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
        OBS_STATE: {"min": [-1.0, -1.0], "max": [1.0, 1.0]},
    }

    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    observation = {
        OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3]),
        OBS_STATE: torch.tensor([0.0, -1.0]),  # Normalized values in [-1, 1]
    }
    transition = create_transition(observation=observation)

    unnormalized_transition = unnormalizer(transition)
    unnormalized_obs = unnormalized_transition[TransitionKey.OBSERVATION]

    # Image should remain unchanged (IDENTITY)
    assert torch.allclose(unnormalized_obs[OBS_IMAGE], observation[OBS_IMAGE])

    # State should be unnormalized (MIN_MAX)
    # (0.0 + 1) / 2 * (1.0 - (-1.0)) + (-1.0) = 0.0
    # (-1.0 + 1) / 2 * (1.0 - (-1.0)) + (-1.0) = -1.0
    expected_state = torch.tensor([0.0, -1.0])
    assert torch.allclose(unnormalized_obs[OBS_STATE], expected_state)


def test_identity_unnormalization_actions():
    """Test that IDENTITY mode skips unnormalization for actions."""
    features = {ACTION: PolicyFeature(FeatureType.ACTION, (2,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.IDENTITY}
    stats = {ACTION: {"min": [-1.0, -2.0], "max": [1.0, 2.0]}}

    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    action = torch.tensor([0.5, -0.8])  # Normalized values
    transition = create_transition(action=action)

    unnormalized_transition = unnormalizer(transition)

    # Action should remain unchanged
    assert torch.allclose(unnormalized_transition[TransitionKey.ACTION], action)


def test_identity_with_missing_stats():
    """Test that IDENTITY mode works even when stats are missing."""
    features = {
        OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        ACTION: PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.IDENTITY,
    }
    stats = {}  # No stats provided

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    observation = {OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3])}
    action = torch.tensor([1.0, -0.5])
    transition = create_transition(observation=observation, action=action)

    # Both should work without errors and return unchanged data
    normalized_transition = normalizer(transition)
    unnormalized_transition = unnormalizer(transition)

    assert torch.allclose(
        normalized_transition[TransitionKey.OBSERVATION][OBS_IMAGE],
        observation[OBS_IMAGE],
    )
    assert torch.allclose(normalized_transition[TransitionKey.ACTION], action)
    assert torch.allclose(
        unnormalized_transition[TransitionKey.OBSERVATION][OBS_IMAGE],
        observation[OBS_IMAGE],
    )
    assert torch.allclose(unnormalized_transition[TransitionKey.ACTION], action)


def test_identity_mixed_with_other_modes():
    """Test IDENTITY mode mixed with other normalization modes."""
    features = {
        OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3,)),
        OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
        ACTION: PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    stats = {
        OBS_IMAGE: {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},  # Will be ignored
        OBS_STATE: {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        ACTION: {"min": [-1.0, -1.0], "max": [1.0, 1.0]},
    }

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    observation = {
        OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3]),
        OBS_STATE: torch.tensor([1.0, -0.5]),
    }
    action = torch.tensor([0.5, 0.0])
    transition = create_transition(observation=observation, action=action)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]
    normalized_action = normalized_transition[TransitionKey.ACTION]

    # Image should remain unchanged (IDENTITY)
    assert torch.allclose(normalized_obs[OBS_IMAGE], observation[OBS_IMAGE])

    # State should be normalized (MEAN_STD)
    expected_state = torch.tensor([1.0, -0.5])  # (x - 0) / 1 = x
    assert torch.allclose(normalized_obs[OBS_STATE], expected_state)

    # Action should be normalized (MIN_MAX) to [-1, 1]
    # 2 * (0.5 - (-1)) / (1 - (-1)) - 1 = 2 * 1.5 / 2 - 1 = 0.5
    # 2 * (0.0 - (-1)) / (1 - (-1)) - 1 = 2 * 1.0 / 2 - 1 = 0.0
    expected_action = torch.tensor([0.5, 0.0])
    assert torch.allclose(normalized_action, expected_action)


def test_identity_defaults_when_not_in_norm_map():
    """Test that IDENTITY is used as default when feature type not in norm_map."""
    features = {
        OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3,)),
        OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
    }
    norm_map = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        # VISUAL not specified, should default to IDENTITY
    }
    stats = {
        OBS_IMAGE: {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
        OBS_STATE: {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
    }

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    observation = {
        OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3]),
        OBS_STATE: torch.tensor([1.0, -0.5]),
    }
    transition = create_transition(observation=observation)

    normalized_transition = normalizer(transition)
    normalized_obs = normalized_transition[TransitionKey.OBSERVATION]

    # Image should remain unchanged (defaults to IDENTITY)
    assert torch.allclose(normalized_obs[OBS_IMAGE], observation[OBS_IMAGE])

    # State should be normalized (explicitly MEAN_STD)
    expected_state = torch.tensor([1.0, -0.5])
    assert torch.allclose(normalized_obs[OBS_STATE], expected_state)


def test_identity_roundtrip():
    """Test that IDENTITY normalization and unnormalization are true inverses."""
    features = {
        OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3,)),
        ACTION: PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.IDENTITY,
    }
    stats = {
        OBS_IMAGE: {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]},
        ACTION: {"min": [-1.0, -1.0], "max": [1.0, 1.0]},
    }

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    original_observation = {OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3])}
    original_action = torch.tensor([0.5, -0.2])
    original_transition = create_transition(observation=original_observation, action=original_action)

    # Normalize then unnormalize
    normalized = normalizer(original_transition)
    roundtrip = unnormalizer(normalized)

    # Should be identical to original
    assert torch.allclose(roundtrip[TransitionKey.OBSERVATION][OBS_IMAGE], original_observation[OBS_IMAGE])
    assert torch.allclose(roundtrip[TransitionKey.ACTION], original_action)


def test_identity_config_serialization():
    """Test that IDENTITY mode is properly saved and loaded in config."""
    features = {
        OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3,)),
        ACTION: PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    stats = {
        OBS_IMAGE: {"mean": [0.5], "std": [0.2]},
        ACTION: {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
    }

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    # Get config
    config = normalizer.get_config()

    # Check that IDENTITY is properly serialized
    assert config["norm_map"]["VISUAL"] == "IDENTITY"
    assert config["norm_map"]["ACTION"] == "MEAN_STD"

    # Create new processor from config (simulating load)
    new_normalizer = NormalizerProcessorStep(
        features=config["features"],
        norm_map=config["norm_map"],
        stats=stats,
        eps=config["eps"],
    )

    # Test that both work the same way
    observation = {OBS_IMAGE: torch.tensor([0.7])}
    action = torch.tensor([1.0, -0.5])
    transition = create_transition(observation=observation, action=action)

    result1 = normalizer(transition)
    result2 = new_normalizer(transition)

    # Results should be identical
    assert torch.allclose(
        result1[TransitionKey.OBSERVATION][OBS_IMAGE],
        result2[TransitionKey.OBSERVATION][OBS_IMAGE],
    )
    assert torch.allclose(result1[TransitionKey.ACTION], result2[TransitionKey.ACTION])


# def test_unsupported_normalization_mode_error():
#     """Test that unsupported normalization modes raise appropriate errors."""
#     features = {OBS_STATE: PolicyFeature(FeatureType.STATE, (2,))}

#     # Create an invalid norm_map (this would never happen in practice, but tests error handling)
#     from enum import Enum

#     class InvalidMode(str, Enum):
#         INVALID = "INVALID"

#     # We can't actually pass an invalid enum to the processor due to type checking,
#     # but we can test the error by manipulating the norm_map after creation
#     norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD}
#     stats = {OBS_STATE: {"mean": [0.0, 0.0], "std": [1.0, 1.0]}}

#     normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

#     # Manually inject an invalid mode to test error handling
#     normalizer.norm_map[FeatureType.STATE] = "INVALID_MODE"

#     observation = {OBS_STATE: torch.tensor([1.0, -0.5])}
#     transition = create_transition(observation=observation)

#     with pytest.raises(ValueError, match="Unsupported normalization mode"):
#         normalizer(transition)


def test_hotswap_stats_basic_functionality():
    """Test that hotswap_stats correctly updates stats in normalizer/unnormalizer steps."""
    # Create initial stats
    initial_stats = {
        OBS_IMAGE: {"mean": np.array([0.5, 0.5, 0.5]), "std": np.array([0.2, 0.2, 0.2])},
        ACTION: {"mean": np.array([0.0, 0.0]), "std": np.array([1.0, 1.0])},
    }

    # Create new stats for hotswapping
    new_stats = {
        OBS_IMAGE: {"mean": np.array([0.3, 0.3, 0.3]), "std": np.array([0.1, 0.1, 0.1])},
        ACTION: {"mean": np.array([0.1, 0.1]), "std": np.array([0.5, 0.5])},
    }

    # Create features and norm_map
    features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }

    # Create processors
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=initial_stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=initial_stats)
    identity = IdentityProcessorStep()

    # Create robot processor
    robot_processor = DataProcessorPipeline(steps=[normalizer, unnormalizer, identity])

    # Hotswap stats
    new_processor = hotswap_stats(robot_processor, new_stats)

    # Check that normalizer and unnormalizer have new stats
    assert new_processor.steps[0].stats == new_stats
    assert new_processor.steps[1].stats == new_stats

    # Check that tensor stats are updated correctly
    expected_tensor_stats = to_tensor(new_stats)
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
        OBS_IMAGE: {"mean": np.array([0.5, 0.5, 0.5]), "std": np.array([0.2, 0.2, 0.2])},
    }

    new_stats = {
        OBS_IMAGE: {"mean": np.array([0.3, 0.3, 0.3]), "std": np.array([0.1, 0.1, 0.1])},
    }

    features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=initial_stats)
    original_processor = DataProcessorPipeline(steps=[normalizer])

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
    """Test that hotswap_stats only modifies NormalizerProcessorStep and UnnormalizerProcessorStep steps."""
    stats = {
        OBS_IMAGE: {"mean": np.array([0.5]), "std": np.array([0.2])},
    }

    new_stats = {
        OBS_IMAGE: {"mean": np.array([0.3]), "std": np.array([0.1])},
    }

    features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}

    # Create mixed steps
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    identity = IdentityProcessorStep()

    robot_processor = DataProcessorPipeline(steps=[normalizer, identity, unnormalizer])

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
        OBS_IMAGE: {"mean": np.array([0.5]), "std": np.array([0.2])},
    }

    empty_stats = {}

    features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=initial_stats)
    robot_processor = DataProcessorPipeline(steps=[normalizer])

    # Hotswap with empty stats
    new_processor = hotswap_stats(robot_processor, empty_stats)

    # Should update to empty stats
    assert new_processor.steps[0].stats == empty_stats
    assert new_processor.steps[0]._tensor_stats == {}


def test_hotswap_stats_no_normalizer_steps():
    """Test hotswap_stats with a processor that has no normalizer/unnormalizer steps."""
    stats = {
        OBS_IMAGE: {"mean": np.array([0.5]), "std": np.array([0.2])},
    }

    # Create processor with only identity steps
    robot_processor = DataProcessorPipeline(steps=[IdentityProcessorStep(), IdentityProcessorStep()])

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
        OBS_IMAGE: {"mean": np.array([0.5]), "std": np.array([0.2])},
    }

    new_stats = {
        OBS_IMAGE: {"mean": np.array([0.3]), "std": np.array([0.1])},
    }

    features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}
    normalize_observation_keys = {OBS_IMAGE}
    eps = 1e-6

    normalizer = NormalizerProcessorStep(
        features=features,
        norm_map=norm_map,
        stats=initial_stats,
        normalize_observation_keys=normalize_observation_keys,
        eps=eps,
    )
    robot_processor = DataProcessorPipeline(steps=[normalizer])

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
        OBS_IMAGE: {"mean": np.array([0.5]), "std": np.array([0.2])},
        ACTION: {"min": np.array([-1.0]), "max": np.array([1.0])},
    }

    new_stats = {
        OBS_IMAGE: {"mean": np.array([0.3]), "std": np.array([0.1])},
        ACTION: {"min": np.array([-2.0]), "max": np.array([2.0])},
    }

    features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }

    # Create multiple normalizers and unnormalizers
    normalizer1 = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=initial_stats)
    normalizer2 = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=initial_stats)
    unnormalizer1 = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=initial_stats)
    unnormalizer2 = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=initial_stats)

    robot_processor = DataProcessorPipeline(steps=[normalizer1, unnormalizer1, normalizer2, unnormalizer2])

    # Hotswap stats
    new_processor = hotswap_stats(robot_processor, new_stats)

    # All normalizer/unnormalizer steps should be updated
    for step in new_processor.steps:
        assert step.stats == new_stats

        # Check tensor stats conversion
        expected_tensor_stats = to_tensor(new_stats)
        for key in expected_tensor_stats:
            for stat_name in expected_tensor_stats[key]:
                torch.testing.assert_close(
                    step._tensor_stats[key][stat_name], expected_tensor_stats[key][stat_name]
                )


def test_hotswap_stats_with_different_data_types():
    """Test hotswap_stats with various data types in stats."""
    initial_stats = {
        OBS_IMAGE: {"mean": np.array([0.5]), "std": np.array([0.2])},
    }

    # New stats with different data types (int, float, list, tuple)
    new_stats = {
        OBS_IMAGE: {
            "mean": [0.3, 0.4, 0.5],  # list
            "std": (0.1, 0.2, 0.3),  # tuple
            "min": 0,  # int
            "max": 1.0,  # float
        },
        ACTION: {
            "mean": np.array([0.1, 0.2]),  # numpy array
            "std": torch.tensor([0.5, 0.6]),  # torch tensor
        },
    }

    features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }

    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=initial_stats)
    robot_processor = DataProcessorPipeline(steps=[normalizer])

    # Hotswap stats
    new_processor = hotswap_stats(robot_processor, new_stats)

    # Check that stats are updated
    assert new_processor.steps[0].stats == new_stats

    # Check that tensor conversion worked correctly
    tensor_stats = new_processor.steps[0]._tensor_stats
    assert isinstance(tensor_stats[OBS_IMAGE]["mean"], torch.Tensor)
    assert isinstance(tensor_stats[OBS_IMAGE]["std"], torch.Tensor)
    assert isinstance(tensor_stats[OBS_IMAGE]["min"], torch.Tensor)
    assert isinstance(tensor_stats[OBS_IMAGE]["max"], torch.Tensor)
    assert isinstance(tensor_stats[ACTION]["mean"], torch.Tensor)
    assert isinstance(tensor_stats[ACTION]["std"], torch.Tensor)

    # Check values
    torch.testing.assert_close(tensor_stats[OBS_IMAGE]["mean"], torch.tensor([0.3, 0.4, 0.5]))
    torch.testing.assert_close(tensor_stats[OBS_IMAGE]["std"], torch.tensor([0.1, 0.2, 0.3]))
    torch.testing.assert_close(tensor_stats[OBS_IMAGE]["min"], torch.tensor(0.0))
    torch.testing.assert_close(tensor_stats[OBS_IMAGE]["max"], torch.tensor(1.0))


def test_hotswap_stats_functional_test():
    """Test that hotswapped processor actually works functionally."""
    # Create test data
    observation = {
        OBS_IMAGE: torch.tensor([[[0.6, 0.7], [0.8, 0.9]], [[0.5, 0.6], [0.7, 0.8]]]),
    }
    action = torch.tensor([0.5, -0.5])
    transition = create_transition(observation=observation, action=action)

    # Initial stats
    initial_stats = {
        OBS_IMAGE: {"mean": np.array([0.5, 0.4]), "std": np.array([0.2, 0.3])},
        ACTION: {"mean": np.array([0.0, 0.0]), "std": np.array([1.0, 1.0])},
    }

    # New stats
    new_stats = {
        OBS_IMAGE: {"mean": np.array([0.3, 0.2]), "std": np.array([0.1, 0.2])},
        ACTION: {"mean": np.array([0.1, -0.1]), "std": np.array([0.5, 0.5])},
    }

    features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(2, 2, 2)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }

    # Create original processor
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=initial_stats)
    original_processor = DataProcessorPipeline(
        steps=[normalizer], to_transition=identity_transition, to_output=identity_transition
    )

    # Process with original stats
    original_result = original_processor(transition)

    # Hotswap stats
    new_processor = hotswap_stats(original_processor, new_stats)

    # Process with new stats
    new_result = new_processor(transition)

    # Results should be different since normalization changed
    assert not torch.allclose(
        original_result[OBS_STR][OBS_IMAGE],
        new_result[OBS_STR][OBS_IMAGE],
        rtol=1e-3,
        atol=1e-3,
    )
    assert not torch.allclose(original_result[ACTION], new_result[ACTION], rtol=1e-3, atol=1e-3)

    # Verify that the new processor is actually using the new stats by checking internal state
    assert new_processor.steps[0].stats == new_stats
    assert torch.allclose(new_processor.steps[0]._tensor_stats[OBS_IMAGE]["mean"], torch.tensor([0.3, 0.2]))
    assert torch.allclose(new_processor.steps[0]._tensor_stats[OBS_IMAGE]["std"], torch.tensor([0.1, 0.2]))
    assert torch.allclose(new_processor.steps[0]._tensor_stats[ACTION]["mean"], torch.tensor([0.1, -0.1]))
    assert torch.allclose(new_processor.steps[0]._tensor_stats[ACTION]["std"], torch.tensor([0.5, 0.5]))

    # Test that normalization actually happens (output should not equal input)
    assert not torch.allclose(new_result[OBS_STR][OBS_IMAGE], observation[OBS_IMAGE])
    assert not torch.allclose(new_result[ACTION], action)


def test_zero_std_uses_eps():
    """When std == 0, (x-mean)/(std+eps) is well-defined; x==mean should map to 0."""
    features = {OBS_STATE: PolicyFeature(FeatureType.STATE, (1,))}
    norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD}
    stats = {OBS_STATE: {"mean": np.array([0.5]), "std": np.array([0.0])}}
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats, eps=1e-6)

    observation = {OBS_STATE: torch.tensor([0.5])}  # equals mean
    out = normalizer(create_transition(observation=observation))
    assert torch.allclose(out[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([0.0]))


def test_min_equals_max_maps_to_minus_one():
    """When min == max, MIN_MAX path maps to -1 after [-1,1] scaling for x==min."""
    features = {OBS_STATE: PolicyFeature(FeatureType.STATE, (1,))}
    norm_map = {FeatureType.STATE: NormalizationMode.MIN_MAX}
    stats = {OBS_STATE: {"min": np.array([2.0]), "max": np.array([2.0])}}
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats, eps=1e-6)

    observation = {OBS_STATE: torch.tensor([2.0])}
    out = normalizer(create_transition(observation=observation))
    assert torch.allclose(out[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([-1.0]))


def test_action_normalized_despite_normalize_observation_keys():
    """Action normalization is independent of normalize_observation_keys filter for observations."""
    features = {
        OBS_STATE: PolicyFeature(FeatureType.STATE, (1,)),
        ACTION: PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {FeatureType.STATE: NormalizationMode.IDENTITY, FeatureType.ACTION: NormalizationMode.MEAN_STD}
    stats = {ACTION: {"mean": np.array([1.0, -1.0]), "std": np.array([2.0, 4.0])}}
    normalizer = NormalizerProcessorStep(
        features=features, norm_map=norm_map, stats=stats, normalize_observation_keys={OBS_STATE}
    )

    transition = create_transition(
        observation={OBS_STATE: torch.tensor([3.0])}, action=torch.tensor([3.0, 3.0])
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
    unnorm_ms = UnnormalizerProcessorStep(
        features={"observation.ms": features["observation.ms"]},
        norm_map={FeatureType.STATE: NormalizationMode.MEAN_STD},
        stats={"observation.ms": {"mean": np.array([1.0, -1.0]), "std": np.array([2.0, 4.0])}},
    )
    unnorm_mm = UnnormalizerProcessorStep(
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
    features = {OBS_STATE: PolicyFeature(FeatureType.STATE, (1,))}
    norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD}
    stats = {OBS_STATE: {"mean": np.array([0.0]), "std": np.array([1.0])}}
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    obs = {OBS_STATE: torch.tensor([1.0]), "observation.unknown": torch.tensor([5.0])}
    tr = create_transition(observation=obs)
    out = normalizer(tr)

    # Unknown key should pass through unchanged and not be tracked
    assert torch.allclose(out[TransitionKey.OBSERVATION]["observation.unknown"], obs["observation.unknown"])


def test_batched_action_normalization():
    features = {ACTION: PolicyFeature(FeatureType.ACTION, (2,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.MEAN_STD}
    stats = {ACTION: {"mean": np.array([1.0, -1.0]), "std": np.array([2.0, 4.0])}}
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    actions = torch.tensor([[1.0, -1.0], [3.0, 3.0]])  # first equals mean → zeros; second → [1, 1]
    out = normalizer(create_transition(action=actions))[TransitionKey.ACTION]
    expected = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    assert torch.allclose(out, expected)


def test_complementary_data_preservation():
    features = {OBS_STATE: PolicyFeature(FeatureType.STATE, (1,))}
    norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD}
    stats = {OBS_STATE: {"mean": np.array([0.0]), "std": np.array([1.0])}}
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    comp = {"existing": 123}
    tr = create_transition(observation={OBS_STATE: torch.tensor([1.0])}, complementary_data=comp)
    out = normalizer(tr)
    new_comp = out[TransitionKey.COMPLEMENTARY_DATA]
    assert new_comp["existing"] == 123


def test_roundtrip_normalize_unnormalize_non_identity():
    features = {
        OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
        ACTION: PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MIN_MAX}
    stats = {
        OBS_STATE: {"mean": np.array([1.0, -1.0]), "std": np.array([2.0, 4.0])},
        ACTION: {"min": np.array([-2.0, 0.0]), "max": np.array([2.0, 4.0])},
    }
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    # Add a time dimension in action for broadcasting check (B,T,D)
    obs = {OBS_STATE: torch.tensor([[3.0, 3.0], [1.0, -1.0]])}
    act = torch.tensor([[[0.0, -1.0], [1.0, 1.0]]])  # shape (1,2,2) already in [-1,1]

    tr = create_transition(observation=obs, action=act)
    out = unnormalizer(normalizer(tr))

    assert torch.allclose(out[TransitionKey.OBSERVATION][OBS_STATE], obs[OBS_STATE], atol=1e-5)
    assert torch.allclose(out[TransitionKey.ACTION], act, atol=1e-5)


def test_dtype_adaptation_bfloat16_input_float32_normalizer():
    """Test automatic dtype adaptation: NormalizerProcessor(float32) adapts to bfloat16 input → bfloat16 output"""
    features = {OBS_STATE: PolicyFeature(FeatureType.STATE, (5,))}
    norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD}
    stats = {
        OBS_STATE: {
            "mean": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "std": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        }
    }

    # Create normalizer configured with float32 dtype
    normalizer = NormalizerProcessorStep(
        features=features, norm_map=norm_map, stats=stats, dtype=torch.float32
    )

    # Verify initial configuration
    assert normalizer.dtype == torch.float32
    for stat_tensor in normalizer._tensor_stats[OBS_STATE].values():
        assert stat_tensor.dtype == torch.float32

    # Create bfloat16 input tensor
    observation = {OBS_STATE: torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.bfloat16)}
    transition = create_transition(observation=observation)

    # Process the transition
    result = normalizer(transition)

    # Verify that:
    # 1. Stats were automatically adapted to bfloat16
    assert normalizer.dtype == torch.bfloat16
    for stat_tensor in normalizer._tensor_stats[OBS_STATE].values():
        assert stat_tensor.dtype == torch.bfloat16

    # 2. Output is in bfloat16
    output_tensor = result[TransitionKey.OBSERVATION][OBS_STATE]
    assert output_tensor.dtype == torch.bfloat16

    # 3. Normalization was applied correctly (mean should be close to original - mean) / std
    expected = (
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.bfloat16)
        - torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.bfloat16)
    ) / torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.bfloat16)
    assert torch.allclose(output_tensor, expected, atol=1e-2)  # bfloat16 has lower precision


def test_stats_override_preservation_in_load_state_dict():
    """
    Test that explicitly provided stats are preserved during load_state_dict.

    This tests the fix for the bug where stats provided via overrides were
    being overwritten when load_state_dict was called.
    """
    # Create original stats
    original_stats = {
        OBS_IMAGE: {"mean": np.array([0.5, 0.5, 0.5]), "std": np.array([0.2, 0.2, 0.2])},
        ACTION: {"mean": np.array([0.0, 0.0]), "std": np.array([1.0, 1.0])},
    }

    # Create override stats (what user wants to use)
    override_stats = {
        OBS_IMAGE: {"mean": np.array([0.3, 0.3, 0.3]), "std": np.array([0.1, 0.1, 0.1])},
        ACTION: {"mean": np.array([0.1, 0.1]), "std": np.array([0.5, 0.5])},
    }

    features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }

    # Create a normalizer with original stats and save its state
    original_normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=original_stats)
    saved_state_dict = original_normalizer.state_dict()

    # Create a new normalizer with override stats (simulating from_pretrained with overrides)
    override_normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=override_stats)

    # Verify that the override stats are initially set correctly
    assert set(override_normalizer.stats.keys()) == set(override_stats.keys())
    for key in override_stats:
        assert set(override_normalizer.stats[key].keys()) == set(override_stats[key].keys())
        for stat_name in override_stats[key]:
            np.testing.assert_array_equal(
                override_normalizer.stats[key][stat_name], override_stats[key][stat_name]
            )
    assert override_normalizer._stats_explicitly_provided is True

    # This is the critical test: load_state_dict should NOT overwrite the override stats
    override_normalizer.load_state_dict(saved_state_dict)

    # After loading state_dict, stats should still be the override stats, not the original stats
    # Check that loaded stats match override stats
    assert set(override_normalizer.stats.keys()) == set(override_stats.keys())
    for key in override_stats:
        assert set(override_normalizer.stats[key].keys()) == set(override_stats[key].keys())
        for stat_name in override_stats[key]:
            np.testing.assert_array_equal(
                override_normalizer.stats[key][stat_name], override_stats[key][stat_name]
            )
    # Compare individual arrays to avoid numpy array comparison ambiguity
    for key in override_stats:
        for stat_name in override_stats[key]:
            assert not np.array_equal(
                override_normalizer.stats[key][stat_name], original_stats[key][stat_name]
            ), f"Stats for {key}.{stat_name} should not match original stats"

    # Verify that _tensor_stats are also correctly set to match the override stats
    expected_tensor_stats = to_tensor(override_stats)
    for key in expected_tensor_stats:
        for stat_name in expected_tensor_stats[key]:
            if isinstance(expected_tensor_stats[key][stat_name], torch.Tensor):
                torch.testing.assert_close(
                    override_normalizer._tensor_stats[key][stat_name], expected_tensor_stats[key][stat_name]
                )


def test_stats_without_override_loads_normally():
    """
    Test that when stats are not explicitly provided (normal case),
    load_state_dict works as before.
    """
    original_stats = {
        OBS_IMAGE: {"mean": np.array([0.5, 0.5, 0.5]), "std": np.array([0.2, 0.2, 0.2])},
        ACTION: {"mean": np.array([0.0, 0.0]), "std": np.array([1.0, 1.0])},
    }

    features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }

    # Create a normalizer with original stats and save its state
    original_normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=original_stats)
    saved_state_dict = original_normalizer.state_dict()

    # Create a new normalizer without stats (simulating normal from_pretrained)
    new_normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats={})

    # Verify that stats are not explicitly provided
    assert new_normalizer._stats_explicitly_provided is False

    # Load state dict - this should work normally and load the saved stats
    new_normalizer.load_state_dict(saved_state_dict)

    # Stats should now match the original stats (normal behavior)
    # Check that all keys and values match
    assert set(new_normalizer.stats.keys()) == set(original_stats.keys())
    for key in original_stats:
        assert set(new_normalizer.stats[key].keys()) == set(original_stats[key].keys())
        for stat_name in original_stats[key]:
            np.testing.assert_allclose(
                new_normalizer.stats[key][stat_name], original_stats[key][stat_name], rtol=1e-6, atol=1e-6
            )


def test_stats_explicit_provided_flag_detection():
    """Test that the _stats_explicitly_provided flag is set correctly in different scenarios."""
    features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    }
    norm_map = {FeatureType.VISUAL: NormalizationMode.MEAN_STD}

    # Test 1: Explicitly provided stats (non-empty dict)
    stats = {OBS_IMAGE: {"mean": [0.5], "std": [0.2]}}
    normalizer1 = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    assert normalizer1._stats_explicitly_provided is True

    # Test 2: Empty stats dict
    normalizer2 = NormalizerProcessorStep(features=features, norm_map=norm_map, stats={})
    assert normalizer2._stats_explicitly_provided is False

    # Test 3: None stats
    normalizer3 = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=None)
    assert normalizer3._stats_explicitly_provided is False

    # Test 4: Stats not provided (defaults to None)
    normalizer4 = NormalizerProcessorStep(features=features, norm_map=norm_map)
    assert normalizer4._stats_explicitly_provided is False


def test_pipeline_from_pretrained_with_stats_overrides():
    """
    Test the actual use case: DataProcessorPipeline.from_pretrained with stat overrides.

    This is an integration test that verifies the fix works in the real scenario
    where users provide stat overrides when loading a pipeline.
    """
    import tempfile

    # Create test data
    features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }

    original_stats = {
        OBS_IMAGE: {"mean": np.array([0.5, 0.5, 0.5]), "std": np.array([0.2, 0.2, 0.2])},
        ACTION: {"mean": np.array([0.0, 0.0]), "std": np.array([1.0, 1.0])},
    }

    override_stats = {
        OBS_IMAGE: {"mean": np.array([0.3, 0.3, 0.3]), "std": np.array([0.1, 0.1, 0.1])},
        ACTION: {"mean": np.array([0.1, 0.1]), "std": np.array([0.5, 0.5])},
    }

    # Create and save a pipeline with the original stats
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=original_stats)
    identity = IdentityProcessorStep()
    original_pipeline = DataProcessorPipeline(steps=[normalizer, identity], name="test_pipeline")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the pipeline
        original_pipeline.save_pretrained(temp_dir)

        # Load the pipeline with stat overrides
        overrides = {"normalizer_processor": {"stats": override_stats}}

        loaded_pipeline = DataProcessorPipeline.from_pretrained(
            temp_dir, config_filename="test_pipeline.json", overrides=overrides
        )

        # The critical test: the loaded pipeline should use override stats, not original stats
        loaded_normalizer = loaded_pipeline.steps[0]
        assert isinstance(loaded_normalizer, NormalizerProcessorStep)

        # Check that loaded stats match override stats
        assert set(loaded_normalizer.stats.keys()) == set(override_stats.keys())
        for key in override_stats:
            assert set(loaded_normalizer.stats[key].keys()) == set(override_stats[key].keys())
            for stat_name in override_stats[key]:
                np.testing.assert_array_equal(
                    loaded_normalizer.stats[key][stat_name], override_stats[key][stat_name]
                )

        # Verify stats don't match original stats
        for key in override_stats:
            for stat_name in override_stats[key]:
                assert not np.array_equal(
                    loaded_normalizer.stats[key][stat_name], original_stats[key][stat_name]
                ), f"Stats for {key}.{stat_name} should not match original stats"

        # Test that the override stats are actually used in processing
        observation = {
            OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3]),
        }
        action = torch.tensor([1.0, -0.5])
        transition = create_transition(observation=observation, action=action)

        # Process with override pipeline
        override_result = loaded_pipeline(transition)

        # Create a reference pipeline with override stats for comparison
        reference_normalizer = NormalizerProcessorStep(
            features=features, norm_map=norm_map, stats=override_stats
        )
        reference_pipeline = DataProcessorPipeline(
            steps=[reference_normalizer, identity],
            to_transition=identity_transition,
            to_output=identity_transition,
        )
        _ = reference_pipeline(transition)

        # The critical part was verified above: loaded_normalizer.stats == override_stats
        # This confirms that override stats are preserved during load_state_dict.
        # Let's just verify the pipeline processes data successfully.
        assert ACTION in override_result
        assert isinstance(override_result[ACTION], torch.Tensor)


def test_dtype_adaptation_device_processor_bfloat16_normalizer_float32():
    """Test policy pipeline scenario: DeviceProcessor(bfloat16) + NormalizerProcessor(float32) → bfloat16 output"""
    from lerobot.processor import DeviceProcessorStep

    features = {OBS_STATE: PolicyFeature(FeatureType.STATE, (3,))}
    norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD}
    stats = {OBS_STATE: {"mean": np.array([0.0, 0.0, 0.0]), "std": np.array([1.0, 1.0, 1.0])}}

    # Create pipeline: DeviceProcessor(bfloat16) → NormalizerProcessor(float32)
    device_processor = DeviceProcessorStep(device=str(auto_select_torch_device()), float_dtype="bfloat16")
    normalizer = NormalizerProcessorStep(
        features=features, norm_map=norm_map, stats=stats, dtype=torch.float32
    )

    # Verify initial normalizer configuration
    assert normalizer.dtype == torch.float32

    # Create CPU input
    observation = {OBS_STATE: torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}
    transition = create_transition(observation=observation)

    # Step 1: DeviceProcessor converts to bfloat16 + moves to CUDA
    processed_1 = device_processor(transition)
    intermediate_tensor = processed_1[TransitionKey.OBSERVATION][OBS_STATE]
    assert intermediate_tensor.dtype == torch.bfloat16
    assert intermediate_tensor.device.type == str(auto_select_torch_device())

    # Step 2: NormalizerProcessor receives bfloat16 input and adapts
    final_result = normalizer(processed_1)
    final_tensor = final_result[TransitionKey.OBSERVATION][OBS_STATE]

    # Verify final output is bfloat16 (automatic adaptation worked)
    assert final_tensor.dtype == torch.bfloat16
    assert final_tensor.device.type == str(auto_select_torch_device())

    # Verify normalizer adapted its internal state
    assert normalizer.dtype == torch.bfloat16
    for stat_tensor in normalizer._tensor_stats[OBS_STATE].values():
        assert stat_tensor.dtype == torch.bfloat16
        assert stat_tensor.device.type == str(auto_select_torch_device())


def test_stats_reconstruction_after_load_state_dict():
    """
    Test that stats dict is properly reconstructed from _tensor_stats after loading.

    This test ensures the bug where stats became empty after loading is fixed.
    The bug occurred when:
    1. Only _tensor_stats were saved via state_dict()
    2. stats field became empty {} after loading
    3. Calling to() method or hotswap_stats would fail because they depend on self.stats
    """

    # Create normalizer with stats
    features = {
        OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
        ACTION: PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {
        FeatureType.VISUAL: NormalizationMode.MEAN_STD,
        FeatureType.STATE: NormalizationMode.MIN_MAX,
        FeatureType.ACTION: NormalizationMode.MEAN_STD,
    }
    stats = {
        OBS_IMAGE: {
            "mean": np.array([0.5, 0.5, 0.5]),
            "std": np.array([0.2, 0.2, 0.2]),
        },
        OBS_STATE: {
            "min": np.array([0.0, -1.0]),
            "max": np.array([1.0, 1.0]),
        },
        ACTION: {
            "mean": np.array([0.0, 0.0]),
            "std": np.array([1.0, 2.0]),
        },
    }

    original_normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    # Save state dict (simulating save/load)
    state_dict = original_normalizer.state_dict()

    # Create new normalizer with empty stats (simulating load)
    new_normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats={})

    # Before fix: this would cause stats to remain empty
    new_normalizer.load_state_dict(state_dict)

    # Verify that stats dict is properly reconstructed from _tensor_stats
    assert new_normalizer.stats is not None
    assert new_normalizer.stats != {}

    # Check that all expected keys are present
    assert OBS_IMAGE in new_normalizer.stats
    assert OBS_STATE in new_normalizer.stats
    assert ACTION in new_normalizer.stats

    # Check that values are correct (converted back from tensors)
    np.testing.assert_allclose(new_normalizer.stats[OBS_IMAGE]["mean"], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(new_normalizer.stats[OBS_IMAGE]["std"], [0.2, 0.2, 0.2])
    np.testing.assert_allclose(new_normalizer.stats[OBS_STATE]["min"], [0.0, -1.0])
    np.testing.assert_allclose(new_normalizer.stats[OBS_STATE]["max"], [1.0, 1.0])
    np.testing.assert_allclose(new_normalizer.stats[ACTION]["mean"], [0.0, 0.0])
    np.testing.assert_allclose(new_normalizer.stats[ACTION]["std"], [1.0, 2.0])

    # Test that methods that depend on self.stats work correctly after loading
    # This would fail before the bug fix because self.stats was empty

    # Test 1: to() method should work without crashing
    try:
        new_normalizer.to(device="cpu", dtype=torch.float32)
        # If we reach here, the bug is fixed
    except (KeyError, AttributeError) as e:
        pytest.fail(f"to() method failed after loading state_dict: {e}")

    # Test 2: hotswap_stats should work
    new_stats = {
        OBS_IMAGE: {"mean": [0.3, 0.3, 0.3], "std": [0.1, 0.1, 0.1]},
        OBS_STATE: {"min": [-1.0, -2.0], "max": [2.0, 2.0]},
        ACTION: {"mean": [0.1, 0.1], "std": [0.5, 0.5]},
    }

    pipeline = DataProcessorPipeline([new_normalizer])
    try:
        new_pipeline = hotswap_stats(pipeline, new_stats)
        # If we reach here, hotswap_stats worked correctly
        assert new_pipeline.steps[0].stats == new_stats
    except (KeyError, AttributeError) as e:
        pytest.fail(f"hotswap_stats failed after loading state_dict: {e}")

    # Test 3: The normalizer should work functionally the same as the original
    observation = {
        OBS_IMAGE: torch.tensor([0.7, 0.5, 0.3]),
        OBS_STATE: torch.tensor([0.5, 0.0]),
    }
    action = torch.tensor([1.0, -0.5])
    transition = create_transition(observation=observation, action=action)

    original_result = original_normalizer(transition)
    new_result = new_normalizer(transition)

    # Results should be identical (within floating point precision)
    torch.testing.assert_close(
        original_result[TransitionKey.OBSERVATION][OBS_IMAGE],
        new_result[TransitionKey.OBSERVATION][OBS_IMAGE],
    )
    torch.testing.assert_close(
        original_result[TransitionKey.OBSERVATION][OBS_STATE],
        new_result[TransitionKey.OBSERVATION][OBS_STATE],
    )
    torch.testing.assert_close(original_result[TransitionKey.ACTION], new_result[TransitionKey.ACTION])

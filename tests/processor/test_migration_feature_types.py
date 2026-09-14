# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

import pytest

from lerobot.configs import FeatureType
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.processor.migrate_policy_normalization import convert_features_to_policy_features


@pytest.mark.parametrize("feature_type", list(FeatureType))
def test_explicit_feature_type_takes_precedence_over_name(feature_type):
    features = {"observation.state": {"type": feature_type.value, "shape": [2]}}
    converted = convert_features_to_policy_features(features)
    assert converted["observation.state"].type is feature_type
    assert converted["observation.state"].shape == (2,)
    assert features["observation.state"]["shape"] == [2]


@pytest.mark.parametrize("include_type", [True, False])
def test_environment_state_only_diffusion_config_remains_valid(include_type):
    features = {
        "observation.state": {"type": "STATE", "shape": [2]},
        "observation.environment_state": {"shape": [16]},
    }
    if include_type:
        features["observation.environment_state"]["type"] = "ENV"
    config = DiffusionConfig(
        device="cpu",
        input_features=convert_features_to_policy_features(features),
        output_features=convert_features_to_policy_features({"action": {"shape": [2]}}),
    )
    config.validate_features()
    assert config.env_state_feature is not None
    assert config.env_state_feature.type is FeatureType.ENV
    assert config.env_state_feature.shape == (16,)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("observation.image", FeatureType.VISUAL),
        ("observation.state", FeatureType.STATE),
        ("action", FeatureType.ACTION),
        ("custom", FeatureType.STATE),
    ],
)
def test_legacy_untyped_features_keep_existing_inference(name, expected):
    feature = convert_features_to_policy_features({name: {"dim": 7}})[name]
    assert feature.type is expected
    assert feature.shape == (7,)


def test_invalid_explicit_type_is_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="observation.state"):
        convert_features_to_policy_features({"observation.state": {"type": "ENVV", "shape": [2]}})

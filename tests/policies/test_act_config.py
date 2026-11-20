"""Tests for ACTConfig.validate_features input acceptance."""

import pytest

from lerobot.configs.policies import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig


def _make_config(input_features):
    return ACTConfig(
        input_features=input_features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )


def test_validate_features_accepts_state_only():
    """forward() reads OBS_STATE, so a state-only input set must validate."""
    cfg = _make_config(
        {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))}
    )
    cfg.validate_features()


def test_validate_features_accepts_env_state():
    cfg = _make_config(
        {"observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(6,))}
    )
    cfg.validate_features()


def test_validate_features_accepts_image():
    cfg = _make_config(
        {"observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96))}
    )
    cfg.validate_features()


def test_validate_features_rejects_empty():
    cfg = _make_config({})
    with pytest.raises(ValueError, match="at least one image"):
        cfg.validate_features()

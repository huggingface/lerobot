#!/usr/bin/env python

from lerobot.configs.types import FeatureType
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features


def test_nonzero_state_feature_is_included():
    features = {OBS_STATE: {"dtype": "float32", "shape": (6,), "names": None}}
    policy_features = dataset_to_policy_features(features)
    assert OBS_STATE in policy_features
    assert policy_features[OBS_STATE].type is FeatureType.STATE
    assert policy_features[OBS_STATE].shape == (6,)


def test_zero_length_state_feature_is_excluded():
    # A PolicyFeature instance is truthy regardless of its shape, so emitting one for
    # a zero-length state would silently look like "this robot has state" to every
    # `if config.robot_state_feature:` / `if OBS_STATE in self.input_features:` check
    # across policies. Omitting the key entirely keeps both styles of check correct.
    features = {OBS_STATE: {"dtype": "float32", "shape": (0,), "names": []}}
    policy_features = dataset_to_policy_features(features)
    assert OBS_STATE not in policy_features


def test_zero_length_env_state_feature_is_excluded():
    features = {OBS_ENV_STATE: {"dtype": "float32", "shape": (0,), "names": []}}
    policy_features = dataset_to_policy_features(features)
    assert OBS_ENV_STATE not in policy_features


def test_zero_length_action_feature_is_still_included():
    # Only state/env are excluded -- a robot with no action space isn't a real case
    # this needs to guard against, and ACTION isn't checked with the same
    # truthiness/membership patterns that motivated excluding STATE/ENV.
    features = {ACTION: {"dtype": "float32", "shape": (0,), "names": []}}
    policy_features = dataset_to_policy_features(features)
    assert ACTION in policy_features
    assert policy_features[ACTION].shape == (0,)

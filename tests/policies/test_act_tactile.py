import pytest

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features


def test_tactile_feature_typing():
    features = {
        "observation.state": {"dtype": "float32", "shape": (10,), "names": None},
        "observation.tactile.sensor_1": {"dtype": "int16", "shape": (6, 6), "names": ["rows", "columns"]},
        "action": {"dtype": "float32", "shape": (6,), "names": None},
    }
    policy_features = dataset_to_policy_features(features)
    assert policy_features["observation.tactile.sensor_1"].type is FeatureType.TACTILE
    assert policy_features["observation.tactile.sensor_1"].shape == (6, 6)
    assert policy_features["observation.state"].type is FeatureType.STATE


def _tactile_input_features(shapes):
    feats = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    }
    for key, shape in shapes.items():
        feats[key] = PolicyFeature(type=FeatureType.TACTILE, shape=shape)
    return feats


def test_tactile_config_property_and_validation():
    input_features = _tactile_input_features(
        {"observation.tactile.sensor_1": (6, 6), "observation.tactile.sensor_2": (4, 8)}
    )
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        use_tactile=True,
    )
    assert list(config.tactile_features) == ["observation.tactile.sensor_1", "observation.tactile.sensor_2"]
    assert config.normalization_mapping["TACTILE"].value == "MEAN_STD"

    with pytest.raises(ValueError):
        ACTConfig(
            input_features=input_features,
            output_features=output_features,
            use_tactile=True,
            tactile_encoder_type="bogus",
        )

    with pytest.raises(ValueError):
        # use_tactile enabled but no tactile features present
        ACTConfig(
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
                f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            },
            output_features=output_features,
            use_tactile=True,
        )

    # Tactile as the sole input modality must be accepted (no images / state / env_state).
    tactile_only = ACTConfig(
        input_features={
            "observation.tactile.sensor_1": PolicyFeature(type=FeatureType.TACTILE, shape=(6, 6))
        },
        output_features=output_features,
        use_tactile=True,
    )
    tactile_only.validate_features()  # must not raise
    assert list(tactile_only.tactile_features) == ["observation.tactile.sensor_1"]

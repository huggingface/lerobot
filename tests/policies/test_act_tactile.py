from lerobot.configs.types import FeatureType
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

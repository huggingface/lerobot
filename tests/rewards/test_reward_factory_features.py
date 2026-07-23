from types import SimpleNamespace

from torch import nn

from lerobot.configs import FeatureType
from lerobot.rewards.factory import make_reward_model
from lerobot.rewards.temporal_siglip_value_function.configuration_temporal_siglip_value_function import (
    TemporalSiglipVFConfig,
)


def test_reward_factory_populates_input_features_from_dataset_meta(monkeypatch):
    from lerobot.rewards import factory

    class FakeReward(nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()
            self.config = config

    monkeypatch.setattr(factory, "get_reward_model_class", lambda name: FakeReward)
    metadata = SimpleNamespace(
        features={
            "observation.images.top": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [14],
                "names": [f"joint_{index}" for index in range(14)],
            },
            "action": {
                "dtype": "float32",
                "shape": [14],
                "names": [f"joint_{index}" for index in range(14)],
            },
        }
    )
    config = TemporalSiglipVFConfig(device="cpu")
    model = make_reward_model(config, dataset_meta=metadata)

    assert model.config.input_features["observation.images.top"].type is FeatureType.VISUAL
    assert model.config.input_features["observation.images.top"].shape == (3, 480, 640)
    assert model.config.input_features["observation.state"].type is FeatureType.STATE
    assert "action" not in model.config.input_features

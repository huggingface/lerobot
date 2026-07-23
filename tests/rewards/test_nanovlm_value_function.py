import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.rewards.factory import get_reward_model_class, make_reward_model_config
from lerobot.rewards.nanovlm_value_function.configuration_nanovlm_value_function import (
    NanoVLMVFConfig,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

CAMERA = "observation.images.top"


def test_config_and_factory_registration():
    config = make_reward_model_config("nanovlm_value_function")
    assert isinstance(config, NanoVLMVFConfig)
    assert get_reward_model_class("nanovlm_value_function").__name__ == "NanoVLMVFRewardModel"


def test_nanovlm_model_forward(monkeypatch):
    from lerobot.rewards.nanovlm_value_function.modeling_nanovlm_value_function import (
        NanoVLMVFRewardModel,
    )

    class FakeVision(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, image):
            return torch.ones(image.shape[0], 4, 6)

    class FakeProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(6, 8)

        def forward(self, features):
            return self.proj(features)

    class FakeDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Embedding(100, 8)

        def forward(self, inputs, attention_mask=None):
            return inputs, None

    class FakeNano(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = SimpleNamespace(lm_hidden_dim=8)
            self.vision_encoder = FakeVision()
            self.MP = FakeProjector()
            self.decoder = FakeDecoder()

        @classmethod
        def from_pretrained(cls, path):
            return cls()

    fake_module = ModuleType("models.vision_language_model")
    fake_module.VisionLanguageModel = FakeNano
    monkeypatch.setitem(sys.modules, "models.vision_language_model", fake_module)

    config = NanoVLMVFConfig(
        device="cpu",
        nanovlm_code_path="third_party/nanoVLM",
    )
    config.input_features = {CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16))}
    model = NanoVLMVFRewardModel(config)
    batch = {
        CAMERA: torch.rand(1, 3, 16, 16),
        CAMERA + ".mask": torch.ones(1, dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.bool),
        "mc_return": torch.tensor([-0.5]),
        "is_terminal": torch.tensor([False]),
    }
    loss, metrics = model(batch)
    assert torch.isfinite(loss)
    assert -1.0 <= metrics["predicted_value_mean"] <= 0.0

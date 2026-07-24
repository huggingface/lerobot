from types import SimpleNamespace

import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.rewards.factory import get_reward_model_class, make_reward_model_config
from lerobot.rewards.temporal_siglip_value_function.configuration_temporal_siglip_value_function import (
    TemporalSiglipVFConfig,
)
from lerobot.rewards.temporal_siglip_value_function.processor_temporal_siglip_value_function import (
    TemporalSiglipImageProcessorStep,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

CAMERAS = ("observation.images.top", "observation.images.left", "observation.images.right")


def _config(**kwargs):
    config = TemporalSiglipVFConfig(
        device="cpu",
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        history_steps=2,
        state_dim=4,
        **kwargs,
    )
    config.input_features = {
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)) for key in CAMERAS},
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }
    return config


def test_config_and_factory_registration():
    config = make_reward_model_config("temporal_siglip_value_function")
    assert isinstance(config, TemporalSiglipVFConfig)
    assert get_reward_model_class("temporal_siglip_value_function").__name__ == "TemporalSiglipVFRewardModel"


def test_history_offsets_are_past_only():
    config = TemporalSiglipVFConfig(history_steps=4, frame_gap=10)
    assert config.observation_delta_indices == [-30, -20, -10, 0]


def test_temporal_image_processor():
    step = TemporalSiglipImageProcessorStep(
        image_resolution=(32, 32),
        image_keys=(CAMERAS[0],),
        history_steps=2,
    )
    transition = {
        TransitionKey.OBSERVATION: {
            CAMERAS[0]: torch.full((1, 2, 3, 20, 16), 128, dtype=torch.uint8),
            CAMERAS[0] + "_is_pad": torch.tensor([[True, False]]),
        }
    }
    observation = step(transition)[TransitionKey.OBSERVATION]
    assert observation[CAMERAS[0]].shape == (1, 2, 3, 32, 32)
    assert observation[CAMERAS[0] + ".mask"].shape == (1, 2)
    assert observation[CAMERAS[0] + ".mask"].tolist() == [[False, True]]
    assert -1.0 <= observation[CAMERAS[0]].min() <= observation[CAMERAS[0]].max() <= 1.0


def test_siglip_tokenizer_builds_missing_attention_mask(monkeypatch):
    from lerobot.rewards.temporal_siglip_value_function import (
        processor_temporal_siglip_value_function as processing,
    )

    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[5, 1, 0, 0]])}

    monkeypatch.setattr(
        processing.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    step = processing.TemporalSiglipTokenizerStep("fake", max_length=4)
    transition = {
        TransitionKey.OBSERVATION: {},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Task: stack blocks."]},
    }
    observation = step(transition)[TransitionKey.OBSERVATION]
    assert observation[OBS_LANGUAGE_ATTENTION_MASK].tolist() == [[True, True, False, False]]


def test_temporal_model_forward(monkeypatch):
    from lerobot.rewards.temporal_siglip_value_function import (
        modeling_temporal_siglip_value_function as modeling,
    )

    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, pixel_values=None, input_ids=None, **kwargs):
            batch = pixel_values.shape[0] if pixel_values is not None else input_ids.shape[0]
            return SimpleNamespace(pooler_output=torch.ones(batch, 8))

    class FakeSiglip(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = FakeEncoder()
            self.text_model = FakeEncoder()
            self.config = SimpleNamespace(
                vision_config=SimpleNamespace(hidden_size=8),
                text_config=SimpleNamespace(hidden_size=8),
            )

    monkeypatch.setattr(modeling.AutoModel, "from_pretrained", lambda *args, **kwargs: FakeSiglip())
    model = modeling.TemporalSiglipVFRewardModel(_config())
    batch = {
        **{key: torch.rand(1, 2, 3, 16, 16) for key in CAMERAS},
        **{key + ".mask": torch.tensor([[False, True]]) for key in CAMERAS},
        OBS_STATE: torch.rand(1, 2, 4),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.bool),
        "mc_return": torch.tensor([-0.5]),
        "is_terminal": torch.tensor([False]),
    }
    loss, metrics = model(batch)
    assert torch.isfinite(loss)
    assert -1.0 <= metrics["predicted_value_mean"] <= 0.0

    model.eval()
    with torch.no_grad():
        eval_loss, eval_metrics = model(batch)
    assert torch.isfinite(eval_loss)
    assert -1.0 <= eval_metrics["predicted_value_mean"] <= 0.0

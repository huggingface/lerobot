import json
import sys
from types import ModuleType, SimpleNamespace

import torch
from PIL import Image
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.rewards.factory import get_reward_model_class, make_reward_model_config
from lerobot.rewards.nanovlm_value_function.configuration_nanovlm_value_function import (
    NanoVLMVFConfig,
)
from lerobot.rewards.nanovlm_value_function.processor_nanovlm_value_function import (
    NANOVLM_ATTENTION_MASK,
    NANOVLM_IMAGES,
    NANOVLM_INPUT_IDS,
    NanoVLMNativeProcessorStep,
)
from lerobot.types import TransitionKey

CAMERA = "observation.images.top"


def test_config_and_factory_registration():
    config = make_reward_model_config("nanovlm_value_function")
    assert isinstance(config, NanoVLMVFConfig)
    assert config.tokenizer_max_length == 8192
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
            self.tokenizer = SimpleNamespace(image_token_id=99)

        def _process_images(self, images, device):
            return torch.cat([image for sample in images for image in sample]).to(device)

        def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
            token_embd = token_embd.clone()
            token_embd[input_ids == self.tokenizer.image_token_id] = image_embd.flatten(0, 1)
            return token_embd

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
        NANOVLM_IMAGES: [[torch.rand(1, 3, 16, 16)]],
        NANOVLM_INPUT_IDS: torch.tensor([[99, 99, 99, 99, 1]]),
        NANOVLM_ATTENTION_MASK: torch.ones(1, 5, dtype=torch.bool),
        "mc_return": torch.tensor([-0.5]),
        "is_terminal": torch.tensor([False]),
    }
    loss, metrics = model(batch)
    assert torch.isfinite(loss)
    assert -1.0 <= metrics["predicted_value_mean"] <= 0.0


def test_native_processor_uses_checkpoint_layout_and_left_padding(monkeypatch, tmp_path):
    config = {
        "lm_tokenizer": "fake",
        "vlm_extra_tokens": {},
        "lm_chat_template": "fake",
        "lm_max_length": 8192,
        "max_img_size": 2048,
        "vit_img_size": 512,
        "resize_to_max_side_len": True,
        "mp_image_token_length": 4,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    class FakeTokenizer:
        pad_token_id = 0
        image_token_id = 99

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert not tokenize and add_generation_prompt
            return messages[0]["content"]

        def __call__(self, prompt, truncation, add_special_tokens):
            assert not truncation and not add_special_tokens
            suffix = [1, 2] if "long" in prompt else [1]
            return {"input_ids": [99] * 4 + suffix, "attention_mask": [1] * (4 + len(suffix))}

    def fake_image_processor(image):
        assert isinstance(image, Image.Image) and image.mode == "RGB"
        return torch.rand(1, 3, 512, 512), (1, 1)

    processors = ModuleType("data.processors")
    processors.get_tokenizer = lambda *args: FakeTokenizer()
    processors.get_image_processor = lambda *args: fake_image_processor
    processors.get_image_string = lambda *args: "<image>"
    monkeypatch.setitem(sys.modules, "data.processors", processors)

    step = NanoVLMNativeProcessorStep(
        pretrained_path=str(tmp_path),
        code_path="third_party/nanoVLM",
        image_keys=(CAMERA,),
        max_length=8192,
    )
    transition = {
        TransitionKey.OBSERVATION: {CAMERA: torch.rand(2, 3, 16, 16)},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["short", "long"]},
    }
    output = step(transition)[TransitionKey.OBSERVATION]

    assert len(output[NANOVLM_IMAGES]) == 2
    assert output[NANOVLM_INPUT_IDS].shape == (2, 6)
    assert output[NANOVLM_INPUT_IDS][0, 0] == 0
    assert not output[NANOVLM_ATTENTION_MASK][0, 0]
    assert output[NANOVLM_ATTENTION_MASK][1].all()

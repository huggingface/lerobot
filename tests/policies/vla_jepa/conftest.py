#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Shared fixtures and helpers for VLA-JEPA tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 2
ACTION_DIM = 3
STATE_DIM = 4
IMAGE_SIZE = 8
ACTION_HORIZON = 4
N_ACTION_STEPS = 2
NUM_VIDEO_FRAMES = 3
QWEN_HIDDEN_SIZE = 16  # hidden size produced by _FakeQwenBackbone

EXPECTED_ACTION_CHUNK_SHAPE = (BATCH_SIZE, ACTION_HORIZON, ACTION_DIM)
EXPECTED_SELECT_ACTION_SHAPE = (BATCH_SIZE, ACTION_DIM)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_config(
    action_dim: int = ACTION_DIM,
    state_dim: int = STATE_DIM,
    action_horizon: int = ACTION_HORIZON,
    num_video_frames: int = NUM_VIDEO_FRAMES,
) -> VLAJEPAConfig:
    config = VLAJEPAConfig(
        input_features={
            f"{OBS_IMAGES}.laptop": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
        device="cpu",
        chunk_size=action_horizon,
        n_action_steps=min(N_ACTION_STEPS, action_horizon),
        action_dim=action_dim,
        state_dim=state_dim,
        num_video_frames=num_video_frames,
        num_action_tokens_per_timestep=2,
        num_embodied_action_tokens_per_instruction=3,
        num_inference_timesteps=2,
        action_hidden_size=QWEN_HIDDEN_SIZE,
        action_model_type="DiT-test",
        action_num_layers=1,
        predictor_depth=1,
        predictor_num_heads=2,
        predictor_mlp_ratio=2.0,
        jepa_tubelet_size=1,
    )
    config.validate_features()
    return config


def make_train_batch(
    batch_size: int = BATCH_SIZE,
    action_dim: int = ACTION_DIM,
    state_dim: int = STATE_DIM,
    action_horizon: int = ACTION_HORIZON,
    num_video_frames: int = NUM_VIDEO_FRAMES,
) -> dict[str, Tensor | list[str]]:
    return {
        f"{OBS_IMAGES}.laptop": torch.rand(batch_size, num_video_frames, 3, IMAGE_SIZE, IMAGE_SIZE),
        OBS_STATE: torch.randn(batch_size, 1, state_dim),
        ACTION: torch.randn(batch_size, action_horizon, action_dim),
        "task": ["pick up the cube"] * batch_size,
    }


def make_inference_batch(
    batch_size: int = BATCH_SIZE,
    state_dim: int = STATE_DIM,
    image_size: int = IMAGE_SIZE,
) -> dict[str, Tensor | list[str]]:
    return {
        f"{OBS_IMAGES}.laptop": torch.rand(batch_size, 3, image_size, image_size),
        OBS_STATE: torch.randn(batch_size, state_dim),
        "task": ["pick up the cube"] * batch_size,
    }


# ---------------------------------------------------------------------------
# Fake external models (replace Qwen3-VL and V-JEPA at test time)
# ---------------------------------------------------------------------------


class _FakeLanguageLayer(nn.Module):
    """Leaf module whose forward hook is captured by _qwen_last_decoder_hidden."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self._hidden_size = hidden_size

    def forward(self, hidden: Tensor, **_: object) -> tuple[Tensor, ...]:
        return (hidden,)


class _FakeLanguageModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self._hidden_size = hidden_size
        self.layers = nn.ModuleList([_FakeLanguageLayer(hidden_size)])

    def forward(self, input_ids: Tensor, **_: object) -> SimpleNamespace:
        batch_size, seq_len = input_ids.shape
        hidden = torch.zeros(batch_size, seq_len, self._hidden_size, device=input_ids.device)
        self.layers[-1](hidden)
        return SimpleNamespace()


class _FakeQwenInnerModel(nn.Module):
    """Mimics the `.model.model` level that _qwen_last_decoder_hidden walks into."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.language_model = _FakeLanguageModel(hidden_size)

    def forward(self, input_ids: Tensor, **kwargs: object) -> SimpleNamespace:
        return self.language_model(input_ids)


class _FakeQwenBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            text_config=SimpleNamespace(hidden_size=hidden_size),
        )
        self.model = _FakeQwenInnerModel(hidden_size)

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def forward(self, input_ids: Tensor, **_: object) -> SimpleNamespace:
        batch_size, seq_len = input_ids.shape
        hidden_size = self.config.hidden_size
        values = torch.arange(
            batch_size * seq_len * hidden_size,
            device=input_ids.device,
            dtype=torch.float32,
        ).view(batch_size, seq_len, hidden_size)
        hidden = values / values.numel() + self.weight
        self.model(input_ids)  # call through so the forward hook on layers[-1] fires
        return SimpleNamespace(hidden_states=[hidden])


class _FakeQwenInterface(nn.Module):
    def __init__(self, config: VLAJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.model = _FakeQwenBackbone(hidden_size=QWEN_HIDDEN_SIZE)

    def expand_tokenizer(self) -> tuple[list[str], list[int], int]:
        max_action_tokens = self.config.chunk_size * self.config.num_action_tokens_per_timestep
        action_tokens = [self.config.special_action_token.format(idx) for idx in range(max_action_tokens)]
        action_token_ids = list(range(1000, 1000 + max_action_tokens))
        return action_tokens, action_token_ids, 2000

    def build_inputs(
        self,
        images: list[list[Tensor]],
        instructions: list[str],
        action_prompt: str,
        embodied_prompt: str,
    ) -> dict[str, Tensor]:
        batch_size = len(images)
        del images, instructions, action_prompt, embodied_prompt
        action_count = (self.config.num_video_frames - 1) * self.config.num_action_tokens_per_timestep
        token_ids = (
            [10]
            + list(range(1000, 1000 + action_count))
            + [2000] * self.config.num_embodied_action_tokens_per_instruction
            + [11]
        )
        return {
            "input_ids": torch.tensor(
                [token_ids] * batch_size,
                device=self.model.device,
                dtype=torch.long,
            )
        }

    @staticmethod
    def to_pixel_values(image_tensor: Tensor) -> Tensor:
        image = image_tensor.detach().float()
        if image.shape[-3] == 1:
            repeats = [1] * image.ndim
            repeats[-3] = 3
            image = image.repeat(*repeats)
        return image


class _FakeVideoEncoder(nn.Module):
    def __init__(self, hidden_size: int = 8, tubelet_size: int = 1) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        # image_size must be >= patch_size (16) so the predictor grid is non-zero.
        # Setting image_size=16 gives a 1x1 grid (1 patch per frame).
        self.config = SimpleNamespace(hidden_size=hidden_size, tubelet_size=tubelet_size, image_size=16)

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def get_vision_features(self, pixel_values_videos: Tensor) -> Tensor:
        batch_size, num_frames = pixel_values_videos.shape[:2]
        hidden_size = self.config.hidden_size
        frame_values = pixel_values_videos.float().mean(dim=(2, 3, 4), keepdim=False)
        return frame_values[:, :, None].expand(batch_size, num_frames, hidden_size)


class _FakeVideoProcessor:
    def __call__(self, videos, return_tensors: str, device=None, **kwargs) -> dict[str, Tensor]:
        assert return_tensors == "pt"
        if isinstance(videos, list):
            pixel_values = torch.stack([torch.as_tensor(v) for v in videos])
        else:
            pixel_values = torch.as_tensor(videos).unsqueeze(0)
        if device is not None:
            pixel_values = pixel_values.to(device)
        return {"pixel_values_videos": pixel_values}


# ---------------------------------------------------------------------------
# A tiny random Qwen3-VL saved with its processor (the real Qwen path, without a download)
# ---------------------------------------------------------------------------

QWEN_SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]
QWEN_CHAT_TEMPLATE = (
    "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n"
    "{% for content in message['content'] %}{% if content['type'] == 'image' %}"
    "<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'text' %}{{ content['text'] }}"
    "{% endif %}{% endfor %}<|im_end|>\n{% endfor -%}"
)
# The tiny image processor upsamples this camera size to 16x16: a grid of 4x4 patches, 4 image tokens.
TINY_QWEN_IMAGE_SIZE = 12


def save_tiny_qwen(path: Path) -> None:
    """A tiny random Qwen3-VL with a word-level tokenizer and the Qwen3-VL processors, as a local checkpoint."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    from tests.policies.common.qwen3_vl_helpers import tiny_qwen3_vl_config

    # The words of the test tasks; the rest of the prompt is unknown words. With the action tokens VLA-JEPA
    # adds, this fits the 32 embedding rows of the tiny model.
    words = ["<|endoftext|>", "[UNK]", "pick", "up", "the", "red", "cube", "stop"]
    vocab = {token: i for i, token in enumerate(words + QWEN_SPECIAL_TOKENS)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(QWEN_SPECIAL_TOKENS)
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="<|endoftext|>")
    fast.chat_template = QWEN_CHAT_TEMPLATE
    fast.save_pretrained(path)
    pixels = {"patch_size": 4, "temporal_patch_size": 2, "merge_size": 2}
    # size, not min_pixels/max_pixels: in transformers 5.5.4 those change the class default for later tests.
    Qwen2VLImageProcessor(
        **pixels,
        size={"shortest_edge": 8 * 8, "longest_edge": 16 * 16},
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    ).save_pretrained(path)
    Qwen3VLVideoProcessor(**pixels).save_pretrained(path)
    config = tiny_qwen3_vl_config()
    config.image_token_id = vocab["<|image_pad|>"]
    config.video_token_id = vocab["<|video_pad|>"]
    config.vision_start_token_id = vocab["<|vision_start|>"]
    config.vision_end_token_id = vocab["<|vision_end|>"]
    torch.manual_seed(0)
    Qwen3VLForConditionalGeneration(config).save_pretrained(path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_qwen_config(tmp_path: Path) -> VLAJEPAConfig:
    """`make_config` on `save_tiny_qwen`, without a world model, with a camera of `TINY_QWEN_IMAGE_SIZE`."""
    save_tiny_qwen(tmp_path)
    config = make_config()
    config.qwen_model_name = str(tmp_path)
    config.enable_world_model = False
    size = TINY_QWEN_IMAGE_SIZE
    config.input_features[f"{OBS_IMAGES}.laptop"] = PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, size, size)
    )
    return config


@pytest.fixture
def patch_vla_jepa_external_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.policies.vla_jepa import modeling_vla_jepa

    monkeypatch.setattr(modeling_vla_jepa, "Qwen3VLInterface", _FakeQwenInterface)
    monkeypatch.setattr(
        modeling_vla_jepa.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: _FakeVideoEncoder(),
    )
    monkeypatch.setattr(
        modeling_vla_jepa.AutoVideoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: _FakeVideoProcessor(),
    )

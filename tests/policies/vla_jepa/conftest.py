#!/usr/bin/env python
"""Shared fixtures and helpers for VLA-JEPA tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
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
) -> dict[str, Tensor | list[str]]:
    return {
        f"{OBS_IMAGES}.laptop": torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE),
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

    @staticmethod
    def _get_torch_dtype(dtype_name: str) -> torch.dtype:
        return torch.float32 if dtype_name == "float32" else torch.bfloat16

    def expand_tokenizer(self) -> tuple[list[str], list[int], int]:
        max_action_tokens = self.config.chunk_size * self.config.num_action_tokens_per_timestep
        action_tokens = [self.config.special_action_token.format(idx) for idx in range(max_action_tokens)]
        action_token_ids = list(range(1000, 1000 + max_action_tokens))
        return action_tokens, action_token_ids, 2000

    def build_inputs(
        self,
        images: list[list[Image.Image]],
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
    def tensor_to_pil(image_tensor: Tensor) -> Image.Image:
        image = image_tensor.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = (image.float().clamp(0, 1) * 255).to(torch.uint8).numpy()
        return Image.fromarray(image)


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
    def __call__(self, videos, return_tensors: str) -> dict[str, Tensor]:
        assert return_tensors == "pt"
        if isinstance(videos, list):
            pixel_values = torch.stack([torch.as_tensor(v) for v in videos])
        else:
            pixel_values = torch.as_tensor(videos).unsqueeze(0)
        return {"pixel_values_videos": pixel_values}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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

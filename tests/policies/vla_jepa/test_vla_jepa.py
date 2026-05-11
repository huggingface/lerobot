#!/usr/bin/env python

from __future__ import annotations

import os
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

pytestmark = pytest.mark.filterwarnings(
    "ignore:In CPU autocast, but the target dtype is not supported:UserWarning"
)


BATCH_SIZE = 2
ACTION_DIM = 3
STATE_DIM = 4
IMAGE_SIZE = 8
ACTION_HORIZON = 4
N_ACTION_STEPS = 2
NUM_VIDEO_FRAMES = 3
EXPECTED_ACTION_CHUNK_SHAPE = (BATCH_SIZE, ACTION_HORIZON, ACTION_DIM)
EXPECTED_SELECT_ACTION_SHAPE = (BATCH_SIZE, ACTION_DIM)
PRETRAINED_REPO_ID = "ginwind/VLA-JEPA"
PRETRAINED_SUBFOLDER = "LIBERO"


def set_seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class _FakeQwenBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            text_config=SimpleNamespace(hidden_size=hidden_size),
        )

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
        return SimpleNamespace(hidden_states=[hidden])


class _FakeQwenInterface(nn.Module):
    def __init__(self, config: VLAJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.model = _FakeQwenBackbone(hidden_size=16)

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
        input_ids = torch.tensor(
            [token_ids] * batch_size,
            device=self.model.device,
            dtype=torch.long,
        )
        return {"input_ids": input_ids}

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
        self.config = SimpleNamespace(hidden_size=hidden_size, tubelet_size=tubelet_size)

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def get_vision_features(self, pixel_values_videos: Tensor) -> Tensor:
        batch_size, num_frames = pixel_values_videos.shape[:2]
        hidden_size = self.config.hidden_size
        frame_values = pixel_values_videos.float().mean(dim=(2, 3, 4), keepdim=False)
        return frame_values[:, :, None].expand(batch_size, num_frames, hidden_size)


class _FakeVideoProcessor:
    def __call__(self, videos: np.ndarray, return_tensors: str) -> dict[str, Tensor]:
        assert return_tensors == "pt"
        return {"pixel_values_videos": torch.as_tensor(videos).unsqueeze(0)}


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


def make_config() -> VLAJEPAConfig:
    config = VLAJEPAConfig(
        input_features={
            f"{OBS_IMAGES}.laptop": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
        },
        device="cpu",
        chunk_size=ACTION_HORIZON,
        n_action_steps=N_ACTION_STEPS,
        future_action_window_size=ACTION_HORIZON - 1,
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        num_video_frames=NUM_VIDEO_FRAMES,
        num_action_tokens_per_timestep=2,
        num_embodied_action_tokens_per_instruction=3,
        num_inference_timesteps=2,
        action_hidden_size=16,
        action_num_layers=1,
        action_num_heads=2,
        action_attention_head_dim=8,
        predictor_depth=1,
        predictor_num_heads=2,
        predictor_mlp_ratio=2.0,
    )
    config.validate_features()
    return config


def make_train_batch(batch_size: int = BATCH_SIZE) -> dict[str, Tensor | list[str]]:
    return {
        f"{OBS_IMAGES}.laptop": torch.rand(batch_size, NUM_VIDEO_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE),
        OBS_STATE: torch.randn(batch_size, 1, STATE_DIM),
        ACTION: torch.randn(batch_size, ACTION_HORIZON, ACTION_DIM),
        "task": ["pick up the cube"] * batch_size,
    }


def make_inference_batch(batch_size: int = BATCH_SIZE) -> dict[str, Tensor | list[str]]:
    return {
        f"{OBS_IMAGES}.laptop": torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE),
        OBS_STATE: torch.randn(batch_size, STATE_DIM),
        "task": ["pick up the cube"] * batch_size,
    }


def test_vla_jepa_training_forward_pass(patch_vla_jepa_external_models: None) -> None:
    set_seed_all(42)
    policy = VLAJEPAPolicy(make_config())
    policy.train()

    batch = make_train_batch()
    batch_before = deepcopy(batch)

    loss, logs = policy.forward(batch)

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert set(logs) == {"action_loss", "wm_loss", "loss"}
    assert logs["action_loss"] > 0
    assert logs["wm_loss"] >= 0

    loss.backward()
    assert any(
        param.grad is not None for param in policy.model.action_model.parameters() if param.requires_grad
    )
    assert set(batch) == set(batch_before)
    for key, value in batch.items():
        if isinstance(value, Tensor):
            assert torch.equal(value, batch_before[key])
        else:
            assert value == batch_before[key]


@torch.no_grad()
def test_vla_jepa_action_generation_shape(
    patch_vla_jepa_external_models: None,
) -> None:
    set_seed_all(42)
    policy = VLAJEPAPolicy(make_config())
    policy.eval()
    batch = make_inference_batch()

    action_chunk = policy.predict_action_chunk(batch)

    assert tuple(action_chunk.shape) == EXPECTED_ACTION_CHUNK_SHAPE
    assert action_chunk.device.type == "cpu"
    assert torch.isfinite(action_chunk).all()

    first_action = policy.select_action(batch)
    second_action = policy.select_action(batch)

    assert tuple(first_action.shape) == EXPECTED_SELECT_ACTION_SHAPE
    assert tuple(second_action.shape) == EXPECTED_SELECT_ACTION_SHAPE
    assert torch.isfinite(first_action).all()
    assert torch.isfinite(second_action).all()


@torch.no_grad()
def test_vla_jepa_inference_reproducibility(
    patch_vla_jepa_external_models: None,
) -> None:
    set_seed_all(42)
    policy = VLAJEPAPolicy(make_config())
    policy.eval()
    batch = make_inference_batch()

    set_seed_all(123)
    actions_1 = policy.predict_action_chunk(batch)

    set_seed_all(123)
    actions_2 = policy.predict_action_chunk(batch)

    assert tuple(actions_1.shape) == EXPECTED_ACTION_CHUNK_SHAPE
    assert torch.allclose(actions_1, actions_2, atol=1e-6)


def test_vla_jepa_pretrained_checkpoint_loads_from_hf_cache() -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    repo_id = os.environ.get("VLA_JEPA_PRETRAINED_REPO_ID", PRETRAINED_REPO_ID)
    subfolder = os.environ.get("VLA_JEPA_PRETRAINED_SUBFOLDER", PRETRAINED_SUBFOLDER).strip("/")
    checkpoint_filename = os.environ.get(
        "VLA_JEPA_PRETRAINED_CHECKPOINT",
        f"{subfolder}/checkpoints/VLA-JEPA-{subfolder}.pt",
    )

    try:
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_filename,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pytest.skip(f"{repo_id}/{checkpoint_filename} is not available in the local Hugging Face cache.")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = (
        checkpoint.get("state_dict")
        or checkpoint.get("model_state_dict")
        or checkpoint.get("model")
        or checkpoint
    )

    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0
    assert all(isinstance(key, str) for key in list(state_dict)[:10])

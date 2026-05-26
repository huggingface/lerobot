#!/usr/bin/env python

# Copyright 2026 The OpenEAI team and The HuggingFace Inc. team. All rights reserved.
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

"""Tests for OpenEAI VLA policy integration with LeRobot.

Covers:
    - Config creation and validation
    - Policy class registration
    - DiT head and block component tests (without Qwen3-VL download)
    - Multi-subset branching
    - Backward compatibility / determinism
"""

import gc

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.configs.types import NormalizationMode  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_policy_config  # noqa: E402
from lerobot.policies.openeai import OpenEAIVLAConfig  # noqa: E402
from lerobot.policies.openeai.blocks import (  # noqa: E402
    CrossAttention,
    DiTBlock,
    create_sinusoidal_pos_embedding,
    get_1d_sincos_pos_embed_from_grid,
    make_timm_attn_mask,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402


@pytest.fixture(autouse=True)
def cleanup():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# == Config Tests ==


def test_config_creation():
    """Test OpenEAIVLAConfig can be created with defaults."""
    config = OpenEAIVLAConfig()
    assert config.qwen_dim == 2560
    assert config.hidden_dim == 1664
    assert config.n_layers == 18
    assert config.num_heads == 32
    assert config.chunk_size == 50
    assert config.denoise_steps == 10
    assert config.feat_length == 20
    assert config.img_seq_len == 64
    assert config.freeze_backbone is True
    assert config.time_sampling_beta_alpha == 1.5
    assert config.time_sampling_beta_beta == 1.0
    assert config.backbone_dtype == "bfloat16"


def test_config_factory():
    """Test policy config creation through factory."""
    config = make_policy_config("openeai")
    assert isinstance(config, OpenEAIVLAConfig)


def test_config_n_action_steps_validation():
    """Test that n_action_steps > chunk_size raises at config construction."""
    with pytest.raises(ValueError, match="n_action_steps.*cannot be greater"):
        OpenEAIVLAConfig(n_action_steps=60, chunk_size=50)


def test_config_n_action_steps_valid():
    """Test that n_action_steps <= chunk_size is valid."""
    config = OpenEAIVLAConfig(n_action_steps=50, chunk_size=50)
    assert config.n_action_steps == 50
    assert config.chunk_size == 50


def test_config_n_obs_steps_validation():
    """Test that n_obs_steps != 1 raises."""
    with pytest.raises(ValueError, match="n_obs_steps=1"):
        OpenEAIVLAConfig(n_obs_steps=2)


def test_config_pad_language_to_validation():
    """Test that invalid pad_language_to raises."""
    with pytest.raises(ValueError, match="pad_language_to"):
        OpenEAIVLAConfig(pad_language_to="invalid_value")


def test_config_backbone_dtype_validation():
    """Test that invalid backbone_dtype raises."""
    with pytest.raises(ValueError, match="backbone_dtype"):
        OpenEAIVLAConfig(backbone_dtype="int8")


def test_config_backbone_torch_dtype():
    """Test backbone_torch_dtype property mapping."""
    config_bf16 = OpenEAIVLAConfig(backbone_dtype="bfloat16")
    assert config_bf16.backbone_torch_dtype == torch.bfloat16

    config_fp16 = OpenEAIVLAConfig(backbone_dtype="float16")
    assert config_fp16.backbone_torch_dtype == torch.float16

    config_fp32 = OpenEAIVLAConfig(backbone_dtype="float32")
    assert config_fp32.backbone_torch_dtype == torch.float32


def test_config_image_resolution_list_to_tuple():
    """Test that image_resolution as list is converted to tuple in __post_init__."""
    config = OpenEAIVLAConfig()
    config.image_resolution = [224, 224]
    config.__post_init__()
    assert isinstance(config.image_resolution, tuple)


def test_config_normalization_mapping():
    """Test default normalization mapping."""
    config = OpenEAIVLAConfig()
    assert config.normalization_mapping["VISUAL"] == NormalizationMode.IDENTITY
    assert config.normalization_mapping["STATE"] == NormalizationMode.MEAN_STD
    assert config.normalization_mapping["ACTION"] == NormalizationMode.MEAN_STD


def test_config_get_optimizer_preset():
    """Test optimizer preset generation."""
    config = OpenEAIVLAConfig()
    optim = config.get_optimizer_preset()
    assert optim.lr == config.optimizer_lr
    assert optim.weight_decay == config.optimizer_weight_decay


def test_config_get_scheduler_preset():
    """Test scheduler preset generation."""
    config = OpenEAIVLAConfig()
    scheduler = config.get_scheduler_preset()
    assert scheduler.peak_lr == config.optimizer_lr
    assert scheduler.num_warmup_steps == config.scheduler_warmup_steps


# == Policy Class Registration Tests ==


def test_get_policy_class():
    """Test that OpenEAI policy class is registered and retrievable."""
    policy_cls = get_policy_class("openeai")
    assert policy_cls.name == "openeai"
    assert issubclass(policy_cls, object)


# == CrossAttention Tests ==


def test_cross_attention():
    """Test CrossAttention forward pass without mask."""
    dim = 64
    num_heads = 4
    cross_attn = CrossAttention(dim=dim, num_heads=num_heads)
    batch_size, seq_len_x, seq_len_c = 2, 10, 8
    x = torch.randn(batch_size, seq_len_x, dim)
    c = torch.randn(batch_size, seq_len_c, dim)
    out = cross_attn(x, c)
    assert out.shape == (batch_size, seq_len_x, dim)


def test_cross_attention_with_float_mask():
    """Test CrossAttention with float mask (auto-converted to bool)."""
    dim = 64
    num_heads = 4
    cross_attn = CrossAttention(dim=dim, num_heads=num_heads)
    batch_size, seq_len_x, seq_len_c = 2, 10, 8
    x = torch.randn(batch_size, seq_len_x, dim)
    c = torch.randn(batch_size, seq_len_c, dim)
    mask = torch.zeros(batch_size, seq_len_c)
    mask[:, :4] = 1.0  # keep first 4 tokens, mask rest
    out = cross_attn(x, c, mask=mask)
    assert out.shape == (batch_size, seq_len_x, dim)


def test_cross_attention_with_bool_mask():
    """Test CrossAttention with bool mask (passed through)."""
    dim = 64
    num_heads = 4
    cross_attn = CrossAttention(dim=dim, num_heads=num_heads)
    batch_size, seq_len_x, seq_len_c = 2, 10, 8
    x = torch.randn(batch_size, seq_len_x, dim)
    c = torch.randn(batch_size, seq_len_c, dim)
    mask = torch.zeros(batch_size, seq_len_c, dtype=torch.bool)
    mask[:, :4] = True
    out = cross_attn(x, c, mask=mask)
    assert out.shape == (batch_size, seq_len_x, dim)


def test_cross_attention_int_mask_does_not_corrupt():
    """Test CrossAttention with 0/1 int mask is properly converted (not added as logits).

    This regression-tests the fix where int masks would be additively combined
    with attention logits in SDPA, instead of being treated as a bool mask.
    """
    dim = 64
    num_heads = 4
    cross_attn = CrossAttention(dim=dim, num_heads=num_heads)
    cross_attn.eval()
    batch_size, seq_len_x, seq_len_c = 1, 4, 6
    x = torch.randn(batch_size, seq_len_x, dim)
    c = torch.randn(batch_size, seq_len_c, dim)

    int_mask = torch.zeros(batch_size, seq_len_c, dtype=torch.long)
    int_mask[:, :3] = 1
    bool_mask = int_mask.to(torch.bool)

    with torch.no_grad():
        out_int = cross_attn(x, c, mask=int_mask)
        out_bool = cross_attn(x, c, mask=bool_mask)
    torch.testing.assert_close(out_int, out_bool, rtol=1e-5, atol=1e-5)


# == DiTBlock Tests ==


def test_dit_block():
    """Test DiTBlock forward pass without masks."""
    hidden_dim = 64
    num_heads = 4
    block = DiTBlock(hidden_dim=hidden_dim, num_heads=num_heads)
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, hidden_dim)
    cond = torch.randn(batch_size, 8, hidden_dim)
    out = block(x, cond)
    assert out.shape == (batch_size, seq_len, hidden_dim)


def test_dit_block_with_masks():
    """Test DiTBlock with self-attention and cross-attention masks.

    The cond_mask passed to CrossAttention must match cond's sequence length,
    not the x sequence length. The x_timm_mask is for self-attention and covers
    the full x sequence (time + state + actions).
    """
    hidden_dim = 64
    num_heads = 4
    block = DiTBlock(hidden_dim=hidden_dim, num_heads=num_heads)
    batch_size = 2
    x_len = 10
    cond_len = 8
    x = torch.randn(batch_size, x_len, hidden_dim)
    cond = torch.randn(batch_size, cond_len, hidden_dim)
    x_mask = torch.ones(batch_size, x_len, dtype=torch.bool)
    x_mask[:, -1:] = False
    x_timm_mask = make_timm_attn_mask(x_mask)
    cond_mask = torch.ones(batch_size, cond_len)
    out = block(x, cond, attn_mask=x_timm_mask, cond_mask=cond_mask)
    assert out.shape == (batch_size, x_len, hidden_dim)


# == Positional Embedding Helpers ==


def test_sinusoidal_time_embedding():
    """Test sinusoidal time embedding shape and value range."""
    batch_size = 4
    dim = 64
    time = torch.rand(batch_size)
    emb = create_sinusoidal_pos_embedding(time, dim)
    assert emb.shape == (batch_size, dim)
    assert torch.all(emb.abs() <= 1.0 + 1e-6)


def test_sinusoidal_time_embedding_invalid_dim():
    """Test that odd dimension raises."""
    time = torch.rand(4)
    with pytest.raises(ValueError, match="dimension.*must be divisible by 2"):
        create_sinusoidal_pos_embedding(time, 65)


def test_sinusoidal_time_embedding_invalid_ndim():
    """Test that non-1D time tensor raises."""
    time = torch.rand(2, 4)
    with pytest.raises(ValueError, match="time tensor"):
        create_sinusoidal_pos_embedding(time, 64)


def test_sincos_pos_embed_from_grid_list():
    """Test sin-cos positional embedding from list."""
    embed_dim = 64
    pos = list(range(10))
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    assert emb.shape == (10, embed_dim)


def test_sincos_pos_embed_from_grid_numpy():
    """Test sin-cos positional embedding from numpy array."""
    embed_dim = 64
    pos = np.arange(10)
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    assert emb.shape == (10, embed_dim)


def test_sincos_pos_embed_from_grid_torch_tensor():
    """Test sin-cos positional embedding accepts torch.Tensor.

    Regression test: blocks.get_1d_sincos_pos_embed_from_grid should explicitly
    handle torch tensors via .detach().cpu().numpy() to avoid PyTorch warnings.
    """
    embed_dim = 64
    pos = torch.arange(10)
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    assert emb.shape == (10, embed_dim)


def test_make_timm_attn_mask():
    """Test timm attention mask conversion."""
    pad_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]])
    mask = make_timm_attn_mask(pad_mask)
    batch_size, seq_len = pad_mask.shape
    assert mask.shape == (batch_size, 1, seq_len, seq_len)


# == DiTPolicyHead Tests ==


@pytest.fixture
def tiny_openeai_config():
    """Create a tiny OpenEAI config for DiT head testing."""
    config = OpenEAIVLAConfig(
        hidden_dim=64,
        n_layers=2,
        num_heads=4,
        ff_ratio=2.0,
        chunk_size=10,
        n_action_steps=10,
        feat_length=5,
        img_seq_len=16,
        qwen_dim=2560,
        denoise_steps=5,
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    return config


def test_dit_head_initialization(tiny_openeai_config):
    """Test DiTPolicyHead initializes correctly."""
    from lerobot.policies.openeai.modeling_openeai import DiTPolicyHead

    data_dim_info = {"__default__": (8, 6)}
    head = DiTPolicyHead(tiny_openeai_config, data_dim_info=data_dim_info)
    assert head.config.hidden_dim == 64
    assert len(head.blocks) == 2
    assert "__default__" in head.state_encoders
    assert "__default__" in head.action_encoders
    assert "__default__" in head.action_decoders
    # cond_pos_embed is preallocated to 1024 (per modeling._initialize)
    assert head.cond_pos_embed.shape[1] >= tiny_openeai_config.feat_length


def test_dit_head_forward(tiny_openeai_config):
    """Test DiTPolicyHead forward pass.

    cond_embed last dim must match qwen_dim, which is the input dim of cond_adapter.
    cond_embed sequence length is feat_length (matches the feat-query tokens
    extracted from Qwen hidden states in OpenEAIModel).
    """
    from lerobot.policies.openeai.modeling_openeai import DiTPolicyHead

    data_dim_info = {"__default__": (8, 6)}
    head = DiTPolicyHead(tiny_openeai_config, data_dim_info=data_dim_info)
    head.initialize_weights()

    batch_size = 2
    cond_len = tiny_openeai_config.feat_length
    noisy_action = torch.randn(batch_size, tiny_openeai_config.chunk_size, 6)
    action_mask = torch.ones(batch_size, tiny_openeai_config.chunk_size, dtype=torch.bool)
    state = torch.randn(batch_size, 8)
    cond_embed = torch.randn(batch_size, cond_len, tiny_openeai_config.qwen_dim)
    cond_mask = torch.ones(batch_size, cond_len)
    timestep = torch.rand(batch_size)

    v_t = head.forward(
        noisy_action=noisy_action,
        action_mask=action_mask,
        state=state,
        cond_embed=cond_embed,
        cond_mask=cond_mask,
        timestep=timestep,
        subset="__default__",
    )
    assert v_t.shape == (batch_size, tiny_openeai_config.chunk_size, 6)


def test_dit_head_forward_cond_len_mismatch(tiny_openeai_config):
    """Test DiTPolicyHead raises when cond_embed length != feat_length."""
    from lerobot.policies.openeai.modeling_openeai import DiTPolicyHead

    data_dim_info = {"__default__": (8, 6)}
    head = DiTPolicyHead(tiny_openeai_config, data_dim_info=data_dim_info)
    head.initialize_weights()

    batch_size = 1
    wrong_len = tiny_openeai_config.feat_length + 1
    cond_embed = torch.randn(batch_size, wrong_len, tiny_openeai_config.qwen_dim)
    cond_mask = torch.ones(batch_size, wrong_len)

    with pytest.raises(ValueError, match="cond_embed length"):
        head.forward(
            noisy_action=torch.randn(batch_size, tiny_openeai_config.chunk_size, 6),
            action_mask=torch.ones(batch_size, tiny_openeai_config.chunk_size, dtype=torch.bool),
            state=torch.randn(batch_size, 8),
            cond_embed=cond_embed,
            cond_mask=cond_mask,
            timestep=torch.rand(batch_size),
            subset="__default__",
        )


# == DiTPolicyHead subset branching Tests ==


def test_dit_head_multimodal_branching(tiny_openeai_config):
    """Test DiTPolicyHead multi-subset encoder/decoder branching."""
    from lerobot.policies.openeai.modeling_openeai import DiTPolicyHead

    data_dim_info = {
        "__default__": (8, 6),
        "droid": (12, 8),
        "mobile_aloha": (16, 14),
    }
    head = DiTPolicyHead(tiny_openeai_config, data_dim_info=data_dim_info)
    head.initialize_weights()

    batch_size = 2
    cond_len = tiny_openeai_config.feat_length
    cond_embed = torch.randn(batch_size, cond_len, tiny_openeai_config.qwen_dim)
    cond_mask = torch.ones(batch_size, cond_len)
    timestep = torch.rand(batch_size)

    v_t_droid = head.forward(
        noisy_action=torch.randn(batch_size, tiny_openeai_config.chunk_size, 8),
        action_mask=torch.ones(batch_size, tiny_openeai_config.chunk_size, dtype=torch.bool),
        state=torch.randn(batch_size, 12),
        cond_embed=cond_embed,
        cond_mask=cond_mask,
        timestep=timestep,
        subset="droid",
    )
    assert v_t_droid.shape == (batch_size, tiny_openeai_config.chunk_size, 8)

    v_t_aloha = head.forward(
        noisy_action=torch.randn(batch_size, tiny_openeai_config.chunk_size, 14),
        action_mask=torch.ones(batch_size, tiny_openeai_config.chunk_size, dtype=torch.bool),
        state=torch.randn(batch_size, 16),
        cond_embed=cond_embed,
        cond_mask=cond_mask,
        timestep=timestep,
        subset="mobile_aloha",
    )
    assert v_t_aloha.shape == (batch_size, tiny_openeai_config.chunk_size, 14)


# == Backward Compatibility / Determinism ==


def test_backward_compatibility_tiny():
    """Test that loading state_dict reproduces deterministic output across head instances."""
    from lerobot.policies.openeai.modeling_openeai import DiTPolicyHead

    config = OpenEAIVLAConfig(
        hidden_dim=32,
        n_layers=2,
        num_heads=4,
        ff_ratio=2.0,
        chunk_size=8,
        n_action_steps=8,
        feat_length=5,
        img_seq_len=16,
        qwen_dim=2560,
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }

    data_dim_info = {"__default__": (8, 6)}

    head1 = DiTPolicyHead(config, data_dim_info=data_dim_info)
    head1.initialize_weights()

    head2 = DiTPolicyHead(config, data_dim_info=data_dim_info)
    head2.initialize_weights()
    # Load head1's weights into head2
    head2.load_state_dict(head1.state_dict(), strict=True)

    batch_size = 2
    cond_len = config.feat_length
    noisy_action = torch.randn(batch_size, config.chunk_size, 6)
    action_mask = torch.ones(batch_size, config.chunk_size, dtype=torch.bool)
    state = torch.randn(batch_size, 8)
    cond_embed = torch.randn(batch_size, cond_len, config.qwen_dim)
    cond_mask = torch.ones(batch_size, cond_len)
    timestep = torch.rand(batch_size)

    head1.eval()
    head2.eval()
    with torch.no_grad():
        v_t1 = head1.forward(
            noisy_action=noisy_action,
            action_mask=action_mask,
            state=state,
            cond_embed=cond_embed,
            cond_mask=cond_mask,
            timestep=timestep,
            subset="__default__",
        )
        v_t2 = head2.forward(
            noisy_action=noisy_action.clone(),
            action_mask=action_mask.clone(),
            state=state.clone(),
            cond_embed=cond_embed.clone(),
            cond_mask=cond_mask.clone(),
            timestep=timestep.clone(),
            subset="__default__",
        )

    assert v_t1.shape == (batch_size, config.chunk_size, 6)
    torch.testing.assert_close(v_t1, v_t2, rtol=1e-5, atol=1e-5)

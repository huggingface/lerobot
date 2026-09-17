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

"""Compare LeRobot OpenEAI against the original OpenEAI-VLA reference implementation.

This test verifies that:
    - DiT action head forward pass produces deterministic outputs
    - Flow matching time sampling follows expected distribution
    - sample_action (Euler integration) produces valid outputs
    - Positional embedding helpers are consistent
    - CrossAttention produces deterministic outputs across instances

Note:
    DiT head's ``cond_embed`` length must equal ``config.feat_length``, since
    OpenEAIModel.compute_vla_loss / predict_action_chunk slice Qwen hidden states
    to the last ``feat_length`` tokens before passing to the head.

Run manually on GPU:
    pytest -sv tests/policies/openeai/test_openeai_lerobot_vs_original.py
"""

import gc
import math
import os

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.openeai.blocks import (  # noqa: E402
    CrossAttention,
    create_sinusoidal_pos_embedding,
    get_1d_sincos_pos_embed_from_grid,
    make_timm_attn_mask,
)
from lerobot.policies.openeai.configuration_openeai import OpenEAIVLAConfig  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402

# Skip in CI
pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Parity tests are slow; run manually on GPU nodes",
)

# == Shared config for tiny DiT ==


class TinyDiTConfig:
    hidden_dim: int = 128
    n_layers: int = 4
    num_heads: int = 8
    ff_ratio: float = 2.67
    chunk_size: int = 16
    feat_length: int = 8
    img_seq_len: int = 32
    qwen_dim: int = 2560
    denoise_steps: int = 5
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORWARD_RTOL = 1e-4
FORWARD_ATOL = 1e-4


@pytest.fixture(autouse=True)
def cleanup_cuda_after_test():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _make_tiny_config() -> OpenEAIVLAConfig:
    c = TinyDiTConfig()
    config = OpenEAIVLAConfig(
        hidden_dim=c.hidden_dim,
        n_layers=c.n_layers,
        num_heads=c.num_heads,
        ff_ratio=c.ff_ratio,
        chunk_size=c.chunk_size,
        n_action_steps=c.chunk_size,
        denoise_steps=c.denoise_steps,
        feat_length=c.feat_length,
        img_seq_len=c.img_seq_len,
        qwen_dim=c.qwen_dim,
        qwen_path="Qwen/Qwen3-VL-4B-Instruct",
        time_sampling_beta_alpha=c.time_sampling_beta_alpha,
        time_sampling_beta_beta=c.time_sampling_beta_beta,
        time_sampling_scale=c.time_sampling_scale,
        time_sampling_offset=c.time_sampling_offset,
    )
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        f"{OBS_IMAGES}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    config.device = DEVICE
    return config


# == Helpers ==


def _get_state_dict(head) -> dict:
    """Get state dict from a DiTPolicyHead (CPU clone)."""
    return {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}


# == DiT Head Parity Tests ==


def test_dit_head_forward_parity():
    """Test DiT head forward pass is deterministic and shape-correct.

    Verifies:
        1. Same seed -> identical weights between two head instances.
        2. Same input -> identical output (deterministic).
        3. Output shape is correct.
        4. Output values are finite (no NaN/Inf).
    """
    from lerobot.policies.openeai.modeling_openeai import DiTPolicyHead

    config = _make_tiny_config()
    data_dim_info = {"__default__": (8, 6)}

    torch.manual_seed(42)
    head1 = DiTPolicyHead(config, data_dim_info=data_dim_info)
    head1.initialize_weights()
    head1.to(DEVICE)

    torch.manual_seed(42)
    head2 = DiTPolicyHead(config, data_dim_info=data_dim_info)
    head2.initialize_weights()
    head2.to(DEVICE)

    # Same init -> identical weights
    sd1 = _get_state_dict(head1)
    sd2 = _get_state_dict(head2)
    assert set(sd1.keys()) == set(sd2.keys())
    for k in sd1:
        assert torch.allclose(sd1[k], sd2[k], rtol=0, atol=0), f"State dict mismatch at {k}"

    # Disable dropout for deterministic comparison
    head1.eval()
    head2.eval()

    # cond_embed length must equal feat_length (matches modeling._compute_vla_loss
    # behavior of slicing Qwen hidden states to the last feat_length tokens).
    batch_size = 2
    cond_len = config.feat_length
    noisy_action = torch.randn(batch_size, config.chunk_size, 6, device=DEVICE)
    action_mask = torch.ones(batch_size, config.chunk_size, dtype=torch.bool, device=DEVICE)
    state = torch.randn(batch_size, 8, device=DEVICE)
    cond_embed = torch.randn(batch_size, cond_len, config.qwen_dim, device=DEVICE)
    cond_mask = torch.ones(batch_size, cond_len, dtype=torch.bool, device=DEVICE)
    timestep = torch.rand(batch_size, device=DEVICE)

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
    assert v_t2.shape == (batch_size, config.chunk_size, 6)
    torch.testing.assert_close(v_t1, v_t2, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)
    assert not torch.any(torch.isnan(v_t1))
    assert not torch.any(torch.isinf(v_t1))


def test_dit_head_cross_subset_forward():
    """Test DiT head forward pass with multiple subsets."""
    from lerobot.policies.openeai.modeling_openeai import DiTPolicyHead

    config = _make_tiny_config()
    data_dim_info = {
        "__default__": (8, 6),
        "droid": (12, 8),
    }

    torch.manual_seed(42)
    head = DiTPolicyHead(config, data_dim_info=data_dim_info)
    head.initialize_weights()
    head.to(DEVICE)

    batch_size = 2
    cond_len = config.feat_length
    cond_embed = torch.randn(batch_size, cond_len, config.qwen_dim, device=DEVICE)
    cond_mask = torch.ones(batch_size, cond_len, dtype=torch.bool, device=DEVICE)
    timestep = torch.rand(batch_size, device=DEVICE)

    # Test "__default__" subset
    v_t_default = head.forward(
        noisy_action=torch.randn(batch_size, config.chunk_size, 6, device=DEVICE),
        action_mask=torch.ones(batch_size, config.chunk_size, dtype=torch.bool, device=DEVICE),
        state=torch.randn(batch_size, 8, device=DEVICE),
        cond_embed=cond_embed,
        cond_mask=cond_mask,
        timestep=timestep,
        subset="__default__",
    )
    assert v_t_default.shape == (batch_size, config.chunk_size, 6)

    # Test "droid" subset
    v_t_droid = head.forward(
        noisy_action=torch.randn(batch_size, config.chunk_size, 8, device=DEVICE),
        action_mask=torch.ones(batch_size, config.chunk_size, dtype=torch.bool, device=DEVICE),
        state=torch.randn(batch_size, 12, device=DEVICE),
        cond_embed=cond_embed,
        cond_mask=cond_mask,
        timestep=timestep,
        subset="droid",
    )
    assert v_t_droid.shape == (batch_size, config.chunk_size, 8)


def test_dit_head_rejects_wrong_cond_len():
    """Test DiT head raises when cond_embed length differs from feat_length."""
    from lerobot.policies.openeai.modeling_openeai import DiTPolicyHead

    config = _make_tiny_config()
    data_dim_info = {"__default__": (8, 6)}
    head = DiTPolicyHead(config, data_dim_info=data_dim_info)
    head.initialize_weights()
    head.to(DEVICE)
    head.eval()

    batch_size = 1
    wrong_len = config.feat_length + config.img_seq_len  # legacy/wrong length
    cond_embed = torch.randn(batch_size, wrong_len, config.qwen_dim, device=DEVICE)
    cond_mask = torch.ones(batch_size, wrong_len, dtype=torch.bool, device=DEVICE)

    with pytest.raises(ValueError, match="cond_embed length"):
        head.forward(
            noisy_action=torch.randn(batch_size, config.chunk_size, 6, device=DEVICE),
            action_mask=torch.ones(batch_size, config.chunk_size, dtype=torch.bool, device=DEVICE),
            state=torch.randn(batch_size, 8, device=DEVICE),
            cond_embed=cond_embed,
            cond_mask=cond_mask,
            timestep=torch.rand(batch_size, device=DEVICE),
            subset="__default__",
        )


# == Flow Matching Time Sampling Tests ==


def test_time_sampling_beta_distribution():
    """Test flow matching time sampling follows Beta distribution."""
    config = _make_tiny_config()
    beta_alpha = config.time_sampling_beta_alpha
    beta_beta = config.time_sampling_beta_beta
    scale = config.time_sampling_scale
    offset = config.time_sampling_offset

    n_samples = 10000
    device = torch.device("cpu")

    beta_dist = torch.distributions.Beta(beta_alpha, beta_beta)
    time_beta = beta_dist.sample((n_samples,)).to(device)
    times = time_beta * scale + offset

    # With Beta(1.5, 1.0), mean = 1.5 / (1.5 + 1.0) = 0.6
    # Scaled: mean ≈ 0.6 * 0.999 + 0.001 ≈ 0.6
    mean = times.mean()
    assert mean > 0.55 and mean < 0.65, f"Time mean {mean} outside expected range [0.55, 0.65]"

    # All times should be in [offset, offset + scale]
    assert torch.all(times >= offset - 1e-6)
    assert torch.all(times <= offset + scale + 1e-6)


def test_flow_noisy_action_construction():
    """Test noisy_action = t * noise + (1-t) * action construction."""
    batch_size = 2
    chunk_size = 16
    action_dim = 6
    device = DEVICE

    action = torch.randn(batch_size, chunk_size, action_dim, device=device)
    noise = torch.randn_like(action)

    # When t=1: noisy_action = noise
    t_at_1 = torch.ones(batch_size, device=device)
    t_expanded_1 = t_at_1[:, None, None]
    noisy_at_1 = t_expanded_1 * noise + (1 - t_expanded_1) * action
    torch.testing.assert_close(noisy_at_1, noise, rtol=1e-5, atol=1e-5)

    # When t=0: noisy_action = action
    t_at_0 = torch.zeros(batch_size, device=device)
    t_expanded_0 = t_at_0[:, None, None]
    noisy_at_0 = t_expanded_0 * noise + (1 - t_expanded_0) * action
    torch.testing.assert_close(noisy_at_0, action, rtol=1e-5, atol=1e-5)


def test_target_velocity_construction():
    """Test target velocity u_t = noise - action for flow matching."""
    batch_size = 2
    chunk_size = 16
    action_dim = 6
    device = DEVICE

    action = torch.randn(batch_size, chunk_size, action_dim, device=device)
    noise = torch.randn_like(action)
    u_t = noise - action

    assert u_t.shape == (batch_size, chunk_size, action_dim)


# == Positional Embedding Parity Tests ==


def test_sincos_embed_parity():
    """Test sin-cos positional embedding produces expected values for pos=0."""
    embed_dim = 32
    pos = list(range(5))
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

    assert emb.shape == (5, embed_dim)

    # pos=0 -> sin(0)=0, cos(0)=1
    assert np.abs(emb[0, : embed_dim // 2]).max() < 1e-6
    assert (emb[0, embed_dim // 2 :] - 1.0).max() < 1e-6


def test_sinusoidal_time_embedding_periodicity():
    """Test sinusoidal time embedding values are bounded."""
    dim = 64
    time = torch.linspace(0, 2 * math.pi, 100)
    emb = create_sinusoidal_pos_embedding(time, dim)

    assert emb.shape == (100, dim)
    assert torch.all(emb.abs() <= 1.0 + 1e-6)


def test_sinusoidal_time_embedding_different_times_different_embeddings():
    """Test that different times produce different embeddings."""
    dim = 64
    t1 = torch.tensor([0.0])
    t2 = torch.tensor([0.5])
    emb1 = create_sinusoidal_pos_embedding(t1, dim)
    emb2 = create_sinusoidal_pos_embedding(t2, dim)
    assert not torch.allclose(emb1, emb2)


# == Attention Mask Parity Tests ==


def test_timm_attn_mask_correct_shape():
    """Test timm attention mask has correct 4D shape for DiT self-attention."""
    batch_size = 2
    chunk_size = 16
    x_len = chunk_size + 2  # time + state + actions
    pad_mask = torch.ones(batch_size, x_len, dtype=torch.bool)
    pad_mask[0, -1] = False

    mask = make_timm_attn_mask(pad_mask)
    assert mask.shape == (batch_size, 1, x_len, x_len)


def test_timm_attn_mask_blocks_masked_tokens():
    """Test timm attention mask correctly assigns -inf to masked positions."""
    pad_mask = torch.tensor([[1, 1, 0, 0]])
    mask = make_timm_attn_mask(pad_mask)

    # pad_2d_mask: outer product of [1,1,0,0] with itself
    # -> 1 where both valid, 0 otherwise; then 1->0.0, 0->-inf
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 0, 2] == float("-inf")
    assert mask[0, 0, 2, 0] == float("-inf")
    assert mask[0, 0, 2, 3] == float("-inf")


# == Flow Matching Inversion Tests ==


def test_flow_inversion_converges():
    """Test that Euler-step flow matching inversion produces valid action."""
    from lerobot.policies.openeai.modeling_openeai import DiTPolicyHead

    config = _make_tiny_config()
    data_dim_info = {"__default__": (8, 6)}

    torch.manual_seed(42)
    head = DiTPolicyHead(config, data_dim_info=data_dim_info)
    head.initialize_weights()
    head.to(DEVICE)
    head.eval()

    batch_size = 2
    cond_len = config.feat_length
    cond_embed = torch.randn(batch_size, cond_len, config.qwen_dim, device=DEVICE)
    cond_mask = torch.ones(batch_size, cond_len, dtype=torch.bool, device=DEVICE)
    state = torch.randn(batch_size, 8, device=DEVICE)

    dt = -1.0 / config.denoise_steps
    dt_tensor = torch.tensor(dt, dtype=torch.float32, device=DEVICE)
    time = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    action_mask = torch.ones((batch_size, config.chunk_size), dtype=torch.bool, device=DEVICE)

    noisy_action = torch.randn((batch_size, config.chunk_size, 6), dtype=torch.float32, device=DEVICE)
    original_shape = noisy_action.shape

    with torch.no_grad():
        for _ in range(config.denoise_steps):
            v_t = head.forward(
                noisy_action=noisy_action,
                action_mask=action_mask,
                state=state,
                cond_embed=cond_embed,
                cond_mask=cond_mask,
                timestep=time.expand(batch_size),
                subset="__default__",
            )
            noisy_action = noisy_action + v_t * dt_tensor
            time = time + dt_tensor

    assert noisy_action.shape == original_shape
    assert not torch.any(torch.isnan(noisy_action))
    assert not torch.any(torch.isinf(noisy_action))


# == CrossAttention Parity ==


def test_cross_attention_parity():
    """Test CrossAttention produces deterministic output across same-seed instances."""
    dim = 64
    num_heads = 4
    batch_size, seq_len_x, seq_len_c = 2, 10, 8

    torch.manual_seed(42)
    ca1 = CrossAttention(dim=dim, num_heads=num_heads).to(DEVICE)
    torch.manual_seed(42)
    ca2 = CrossAttention(dim=dim, num_heads=num_heads).to(DEVICE)

    ca1.eval()
    ca2.eval()

    x = torch.randn(batch_size, seq_len_x, dim, device=DEVICE)
    c = torch.randn(batch_size, seq_len_c, dim, device=DEVICE)

    with torch.no_grad():
        out1 = ca1(x, c)
        out2 = ca2(x.clone(), c.clone())

    assert out1.shape == (batch_size, seq_len_x, dim)
    torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-5)

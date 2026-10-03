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

"""Behavior-pinning tests for the shared VLA helpers.

These helpers are the canonical versions of functions that used to be copy-pasted across
the openpi-derived policies (pi0, pi05, pi0_fast, smolvla, eo1, xvla). The expected
values below encode the historical per-policy behavior exactly; a failure here means a
behavior change that would silently affect released checkpoints.
"""

import contextlib
import copy
import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from lerobot.policies.common.vla_utils import (
    create_sinusoidal_pos_embedding,
    fuse_action_time_embedding,
    make_att_2d_masks,
    pad_vector,
    prepare_attention_masks_4d,
    resize_with_pad,
    resize_with_pad_torch,
)
from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE


def test_create_sinusoidal_pos_embedding_matches_openpi_formula():
    time = torch.tensor([0.0, 0.25, 1.0])
    dim, min_period, max_period = 8, 4e-3, 4.0
    emb = create_sinusoidal_pos_embedding(time, dim, min_period, max_period, device=torch.device("cpu"))

    assert emb.shape == (3, dim)
    # Independent recomputation of the openpi formula in float64.
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi
    sin_input = scaling[None, :] * time.to(torch.float64)[:, None]
    expected = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    torch.testing.assert_close(emb, expected, rtol=1e-9, atol=1e-9)


def test_create_sinusoidal_pos_embedding_per_action_time_matches_scalar():
    """Per-action time (training-time RTC) must embed each row exactly like the scalar path."""
    dim, min_period, max_period = 8, 4e-3, 4.0
    per_action = torch.tensor([[0.0, 0.25, 0.25], [1.0, 0.5, 0.5]])
    emb = create_sinusoidal_pos_embedding(per_action, dim, min_period, max_period, device=torch.device("cpu"))

    assert emb.shape == (2, 3, dim)
    flat = create_sinusoidal_pos_embedding(
        per_action.reshape(-1), dim, min_period, max_period, device=torch.device("cpu")
    )
    torch.testing.assert_close(emb.reshape(-1, dim), flat)


def test_create_sinusoidal_pos_embedding_validation():
    with pytest.raises(ValueError, match="divisible by 2"):
        create_sinusoidal_pos_embedding(torch.zeros(2), 7, 4e-3, 4.0, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="must have shape"):
        create_sinusoidal_pos_embedding(torch.zeros(2, 2, 2), 8, 4e-3, 4.0, device=torch.device("cpu"))


BATCH, HORIZON, ACTION_DIM, WIDTH = 3, 4, 6, 16
MIN_PERIOD, MAX_PERIOD = 4e-3, 4.0

# The checkpoint keys pi0, smolvla and eo1 all ship for this block. `fuse_action_time_embedding`
# owns no parameters, so migrating to it must leave these byte-for-byte identical.
ACTION_TIME_STATE_DICT_KEYS = [
    "action_in_proj.bias",
    "action_in_proj.weight",
    "action_time_mlp_in.bias",
    "action_time_mlp_in.weight",
    "action_time_mlp_out.bias",
    "action_time_mlp_out.weight",
]


class _ActionTimeLayers(nn.Module):
    """The layers every adopter owns, under the attribute names their checkpoints use."""

    def __init__(self, dtype=None):
        super().__init__()
        self.action_in_proj = nn.Linear(ACTION_DIM, WIDTH, dtype=dtype)
        self.action_time_mlp_in = nn.Linear(2 * WIDTH, WIDTH, dtype=dtype)
        self.action_time_mlp_out = nn.Linear(WIDTH, WIDTH, dtype=dtype)


def _call_directly(func, *args):
    return func(*args)


def _checkpointed(func, *args):
    """The `_apply_checkpoint` branch pi0 and eo1 take while training."""
    return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, preserve_rng_state=False)


def _pi0_reference(layers, noisy_actions, timestep, apply_checkpoint):
    """Verbatim pi0 `embed_suffix` block, copied before the extraction."""
    time_emb = create_sinusoidal_pos_embedding(
        timestep,
        layers.action_in_proj.out_features,
        min_period=MIN_PERIOD,
        max_period=MAX_PERIOD,
        device=timestep.device,
    )
    time_emb = time_emb.type(dtype=timestep.dtype)

    def action_proj_func(noisy_actions):
        return layers.action_in_proj(noisy_actions)

    action_emb = apply_checkpoint(action_proj_func, noisy_actions)

    time_emb = time_emb[:, None, :].expand_as(action_emb)
    action_time_emb = torch.cat([action_emb, time_emb], dim=2)

    def mlp_func(action_time_emb):
        x = layers.action_time_mlp_in(action_time_emb)
        x = F.silu(x)
        return layers.action_time_mlp_out(x)

    return apply_checkpoint(mlp_func, action_time_emb)


def _pi0_migrated(layers, noisy_actions, timestep, apply_checkpoint):
    def action_proj_func(noisy_actions):
        return layers.action_in_proj(noisy_actions)

    def mlp_func(action_time_emb):
        x = layers.action_time_mlp_in(action_time_emb)
        x = F.silu(x)
        return layers.action_time_mlp_out(x)

    return fuse_action_time_embedding(
        noisy_actions,
        timestep,
        action_proj=action_proj_func,
        action_time_mlp=mlp_func,
        embedding_width=layers.action_in_proj.out_features,
        min_period=MIN_PERIOD,
        max_period=MAX_PERIOD,
        apply_checkpoint=apply_checkpoint,
    )


def _smolvla_reference(layers, noisy_actions, timestep):
    """Verbatim smolvla `embed_suffix` block, copied before the extraction."""
    action_emb = layers.action_in_proj(noisy_actions)
    device = action_emb.device
    dtype = action_emb.dtype
    time_emb = create_sinusoidal_pos_embedding(
        timestep,
        WIDTH,
        MIN_PERIOD,
        MAX_PERIOD,
        device=device,
    )
    time_emb = time_emb.type(dtype=dtype)

    time_emb = time_emb[:, None, :].expand_as(action_emb)
    action_time_emb = torch.cat([action_emb, time_emb], dim=2)

    action_time_emb = layers.action_time_mlp_in(action_time_emb)
    action_time_emb = F.silu(action_time_emb)  # swish == silu
    return layers.action_time_mlp_out(action_time_emb)


def _smolvla_migrated(layers, noisy_actions, timestep):
    def mlp_func(action_time_emb):
        action_time_emb = layers.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        return layers.action_time_mlp_out(action_time_emb)

    return fuse_action_time_embedding(
        noisy_actions,
        timestep,
        action_proj=layers.action_in_proj,
        action_time_mlp=mlp_func,
        embedding_width=WIDTH,
        min_period=MIN_PERIOD,
        max_period=MAX_PERIOD,
    )


def _eo1_autocast_context(layers, force_fp32):
    """eo1's `flow_head_autocast_context`."""
    if force_fp32:
        return torch.autocast(device_type=layers.action_in_proj.weight.device.type, enabled=False)
    return contextlib.nullcontext()


def _eo1_reference(layers, noisy_actions, timestep, apply_checkpoint, force_fp32):
    """Verbatim eo1 `embed_suffix` block, copied before the extraction."""

    def action_proj_func(noisy_actions):
        with _eo1_autocast_context(layers, force_fp32):
            noisy_actions = noisy_actions.to(dtype=layers.action_in_proj.weight.dtype)
            return layers.action_in_proj(noisy_actions)

    action_embs = apply_checkpoint(action_proj_func, noisy_actions)
    time_embs = create_sinusoidal_pos_embedding(
        timestep,
        WIDTH,
        min_period=MIN_PERIOD,
        max_period=MAX_PERIOD,
        device=action_embs.device,
    )
    time_embs = time_embs.to(dtype=action_embs.dtype)
    time_embs = time_embs[:, None, :].expand_as(action_embs)
    action_time_embs = torch.cat([action_embs, time_embs], dim=2)

    def mlp_func(action_time_embs):
        with _eo1_autocast_context(layers, force_fp32):
            action_time_embs = action_time_embs.to(dtype=layers.action_time_mlp_in.weight.dtype)
            action_time_embs = layers.action_time_mlp_in(action_time_embs)
            action_time_embs = F.silu(action_time_embs)
            return layers.action_time_mlp_out(action_time_embs)

    return apply_checkpoint(mlp_func, action_time_embs)


def _eo1_migrated(layers, noisy_actions, timestep, apply_checkpoint, force_fp32):
    def action_proj_func(noisy_actions):
        with _eo1_autocast_context(layers, force_fp32):
            noisy_actions = noisy_actions.to(dtype=layers.action_in_proj.weight.dtype)
            return layers.action_in_proj(noisy_actions)

    def mlp_func(action_time_embs):
        with _eo1_autocast_context(layers, force_fp32):
            action_time_embs = action_time_embs.to(dtype=layers.action_time_mlp_in.weight.dtype)
            action_time_embs = layers.action_time_mlp_in(action_time_embs)
            action_time_embs = F.silu(action_time_embs)
            return layers.action_time_mlp_out(action_time_embs)

    return fuse_action_time_embedding(
        noisy_actions,
        timestep,
        action_proj=action_proj_func,
        action_time_mlp=mlp_func,
        embedding_width=WIDTH,
        min_period=MIN_PERIOD,
        max_period=MAX_PERIOD,
        apply_checkpoint=apply_checkpoint,
    )


def _suffix_inputs(requires_grad=False):
    torch.manual_seed(0)
    noisy_actions = torch.randn(BATCH, HORIZON, ACTION_DIM, requires_grad=requires_grad)
    timestep = torch.rand(BATCH, dtype=torch.float32)
    return noisy_actions, timestep


def _assert_grads_equal(reference_layers, migrated_layers):
    reference_grads = {name: p.grad for name, p in reference_layers.named_parameters()}
    for name, param in migrated_layers.named_parameters():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.equal(param.grad, reference_grads[name]), f"gradient mismatch on {name}"


@pytest.mark.parametrize("apply_checkpoint", [_call_directly, _checkpointed])
def test_fuse_action_time_embedding_matches_pi0_block(apply_checkpoint):
    torch.manual_seed(1)
    reference_layers = _ActionTimeLayers()
    migrated_layers = copy.deepcopy(reference_layers)
    noisy_actions, timestep = _suffix_inputs()

    expected = _pi0_reference(reference_layers, noisy_actions, timestep, apply_checkpoint)
    actual = _pi0_migrated(migrated_layers, noisy_actions, timestep, apply_checkpoint)

    assert torch.equal(actual, expected)
    assert actual.shape == (BATCH, HORIZON, WIDTH)


@pytest.mark.parametrize("apply_checkpoint", [_call_directly, _checkpointed])
def test_fuse_action_time_embedding_matches_pi0_block_under_bf16_autocast(apply_checkpoint):
    """pi0 trains under bf16 autocast, which is where the two blocks disagree about dtype.

    The historical block cast the time embedding to the timestep's dtype (fp32) and let the
    concatenation promote; the helper casts it to the action embedding's (bf16 here). The MLP's
    autocast wrapper rounds its whole input to bf16 either way, and that rounding is elementwise,
    so doing it before or after the concatenation lands on the same bits.
    """
    torch.manual_seed(7)
    reference_layers = _ActionTimeLayers(dtype=torch.float32)
    migrated_layers = copy.deepcopy(reference_layers)
    noisy_actions, timestep = _suffix_inputs()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        expected = _pi0_reference(reference_layers, noisy_actions, timestep, apply_checkpoint)
        actual = _pi0_migrated(migrated_layers, noisy_actions, timestep, apply_checkpoint)

    assert expected.dtype == torch.bfloat16 and actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


def test_fuse_action_time_embedding_pi0_time_embedding_follows_the_action_dtype():
    """Pins the one case where the helper deliberately departs from the historical pi0 block.

    With bf16 weights and no autocast to reconcile them, casting the time embedding to the
    timestep's fp32 built a mixed-dtype concatenation that the MLP rejects outright. Following
    the action embedding's dtype instead is what makes the block dtype-agnostic.
    """
    torch.manual_seed(8)
    layers = _ActionTimeLayers(dtype=torch.bfloat16)
    noisy_actions, timestep = _suffix_inputs()
    noisy_actions = noisy_actions.to(torch.bfloat16)

    with pytest.raises(RuntimeError, match="same dtype"):
        _pi0_reference(layers, noisy_actions, timestep, _call_directly)

    assert _pi0_migrated(layers, noisy_actions, timestep, _call_directly).dtype == torch.bfloat16


def test_fuse_action_time_embedding_matches_smolvla_block():
    torch.manual_seed(2)
    reference_layers = _ActionTimeLayers()
    migrated_layers = copy.deepcopy(reference_layers)
    noisy_actions, timestep = _suffix_inputs()

    expected = _smolvla_reference(reference_layers, noisy_actions, timestep)
    actual = _smolvla_migrated(migrated_layers, noisy_actions, timestep)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("apply_checkpoint", [_call_directly, _checkpointed])
@pytest.mark.parametrize("outer_autocast", [False, True])
@pytest.mark.parametrize("force_fp32", [False, True])
def test_fuse_action_time_embedding_matches_eo1_block(apply_checkpoint, outer_autocast, force_fp32):
    """eo1 threads its weight-dtype casts and autocast context through the two callables."""
    torch.manual_seed(3)
    reference_layers = _ActionTimeLayers(dtype=torch.float32)
    migrated_layers = copy.deepcopy(reference_layers)
    noisy_actions, timestep = _suffix_inputs()

    autocast = (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        if outer_autocast
        else contextlib.nullcontext()
    )
    with autocast:
        expected = _eo1_reference(reference_layers, noisy_actions, timestep, apply_checkpoint, force_fp32)
        actual = _eo1_migrated(migrated_layers, noisy_actions, timestep, apply_checkpoint, force_fp32)

    assert torch.equal(actual, expected)
    # Under bf16 autocast the flow head runs in bf16 unless eo1 forces it back to fp32.
    expected_dtype = torch.float32 if force_fp32 or not outer_autocast else torch.bfloat16
    assert actual.dtype == expected_dtype


@pytest.mark.parametrize("apply_checkpoint", [_call_directly, _checkpointed])
def test_fuse_action_time_embedding_pi0_gradients_match(apply_checkpoint):
    torch.manual_seed(4)
    reference_layers = _ActionTimeLayers()
    migrated_layers = copy.deepcopy(reference_layers)
    reference_actions, timestep = _suffix_inputs(requires_grad=True)
    migrated_actions = reference_actions.detach().clone().requires_grad_(True)

    _pi0_reference(reference_layers, reference_actions, timestep, apply_checkpoint).sum().backward()
    _pi0_migrated(migrated_layers, migrated_actions, timestep, apply_checkpoint).sum().backward()

    # Both stages are differentiable and identical: the projection *and* the MLP.
    _assert_grads_equal(reference_layers, migrated_layers)
    assert torch.equal(migrated_actions.grad, reference_actions.grad)


@pytest.mark.parametrize("apply_checkpoint", [_call_directly, _checkpointed])
def test_fuse_action_time_embedding_pi0_gradients_match_under_bf16_autocast(apply_checkpoint):
    torch.manual_seed(9)
    reference_layers = _ActionTimeLayers(dtype=torch.float32)
    migrated_layers = copy.deepcopy(reference_layers)
    reference_actions, timestep = _suffix_inputs(requires_grad=True)
    migrated_actions = reference_actions.detach().clone().requires_grad_(True)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _pi0_reference(reference_layers, reference_actions, timestep, apply_checkpoint).sum().backward()
        _pi0_migrated(migrated_layers, migrated_actions, timestep, apply_checkpoint).sum().backward()

    _assert_grads_equal(reference_layers, migrated_layers)
    assert torch.equal(migrated_actions.grad, reference_actions.grad)


def test_fuse_action_time_embedding_smolvla_gradients_match():
    torch.manual_seed(5)
    reference_layers = _ActionTimeLayers()
    migrated_layers = copy.deepcopy(reference_layers)
    reference_actions, timestep = _suffix_inputs(requires_grad=True)
    migrated_actions = reference_actions.detach().clone().requires_grad_(True)

    _smolvla_reference(reference_layers, reference_actions, timestep).sum().backward()
    _smolvla_migrated(migrated_layers, migrated_actions, timestep).sum().backward()

    _assert_grads_equal(reference_layers, migrated_layers)
    assert torch.equal(migrated_actions.grad, reference_actions.grad)


@pytest.mark.parametrize("outer_autocast", [False, True])
@pytest.mark.parametrize("force_fp32", [False, True])
def test_fuse_action_time_embedding_eo1_gradients_match(outer_autocast, force_fp32):
    torch.manual_seed(6)
    reference_layers = _ActionTimeLayers(dtype=torch.float32)
    migrated_layers = copy.deepcopy(reference_layers)
    reference_actions, timestep = _suffix_inputs(requires_grad=True)
    migrated_actions = reference_actions.detach().clone().requires_grad_(True)

    autocast = (
        torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        if outer_autocast
        else contextlib.nullcontext()
    )
    with autocast:
        _eo1_reference(
            reference_layers, reference_actions, timestep, _checkpointed, force_fp32
        ).sum().backward()
        _eo1_migrated(migrated_layers, migrated_actions, timestep, _checkpointed, force_fp32).sum().backward()

    _assert_grads_equal(reference_layers, migrated_layers)
    assert torch.equal(migrated_actions.grad, reference_actions.grad)


def test_fuse_action_time_embedding_leaves_state_dict_keys_untouched():
    """The helper owns no parameters, so no adopter's checkpoint keys can move."""
    reference_layers = _ActionTimeLayers()
    migrated_layers = copy.deepcopy(reference_layers)
    noisy_actions, timestep = _suffix_inputs()

    keys_before = sorted(reference_layers.state_dict().keys())
    assert keys_before == ACTION_TIME_STATE_DICT_KEYS

    _pi0_migrated(migrated_layers, noisy_actions, timestep, _call_directly)
    _smolvla_migrated(migrated_layers, noisy_actions, timestep)
    _eo1_migrated(migrated_layers, noisy_actions, timestep, _call_directly, force_fp32=True)

    assert sorted(migrated_layers.state_dict().keys()) == keys_before


def test_fuse_action_time_embedding_checkpoints_both_stages():
    layers = _ActionTimeLayers()
    noisy_actions, timestep = _suffix_inputs()
    wrapped = []

    def recording_apply_checkpoint(func, *args):
        wrapped.append(func)
        return func(*args)

    _pi0_migrated(layers, noisy_actions, timestep, recording_apply_checkpoint)

    assert len(wrapped) == 2, "both the action projection and the MLP must be checkpointed"


def test_fuse_action_time_embedding_requires_scalar_timesteps():
    """This extraction is scoped to one timestep per batch element; the sinusoid stays broader."""
    layers = _ActionTimeLayers()
    noisy_actions, _ = _suffix_inputs()
    per_action_timestep = torch.rand(BATCH, HORIZON)

    with pytest.raises(ValueError, match=r"must have shape \(batch_size,\)"):
        _smolvla_migrated(layers, noisy_actions, per_action_timestep)

    # The shared sinusoid keeps its (batch, action_horizon) support for future consumers.
    assert create_sinusoidal_pos_embedding(
        per_action_timestep, WIDTH, MIN_PERIOD, MAX_PERIOD, device=torch.device("cpu")
    ).shape == (BATCH, HORIZON, WIDTH)


def test_make_att_2d_masks_docstring_cases():
    # Pure causal attention: [[1 1 1]]
    pad = torch.ones(1, 3, dtype=torch.bool)
    att = torch.tensor([[1, 1, 1]], dtype=torch.int32)
    expected = torch.tensor([[[1, 0, 0], [1, 1, 0], [1, 1, 1]]], dtype=torch.bool)
    assert torch.equal(make_att_2d_masks(pad, att), expected)

    # Prefix-LM: [[0 0 1 1]] -> first two tokens attend bidirectionally, rest causal.
    att = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
    pad = torch.ones(1, 4, dtype=torch.bool)
    expected = torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]], dtype=torch.bool)
    assert torch.equal(make_att_2d_masks(pad, att), expected)

    # Padding removes rows and columns.
    pad = torch.tensor([[True, True, False]])
    att = torch.tensor([[0, 1, 1]], dtype=torch.int32)
    out = make_att_2d_masks(pad, att)
    assert not out[0, :, 2].any() and not out[0, 2, :].any()


def test_make_att_2d_masks_validation():
    with pytest.raises(ValueError):
        make_att_2d_masks(torch.ones(3, dtype=torch.bool), torch.ones(1, 3, dtype=torch.int32))
    with pytest.raises(ValueError):
        make_att_2d_masks(torch.ones(1, 3, dtype=torch.bool), torch.ones(3, dtype=torch.int32))


def test_prepare_attention_masks_4d():
    masks = torch.tensor([[[True, False], [False, True]]])
    out = prepare_attention_masks_4d(masks)
    assert out.shape == (1, 1, 2, 2)
    expected = torch.tensor([[[[0.0, OPENPI_ATTENTION_MASK_VALUE], [OPENPI_ATTENTION_MASK_VALUE, 0.0]]]])
    assert torch.equal(out, expected)

    out_bf16 = prepare_attention_masks_4d(masks, dtype=torch.bfloat16)
    assert out_bf16.dtype == torch.bfloat16
    assert torch.equal(out_bf16, expected.to(torch.bfloat16))


def test_pad_vector_openpi_semantics():
    v = torch.arange(6.0).reshape(2, 3)
    padded = pad_vector(v, 5)
    assert padded.shape == (2, 5)
    assert torch.equal(padded[:, :3], v) and not padded[:, 3:].any()
    # Already large enough (>=): returned unchanged, same object.
    assert pad_vector(v, 3) is v
    assert pad_vector(v, 2) is v
    # 3D input.
    v3 = torch.ones(2, 4, 3)
    assert pad_vector(v3, 7).shape == (2, 4, 7)


def test_pad_vector_truncate_semantics():
    v = torch.arange(6.0).reshape(2, 3)
    out = pad_vector(v, 2, truncate=True)
    assert out.shape == (2, 2) and torch.equal(out, v[:, :2])
    out = pad_vector(v, 5, truncate=True)
    assert out.shape == (2, 5) and torch.equal(out[:, :3], v) and not out[:, 3:].any()
    assert pad_vector(v, 0, truncate=True).shape == (2, 0)
    assert pad_vector(v, 3, truncate=True) is v


@pytest.mark.parametrize("channels_last", [True, False])
def test_resize_with_pad_torch_centered(channels_last):
    img = torch.rand(2, 3, 30, 60) if not channels_last else torch.rand(2, 30, 60, 3)
    out = resize_with_pad_torch(img, 64, 64)
    if channels_last:
        assert out.shape == (2, 64, 64, 3)
        # Aspect ratio preserved: 30x60 -> 32x64, padded 16 top and 16 bottom (centered).
        assert not out[:, :16].any() and not out[:, -16:].any()
        assert out[:, 16:48].abs().sum() > 0
    else:
        assert out.shape == (2, 3, 64, 64)
        assert not out[:, :, :16].any() and not out[:, :, -16:].any()


def test_resize_with_pad_torch_uint8_roundtrip():
    img = (torch.rand(1, 3, 20, 20) * 255).to(torch.uint8)
    out = resize_with_pad_torch(img, 40, 40)
    assert out.dtype == torch.uint8 and out.shape == (1, 3, 40, 40)
    with pytest.raises(ValueError, match="Unsupported image dtype"):
        resize_with_pad_torch(torch.rand(1, 3, 8, 8, dtype=torch.float64), 16, 16)


def test_resize_with_pad_top_left():
    img = torch.rand(2, 3, 30, 60)
    out = resize_with_pad(img, 64, 64, pad_value=-1.0)
    assert out.shape == (2, 3, 64, 64)
    # 30x60 -> 32x64; this variant pads on the TOP only (32 rows of pad_value).
    assert torch.equal(out[:, :, :32], torch.full((2, 3, 32, 64), -1.0))
    assert out[:, :, 32:].min() >= 0
    # No-op fast path returns the same object.
    assert resize_with_pad(img, 30, 60, pad_value=0.0) is img
    with pytest.raises(ValueError, match="expected"):
        resize_with_pad(torch.rand(3, 8, 8), 16, 16, pad_value=0.0)


def test_clone_past_key_values():
    pytest.importorskip("transformers")
    from transformers import DynamicCache

    from lerobot.policies.common.vla_utils import clone_past_key_values

    cache = DynamicCache()
    keys, values = torch.rand(1, 2, 4, 8), torch.rand(1, 2, 4, 8)
    cache.update(keys, values, 0)
    cloned = clone_past_key_values(cache)
    (ck, cv, _), (ok, ov, _) = next(iter(cloned)), next(iter(cache))
    assert torch.equal(ck, ok) and torch.equal(cv, ov)
    # Deep copy: mutating the clone must not touch the original.
    ck.zero_()
    assert not torch.equal(ck, ok)


def test_clone_past_key_values_is_fullgraph_compilable():
    pytest.importorskip("transformers")
    from transformers import DynamicCache

    from lerobot.policies.common.vla_utils import clone_past_key_values

    cache = DynamicCache()
    keys, values = torch.rand(1, 2, 4, 8), torch.rand(1, 2, 4, 8)
    cache.update(keys, values, 0)

    compiled_clone = torch.compile(clone_past_key_values, backend="eager", fullgraph=True)
    cloned = compiled_clone(cache)

    (cloned_keys, cloned_values, _), (original_keys, original_values, _) = (
        next(iter(cloned)),
        next(iter(cache)),
    )
    assert torch.equal(cloned_keys, original_keys)
    assert torch.equal(cloned_values, original_values)

#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest
import torch

pytest.importorskip("transformers")

from transformers.models.gemma.configuration_gemma import GemmaConfig  # noqa: E402
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding  # noqa: E402

from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import (  # noqa: E402
    _build_flow_matching_inputs,
    _reduce_training_rtc_loss,
    compute_layer_complete,
)
from lerobot.policies.pi_gemma import _get_pi_gemma_decoder_layer_base  # noqa: E402


def test_pi05_training_rtc_uses_clean_prefix_and_per_token_time():
    actions = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    noise = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
    time = torch.tensor([0.25])
    prefix_mask = torch.tensor([[True, True, False, False]])

    x_t, model_time = _build_flow_matching_inputs(actions, noise, time, prefix_mask)

    assert model_time.tolist() == [[0.0, 0.0, 0.25, 0.25]]
    assert torch.equal(x_t[:, :2], actions[:, :2])
    assert torch.equal(x_t[:, 2:], 0.25 * noise[:, 2:] + 0.75 * actions[:, 2:])


def test_pi05_training_rtc_loss_excludes_clean_prefix():
    losses = torch.tensor([[[100.0], [100.0], [2.0], [4.0]]])
    prefix_mask = torch.tensor([[True, True, False, False]])

    loss = _reduce_training_rtc_loss(losses, prefix_mask, reduction="mean")

    assert loss.item() == pytest.approx(3.0)


@pytest.mark.parametrize("max_delay", [-1, 5])
def test_pi05_config_rejects_invalid_training_rtc_delay(max_delay):
    with pytest.raises(ValueError, match="rtc_training_max_delay"):
        PI05Config(chunk_size=5, n_action_steps=5, rtc_training_max_delay=max_delay)


# ``compute_layer_complete`` hardcodes 8 attention heads when reshaping the joint
# attention output, so the miniature stack below has to match that.
_HEADS, _HEAD_DIM = 8, 4
_WIDTH = _HEADS * _HEAD_DIM


def _mini_gemma_config(use_adarms: bool) -> GemmaConfig:
    config = GemmaConfig(
        hidden_size=_WIDTH,
        intermediate_size=2 * _WIDTH,
        num_hidden_layers=1,
        num_attention_heads=_HEADS,
        num_key_value_heads=_HEADS,
        head_dim=_HEAD_DIM,
        vocab_size=32,
    )
    config.use_adarms = use_adarms
    config.adarms_cond_dim = _WIDTH if use_adarms else None
    config._attn_implementation = "eager"  # noqa: SLF001
    return config


def _mini_two_stream_layers() -> tuple:
    """VLM stream without AdaRMS, action-expert stream with it — the pi05 layout."""
    layer_cls = _get_pi_gemma_decoder_layer_base()
    vlm_config = _mini_gemma_config(use_adarms=False)
    expert_config = _mini_gemma_config(use_adarms=True)
    layers = (layer_cls(vlm_config, 0), layer_cls(expert_config, 0))
    return layers, GemmaRotaryEmbedding(vlm_config)


def test_pi05_expert_layer_accepts_per_action_time_conditioning():
    """The pi05 training path feeds a (B, chunk, width) AdaRMS cond to the action expert.

    pi05 injects the flow timestep only through AdaRMS, so training-time RTC's per-action
    timesteps have to survive this two-stream layer unchanged in shape.
    """
    torch.manual_seed(0)
    batch, prefix_tokens, chunk = 2, 3, 4
    layers, rotary_emb = _mini_two_stream_layers()

    inputs_embeds = [torch.randn(batch, prefix_tokens, _WIDTH), torch.randn(batch, chunk, _WIDTH)]
    total = prefix_tokens + chunk
    position_ids = torch.arange(total)[None].expand(batch, total)
    attention_mask = torch.zeros(batch, 1, total, total)
    # Expert stream only: prefix (VLM) tokens keep the plain norm, as in PI05FlowMatching.
    adarms_cond = [None, torch.randn(batch, chunk, _WIDTH)]

    out_prefix, out_suffix = compute_layer_complete(
        inputs_embeds,
        attention_mask,
        position_ids,
        adarms_cond,
        layers=layers,
        rotary_emb=rotary_emb,
    )

    assert out_prefix.shape == (batch, prefix_tokens, _WIDTH)
    assert out_suffix.shape == (batch, chunk, _WIDTH)


def test_pi05_expert_layer_per_action_time_matches_scalar_when_uniform():
    """A per-action cond repeated across the chunk must reproduce the scalar-cond output."""
    torch.manual_seed(0)
    batch, prefix_tokens, chunk = 2, 3, 4
    layers, rotary_emb = _mini_two_stream_layers()
    # dense is zero-initialised by design; give it signal so a broken modulation shows up.
    torch.nn.init.normal_(layers[1].input_layernorm.dense.weight, std=0.02)
    torch.nn.init.normal_(layers[1].post_attention_layernorm.dense.weight, std=0.02)

    inputs_embeds = [torch.randn(batch, prefix_tokens, _WIDTH), torch.randn(batch, chunk, _WIDTH)]
    total = prefix_tokens + chunk
    position_ids = torch.arange(total)[None].expand(batch, total)
    attention_mask = torch.zeros(batch, 1, total, total)
    cond = torch.randn(batch, _WIDTH)

    def run(expert_cond):
        return compute_layer_complete(
            [t.clone() for t in inputs_embeds],
            attention_mask,
            position_ids,
            [None, expert_cond],
            layers=layers,
            rotary_emb=rotary_emb,
        )[1]

    torch.testing.assert_close(run(cond[:, None, :].expand(batch, chunk, _WIDTH)), run(cond))

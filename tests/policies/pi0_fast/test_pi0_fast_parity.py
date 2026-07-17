#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
pytest.importorskip("scipy")

from lerobot.configs import NormalizationMode  # noqa: E402
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig  # noqa: E402
from lerobot.policies.pi0_fast.modeling_pi0_fast import (  # noqa: E402
    PI0FastPytorch,
    PI0FastPolicy,
    _gather_last_valid_language_hidden,
    _reduce_fast_token_loss,
)
from lerobot.policies.pi0_fast.processor_pi0_fast import (  # noqa: E402
    Pi0FastPrepareStateAndLanguageTokenizerProcessorStep,
)
from lerobot.processor.tokenizer_processor import ActionTokenizerProcessorStep  # noqa: E402
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402


class _FakePaliGemmaTokenizer:
    vocab_size = 1000
    bos_token_id = 2
    eos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        if text == "Action: ":
            return [10, 11]
        if text == "|":
            return [12, self.eos_token_id] if add_special_tokens else [12]
        return [900, 901]


def test_pi0_fast_uses_openpi_quantile_normalization_by_default():
    config = PI0FastConfig()

    assert config.normalization_mapping == {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.QUANTILES,
        "ACTION": NormalizationMode.QUANTILES,
    }


def test_pi0_fast_action_tokens_have_no_second_bos():
    step = ActionTokenizerProcessorStep.__new__(ActionTokenizerProcessorStep)
    step.max_action_tokens = 8
    step.fast_skip_tokens = 128
    step.prepend_bos = False
    step.action_tokenizer = lambda _: torch.tensor([4, 9])
    step._paligemma_tokenizer = _FakePaliGemmaTokenizer()

    tokens, mask = step._tokenize_action(torch.zeros(1, 2, 1))

    mapped = [1000 - 1 - 128 - token for token in (4, 9)]
    assert tokens[0, :6].tolist() == [10, 11, *mapped, 12, 1]
    assert _FakePaliGemmaTokenizer.bos_token_id not in tokens[0, mask[0]].tolist()


def test_action_tokenizer_serializes_bos_layout():
    step = ActionTokenizerProcessorStep.__new__(ActionTokenizerProcessorStep)
    step.trust_remote_code = True
    step.max_action_tokens = 8
    step.fast_skip_tokens = 128
    step.paligemma_tokenizer_name = "paligemma"
    step.prepend_bos = False
    step.action_tokenizer_name = "fast"
    step.action_tokenizer_input_object = None

    assert step.get_config() == {
        "trust_remote_code": True,
        "max_action_tokens": 8,
        "fast_skip_tokens": 128,
        "paligemma_tokenizer_name": "paligemma",
        "prepend_bos": False,
        "action_tokenizer_name": "fast",
    }


def test_pi0_fast_prompt_canonicalizes_task_to_lowercase():
    step = Pi0FastPrepareStateAndLanguageTokenizerProcessorStep()
    transition = {
        TransitionKey.OBSERVATION: {"observation.state": torch.zeros(1, 2)},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["  Pick_UP\nCube  "]},
    }

    result = step(transition)

    assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == ["Task: pick up cube, State: 128 128;\n"]


def test_last_language_hidden_uses_attention_mask_not_padding():
    hidden = torch.arange(2 * 7, dtype=torch.float32).reshape(2, 7, 1)
    language_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

    gathered = _gather_last_valid_language_hidden(hidden, language_mask, image_token_count=3)

    assert gathered[:, 0].tolist() == [hidden[0, 4, 0].item(), hidden[1, 5, 0].item()]


def test_fast_ce_averages_each_sample_before_the_batch():
    token_loss = torch.tensor([[2.0, 99.0, 99.0], [6.0, 6.0, 6.0]])
    mask = torch.tensor([[True, False, False], [True, True, True]])

    loss = _reduce_fast_token_loss(token_loss, mask)

    assert loss.item() == pytest.approx(4.0)


class _TokenFromHiddenHead(nn.Module):
    def forward(self, hidden):
        logits = torch.full((*hidden.shape[:-1], 10), -100.0)
        logits.scatter_(-1, hidden.long(), 100.0)
        return logits


class _ScriptedPaliGemma:
    def __init__(self):
        q_proj = SimpleNamespace(weight=torch.empty(1))
        language_model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(q_proj=q_proj))]
        )
        self.paligemma = SimpleNamespace(
            lm_head=_TokenFromHiddenHead(),
            model=SimpleNamespace(language_model=language_model),
        )
        self.calls = 0

    def forward(self, inputs_embeds, **_kwargs):
        scripted_tokens = ([5, 6], [1, 7], [9, 1])
        token_ids = torch.tensor(scripted_tokens[self.calls], dtype=torch.float32)
        self.calls += 1
        hidden = token_ids[:, None, None].expand(-1, inputs_embeds[0].shape[1], -1).clone()
        return (hidden, None), object()

    @staticmethod
    def embed_language_tokens(tokens):
        return tokens.to(dtype=torch.float32).unsqueeze(-1)


def _make_scripted_generation_model(captured):
    model = PI0FastPytorch.__new__(PI0FastPytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(max_action_tokens=4)
    model._paligemma_tokenizer = SimpleNamespace(eos_token_id=1)
    model.paligemma_with_expert = _ScriptedPaliGemma()
    model._prepare_attention_masks_4d = lambda masks, dtype: masks

    def embed_prefix(_images, _img_masks, tokens, masks, **_kwargs):
        captured.append(tokens.clone())
        image_masks = torch.ones(tokens.shape[0], 1, dtype=torch.bool)
        pad_masks = torch.cat([image_masks, masks], dim=1)
        embeddings = torch.zeros(tokens.shape[0], pad_masks.shape[1], 1)
        attention = torch.ones(tokens.shape[0], pad_masks.shape[1], pad_masks.shape[1], dtype=torch.bool)
        return embeddings, pad_masks, attention, 1, 0

    model.embed_prefix_fast = embed_prefix
    return model


@pytest.mark.parametrize("sampler_name", ["sample_actions_fast", "sample_actions_fast_kv_cache"])
def test_fast_generators_stop_each_sample_at_eos_without_boundary_bos(sampler_name):
    captured = []
    model = _make_scripted_generation_model(captured)
    tokens = torch.tensor([[2, 3, 0, 0], [2, 3, 4, 0]])
    masks = torch.tensor([[True, True, False, False], [True, True, True, False]])

    generated = getattr(model, sampler_name)(
        [], [], tokens, masks, max_decoding_steps=4, temperature=0.0
    )

    assert torch.equal(generated, torch.tensor([[5, 1, 0, 0], [6, 7, 1, 0]]))
    assert torch.equal(captured[0], tokens)


def test_predict_action_chunk_decodes_full_chunk(monkeypatch):
    policy = PI0FastPolicy.__new__(PI0FastPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        chunk_size=8,
        n_action_steps=3,
        output_features={"action": SimpleNamespace(shape=(4,))},
        temperature=0.0,
        max_decoding_steps=16,
        use_kv_cache=False,
    )
    policy.model = SimpleNamespace(
        sample_actions_fast=lambda *args, **kwargs: torch.ones(1, 4, dtype=torch.long)
    )
    monkeypatch.setattr(policy, "_preprocess_images", lambda batch: ([], []))
    captured = {}

    def detokenize(tokens, action_horizon, action_dim):
        captured["shape"] = (action_horizon, action_dim)
        return torch.zeros(1, action_horizon, action_dim)

    monkeypatch.setattr(policy, "detokenize_actions", detokenize)
    batch = {
        OBS_LANGUAGE_TOKENS: torch.ones(1, 2, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 2, dtype=torch.bool),
    }

    actions = policy.predict_action_chunk(batch)

    assert captured["shape"] == (8, 4)
    assert actions.shape == (1, 8, 4)


class _FakeActionTokenizer:
    @staticmethod
    def decode(tokens, time_horizon, action_dim):
        if tokens != [[-119]]:
            raise ValueError("invalid FAST token")
        return [np.arange(time_horizon * action_dim).reshape(time_horizon, action_dim)]


class _FakeDecodeTokenizer(_FakePaliGemmaTokenizer):
    decoded_text = ""

    def decode(self, _tokens):
        return self.decoded_text

    def encode(self, text, add_special_tokens=True):
        if text == "codes":
            return [990]
        return super().encode(text, add_special_tokens=add_special_tokens)


@pytest.mark.parametrize("decoded_text", ["", "not an action", "Action: bad"])
def test_malformed_fast_generation_returns_zero_actions(decoded_text):
    policy = PI0FastPolicy.__new__(PI0FastPolicy)
    nn.Module.__init__(policy)
    tokenizer = _FakeDecodeTokenizer()
    tokenizer.decoded_text = decoded_text
    policy._paligemma_tokenizer = tokenizer
    policy.action_tokenizer = _FakeActionTokenizer()
    policy.config = SimpleNamespace(fast_skip_tokens=128)

    actions = policy.detokenize_actions(torch.tensor([[7, 1, 0]]), action_horizon=2, action_dim=2)

    assert actions.shape == (1, 2, 2)
    assert torch.equal(actions, torch.zeros_like(actions))


def test_valid_fast_generation_decodes_exact_shape():
    policy = PI0FastPolicy.__new__(PI0FastPolicy)
    nn.Module.__init__(policy)
    tokenizer = _FakeDecodeTokenizer()
    tokenizer.decoded_text = "Action: codes|"
    policy._paligemma_tokenizer = tokenizer
    policy.action_tokenizer = _FakeActionTokenizer()
    policy.config = SimpleNamespace(fast_skip_tokens=128)

    actions = policy.detokenize_actions(torch.tensor([[7, 1, 0]]), action_horizon=2, action_dim=2)

    assert actions.shape == (1, 2, 2)
    assert np.isfinite(actions.numpy()).all()

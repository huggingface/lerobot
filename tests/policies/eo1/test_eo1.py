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

"""Smoke tests for EO1's public LeRobot policy interface."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.eo1.modeling_eo1 import EO1Policy
from lerobot.utils.constants import ACTION, OBS_STATE

HIDDEN_SIZE = 8
STATE_DIM = 4
ACTION_DIM = 3
CHUNK_SIZE = 3
N_ACTION_STEPS = 2
MAX_ACTION_DIM = 6
STATE_TOKEN_ID = 5
ACTION_TOKEN_ID = 6


class DummyVLMBackbone(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=hidden_size))

    @property
    def model(self):
        return self

    def get_input_embeddings(self):
        return self.embedding

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ):
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            text_positions = torch.arange(seq_len, device=input_ids.device).expand(batch_size, -1)
        else:
            text_positions = attention_mask.long().cumsum(-1) - 1
            text_positions = text_positions.masked_fill(attention_mask == 0, 0)
        position_ids = text_positions.view(1, batch_size, seq_len).expand(3, batch_size, seq_len)
        rope_deltas = torch.zeros(batch_size, 1, dtype=torch.long, device=input_ids.device)
        return position_ids, rope_deltas

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        return gradient_checkpointing_kwargs

    def gradient_checkpointing_disable(self):
        return None

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            past_key_values=SimpleNamespace(crop=lambda prefix_len: None),
        )


def make_eo1_config():
    from lerobot.policies.eo1.configuration_eo1 import EO1Config

    return EO1Config(
        device="cpu",
        dtype="float32",
        vlm_base="dummy-qwen",
        vlm_config={},
        chunk_size=CHUNK_SIZE,
        n_action_steps=N_ACTION_STEPS,
        max_state_dim=STATE_DIM,
        max_action_dim=MAX_ACTION_DIM,
        num_denoise_steps=2,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
        },
    )


def make_policy_batch(include_action: bool) -> dict[str, torch.Tensor | int]:
    batch_size = 1
    seq_len = CHUNK_SIZE + 4
    input_ids = torch.tensor(
        [[11, STATE_TOKEN_ID, 12, ACTION_TOKEN_ID, ACTION_TOKEN_ID, ACTION_TOKEN_ID, 13]],
        dtype=torch.long,
    )
    assert input_ids.shape == (batch_size, seq_len)

    batch: dict[str, torch.Tensor | int] = {
        OBS_STATE: torch.randn(batch_size, STATE_DIM, dtype=torch.float32),
        "input_ids": input_ids,
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "pixel_values": torch.zeros(batch_size, 3, 4, 4, dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        "mm_token_type_ids": torch.zeros(batch_size, seq_len, dtype=torch.int32),
        "state_token_id": STATE_TOKEN_ID,
        "action_token_id": ACTION_TOKEN_ID,
    }
    if include_action:
        batch[ACTION] = torch.randn(batch_size, CHUNK_SIZE, ACTION_DIM, dtype=torch.float32)
    return batch


def test_lerobot_eo1_forward_pass(monkeypatch):
    monkeypatch.setattr(
        "lerobot.policies.eo1.modeling_eo1.Qwen2_5_VLForConditionalGeneration.from_pretrained",
        lambda *args, **kwargs: DummyVLMBackbone(HIDDEN_SIZE),
    )
    policy = EO1Policy(make_eo1_config())

    loss, metrics = policy.forward(make_policy_batch(include_action=True))

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert metrics["loss"] == pytest.approx(loss.item())


def test_lerobot_eo1_inference(monkeypatch):
    monkeypatch.setattr(
        "lerobot.policies.eo1.modeling_eo1.Qwen2_5_VLForConditionalGeneration.from_pretrained",
        lambda *args, **kwargs: DummyVLMBackbone(HIDDEN_SIZE),
    )
    policy = EO1Policy(make_eo1_config())

    sample_calls = {"count": 0}
    fixed_chunk = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3, 9.0, 9.0, 9.0],
                [1.1, 1.2, 1.3, 9.0, 9.0, 9.0],
                [2.1, 2.2, 2.3, 9.0, 9.0, 9.0],
            ]
        ],
        dtype=torch.float32,
    )

    def fake_sample_actions(**kwargs):
        sample_calls["count"] += 1
        return fixed_chunk

    monkeypatch.setattr(policy.model, "sample_actions", fake_sample_actions)

    batch = make_policy_batch(include_action=False)
    action_0 = policy.select_action(batch)
    action_1 = policy.select_action(batch)

    torch.testing.assert_close(action_0, fixed_chunk[:, 0, :ACTION_DIM])
    torch.testing.assert_close(action_1, fixed_chunk[:, 1, :ACTION_DIM])
    assert sample_calls["count"] == 1

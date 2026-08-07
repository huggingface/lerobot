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
from lerobot.policies.eo1.configuration_eo1 import EO1Config
from lerobot.policies.eo1.modeling_eo1 import EO1Policy
from lerobot.policies.eo1.processor_eo1 import make_eo1_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

HIDDEN_SIZE = 8
STATE_DIM = 4
ACTION_DIM = 3
CHUNK_SIZE = 3
N_ACTION_STEPS = 2
MAX_ACTION_DIM = 6
STATE_TOKEN_ID = 5
ACTION_TOKEN_ID = 6


def test_eo1_defaults_match_released_base_checkpoint():
    config = EO1Config(vlm_config={}, device="cpu")

    assert config.chunk_size == 16
    assert config.n_action_steps == 16
    assert config.max_state_dim == 32
    assert config.max_action_dim == 32
    assert config.num_denoise_steps == 10


class DummyVLMBackbone(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
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

    def generate(self, input_ids, **kwargs):
        del kwargs
        suffix = torch.tensor([[7, 8]], device=input_ids.device).expand(input_ids.shape[0], -1)
        return torch.cat([input_ids, suffix], dim=1)


class DummyTextProcessor:
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        batch_size = len(messages)
        return {
            "input_ids": torch.tensor([[1, 2]]).expand(batch_size, -1),
            "attention_mask": torch.ones(batch_size, 2, dtype=torch.long),
            "pixel_values": torch.zeros(batch_size, 3, 2, 2),
            "image_grid_thw": torch.ones(batch_size, 3, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(batch_size, 2, dtype=torch.long),
        }

    def batch_decode(self, token_ids, **kwargs):
        del kwargs
        assert torch.equal(token_ids, torch.tensor([[7, 8]]))
        return ["the cup is left of the plate"]


def make_eo1_config():
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


def test_lerobot_eo1_joint_text_and_action_supervision(monkeypatch):
    monkeypatch.setattr(
        "lerobot.policies.eo1.modeling_eo1.Qwen2_5_VLForConditionalGeneration.from_pretrained",
        lambda *args, **kwargs: DummyVLMBackbone(HIDDEN_SIZE),
    )
    policy = EO1Policy(make_eo1_config())
    batch = make_policy_batch(include_action=True)
    labels = torch.full_like(batch["input_ids"], -100)
    labels[:, 2] = batch["input_ids"][:, 2]
    batch["text_labels"] = labels

    loss, metrics = policy.forward(batch)

    assert torch.isfinite(loss)
    assert metrics["flow_loss"] > 0
    assert metrics["text_loss"] > 0
    loss.backward()
    assert policy.model.vlm_backbone.lm_head.weight.grad is not None


def test_lerobot_eo1_text_only_row_skips_flow(monkeypatch):
    monkeypatch.setattr(
        "lerobot.policies.eo1.modeling_eo1.Qwen2_5_VLForConditionalGeneration.from_pretrained",
        lambda *args, **kwargs: DummyVLMBackbone(HIDDEN_SIZE),
    )
    policy = EO1Policy(make_eo1_config())
    batch = make_policy_batch(include_action=True)
    batch["input_ids"] = torch.tensor([[11, STATE_TOKEN_ID, 12, 13, 14, 15, 16]])
    labels = torch.full_like(batch["input_ids"], -100)
    labels[:, 3] = batch["input_ids"][:, 3]
    batch["text_labels"] = labels

    loss, metrics = policy.forward(batch)

    assert torch.isfinite(loss)
    assert "flow_loss" not in metrics
    assert metrics["text_loss"] > 0


def test_lerobot_eo1_exposes_image_conditioned_text_generation(monkeypatch):
    monkeypatch.setattr(
        "lerobot.policies.eo1.modeling_eo1.Qwen2_5_VLForConditionalGeneration.from_pretrained",
        lambda *args, **kwargs: DummyVLMBackbone(HIDDEN_SIZE),
    )
    policy = EO1Policy(make_eo1_config())
    policy._text_processor = DummyTextProcessor()
    batch = {
        OBS_STATE: torch.zeros(1, STATE_DIM),
        "observation.images.image": torch.zeros(1, 3, 16, 16),
        "task": ["clear the table"],
    }

    output = policy.generate_texts(batch, kind="vqa", user_text="Where is the cup?")

    assert output == ["the cup is left of the plate"]
    # The runtime contract is the single-sample form, decoding with `config.generation`.
    assert policy.generate_text(batch, kind="vqa", user_text="Where is the cup?") == (
        "the cup is left of the plate"
    )
    assert policy.supports_text_generation()


def test_eo1_recipe_processor_builds_sparse_joint_labels():
    pytest.importorskip("datasets", reason="language recipes require lerobot[dataset]")
    config = make_eo1_config()
    config.vlm_base = "Qwen/Qwen2.5-VL-3B-Instruct"
    config.recipe_path = "recipes/subtask_joint.yaml"
    preprocessor, _ = make_eo1_pre_post_processors(
        config,
        dataset_stats={
            OBS_STATE: {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)},
            ACTION: {"mean": torch.zeros(ACTION_DIM), "std": torch.ones(ACTION_DIM)},
        },
    )
    batch = {
        OBS_STATE: torch.zeros(1, STATE_DIM),
        ACTION: torch.zeros(1, CHUNK_SIZE, ACTION_DIM),
        "observation.images.image": torch.zeros(1, 3, 56, 56),
        "task": ["clear the table"],
        "timestamp": torch.tensor([0.0]),
        "index": torch.tensor([7]),
        "language_persistent": [
            [
                {
                    "role": "assistant",
                    "content": "pick up the red block",
                    "style": "subtask",
                    "timestamp": 0.0,
                    "camera": None,
                    "tool_calls": None,
                }
            ]
        ],
        "language_events": [[]],
    }

    processed = preprocessor(batch)

    labels = processed["text_labels"]
    assert labels.shape == processed["input_ids"].shape
    assert (labels != -100).any()
    action_token_id = processed["action_token_id"]
    assert (processed["input_ids"] == action_token_id).sum() == CHUNK_SIZE
    assert not (labels == action_token_id).any()

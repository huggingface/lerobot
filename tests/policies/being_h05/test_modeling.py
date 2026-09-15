#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3MLP

from lerobot.lerobot_types import TransitionKey
from lerobot.policies.being_h05.configuration_being_h05 import BeingH05Config
from lerobot.policies.being_h05.modeling_being_h05 import (
    ActionEncoder,
    BeingH05Policy,
    BeingH05Qwen3ForCausalLM,
    MPGEnhancement,
    _selective_text_cross_entropy,
)
from lerobot.policies.being_h05.processor_being_h05 import (
    BEING_H05_MESSAGE_TOKEN_IDS,
    BeingH05TokenizerStep,
)


def test_action_encoder_batched_and_packed_paths_match():
    encoder = ActionEncoder(action_dim=6, hidden_size=16)
    actions = torch.randn(2, 4, 6)
    timesteps = torch.tensor([2, 7])

    batched = encoder(actions, timesteps)
    packed = encoder(actions.flatten(0, 1), timesteps[:, None].expand(-1, 4).flatten())

    torch.testing.assert_close(batched.flatten(0, 1), packed)


def test_mot_reuses_transformers_qwen3_primitives_and_checkpoint_names():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config.expert_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config.qk_norm = True
    model = BeingH05Qwen3ForCausalLM(config)
    layer = model.model.layers[0]

    assert isinstance(layer.self_attn, Qwen3Attention)
    assert isinstance(layer.mlp, Qwen3MLP)
    assert isinstance(layer.mlp_mot_gen, Qwen3MLP)
    state_dict = model.state_dict()
    assert "model.layers.0.self_attn.q_proj.weight" in state_dict
    assert "model.layers.0.self_attn.q_proj_mot_gen.weight" in state_dict
    assert "model.layers.0.mlp_mot_gen.gate_proj.weight" in state_dict


def test_understanding_kv_cache_matches_full_decode():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config.expert_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config._attn_implementation = "sdpa"
    model = BeingH05Qwen3ForCausalLM(config).eval()
    inputs = torch.randn(1, 6, config.hidden_size)

    full_logits, _ = model.forward_understanding(inputs)
    _, cache = model.forward_understanding(inputs[:, :-1], use_cache=True)
    cached_logits, _ = model.forward_understanding(
        inputs[:, -1:],
        past_key_values=cache,
        use_cache=True,
    )

    torch.testing.assert_close(cached_logits[:, -1], full_logits[:, -1], rtol=1e-5, atol=1e-5)


def test_released_zero_strength_mpg_is_noop():
    module = MPGEnhancement(
        obs_feature_dim=16,
        action_feature_dim=8,
        embedding_dim=16,
        num_projections=4,
        lambda_strength=0.0,
        use_stop_gradient=True,
        gate_temperature=2.0,
    )
    observations = torch.randn(1, 5, 16)
    actions = torch.randn(1, 4, 8)

    assert module(observations, actions) is observations


def test_scheduler_preset_supplies_peak_and_decay_lr():
    config = BeingH05Config(device="cpu")

    preset = config.get_scheduler_preset()

    assert preset.peak_lr == config.optimizer_lr
    assert preset.decay_lr == config.scheduler_decay_lr


class _FakeTokenizer:
    def __init__(self):
        self.ids: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        if text not in self.ids:
            self.ids[text] = len(self.ids) + 20
        return [self.ids[text]]


class _PackingModel(nn.Module):
    num_image_token = 2
    system_message = "system prompt"


def _packing_policy() -> BeingH05Policy:
    policy = object.__new__(BeingH05Policy)
    nn.Module.__init__(policy)
    policy.config = BeingH05Config(
        device="cpu",
        chunk_size=2,
        n_action_steps=1,
        author_config={"action_chunk_length": 2},
    )
    policy.tokenizer = _FakeTokenizer()
    policy.model = _PackingModel()
    policy._bos = 1
    policy._eos = 2
    policy._newline = 3
    policy._image_start = 4
    policy._image_end = 5
    policy._state_start = 6
    policy._state_end = 7
    return policy


def _recipe_batch(predict_actions: bool) -> dict:
    batch = {
        "being_h05.state": torch.zeros(1, 200),
        "being_h05.pixel_values": torch.zeros(1, 1, 3, 224, 224),
        "being_h05.image_valid": torch.ones(1, 1, dtype=torch.bool),
        "being_h05_messages": [
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        ],
        "being_h05_target_message_indices": [[1]],
        "being_h05_predict_actions": torch.tensor([predict_actions]),
        "action": torch.zeros(1, 2, 200),
        "being_h05.action_valid": torch.ones(1, 2, 200, dtype=torch.bool),
    }
    step = BeingH05TokenizerStep(tokenizer_name="unused", system_message="system prompt")
    step._tokenizer = _FakeTokenizer()
    tokenized = step(
        {
            TransitionKey.COMPLEMENTARY_DATA: {
                "being_h05_messages": batch["being_h05_messages"],
            }
        }
    )[TransitionKey.COMPLEMENTARY_DATA]
    batch.update(tokenized)
    return batch


def test_recipe_packing_masks_headers_and_supervises_assistant_content_and_eos():
    policy = _packing_policy()

    packed = policy._pack_model_inputs(_recipe_batch(predict_actions=True), training=True)

    supervised = packed["packed_text_labels"][packed["packed_text_labels"].ne(-100)]
    answer_id = _recipe_batch(True)[BEING_H05_MESSAGE_TOKEN_IDS][0][1]["content_ids"][0]
    assert supervised.tolist() == [answer_id, policy._eos]
    assert packed["padded_action_mask"].all()


def test_high_level_recipe_sample_masks_the_action_loss():
    policy = _packing_policy()

    packed = policy._pack_model_inputs(_recipe_batch(predict_actions=False), training=True)

    assert not packed["padded_action_mask"].any()
    assert packed["packed_text_labels"].ne(-100).sum().item() == 2


def test_selective_text_loss_projects_only_supervised_positions_and_backpropagates():
    config = Qwen3Config(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
    )
    config.expert_config = Qwen3Config(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
    )
    model = BeingH05Qwen3ForCausalLM(config)
    hidden = torch.randn(4, config.hidden_size, requires_grad=True)
    labels = torch.tensor([-100, 2, -100, 5])

    loss = _selective_text_cross_entropy(model, hidden, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert hidden.grad is not None
    assert not hidden.grad[[0, 2]].any()
    assert hidden.grad[[1, 3]].abs().sum() > 0

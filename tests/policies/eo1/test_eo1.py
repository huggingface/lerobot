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
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import RenderRuntimeMessagesStep, RenderTrainingMessagesStep
from lerobot.utils.constants import ACTION, OBS_STATE, QUERY_KIND, QUERY_TEXT

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


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    @staticmethod
    def add_tokens(*args, **kwargs):
        return 0

    @staticmethod
    def convert_tokens_to_ids(token):
        return {"<|state_pad|>": STATE_TOKEN_ID, "<|action_pad|>": ACTION_TOKEN_ID}.get(token, 3)


class DummyTextProcessor:
    tokenizer = DummyTokenizer()

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
    batch = make_policy_batch(include_action=False)

    assert EO1Policy.generate_text is not PreTrainedPolicy.generate_text
    assert EO1Policy.supports_text_generation is not PreTrainedPolicy.supports_text_generation
    assert policy.generate_text(batch) == "the cup is left of the plate"
    assert policy.supports_text_generation()
    assert not hasattr(EO1Policy, "generate_texts")


def test_eo1_default_processor_owns_runtime_prompt_rendering(monkeypatch):
    monkeypatch.setattr(
        "lerobot.policies.eo1.processor_eo1.Qwen2_5_VLProcessor.from_pretrained",
        lambda *args, **kwargs: DummyTextProcessor(),
    )
    config = make_eo1_config()
    preprocessor, _ = make_eo1_pre_post_processors(
        config,
        dataset_stats={
            OBS_STATE: {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)},
            ACTION: {"mean": torch.zeros(ACTION_DIM), "std": torch.ones(ACTION_DIM)},
        },
    )

    processed = preprocessor(
        {
            OBS_STATE: torch.zeros(STATE_DIM),
            "observation.images.image": torch.zeros(3, 16, 16),
            "task": "current subtask",
            QUERY_KIND: "next_subtask",
            QUERY_TEXT: "clear the table",
        }
    )

    assert isinstance(preprocessor.steps[0], RenderRuntimeMessagesStep)
    assert isinstance(preprocessor.steps[1], RenderTrainingMessagesStep)
    assert processed["input_ids"].shape == (1, 2)
    assert "messages" not in processed
    assert QUERY_KIND not in processed
    assert QUERY_TEXT not in processed
    assert not hasattr(EO1Policy, "prepare_runtime_action_batch")


def test_eo1_recipe_processor_builds_sparse_joint_labels():
    pytest.importorskip("datasets", reason="language recipes require lerobot[dataset]")
    config = make_eo1_config()
    config.vlm_base = "Qwen/Qwen2.5-VL-3B-Instruct"
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


@pytest.mark.parametrize("recipe_enabled", [False, True])
def test_recipe_config_round_trip_and_optional_rendering(recipe_enabled):
    import draccus

    config = make_eo1_config()
    if not recipe_enabled:
        config.recipe = None
    restored = draccus.decode(EO1Config, draccus.encode(config))
    assert restored.recipe == config.recipe
    runtime = RenderRuntimeMessagesStep(restored.recipe)
    training = RenderTrainingMessagesStep(restored.recipe)
    if recipe_enabled:
        assert runtime.recipe is not None
        assert training.recipe == runtime.recipe
    else:
        from lerobot.processor.converters import create_transition

        transition = create_transition(action=torch.ones(1), complementary_data={"task": "tidy"})
        assert training(transition) is transition


@pytest.mark.parametrize("system_prompt", [None, "Use concise robot instructions."])
@pytest.mark.parametrize("include_user", [False, True])
def test_model_messages_preserve_system_prompt_and_target_alignment(system_prompt, include_user):
    from lerobot.datasets.recipe import MessageTurn, TrainingRecipe, render_message_turns
    from lerobot.lerobot_types import TransitionKey
    from lerobot.policies.eo1.processor_eo1 import EO1PrepareModelMessagesStep
    from lerobot.processor.converters import create_transition

    turns = []
    if system_prompt is not None:
        turns.append(MessageTurn(role="system", content=system_prompt, stream="high_level"))
    if include_user:
        turns.append(MessageTurn(role="user", content="${task}", stream="high_level"))
    turns.append(MessageTurn(role="assistant", content="${subtask}", stream="high_level", target=True))
    recipe = TrainingRecipe(messages=turns)
    rendered = render_message_turns(recipe.messages, {"task": "tidy", "subtask": "pick up the cup"})
    data = {key: [value] for key, value in rendered.items()}
    data["task"] = ["tidy"]
    config = make_eo1_config()
    transition = create_transition(
        observation={
            OBS_STATE: torch.zeros(1, STATE_DIM),
            "observation.images.image": torch.zeros(1, 3, 8, 8),
        },
        complementary_data=data,
    )
    step = EO1PrepareModelMessagesStep(config.input_features, CHUNK_SIZE)
    output = step(transition)[TransitionKey.COMPLEMENTARY_DATA]
    messages = output["messages"][0]
    expected_roles = ["user", "assistant"] if system_prompt is None else ["system", "user", "assistant"]
    assert [message["role"] for message in messages] == expected_roles
    if system_prompt is not None:
        assert messages[0]["content"] == [{"type": "text", "text": system_prompt}]
    assert output["target_message_indices"] == [[1 if system_prompt is None else 2]]
    assert messages[-1]["content"] == [{"type": "text", "text": "pick up the cup"}]
    assert any(block["type"] == "image" for block in messages[-2]["content"])


@pytest.mark.parametrize("query_kind", ["vqa", "next_subtask", None])
def test_model_messages_use_recipe_system_prompt_and_only_fall_back_for_action_inputs(query_kind):
    from lerobot.lerobot_types import TransitionKey
    from lerobot.policies.eo1.processor_eo1 import EO1PrepareModelMessagesStep
    from lerobot.processor.converters import create_transition

    config = make_eo1_config()
    assert config.recipe["messages"][0] == {
        "role": "system",
        "content": "You are a helpful physical assistant.",
        "stream": "low_level",
    }
    data = {"task": ["tidy"]}
    if query_kind:
        data.update({QUERY_KIND: query_kind, QUERY_TEXT: "tidy"})
    transition = create_transition(
        observation={
            OBS_STATE: torch.zeros(1, STATE_DIM),
            "observation.images.image": torch.zeros(1, 3, 8, 8),
        },
        complementary_data=data,
    )
    transition = RenderRuntimeMessagesStep(config.recipe)(transition)
    output = EO1PrepareModelMessagesStep(config.input_features, CHUNK_SIZE)(transition)
    messages = output[TransitionKey.COMPLEMENTARY_DATA]["messages"][0]
    system_messages = [message for message in messages if message["role"] == "system"]
    if query_kind == "vqa":
        assert system_messages == []
    else:
        assert system_messages == [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful physical assistant."}]}
        ]


def test_saved_message_processor_registry_name_still_loads():
    from lerobot.policies.eo1.processor_eo1 import EO1PrepareModelMessagesStep
    from lerobot.processor import PolicyProcessorPipeline

    config = make_eo1_config()
    step = EO1PrepareModelMessagesStep(config.input_features, CHUNK_SIZE)
    pipeline = PolicyProcessorPipeline(steps=[step])
    saved_config = pipeline.get_config()
    assert saved_config["steps"][0]["registry_name"] == "eo1_conversation_template_processor"
    restored = PolicyProcessorPipeline.from_config(saved_config)
    assert isinstance(restored.steps[0], EO1PrepareModelMessagesStep)


@pytest.mark.parametrize("query_kind", ["next_subtask", "vqa"])
@pytest.mark.parametrize("include_user", [False, True])
def test_training_and_generation_share_state_conditioned_prefix(query_kind, include_user):
    from lerobot.datasets.recipe import MessageTurn, TrainingRecipe, render_message_turns
    from lerobot.lerobot_types import TransitionKey
    from lerobot.policies.eo1.processor_eo1 import EO1PrepareModelMessagesStep
    from lerobot.processor.converters import create_transition

    turns = [MessageTurn(role="system", content="Robot assistant", stream="high_level")]
    if include_user:
        turns.append(MessageTurn(role="user", content="${task}", stream="high_level"))
    turns.append(MessageTurn(role="assistant", content="${subtask}", stream="high_level", target=True))
    recipe = TrainingRecipe(messages=turns)
    rendered = render_message_turns(recipe.messages, {"task": "tidy", "subtask": "pick up cup"})
    config = make_eo1_config()
    formatter = EO1PrepareModelMessagesStep(config.input_features, CHUNK_SIZE)
    observation = {
        OBS_STATE: torch.zeros(1, STATE_DIM),
        "observation.images.image": torch.zeros(1, 3, 8, 8),
    }
    training = formatter(
        create_transition(
            observation=observation,
            complementary_data={"task": ["tidy"], **{key: [value] for key, value in rendered.items()}},
        )
    )[TransitionKey.COMPLEMENTARY_DATA]
    request = create_transition(
        observation=observation,
        complementary_data={"task": ["tidy"], QUERY_KIND: query_kind, QUERY_TEXT: "tidy"},
    )
    runtime = formatter(RenderRuntimeMessagesStep(recipe)(request))[TransitionKey.COMPLEMENTARY_DATA]
    runtime_user = next(message for message in runtime["messages"][0] if message["role"] == "user")
    training_user = next(message for message in training["messages"][0] if message["role"] == "user")
    state_text = "<|state_start|><|state_pad|><|state_end|>"
    for user in (training_user, runtime_user):
        assert sum(block.get("text") == state_text for block in user["content"]) == 1
    if query_kind == "next_subtask" or include_user:
        # Compare multimodal prefixes without asking Python to compare image tensors as booleans.
        assert len(training_user["content"]) == len(runtime_user["content"])
        for training_block, runtime_block in zip(
            training_user["content"], runtime_user["content"], strict=True
        ):
            assert training_block["type"] == runtime_block["type"]
            if training_block["type"] == "image":
                torch.testing.assert_close(training_block["image"], runtime_block["image"])
            else:
                assert training_block == runtime_block
    assert runtime["target_message_indices"] == [[]]


def test_real_qwen_generation_uses_projected_state_and_cached_image_prefix(monkeypatch):
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    qwen_config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0, "mrope_section": [1, 1, 2]},
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "window_size": 4,
            "out_hidden_size": 32,
            "fullatt_block_indexes": [0],
        },
        image_token_id=40,
        video_token_id=43,
        vision_start_token_id=41,
        vision_end_token_id=42,
    )
    backbone = Qwen2_5_VLForConditionalGeneration(qwen_config).eval()
    monkeypatch.setattr(
        "lerobot.policies.eo1.modeling_eo1.Qwen2_5_VLForConditionalGeneration.from_pretrained",
        lambda *args, **kwargs: backbone,
    )
    policy = EO1Policy(make_eo1_config())
    with torch.no_grad():
        policy.model.state_proj.weight.fill_(0.25)
        policy.model.state_proj.bias.zero_()

    decoded = []
    generated = []
    prefill_embeddings = []
    cache_lengths = []
    image_calls = []

    def decode(token_ids, **kwargs):
        decoded.append(token_ids.clone())
        return ["generated response"]

    policy._text_processor = SimpleNamespace(tokenizer=DummyTokenizer(), batch_decode=decode)
    real_generate = backbone.generate

    def generate_three_tokens(**kwargs):
        kwargs.update(max_new_tokens=3, min_new_tokens=3, eos_token_id=None)
        result = real_generate(**kwargs)
        generated.append(result.clone())
        return result

    monkeypatch.setattr(backbone, "generate", generate_three_tokens)

    def capture_prefill(module, args, kwargs):
        cache = kwargs.get("past_key_values")
        cache_lengths.append(0 if cache is None else cache.get_seq_length())
        if kwargs.get("inputs_embeds") is not None:
            prefill_embeddings.append(kwargs["inputs_embeds"].detach().clone())

    prefill_hook = backbone.model.register_forward_pre_hook(capture_prefill, with_kwargs=True)
    image_hook = backbone.model.visual.register_forward_hook(lambda *args: image_calls.append(True))
    input_ids = torch.tensor([[0, 1, 41, 40, 42, STATE_TOKEN_ID, 10]])
    batch = {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(0).long(),
        "pixel_values": torch.zeros(4, 12),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "mm_token_type_ids": torch.tensor([[0, 0, 0, 1, 0, 0, 0]]),
        "state_token_id": STATE_TOKEN_ID,
        "action_token_id": ACTION_TOKEN_ID,
    }
    try:
        for state in (torch.zeros(1, STATE_DIM - 1), torch.ones(1, STATE_DIM - 1)):
            assert policy.generate_text({**batch, OBS_STATE: state}) == "generated response"
    finally:
        prefill_hook.remove()
        image_hook.remove()

    assert len(prefill_embeddings) == 2
    torch.testing.assert_close(prefill_embeddings[0][:, 5], torch.zeros(1, 32))
    # Three state coordinates equal one; the fourth is zero-padded before projection.
    torch.testing.assert_close(prefill_embeddings[1][:, 5], torch.full((1, 32), 0.75))
    non_state = input_ids[0] != STATE_TOKEN_ID
    torch.testing.assert_close(prefill_embeddings[0][:, non_state], prefill_embeddings[1][:, non_state])
    assert cache_lengths == [0, 7, 8, 0, 7, 8]
    assert len(image_calls) == 2  # Encode the image once per request, not per generated token.
    for full_output, decoded_tokens in zip(generated, decoded, strict=True):
        torch.testing.assert_close(full_output[:, :7], input_ids)
        torch.testing.assert_close(decoded_tokens, full_output[:, 7:])
        assert decoded_tokens.shape == (1, 3)

#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Test script to verify Wall-X policy integration with LeRobot"""

from types import SimpleNamespace

import pytest
import torch

# Skip if required dependencies are not available
pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("torchdiffeq")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.factory import make_policy_config, make_pre_post_processors  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.policies.wall_x import (
    WallXConfig,  # noqa: E402
)
from lerobot.policies.wall_x.modeling_wall_x import Qwen2_5_VLMoEForAction, WallXPolicy  # noqa: E402
from lerobot.policies.wall_x.processor_wall_x import make_wall_x_pre_post_processors  # noqa: E402
from lerobot.policies.wall_x.qwen_model import Qwen2_5_VLMoEModel, Qwen2_5_VLTextConfig  # noqa: E402
from lerobot.policies.wall_x.utils import _extract_text_target_spans  # noqa: E402
from lerobot.processor import RenderMessagesStep  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from tests.utils import require_cuda, require_hf_token  # noqa: E402


def test_moe_model_captures_requested_hidden_states_and_attentions():
    hidden_size = 16
    expert_config = {
        "hidden_size": hidden_size,
        "intermediate_size": 32,
        "hidden_act": "silu",
    }
    config = Qwen2_5_VLTextConfig(
        vocab_size=32,
        hidden_size=hidden_size,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
            "mrope_section": [1, 1, 0],
        },
        num_experts=2,
        experts=[expert_config, expert_config],
        dim_inputs=(hidden_size, hidden_size),
        mlp_moe=True,
    )
    config._attn_implementation = "eager"
    model = Qwen2_5_VLMoEModel(config)
    input_ids = torch.tensor([[1, 2, 3]])

    output = model(
        input_ids=input_ids,
        moe_token_types=torch.zeros_like(input_ids),
        output_hidden_states=True,
        output_attentions=True,
    )

    assert len(output.hidden_states) == config.num_hidden_layers + 1
    assert len(output.attentions) == config.num_hidden_layers


def _make_unloaded_policy(**config_values):
    policy = WallXPolicy.__new__(WallXPolicy)
    torch.nn.Module.__init__(policy)
    config_values.setdefault("text_temperature", 0.0)
    config_values.setdefault("text_top_p", 1.0)
    policy.config = SimpleNamespace(chunk_size=3, **config_values)
    return policy


def test_recipe_prompt_targets_only_selected_assistant_and_keeps_action_supervision():
    policy = _make_unloaded_policy()
    prompt, predicts_action = policy._format_recipe_text(
        [
            {"role": "user", "content": "What should the robot do?"},
            {"role": "assistant", "content": "Reach for the cup."},
        ],
        ["high_level", "low_level"],
        [1],
        "pick up the cup",
        ["observation.images.face_view"],
    )

    clean_prompt, spans = _extract_text_target_spans(prompt)
    assert predicts_action
    assert len(spans) == 1
    assert clean_prompt[slice(*spans[0])] == "Reach for the cup.<|im_end|>"
    assert clean_prompt.count("<|action|>") == 3
    assert "Proprioception: <|propri|>" in clean_prompt


def test_policy_combines_text_and_flow_losses_with_configured_weights(monkeypatch):
    policy = _make_unloaded_policy(flow_loss_weight=2.0, text_loss_weight=0.25)
    policy.model = SimpleNamespace(
        __call__=None,
    )
    outputs = SimpleNamespace(
        loss=torch.tensor(99.0),
        flow_loss=torch.tensor(3.0),
        cross_entropy_loss=torch.tensor(4.0),
        channel_loss_dict=None,
    )
    policy.model = lambda **kwargs: outputs
    monkeypatch.setattr(policy, "preprocess_inputs", lambda batch, compute_position_ids: batch)

    loss, metrics = policy.forward(
        {"input_ids": torch.ones(1, 1, dtype=torch.long), "messages": [[{"role": "user"}]]}
    )

    assert loss.item() == 7.0
    assert metrics["flow_loss"].item() == 3.0
    assert metrics["cross_entropy_loss"].item() == 4.0


def test_policy_preserves_original_action_only_loss(monkeypatch):
    policy = _make_unloaded_policy(flow_loss_weight=2.0, text_loss_weight=0.25)
    outputs = SimpleNamespace(
        loss=torch.tensor(9.0),
        flow_loss=torch.tensor(3.0),
        cross_entropy_loss=torch.tensor(4.0),
        channel_loss_dict=None,
    )
    policy.model = lambda **kwargs: outputs
    monkeypatch.setattr(policy, "preprocess_inputs", lambda batch, compute_position_ids: batch)

    loss, _ = policy.forward({"input_ids": torch.ones(1, 1, dtype=torch.long)})

    assert loss.item() == 9.0


def test_generation_preparation_synthesizes_cache_positions():
    input_ids = torch.tensor([[1, 2, 3]])

    prepared = Qwen2_5_VLMoEForAction.prepare_inputs_for_generation(
        SimpleNamespace(),
        input_ids,
        attention_mask=torch.ones_like(input_ids),
    )

    assert torch.equal(prepared["cache_position"], torch.arange(3))


def test_policy_exposes_text_generation(monkeypatch):
    assert WallXPolicy.generate_text is not PreTrainedPolicy.generate_text
    assert WallXPolicy.supports_text_generation is not PreTrainedPolicy.supports_text_generation
    assert not hasattr(WallXPolicy, "generate_texts")

    class Inputs(dict):
        __getattr__ = dict.__getitem__

    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 0

        @staticmethod
        def batch_decode(token_ids, **kwargs):
            del kwargs
            assert torch.equal(token_ids, torch.tensor([[7, 8]]))
            return ["move toward the cup"]

    class Model:
        processor = SimpleNamespace(tokenizer=Tokenizer())

        @staticmethod
        def generate(input_ids, **kwargs):
            del kwargs
            return torch.cat([input_ids, torch.tensor([[7, 8]])], dim=1)

    policy = _make_unloaded_policy()
    policy.model = Model()
    inputs = Inputs(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3))
    received_batches = []

    def build_text_inputs(batch):
        received_batches.append(batch)
        return inputs

    monkeypatch.setattr(policy, "_build_text_inputs", build_text_inputs)

    batch = {
        "observation.state": torch.zeros(1, 7),
        "task": ["pick up the cup"],
        "messages": [[{"role": "user", "content": "pick up the cup"}]],
    }

    assert policy.generate_text(batch) == "move toward the cup"
    assert received_batches == [batch]
    assert policy.supports_text_generation()
    assert policy._format_generation_messages(
        [{"role": "user", "content": "clear the table\nPredict the next action in language.\n"}],
        ["observation.images.face_view"],
    ).endswith(
        "Instruction: clear the table\nPredict the next action in language.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def test_wall_x_runtime_query_is_rendered_by_the_default_input_pipeline():
    config = WallXConfig(device="cpu")
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.images.face_view": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    stats = {
        "observation.state": {"mean": torch.zeros(7), "std": torch.ones(7)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
    }

    preprocessor, _ = make_pre_post_processors(config, dataset_stats=stats)
    batch = preprocessor(
        {
            "observation.state": torch.zeros(7),
            "observation.images.face_view": torch.zeros(3, 8, 8),
            "task": "current subtask",
            "query_kind": "next_subtask",
            "text": "clear the table",
        }
    )

    assert isinstance(preprocessor.steps[0], RenderMessagesStep)
    assert preprocessor.steps[0].render_training is False
    assert batch["messages"] == [
        [{"role": "user", "content": "clear the table\nPredict the next action in language.\n"}]
    ]
    assert "query_kind" not in batch
    assert "text" not in batch


def test_wall_x_loads_an_explicit_external_recipe(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text("messages:\n  - {role: user, content: '${task}', stream: low_level}\n")
    config = WallXConfig(device="cpu", recipe_path=str(path))

    assert config.recipe is not None
    assert config.recipe.messages is not None


@require_cuda
@require_hf_token
def test_policy_instantiation():
    # Create config
    set_seed(42)
    config = WallXConfig(device="cuda")

    # Set up input_features and output_features in the config
    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(7,),
        ),
        "observation.images.face_view": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
    }

    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(7,),
        ),
    }

    # Create dummy dataset stats
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
        },
        "observation.images.face_view": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
        },
    }

    # Instantiate policy
    policy = WallXPolicy(config)
    preprocessor, postprocessor = make_wall_x_pre_post_processors(config=config, dataset_stats=dataset_stats)
    # Test forward pass with dummy data
    batch_size = 1
    device = config.device
    batch = {
        "observation.state": torch.randn(batch_size, 7, dtype=torch.float32, device=device),
        "action": torch.randn(batch_size, config.chunk_size, 7, dtype=torch.float32, device=device),
        "observation.images.face_view": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),  # Use rand for [0,1] range
        "task": ["Pick up the object"] * batch_size,
    }
    batch = preprocessor(batch)
    try:
        loss, loss_dict = policy.forward(batch)
        print(f"Forward pass successful. Loss: {loss_dict['loss']:.4f}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise

    # Test inference
    batch = {
        "observation.state": torch.randn(batch_size, 7, dtype=torch.float32, device=device),
        "observation.images.face_view": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),  # Use rand for [0,1] range
        "task": ["Pick up the object"] * batch_size,
    }
    batch = preprocessor(batch)
    try:
        with torch.no_grad():
            action = policy.select_action(batch)
            action = postprocessor(action)
            print(f"Action: {action}")
        print(f"Action prediction successful. Action shape: {action.shape}")
    except Exception as e:
        print(f"Action prediction failed: {e}")
        raise


@require_cuda
@require_hf_token
def test_config_creation():
    """Test policy config creation through factory."""
    try:
        config = make_policy_config(
            policy_type="wall_x",
        )
        print("Config created successfully through factory")
        print(f"  Config type: {type(config).__name__}")
    except Exception as e:
        print(f"Config creation failed: {e}")
        raise


def test_subtask_prompt_is_token_exact_with_the_trained_template():
    """WALL-OSS declares its mode in the user turn, so the template must match upstream."""
    from lerobot.policies.wall_x.utils import get_wallx_normal_text

    img_keys = ["observation.images.face_view"]
    upstream, generated_subtask = get_wallx_normal_text(
        {"instruction": "clear the table", "subtask_generation": "pick the cup"},
        action_chunk_size=1,
        frame_idx=0,
        priority_order=None,
        img_keys=list(img_keys),
        generate_subtask_ratio=1.0,
    )
    assert generated_subtask

    ours = _make_unloaded_policy()._format_generation_messages(
        [{"role": "user", "content": "clear the table\nPredict the next action in language.\n"}],
        list(img_keys),
    )

    # Compare the user turn only: upstream appends its own assistant target, ours ends
    # at the generation prompt.
    assert ours.split("<|im_start|>assistant")[0] == upstream.split("<|im_start|>assistant")[0]

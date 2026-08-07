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

from lerobot.configs import TextKind  # noqa: E402
from lerobot.policies.factory import make_policy_config  # noqa: E402
from lerobot.policies.wall_x import (
    WallXConfig,  # noqa: E402
)
from lerobot.policies.wall_x.modeling_wall_x import (  # noqa: E402
    Qwen2_5_VLMoEForAction,
    WallXPolicy,
)
from lerobot.policies.wall_x.processor_wall_x import make_wall_x_pre_post_processors  # noqa: E402
from lerobot.policies.wall_x.qwen_model import Qwen2_5_VLMoEModel, Qwen2_5_VLTextConfig  # noqa: E402
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


def _make_unloaded_policy():
    policy = WallXPolicy.__new__(WallXPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        text_temperature=0.0,
        text_top_p=1.0,
    )
    return policy


def test_policy_exposes_grounded_text_generation(monkeypatch):
    class Inputs(dict):
        __getattr__ = dict.__getitem__

    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 0

        @staticmethod
        def batch_decode(token_ids, **kwargs):
            del kwargs
            assert torch.equal(token_ids, torch.tensor([[7, 8]]))
            return ["The mug is beside the bowl."]

    class Model:
        processor = SimpleNamespace(tokenizer=Tokenizer())

        @staticmethod
        def generate(input_ids, **kwargs):
            del kwargs
            return torch.cat([input_ids, torch.tensor([[7, 8]])], dim=1)

    policy = _make_unloaded_policy()
    policy.model = Model()
    inputs = Inputs(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3))
    monkeypatch.setattr(policy, "_build_text_inputs", lambda *args, **kwargs: inputs)

    batch = {"observation.state": torch.zeros(1, 7), "task": "pick up the cup"}
    assert (
        policy.generate_text(batch, kind=TextKind.VQA, user_text="Where is the mug?")
        == "The mug is beside the bowl."
    )
    assert policy.supports_text_generation()

    prompt = policy._format_text_prompt(
        "Where is the mug?",
        TextKind.VQA,
        ["observation.images.face_view"],
    )
    assert "Observation: front view:" in prompt
    assert "Instruction: Where is the mug?" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_text_generation_synthesizes_missing_cache_position():
    class Cache:
        @staticmethod
        def get_seq_length():
            return 3

    pixel_values = torch.ones(1, 3, 4, 4)
    inputs = Qwen2_5_VLMoEForAction.prepare_inputs_for_generation(
        object(),
        torch.tensor([[9]]),
        past_key_values=Cache(),
        pixel_values=pixel_values,
    )

    assert torch.equal(inputs["cache_position"], torch.tensor([3]))
    assert torch.equal(inputs["input_ids"], torch.tensor([[9]]))
    # A continuation token reuses the KV cache, so the image is not encoded again.
    assert inputs["pixel_values"] is None


@require_cuda
@require_hf_token
def test_policy_instantiation():
    # Create config
    set_seed(42)
    config = WallXConfig(device="cuda")

    # Set up input_features and output_features in the config
    from lerobot.configs.types import FeatureType, PolicyFeature

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

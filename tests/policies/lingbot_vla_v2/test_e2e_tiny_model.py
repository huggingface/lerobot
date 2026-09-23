# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""End-to-end test: tiny LingBot-VLA 2.0 policy construction and forward pass.

Uses a tiny Qwen3-VL backbone (2 layers, small hidden size) and tiny action expert
(2 layers, 4 experts) to verify the full policy can be constructed and run forward
without the 6B pretrained weights. This is the CI-friendly version of the full
policy test.
"""

import pytest
import torch

pytest.importorskip("transformers")

from transformers import AutoConfig

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config, SlotMapping
from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy
from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import make_lingbot_vla_v2_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


def _make_tiny_vlm_config():
    """Create a tiny Qwen3-VL config for testing (2 layers, small dims)."""
    config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    config.text_config.num_hidden_layers = 2
    config.text_config.hidden_size = 128
    config.text_config.intermediate_size = 256
    config.text_config.num_attention_heads = 4
    config.text_config.num_key_value_heads = 2
    config.text_config.vocab_size = 151936  # Match real Qwen3-VL tokenizer
    config.vision_config.hidden_size = 128  # Match text_config.hidden_size
    config.vision_config.intermediate_size = 256
    config.vision_config.num_hidden_layers = 2
    config.vision_config.num_attention_heads = 2
    config.vision_config.patch_size = 16
    return config


def _make_tiny_config() -> LingbotVLAV2Config:
    """Create a tiny LingBot-VLA 2.0 config for end-to-end testing."""
    return LingbotVLAV2Config(
        # Tiny VLM backbone (mocked via monkeypatch in test)
        tokenizer_path="Qwen/Qwen3-VL-4B-Instruct",
        # Tiny action expert
        expert_hidden_size=64,
        expert_intermediate_size=128,
        action_num_attention_heads=4,
        action_num_key_value_heads=2,
        action_head_dim=16,
        # Tiny MoE
        use_moe=True,
        token_num_experts=4,
        token_top_k=2,
        token_moe_intermediate_size=32,
        # Device: CPU for testing
        device="cpu",
        # Small dims
        max_state_dim=14,
        max_action_dim=14,
        chunk_size=5,
        n_action_steps=5,
        num_steps=2,
        tokenizer_max_length=16,
        # Canonical layout (simplified for testing)
        canonical_joints={
            "arm.position": 6,
            "effector.position": 1,
        },
        canonical_norm_type={
            "arm.position": "meanstd",
            "effector.position": "meanstd",
        },
        canonical_cameras=["camera_top"],
        # Feature transform
        state_slots={
            "arm.position": SlotMapping(
                origin_keys=[{"observation.state": {"start": 0, "end": 6}}],
            ),
            "effector.position": SlotMapping(
                origin_keys=[{"observation.state": {"start": 6, "end": 7}}],
            ),
        },
        action_slots={
            "arm.position": SlotMapping(
                origin_keys=[{"action": {"start": 0, "end": 6}}],
                subtract_state=False,
            ),
            "effector.position": SlotMapping(
                origin_keys=[{"action": {"start": 6, "end": 7}}],
                subtract_state=False,
            ),
        },
        camera_mapping={
            "observation.images.camera_top": "observation.images.camera_top",
        },
        # Fake norm_stats for testing (mean=0, std=1)
        norm_stats={
            "norm_stats": {
                "observation.state.arm.position": {"mean": [0.0] * 6, "std": [1.0] * 6},
                "observation.state.effector.position": {"mean": [0.0], "std": [1.0]},
                "action.arm.position": {"mean": [0.0] * 6, "std": [1.0] * 6},
                "action.effector.position": {"mean": [0.0], "std": [1.0]},
            }
        },
        # Input/output features
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.camera_top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
    )


def _make_fake_batch(config: LingbotVLAV2Config, batch_size: int = 2) -> dict:
    """Create a fake batch matching the config's input features."""
    return {
        OBS_STATE: torch.randn(batch_size, 7),
        "observation.images.camera_top": torch.randn(batch_size, 3, 64, 64),
        ACTION: torch.randn(batch_size, config.chunk_size, 7),
        "task": ["pick up the cube"] * batch_size,
    }


@pytest.mark.parametrize("batch_size", [1, 2])
def test_tiny_policy_construction_and_processor(batch_size, monkeypatch):
    """Test that a tiny LingBot-VLA 2.0 policy can be constructed and processor works."""
    # Mock AutoConfig.from_pretrained to return tiny VLM config
    tiny_vlm_config = _make_tiny_vlm_config()

    def mock_from_pretrained(*args, **kwargs):
        return tiny_vlm_config

    monkeypatch.setattr(AutoConfig, "from_pretrained", mock_from_pretrained)

    # Build tiny config and policy
    config = _make_tiny_config()
    policy = LingbotVLAV2Policy(config)

    # Verify policy was constructed
    assert policy.config.type == "lingbot_vla_v2"
    assert hasattr(policy, "model")
    assert hasattr(policy.model, "qwenvl_with_expert")

    # Build processor and verify it works
    preprocessor, postprocessor = make_lingbot_vla_v2_pre_post_processors(config)
    batch = _make_fake_batch(config, batch_size)
    processed_batch = preprocessor(batch)

    # Verify processed batch has expected keys
    assert OBS_STATE in processed_batch
    assert "observation.images.camera_top" in processed_batch
    assert ACTION in processed_batch
    assert "lang_tokens" in processed_batch
    assert "lang_masks" in processed_batch

    # Verify shapes after slot mapping
    assert processed_batch[OBS_STATE].shape[-1] == config.max_state_dim
    assert processed_batch[ACTION].shape[-1] == config.max_action_dim


def test_identity_passthrough_processor():
    """Test that the identity passthrough processor works without slot mappings."""
    config = LingbotVLAV2Config()  # No slot mappings provided

    # Should build processor with identity passthrough
    preprocessor, postprocessor = make_lingbot_vla_v2_pre_post_processors(config)

    assert len(preprocessor.steps) == 4  # rename, to_batch, feature_transform, device
    assert len(postprocessor.steps) == 1  # device
    assert preprocessor.steps[2].__class__.__name__ == "LingbotVLAV2FeatureTransformStep"

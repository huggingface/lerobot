import pytest
import torch

pytest.importorskip("transformers")

from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3MLP

from lerobot.policies.being_h05.configuration_being_h05 import BeingH05Config
from lerobot.policies.being_h05.modeling_being_h05 import (
    ActionEncoder,
    BeingH05Qwen3ForCausalLM,
    MPGEnhancement,
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

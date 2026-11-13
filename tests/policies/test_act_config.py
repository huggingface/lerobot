"""Tests for ACTConfig.validate_features input acceptance."""

import pytest
import torch

from lerobot.configs.policies import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def _make_config(input_features):
    return ACTConfig(
        input_features=input_features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )


def test_validate_features_accepts_state_only():
    """forward() reads OBS_STATE, so a state-only input set must validate."""
    cfg = _make_config({"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))})
    cfg.validate_features()


def test_validate_features_accepts_env_state():
    cfg = _make_config({"observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(6,))})
    cfg.validate_features()


def test_validate_features_accepts_image():
    cfg = _make_config({"observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96))})
    cfg.validate_features()


def test_validate_features_rejects_empty():
    cfg = _make_config({})
    with pytest.raises(ValueError, match="at least one image"):
        cfg.validate_features()


def _make_tiny_state_only_policy(use_vae: bool = True) -> ACTPolicy:
    cfg = ACTConfig(
        device="cpu",
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        chunk_size=3,
        n_action_steps=3,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=4,
        use_vae=use_vae,
        pretrained_backbone_weights=None,
    )
    return ACTPolicy(cfg)


def test_state_only_predict_action_chunk():
    """Inference end-to-end with only observation.state in the batch (repro of #4921)."""
    policy = _make_tiny_state_only_policy().eval()
    actions = policy.predict_action_chunk({"observation.state": torch.zeros(2, 6)})
    assert actions.shape[0] == 2
    assert actions.shape[-1] == 2


def test_state_only_select_action():
    policy = _make_tiny_state_only_policy().eval()
    action = policy.select_action({"observation.state": torch.zeros(1, 6)})
    assert action.shape == (1, 2)


@pytest.mark.parametrize("use_vae", [False, True])
def test_state_only_training_forward(use_vae):
    """Training forward/backward with only observation.state (plus actions) in the batch."""
    policy = _make_tiny_state_only_policy(use_vae=use_vae)
    policy.train()
    batch = {
        "observation.state": torch.randn(2, 6),
        "action": torch.randn(2, 3, 2),
        "action_is_pad": torch.zeros(2, 3, dtype=torch.bool),
    }
    loss, loss_dict = policy.forward(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert "l1_loss" in loss_dict
    if use_vae:
        assert "kld_loss" in loss_dict

import pytest

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies import HyVLAConfig
from lerobot.policies.factory import get_policy_class, make_policy_config


def test_hy_vla_factory_registration():
    config = make_policy_config("hy_vla", device="cpu")
    assert isinstance(config, HyVLAConfig)
    assert get_policy_class("hy_vla").__name__ == "HyVLAPolicy"


def test_umi_contract():
    config = HyVLAConfig(device="cpu")
    assert config.chunk_size == 50
    assert config.physical_action_horizon == 50
    assert config.num_steps == 10
    assert config.max_state_dim == config.max_action_dim == 32


def test_robotwin_rel_abs_contract():
    config = HyVLAConfig(
        device="cpu",
        chunk_size=40,
        n_action_steps=40,
        action_representation="relative_absolute",
        action_decode_mode="blend",
        embodiment="robotwin_dual_arm",
        native_quaternion_order="wxyz",
        use_video_encoder=True,
        img_history_size=6,
        img_history_interval=5,
        execution_horizon=7,
    )
    assert config.physical_action_horizon == 20
    assert config.execution_horizon == 7
    assert config.action_delta_indices == list(range(20))


def test_unnamed_mobile_action_is_rejected():
    with pytest.raises(ValueError, match="umi_dual_arm action"):
        HyVLAConfig(
            device="cpu",
            output_features={"action": PolicyFeature(FeatureType.ACTION, (12,))},
        )

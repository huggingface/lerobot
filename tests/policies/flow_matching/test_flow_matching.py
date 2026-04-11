import torch

from lerobot.policies.factory import make_policy_config
from lerobot.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.policies.flow_matching.modeling_flow_matching import FlowMatchingPolicy


def test_flow_matching_config_factory():
    cfg = make_policy_config("flow_matching", max_horizon=8, device="cpu")
    assert isinstance(cfg, FlowMatchingConfig)
    assert cfg.action_delta_indices == list(range(8))


def test_flow_matching_policy_forward_and_select_action_cpu():
    torch.manual_seed(0)

    cfg = FlowMatchingConfig(
        action_dim=14,
        qpos_dim=13,
        num_cameras=1,
        max_horizon=8,
        num_sampling_steps=2,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        pretrained_backbone_weights=None,
        device="cpu",
    )
    policy = FlowMatchingPolicy(config=cfg)

    batch_size = 2
    train_batch = {
        "observation.state": torch.rand(batch_size, 13),
        "observation.images.cam_high": torch.rand(batch_size, 3, 64, 64),
        "action": torch.rand(batch_size, 8, 14),
    }

    loss, info = policy(train_batch)

    assert torch.isfinite(loss)
    assert loss.dim() == 0
    assert "mse" in info

    infer_batch = {
        "observation.state": train_batch["observation.state"],
        "observation.images.cam_high": train_batch["observation.images.cam_high"],
    }

    action = policy.select_action(infer_batch)
    assert action.shape == (batch_size, 14)

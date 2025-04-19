import pytest
import torch
from torch import nn

from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.sac.modeling_sac import SACPolicy


@pytest.fixture
def sac_policy():
    config = SACConfig(
        num_critics=2,
        num_subsample_critics=1,
        discount=0.99,
        temperature_init=0.1,
        target_entropy=None,
        num_discrete_actions=None,
        use_backup_entropy=True,
        actor_network_kwargs={"hidden_dims": [256, 256]},
        discrete_critic_network_kwargs={"hidden_dims": [256, 256]},
        policy_kwargs={"use_tanh_squash": True},
    )
    return SACPolicy(config=config)


def test_compute_loss_critic(sac_policy):
    observations = torch.randn(4, 3, 84, 84)
    actions = torch.randn(4, 4)
    rewards = torch.randn(4)
    next_observations = torch.randn(4, 3, 84, 84)
    done = torch.zeros(4)

    loss = sac_policy.compute_loss_critic(observations, actions, rewards, next_observations, done)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor."


def test_compute_loss_actor(sac_policy):
    observations = torch.randn(4, 3, 84, 84)

    loss = sac_policy.compute_loss_actor(observations)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor."


def test_compute_loss_temperature(sac_policy):
    observations = torch.randn(4, 3, 84, 84)

    loss = sac_policy.compute_loss_temperature(observations)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor."


def test_init_normalization(sac_policy):
    dataset_stats = {
        "observation.image": {"mean": [0.5], "std": [0.5]},
        "action": {"mean": [0.0], "std": [1.0]},
    }
    sac_policy._init_normalization(dataset_stats)
    assert isinstance(sac_policy.normalize_inputs, nn.Module), "Normalization should be a module."

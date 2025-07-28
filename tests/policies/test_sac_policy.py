# !/usr/bin/env python

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

import math

import pytest
import torch
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import MLP, SACPolicy
from lerobot.utils.random_utils import seeded_context, set_seed

try:
    import transformers  # noqa: F401

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.fixture(autouse=True)
def set_random_seed():
    seed = 42
    set_seed(seed)


def test_mlp_with_default_args():
    mlp = MLP(input_dim=10, hidden_dims=[256, 256])

    x = torch.randn(10)
    y = mlp(x)
    assert y.shape == (256,)


def test_mlp_with_batch_dim():
    mlp = MLP(input_dim=10, hidden_dims=[256, 256])
    x = torch.randn(2, 10)
    y = mlp(x)
    assert y.shape == (2, 256)


def test_forward_with_empty_hidden_dims():
    mlp = MLP(input_dim=10, hidden_dims=[])
    x = torch.randn(1, 10)
    assert mlp(x).shape == (1, 10)


def test_mlp_with_dropout():
    mlp = MLP(input_dim=10, hidden_dims=[256, 256, 11], dropout_rate=0.1)
    x = torch.randn(1, 10)
    y = mlp(x)
    assert y.shape == (1, 11)

    drop_out_layers_count = sum(isinstance(layer, nn.Dropout) for layer in mlp.net)
    assert drop_out_layers_count == 2


def test_mlp_with_custom_final_activation():
    mlp = MLP(input_dim=10, hidden_dims=[256, 256], final_activation=torch.nn.Tanh())
    x = torch.randn(1, 10)
    y = mlp(x)
    assert y.shape == (1, 256)
    assert (y >= -1).all() and (y <= 1).all()


def test_sac_policy_with_default_args():
    with pytest.raises(ValueError, match="should be an instance of class `PreTrainedConfig`"):
        SACPolicy()


def create_dummy_state(batch_size: int, state_dim: int = 10) -> Tensor:
    return {
        "observation.state": torch.randn(batch_size, state_dim),
    }


def create_dummy_with_visual_input(batch_size: int, state_dim: int = 10) -> Tensor:
    return {
        "observation.image": torch.randn(batch_size, 3, 84, 84),
        "observation.state": torch.randn(batch_size, state_dim),
    }


def create_dummy_action(batch_size: int, action_dim: int = 10) -> Tensor:
    return torch.randn(batch_size, action_dim)


def create_default_train_batch(
    batch_size: int = 8, state_dim: int = 10, action_dim: int = 10
) -> dict[str, Tensor]:
    return {
        "action": create_dummy_action(batch_size, action_dim),
        "reward": torch.randn(batch_size),
        "state": create_dummy_state(batch_size, state_dim),
        "next_state": create_dummy_state(batch_size, state_dim),
        "done": torch.randn(batch_size),
    }


def create_train_batch_with_visual_input(
    batch_size: int = 8, state_dim: int = 10, action_dim: int = 10
) -> dict[str, Tensor]:
    return {
        "action": create_dummy_action(batch_size, action_dim),
        "reward": torch.randn(batch_size),
        "state": create_dummy_with_visual_input(batch_size, state_dim),
        "next_state": create_dummy_with_visual_input(batch_size, state_dim),
        "done": torch.randn(batch_size),
    }


def create_observation_batch(batch_size: int = 8, state_dim: int = 10) -> dict[str, Tensor]:
    return {
        "observation.state": torch.randn(batch_size, state_dim),
    }


def create_observation_batch_with_visual_input(batch_size: int = 8, state_dim: int = 10) -> dict[str, Tensor]:
    return {
        "observation.state": torch.randn(batch_size, state_dim),
        "observation.image": torch.randn(batch_size, 3, 84, 84),
    }


def make_optimizers(policy: SACPolicy, has_discrete_action: bool = False) -> dict[str, torch.optim.Optimizer]:
    """Create optimizers for the SAC policy."""
    optimizer_actor = torch.optim.Adam(
        # Handle the case of shared encoder where the encoder weights are not optimized with the actor gradient
        params=[
            p
            for n, p in policy.actor.named_parameters()
            if not policy.config.shared_encoder or not n.startswith("encoder")
        ],
        lr=policy.config.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(
        params=policy.critic_ensemble.parameters(),
        lr=policy.config.critic_lr,
    )
    optimizer_temperature = torch.optim.Adam(
        params=[policy.log_alpha],
        lr=policy.config.critic_lr,
    )

    optimizers = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
        "temperature": optimizer_temperature,
    }

    if has_discrete_action:
        optimizers["discrete_critic"] = torch.optim.Adam(
            params=policy.discrete_critic.parameters(),
            lr=policy.config.critic_lr,
        )

    return optimizers


def create_default_config(
    state_dim: int, continuous_action_dim: int, has_discrete_action: bool = False
) -> SACConfig:
    action_dim = continuous_action_dim
    if has_discrete_action:
        action_dim += 1

    config = SACConfig(
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(continuous_action_dim,))},
        dataset_stats={
            "observation.state": {
                "min": [0.0] * state_dim,
                "max": [1.0] * state_dim,
            },
            "action": {
                "min": [0.0] * continuous_action_dim,
                "max": [1.0] * continuous_action_dim,
            },
        },
    )
    config.validate_features()
    return config


def create_config_with_visual_input(
    state_dim: int, continuous_action_dim: int, has_discrete_action: bool = False
) -> SACConfig:
    config = create_default_config(
        state_dim=state_dim,
        continuous_action_dim=continuous_action_dim,
        has_discrete_action=has_discrete_action,
    )
    config.input_features["observation.image"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84))
    config.dataset_stats["observation.image"] = {
        "mean": torch.randn(3, 1, 1),
        "std": torch.randn(3, 1, 1),
    }

    # Let make tests a little bit faster
    config.state_encoder_hidden_dim = 32
    config.latent_dim = 32

    config.validate_features()
    return config


@pytest.mark.parametrize("batch_size,state_dim,action_dim", [(2, 6, 6), (1, 10, 10)])
def test_sac_policy_with_default_config(batch_size: int, state_dim: int, action_dim: int):
    batch = create_default_train_batch(batch_size=batch_size, action_dim=action_dim, state_dim=state_dim)
    config = create_default_config(state_dim=state_dim, continuous_action_dim=action_dim)

    policy = SACPolicy(config=config)
    policy.train()

    optimizers = make_optimizers(policy)

    cirtic_loss = policy.forward(batch, model="critic")["loss_critic"]
    assert cirtic_loss.item() is not None
    assert cirtic_loss.shape == ()
    cirtic_loss.backward()
    optimizers["critic"].step()

    actor_loss = policy.forward(batch, model="actor")["loss_actor"]
    assert actor_loss.item() is not None
    assert actor_loss.shape == ()

    actor_loss.backward()
    optimizers["actor"].step()

    temperature_loss = policy.forward(batch, model="temperature")["loss_temperature"]
    assert temperature_loss.item() is not None
    assert temperature_loss.shape == ()

    temperature_loss.backward()
    optimizers["temperature"].step()

    policy.eval()
    with torch.no_grad():
        observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
        selected_action = policy.select_action(observation_batch)
        assert selected_action.shape == (batch_size, action_dim)


@pytest.mark.parametrize("batch_size,state_dim,action_dim", [(2, 6, 6), (1, 10, 10)])
def test_sac_policy_with_visual_input(batch_size: int, state_dim: int, action_dim: int):
    config = create_config_with_visual_input(state_dim=state_dim, continuous_action_dim=action_dim)
    policy = SACPolicy(config=config)

    batch = create_train_batch_with_visual_input(
        batch_size=batch_size, state_dim=state_dim, action_dim=action_dim
    )

    policy.train()

    optimizers = make_optimizers(policy)

    cirtic_loss = policy.forward(batch, model="critic")["loss_critic"]
    assert cirtic_loss.item() is not None
    assert cirtic_loss.shape == ()
    cirtic_loss.backward()
    optimizers["critic"].step()

    actor_loss = policy.forward(batch, model="actor")["loss_actor"]
    assert actor_loss.item() is not None
    assert actor_loss.shape == ()

    actor_loss.backward()
    optimizers["actor"].step()

    temperature_loss = policy.forward(batch, model="temperature")["loss_temperature"]
    assert temperature_loss.item() is not None
    assert temperature_loss.shape == ()

    temperature_loss.backward()
    optimizers["temperature"].step()

    policy.eval()
    with torch.no_grad():
        observation_batch = create_observation_batch_with_visual_input(
            batch_size=batch_size, state_dim=state_dim
        )
        selected_action = policy.select_action(observation_batch)
        assert selected_action.shape == (batch_size, action_dim)


# Let's check best candidates for pretrained encoders
@pytest.mark.parametrize(
    "batch_size,state_dim,action_dim,vision_encoder_name",
    [(1, 6, 6, "helper2424/resnet10"), (1, 6, 6, "facebook/convnext-base-224")],
)
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers are not installed")
def test_sac_policy_with_pretrained_encoder(
    batch_size: int, state_dim: int, action_dim: int, vision_encoder_name: str
):
    config = create_config_with_visual_input(state_dim=state_dim, continuous_action_dim=action_dim)
    config.vision_encoder_name = vision_encoder_name
    policy = SACPolicy(config=config)
    policy.train()

    batch = create_train_batch_with_visual_input(
        batch_size=batch_size, state_dim=state_dim, action_dim=action_dim
    )

    optimizers = make_optimizers(policy)

    cirtic_loss = policy.forward(batch, model="critic")["loss_critic"]
    assert cirtic_loss.item() is not None
    assert cirtic_loss.shape == ()
    cirtic_loss.backward()
    optimizers["critic"].step()

    actor_loss = policy.forward(batch, model="actor")["loss_actor"]
    assert actor_loss.item() is not None
    assert actor_loss.shape == ()


def test_sac_policy_with_shared_encoder():
    batch_size = 2
    action_dim = 10
    state_dim = 10
    config = create_config_with_visual_input(state_dim=state_dim, continuous_action_dim=action_dim)
    config.shared_encoder = True

    policy = SACPolicy(config=config)
    policy.train()

    batch = create_train_batch_with_visual_input(
        batch_size=batch_size, state_dim=state_dim, action_dim=action_dim
    )

    policy.train()

    optimizers = make_optimizers(policy)

    cirtic_loss = policy.forward(batch, model="critic")["loss_critic"]
    assert cirtic_loss.item() is not None
    assert cirtic_loss.shape == ()
    cirtic_loss.backward()
    optimizers["critic"].step()

    actor_loss = policy.forward(batch, model="actor")["loss_actor"]
    assert actor_loss.item() is not None
    assert actor_loss.shape == ()

    actor_loss.backward()
    optimizers["actor"].step()


def test_sac_policy_with_discrete_critic():
    batch_size = 2
    continuous_action_dim = 9
    full_action_dim = continuous_action_dim + 1  # the last action is discrete
    state_dim = 10
    config = create_config_with_visual_input(
        state_dim=state_dim, continuous_action_dim=continuous_action_dim, has_discrete_action=True
    )

    num_discrete_actions = 5
    config.num_discrete_actions = num_discrete_actions

    policy = SACPolicy(config=config)
    policy.train()

    batch = create_train_batch_with_visual_input(
        batch_size=batch_size, state_dim=state_dim, action_dim=full_action_dim
    )

    policy.train()

    optimizers = make_optimizers(policy, has_discrete_action=True)

    cirtic_loss = policy.forward(batch, model="critic")["loss_critic"]
    assert cirtic_loss.item() is not None
    assert cirtic_loss.shape == ()
    cirtic_loss.backward()
    optimizers["critic"].step()

    discrete_critic_loss = policy.forward(batch, model="discrete_critic")["loss_discrete_critic"]
    assert discrete_critic_loss.item() is not None
    assert discrete_critic_loss.shape == ()
    discrete_critic_loss.backward()
    optimizers["discrete_critic"].step()

    actor_loss = policy.forward(batch, model="actor")["loss_actor"]
    assert actor_loss.item() is not None
    assert actor_loss.shape == ()

    actor_loss.backward()
    optimizers["actor"].step()

    policy.eval()
    with torch.no_grad():
        observation_batch = create_observation_batch_with_visual_input(
            batch_size=batch_size, state_dim=state_dim
        )
        selected_action = policy.select_action(observation_batch)
        assert selected_action.shape == (batch_size, full_action_dim)

        discrete_actions = selected_action[:, -1].long()
        discrete_action_values = set(discrete_actions.tolist())

        assert all(action in range(num_discrete_actions) for action in discrete_action_values), (
            f"Discrete action {discrete_action_values} is not in range({num_discrete_actions})"
        )


def test_sac_policy_with_default_entropy():
    config = create_default_config(continuous_action_dim=10, state_dim=10)
    policy = SACPolicy(config=config)
    assert policy.target_entropy == -5.0


def test_sac_policy_default_target_entropy_with_discrete_action():
    config = create_config_with_visual_input(state_dim=10, continuous_action_dim=6, has_discrete_action=True)
    policy = SACPolicy(config=config)
    assert policy.target_entropy == -3.0


def test_sac_policy_with_predefined_entropy():
    config = create_default_config(state_dim=10, continuous_action_dim=6)
    config.target_entropy = -3.5

    policy = SACPolicy(config=config)
    assert policy.target_entropy == pytest.approx(-3.5)


def test_sac_policy_update_temperature():
    config = create_default_config(continuous_action_dim=10, state_dim=10)
    policy = SACPolicy(config=config)

    assert policy.temperature == pytest.approx(1.0)
    policy.log_alpha.data = torch.tensor([math.log(0.1)])
    policy.update_temperature()
    assert policy.temperature == pytest.approx(0.1)


def test_sac_policy_update_target_network():
    config = create_default_config(state_dim=10, continuous_action_dim=6)
    config.critic_target_update_weight = 1.0

    policy = SACPolicy(config=config)
    policy.train()

    for p in policy.critic_ensemble.parameters():
        p.data = torch.ones_like(p.data)

    policy.update_target_networks()
    for p in policy.critic_target.parameters():
        assert torch.allclose(p.data, torch.ones_like(p.data)), (
            f"Target network {p.data} is not equal to {torch.ones_like(p.data)}"
        )


@pytest.mark.parametrize("num_critics", [1, 3])
def test_sac_policy_with_critics_number_of_heads(num_critics: int):
    batch_size = 2
    action_dim = 10
    state_dim = 10
    config = create_config_with_visual_input(state_dim=state_dim, continuous_action_dim=action_dim)
    config.num_critics = num_critics

    policy = SACPolicy(config=config)
    policy.train()

    assert len(policy.critic_ensemble.critics) == num_critics

    batch = create_train_batch_with_visual_input(
        batch_size=batch_size, state_dim=state_dim, action_dim=action_dim
    )

    policy.train()

    optimizers = make_optimizers(policy)

    cirtic_loss = policy.forward(batch, model="critic")["loss_critic"]
    assert cirtic_loss.item() is not None
    assert cirtic_loss.shape == ()
    cirtic_loss.backward()
    optimizers["critic"].step()


def test_sac_policy_save_and_load(tmp_path):
    root = tmp_path / "test_sac_save_and_load"

    state_dim = 10
    action_dim = 10
    batch_size = 2

    config = create_default_config(state_dim=state_dim, continuous_action_dim=action_dim)
    policy = SACPolicy(config=config)
    policy.eval()
    policy.save_pretrained(root)
    loaded_policy = SACPolicy.from_pretrained(root, config=config)
    loaded_policy.eval()

    batch = create_default_train_batch(batch_size=1, state_dim=10, action_dim=10)

    with torch.no_grad():
        with seeded_context(12):
            # Collect policy values before saving
            cirtic_loss = policy.forward(batch, model="critic")["loss_critic"]
            actor_loss = policy.forward(batch, model="actor")["loss_actor"]
            temperature_loss = policy.forward(batch, model="temperature")["loss_temperature"]

            observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
            actions = policy.select_action(observation_batch)

        with seeded_context(12):
            # Collect policy values after loading
            loaded_cirtic_loss = loaded_policy.forward(batch, model="critic")["loss_critic"]
            loaded_actor_loss = loaded_policy.forward(batch, model="actor")["loss_actor"]
            loaded_temperature_loss = loaded_policy.forward(batch, model="temperature")["loss_temperature"]

            loaded_observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
            loaded_actions = loaded_policy.select_action(loaded_observation_batch)

        assert policy.state_dict().keys() == loaded_policy.state_dict().keys()
        for k in policy.state_dict():
            assert torch.allclose(policy.state_dict()[k], loaded_policy.state_dict()[k], atol=1e-6)

        # Compare values before and after saving and loading
        # They should be the same
        assert torch.allclose(cirtic_loss, loaded_cirtic_loss)
        assert torch.allclose(actor_loss, loaded_actor_loss)
        assert torch.allclose(temperature_loss, loaded_temperature_loss)
        assert torch.allclose(actions, loaded_actions)

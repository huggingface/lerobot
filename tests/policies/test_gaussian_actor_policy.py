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

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig  # noqa: E402
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import MLP, GaussianActorPolicy  # noqa: E402
from lerobot.rl.algorithms.sac import SACAlgorithm, SACAlgorithmConfig  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE  # noqa: E402
from lerobot.utils.random_utils import seeded_context, set_seed  # noqa: E402

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


def test_gaussian_actor_policy_with_default_args():
    with pytest.raises(ValueError, match="should be an instance of class `PreTrainedConfig`"):
        GaussianActorPolicy()


def create_dummy_state(batch_size: int, state_dim: int = 10) -> Tensor:
    return {
        OBS_STATE: torch.randn(batch_size, state_dim),
    }


def create_dummy_with_visual_input(batch_size: int, state_dim: int = 10) -> Tensor:
    return {
        OBS_IMAGE: torch.randn(batch_size, 3, 84, 84),
        OBS_STATE: torch.randn(batch_size, state_dim),
    }


def create_dummy_action(batch_size: int, action_dim: int = 10) -> Tensor:
    return torch.randn(batch_size, action_dim)


def create_default_train_batch(
    batch_size: int = 8, state_dim: int = 10, action_dim: int = 10
) -> dict[str, Tensor]:
    return {
        ACTION: create_dummy_action(batch_size, action_dim),
        "reward": torch.randn(batch_size),
        "state": create_dummy_state(batch_size, state_dim),
        "next_state": create_dummy_state(batch_size, state_dim),
        "done": torch.randn(batch_size),
    }


def create_train_batch_with_visual_input(
    batch_size: int = 8, state_dim: int = 10, action_dim: int = 10
) -> dict[str, Tensor]:
    return {
        ACTION: create_dummy_action(batch_size, action_dim),
        "reward": torch.randn(batch_size),
        "state": create_dummy_with_visual_input(batch_size, state_dim),
        "next_state": create_dummy_with_visual_input(batch_size, state_dim),
        "done": torch.randn(batch_size),
    }


def create_observation_batch(batch_size: int = 8, state_dim: int = 10) -> dict[str, Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, state_dim),
    }


def create_observation_batch_with_visual_input(batch_size: int = 8, state_dim: int = 10) -> dict[str, Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, state_dim),
        OBS_IMAGE: torch.randn(batch_size, 3, 84, 84),
    }


def create_default_config(
    state_dim: int, continuous_action_dim: int, has_discrete_action: bool = False
) -> GaussianActorConfig:
    action_dim = continuous_action_dim
    if has_discrete_action:
        action_dim += 1

    config = GaussianActorConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(continuous_action_dim,))},
        dataset_stats={
            OBS_STATE: {
                "min": [0.0] * state_dim,
                "max": [1.0] * state_dim,
            },
            ACTION: {
                "min": [0.0] * continuous_action_dim,
                "max": [1.0] * continuous_action_dim,
            },
        },
    )
    config.validate_features()
    return config


def create_config_with_visual_input(
    state_dim: int, continuous_action_dim: int, has_discrete_action: bool = False
) -> GaussianActorConfig:
    config = create_default_config(
        state_dim=state_dim,
        continuous_action_dim=continuous_action_dim,
        has_discrete_action=has_discrete_action,
    )
    config.input_features[OBS_IMAGE] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84))
    config.dataset_stats[OBS_IMAGE] = {
        "mean": torch.randn(3, 1, 1),
        "std": torch.randn(3, 1, 1),
    }

    config.state_encoder_hidden_dim = 32
    config.latent_dim = 32

    config.validate_features()
    return config


def _make_algorithm(config: GaussianActorConfig) -> tuple[SACAlgorithm, GaussianActorPolicy]:
    """Helper to create policy + algorithm pair for tests that need critics."""
    policy = GaussianActorPolicy(config=config)
    policy.train()
    algo_config = SACAlgorithmConfig.from_policy_config(config)
    algorithm = SACAlgorithm(policy=policy, config=algo_config)
    algorithm.make_optimizers_and_scheduler()
    return algorithm, policy


@pytest.mark.parametrize("batch_size,state_dim,action_dim", [(2, 6, 6), (1, 10, 10)])
def test_gaussian_actor_policy_select_action(batch_size: int, state_dim: int, action_dim: int):
    config = create_default_config(state_dim=state_dim, continuous_action_dim=action_dim)
    policy = GaussianActorPolicy(config=config)
    policy.eval()

    with torch.no_grad():
        observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
        selected_action = policy.select_action(observation_batch)
        # squeeze(0) removes batch dim when batch_size==1
        assert selected_action.shape[-1] == action_dim


def test_gaussian_actor_policy_select_action_with_discrete():
    """select_action should return continuous + discrete actions."""
    config = create_default_config(state_dim=10, continuous_action_dim=6)
    config.num_discrete_actions = 3
    policy = GaussianActorPolicy(config=config)
    policy.eval()

    with torch.no_grad():
        observation_batch = create_observation_batch(batch_size=1, state_dim=10)
        # Squeeze to unbatched (single observation)
        observation_batch = {k: v.squeeze(0) for k, v in observation_batch.items()}
        selected_action = policy.select_action(observation_batch)
        assert selected_action.shape[-1] == 7  # 6 continuous + 1 discrete


@pytest.mark.parametrize("batch_size,state_dim,action_dim", [(2, 6, 6), (1, 10, 10)])
def test_gaussian_actor_policy_forward(batch_size: int, state_dim: int, action_dim: int):
    config = create_default_config(state_dim=state_dim, continuous_action_dim=action_dim)
    policy = GaussianActorPolicy(config=config)
    policy.eval()

    batch = create_default_train_batch(batch_size=batch_size, action_dim=action_dim, state_dim=state_dim)
    with torch.no_grad():
        output = policy.forward(batch)
        assert "action" in output
        assert "log_prob" in output
        assert "action_mean" in output
        assert output["action"].shape == (batch_size, action_dim)


@pytest.mark.parametrize("batch_size,state_dim,action_dim", [(2, 6, 6), (1, 10, 10)])
def test_gaussian_actor_training_through_sac(batch_size: int, state_dim: int, action_dim: int):
    config = create_default_config(state_dim=state_dim, continuous_action_dim=action_dim)
    algorithm, policy = _make_algorithm(config)

    batch = create_default_train_batch(batch_size=batch_size, action_dim=action_dim, state_dim=state_dim)
    forward_batch = algorithm._prepare_forward_batch(batch)

    critic_loss = algorithm._compute_loss_critic(forward_batch)
    assert critic_loss.item() is not None
    assert critic_loss.shape == ()
    algorithm.optimizers["critic"].zero_grad()
    critic_loss.backward()
    algorithm.optimizers["critic"].step()

    actor_loss = algorithm._compute_loss_actor(forward_batch)
    assert actor_loss.item() is not None
    assert actor_loss.shape == ()
    algorithm.optimizers["actor"].zero_grad()
    actor_loss.backward()
    algorithm.optimizers["actor"].step()

    temp_loss = algorithm._compute_loss_temperature(forward_batch)
    assert temp_loss.item() is not None
    assert temp_loss.shape == ()
    algorithm.optimizers["temperature"].zero_grad()
    temp_loss.backward()
    algorithm.optimizers["temperature"].step()


@pytest.mark.parametrize("batch_size,state_dim,action_dim", [(2, 6, 6), (1, 10, 10)])
def test_gaussian_actor_training_with_visual_input(batch_size: int, state_dim: int, action_dim: int):
    config = create_config_with_visual_input(state_dim=state_dim, continuous_action_dim=action_dim)
    algorithm, policy = _make_algorithm(config)

    batch = create_train_batch_with_visual_input(
        batch_size=batch_size, state_dim=state_dim, action_dim=action_dim
    )
    forward_batch = algorithm._prepare_forward_batch(batch)

    critic_loss = algorithm._compute_loss_critic(forward_batch)
    assert critic_loss.item() is not None
    assert critic_loss.shape == ()
    algorithm.optimizers["critic"].zero_grad()
    critic_loss.backward()
    algorithm.optimizers["critic"].step()

    actor_loss = algorithm._compute_loss_actor(forward_batch)
    assert actor_loss.item() is not None
    assert actor_loss.shape == ()
    algorithm.optimizers["actor"].zero_grad()
    actor_loss.backward()
    algorithm.optimizers["actor"].step()

    policy.eval()
    with torch.no_grad():
        observation_batch = create_observation_batch_with_visual_input(
            batch_size=batch_size, state_dim=state_dim
        )
        selected_action = policy.select_action(observation_batch)
        assert selected_action.shape[-1] == action_dim


@pytest.mark.parametrize(
    "batch_size,state_dim,action_dim,vision_encoder_name",
    [(1, 6, 6, "lerobot/resnet10"), (1, 6, 6, "facebook/convnext-base-224")],
)
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers are not installed")
def test_gaussian_actor_policy_with_pretrained_encoder(
    batch_size: int, state_dim: int, action_dim: int, vision_encoder_name: str
):
    config = create_config_with_visual_input(state_dim=state_dim, continuous_action_dim=action_dim)
    config.vision_encoder_name = vision_encoder_name
    algorithm, policy = _make_algorithm(config)

    batch = create_train_batch_with_visual_input(
        batch_size=batch_size, state_dim=state_dim, action_dim=action_dim
    )
    forward_batch = algorithm._prepare_forward_batch(batch)

    critic_loss = algorithm._compute_loss_critic(forward_batch)
    assert critic_loss.item() is not None
    assert critic_loss.shape == ()
    algorithm.optimizers["critic"].zero_grad()
    critic_loss.backward()
    algorithm.optimizers["critic"].step()

    actor_loss = algorithm._compute_loss_actor(forward_batch)
    assert actor_loss.item() is not None
    assert actor_loss.shape == ()


def test_gaussian_actor_training_with_shared_encoder():
    batch_size = 2
    action_dim = 10
    state_dim = 10
    config = create_config_with_visual_input(state_dim=state_dim, continuous_action_dim=action_dim)
    config.shared_encoder = True

    algorithm, policy = _make_algorithm(config)

    batch = create_train_batch_with_visual_input(
        batch_size=batch_size, state_dim=state_dim, action_dim=action_dim
    )
    forward_batch = algorithm._prepare_forward_batch(batch)

    critic_loss = algorithm._compute_loss_critic(forward_batch)
    assert critic_loss.shape == ()
    algorithm.optimizers["critic"].zero_grad()
    critic_loss.backward()
    algorithm.optimizers["critic"].step()

    actor_loss = algorithm._compute_loss_actor(forward_batch)
    assert actor_loss.shape == ()
    algorithm.optimizers["actor"].zero_grad()
    actor_loss.backward()
    algorithm.optimizers["actor"].step()


def test_gaussian_actor_training_with_discrete_critic():
    batch_size = 2
    continuous_action_dim = 9
    full_action_dim = continuous_action_dim + 1
    state_dim = 10
    config = create_config_with_visual_input(
        state_dim=state_dim, continuous_action_dim=continuous_action_dim, has_discrete_action=True
    )
    config.num_discrete_actions = 5

    algorithm, policy = _make_algorithm(config)

    batch = create_train_batch_with_visual_input(
        batch_size=batch_size, state_dim=state_dim, action_dim=full_action_dim
    )
    forward_batch = algorithm._prepare_forward_batch(batch)

    critic_loss = algorithm._compute_loss_critic(forward_batch)
    assert critic_loss.shape == ()
    algorithm.optimizers["critic"].zero_grad()
    critic_loss.backward()
    algorithm.optimizers["critic"].step()

    discrete_critic_loss = algorithm._compute_loss_discrete_critic(forward_batch)
    assert discrete_critic_loss.shape == ()
    algorithm.optimizers["discrete_critic"].zero_grad()
    discrete_critic_loss.backward()
    algorithm.optimizers["discrete_critic"].step()

    actor_loss = algorithm._compute_loss_actor(forward_batch)
    assert actor_loss.shape == ()
    algorithm.optimizers["actor"].zero_grad()
    actor_loss.backward()
    algorithm.optimizers["actor"].step()

    policy.eval()
    with torch.no_grad():
        observation_batch = create_observation_batch_with_visual_input(
            batch_size=batch_size, state_dim=state_dim
        )
        # Policy.select_action now handles both continuous + discrete
        selected_action = policy.select_action({k: v.squeeze(0) for k, v in observation_batch.items()})
        assert selected_action.shape[-1] == continuous_action_dim + 1


def test_sac_algorithm_target_entropy():
    """Target entropy is an SAC hyperparameter and lives on the algorithm."""
    config = create_default_config(continuous_action_dim=10, state_dim=10)
    algorithm, _ = _make_algorithm(config)
    assert algorithm.target_entropy == -5.0


def test_sac_algorithm_target_entropy_with_discrete_action():
    config = create_config_with_visual_input(state_dim=10, continuous_action_dim=6, has_discrete_action=True)
    config.num_discrete_actions = 5
    algorithm, _ = _make_algorithm(config)
    assert algorithm.target_entropy == -3.5


def test_sac_algorithm_temperature():
    import math

    config = create_default_config(continuous_action_dim=10, state_dim=10)
    algo_config = SACAlgorithmConfig.from_policy_config(config)
    policy = GaussianActorPolicy(config=config)
    algorithm = SACAlgorithm(policy=policy, config=algo_config)

    assert algorithm.temperature == pytest.approx(1.0)
    algorithm.log_alpha.data = torch.tensor([math.log(0.1)])
    assert algorithm.temperature == pytest.approx(0.1)


def test_sac_algorithm_update_target_network():
    config = create_default_config(state_dim=10, continuous_action_dim=6)
    algo_config = SACAlgorithmConfig.from_policy_config(config)
    algo_config.critic_target_update_weight = 1.0
    policy = GaussianActorPolicy(config=config)
    algorithm = SACAlgorithm(policy=policy, config=algo_config)

    for p in algorithm.critic_ensemble.parameters():
        p.data = torch.ones_like(p.data)

    algorithm._update_target_networks()
    for p in algorithm.critic_target.parameters():
        assert torch.allclose(p.data, torch.ones_like(p.data))


@pytest.mark.parametrize("num_critics", [1, 3])
def test_sac_algorithm_with_critics_number_of_heads(num_critics: int):
    batch_size = 2
    action_dim = 10
    state_dim = 10
    config = create_config_with_visual_input(state_dim=state_dim, continuous_action_dim=action_dim)

    policy = GaussianActorPolicy(config=config)
    policy.train()
    algo_config = SACAlgorithmConfig.from_policy_config(config)
    algo_config.num_critics = num_critics
    algorithm = SACAlgorithm(policy=policy, config=algo_config)
    algorithm.make_optimizers_and_scheduler()

    assert len(algorithm.critic_ensemble.critics) == num_critics

    batch = create_train_batch_with_visual_input(
        batch_size=batch_size, state_dim=state_dim, action_dim=action_dim
    )
    forward_batch = algorithm._prepare_forward_batch(batch)

    critic_loss = algorithm._compute_loss_critic(forward_batch)
    assert critic_loss.shape == ()
    algorithm.optimizers["critic"].zero_grad()
    critic_loss.backward()
    algorithm.optimizers["critic"].step()


def test_gaussian_actor_policy_save_and_load(tmp_path):
    """Test that the policy can be saved and loaded from pretrained."""
    root = tmp_path / "test_gaussian_actor_save_and_load"

    state_dim = 10
    action_dim = 10
    batch_size = 2

    config = create_default_config(state_dim=state_dim, continuous_action_dim=action_dim)
    policy = GaussianActorPolicy(config=config)
    policy.eval()
    policy.save_pretrained(root)
    loaded_policy = GaussianActorPolicy.from_pretrained(root, config=config)
    loaded_policy.eval()

    assert policy.state_dict().keys() == loaded_policy.state_dict().keys()
    for k in policy.state_dict():
        assert torch.allclose(policy.state_dict()[k], loaded_policy.state_dict()[k], atol=1e-6)

    with torch.no_grad():
        with seeded_context(12):
            observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
            actions = policy.select_action(observation_batch)

        with seeded_context(12):
            loaded_observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
            loaded_actions = loaded_policy.select_action(loaded_observation_batch)

        assert torch.allclose(actions, loaded_actions)


def test_gaussian_actor_policy_save_and_load_with_discrete_critic(tmp_path):
    """Discrete critic should be saved/loaded as part of the policy."""
    root = tmp_path / "test_gaussian_actor_save_and_load_discrete"

    state_dim = 10
    action_dim = 6

    config = create_default_config(state_dim=state_dim, continuous_action_dim=action_dim)
    config.num_discrete_actions = 3
    policy = GaussianActorPolicy(config=config)
    policy.eval()
    policy.save_pretrained(root)

    loaded_policy = GaussianActorPolicy.from_pretrained(root, config=config)
    loaded_policy.eval()

    assert loaded_policy.discrete_critic is not None
    dc_keys = [k for k in loaded_policy.state_dict() if k.startswith("discrete_critic.")]
    assert len(dc_keys) > 0

    for k in policy.state_dict():
        assert torch.allclose(policy.state_dict()[k], loaded_policy.state_dict()[k], atol=1e-6)

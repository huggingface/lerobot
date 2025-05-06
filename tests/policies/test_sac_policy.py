import pytest
import torch
from torch import Tensor, nn

from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.sac.modeling_sac import MLP, SACPolicy
from lerobot.common.utils.random_utils import seeded_context
from lerobot.configs.types import FeatureType, PolicyFeature


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


def make_optimizers(policy: SACPolicy) -> dict[str, torch.optim.Optimizer]:
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
    return optimizers


def create_default_config(state_dim: int, action_dim: int) -> SACConfig:
    config = SACConfig(
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        dataset_stats={
            "observation.state": {
                "min": [0.0] * state_dim,
                "max": [1.0] * state_dim,
            },
            "action": {
                "min": [0.0] * action_dim,
                "max": [1.0] * action_dim,
            },
        },
    )
    config.validate_features()
    return config


def create_config_with_visual_input(state_dim: int, action_dim: int) -> SACConfig:
    config = create_default_config(state_dim=state_dim, action_dim=action_dim)
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
    batch = create_default_train_batch(batch_size=batch_size, state_dim=state_dim, action_dim=action_dim)
    config = create_default_config(state_dim=state_dim, action_dim=action_dim)

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
    config = create_config_with_visual_input(state_dim=state_dim, action_dim=action_dim)
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
def test_sac_policy_with_pretrained_encoder(
    batch_size: int, state_dim: int, action_dim: int, vision_encoder_name: str
):
    config = create_config_with_visual_input(state_dim=state_dim, action_dim=action_dim)
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
    config = create_config_with_visual_input(state_dim=state_dim, action_dim=action_dim)
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
    action_dim = 10
    state_dim = 10
    config = create_config_with_visual_input(state_dim=state_dim, action_dim=action_dim)
    config.num_discrete_actions = 3

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


def test_sac_policy_with_max_entropy():
    pass


def test_sac_policy_with_frozen_encoder():
    pass


@pytest.mark.parametrize("num_critics", [1, 3])
def test_sac_policy_with_critics_number_of_heads(num_critics: int):
    batch_size = 2
    action_dim = 10
    state_dim = 10
    config = create_config_with_visual_input(state_dim=state_dim, action_dim=action_dim)
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

    with seeded_context(0):
        config = create_default_config(state_dim=10, action_dim=10)
        policy = SACPolicy(config=config)
        policy.eval()
        policy.save_pretrained(root)
        loaded_policy = SACPolicy.from_pretrained(root)
        loaded_policy.eval()

        batch = create_default_train_batch(batch_size=1, state_dim=10, action_dim=10)

        with torch.no_grad():
            # Collect policy values before saving
            cirtic_loss = policy.forward(batch, model="critic")["loss_critic"]
            actor_loss = policy.forward(batch, model="actor")["loss_actor"]
            temperature_loss = policy.forward(batch, model="temperature")["loss_temperature"]

            # Collect policy values after loading
            loaded_cirtic_loss = loaded_policy.forward(batch, model="critic")["loss_critic"]
            loaded_actor_loss = loaded_policy.forward(batch, model="actor")["loss_actor"]
            loaded_temperature_loss = loaded_policy.forward(batch, model="temperature")["loss_temperature"]

            # Compare values before and after saving and loading
            # They should be the same
            assert torch.allclose(cirtic_loss, loaded_cirtic_loss)
            assert torch.allclose(actor_loss, loaded_actor_loss)
            assert torch.allclose(temperature_loss, loaded_temperature_loss)

#!/usr/bin/env python

# Copyright 2025 Bryson Jones and The HuggingFace Inc. team. All rights reserved.
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

"""Test script for Multi-Task DiT policy.

To run tests with GPU on Modal (temporary script):
    modal run run_tests_modal.py

To run tests locally:
    python -m pytest tests/policies/test_multi_task_dit_policy.py -v
"""

import pytest
import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import (
    DiffusionConfig,
    FlowMatchingConfig,
    MultiTaskDiTConfig,
)
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.random_utils import seeded_context, set_seed


@pytest.fixture(autouse=True)
def set_random_seed():
    seed = 17
    set_seed(seed)


def create_train_batch(
    batch_size: int = 2,
    n_obs_steps: int = 2,
    horizon: int = 16,
    state_dim: int = 10,
    action_dim: int = 10,
    height: int = 224,
    width: int = 224,
) -> dict[str, Tensor]:
    """Create a training batch with visual input and text."""
    return {
        "observation.state": torch.randn(batch_size, n_obs_steps, state_dim),
        f"{OBS_IMAGES}.laptop": torch.rand(batch_size, n_obs_steps, 3, height, width),
        ACTION: torch.randn(batch_size, horizon, action_dim),
        "task": ["pick up the cube"] * batch_size,
    }


def create_observation_batch(
    batch_size: int = 2, state_dim: int = 10, height: int = 224, width: int = 224
) -> dict:
    """Create observation batch for inference for a single timestep."""
    return {
        "observation.state": torch.randn(batch_size, state_dim),
        f"{OBS_IMAGES}.laptop": torch.rand(batch_size, 3, height, width),
        "task": ["pick up the red cube"] * batch_size,
    }


def create_config(
    state_dim: int = 10,
    action_dim: int = 10,
    n_obs_steps: int = 2,
    horizon: int = 16,
    n_action_steps: int = 8,
    with_visual: bool = True,
    height: int = 224,
    width: int = 224,
) -> MultiTaskDiTConfig:
    """Create a MultiTaskDiT config for testing.

    Args:
        state_dim: Dimension of state observations
        action_dim: Dimension of actions
        n_obs_steps: Number of observation steps
        horizon: Action prediction horizon
        n_action_steps: Number of action steps to execute
        with_visual: Whether to include visual input (default: True)
        height: Image height (only used if with_visual=True)
        width: Image width (only used if with_visual=True)
    """
    input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))}

    if with_visual:
        input_features[f"{OBS_IMAGES}.laptop"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, height, width)
        )

    config = MultiTaskDiTConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
    )

    # Use smaller model for faster tests
    config.transformer.hidden_dim = 128
    config.transformer.num_layers = 2
    config.transformer.num_heads = 4

    config.validate_features()
    return config


@pytest.mark.parametrize("batch_size,state_dim,action_dim", [(2, 10, 10), (1, 6, 6)])
def test_multi_task_dit_policy_forward(batch_size: int, state_dim: int, action_dim: int):
    """Test forward pass (training mode)."""
    n_obs_steps = 2
    horizon = 16
    n_action_steps = 8

    config = create_config(
        state_dim=state_dim,
        action_dim=action_dim,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
    )

    policy = MultiTaskDiTPolicy(config=config)
    policy.train()

    batch = create_train_batch(
        batch_size=batch_size,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    # Test forward pass
    loss, _ = policy.forward(batch)
    assert loss is not None
    assert loss.item() is not None
    assert loss.shape == ()

    # Test backward pass
    loss.backward()


@pytest.mark.parametrize("batch_size,state_dim,action_dim", [(2, 10, 10), (1, 6, 6)])
def test_multi_task_dit_policy_select_action(batch_size: int, state_dim: int, action_dim: int):
    """Test select_action (inference mode)."""
    n_obs_steps = 2
    horizon = 16
    n_action_steps = 8

    config = create_config(
        state_dim=state_dim,
        action_dim=action_dim,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
    )

    policy = MultiTaskDiTPolicy(config=config)
    policy.eval()
    policy.reset()  # Reset queues before inference

    with torch.no_grad():
        observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
        selected_action = policy.select_action(observation_batch)
        assert selected_action.shape == (batch_size, action_dim)


def test_multi_task_dit_policy_diffusion_objective():
    """Test policy with diffusion objective."""
    batch_size = 2
    state_dim = 10
    action_dim = 10
    n_obs_steps = 2
    horizon = 16
    n_action_steps = 8

    config = create_config(
        state_dim=state_dim,
        action_dim=action_dim,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
    )
    config.objective = DiffusionConfig(
        noise_scheduler_type="DDPM",
        num_train_timesteps=100,
        num_inference_steps=10,
    )

    policy = MultiTaskDiTPolicy(config=config)
    policy.train()

    batch = create_train_batch(
        batch_size=batch_size,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    # Test forward pass
    loss, _ = policy.forward(batch)
    assert loss is not None
    assert loss.item() is not None

    # Test inference
    policy.eval()
    with torch.no_grad():
        observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
        selected_action = policy.select_action(observation_batch)
        assert selected_action.shape == (batch_size, action_dim)


def test_multi_task_dit_policy_flow_matching_objective():
    """Test policy with flow matching objective."""
    batch_size = 2
    state_dim = 10
    action_dim = 10
    n_obs_steps = 2
    horizon = 16
    n_action_steps = 8

    config = create_config(
        state_dim=state_dim,
        action_dim=action_dim,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
    )
    config.objective = FlowMatchingConfig(
        sigma_min=0.0,
        num_integration_steps=10,  # Use fewer steps for faster tests
        integration_method="euler",
    )

    policy = MultiTaskDiTPolicy(config=config)
    policy.train()

    batch = create_train_batch(
        batch_size=batch_size,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    # Test forward pass
    loss, _ = policy.forward(batch)
    assert loss is not None
    assert loss.item() is not None

    # Test inference
    policy.eval()
    with torch.no_grad():
        observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
        selected_action = policy.select_action(observation_batch)
        assert selected_action.shape == (batch_size, action_dim)


def test_multi_task_dit_policy_save_and_load(tmp_path):
    """Test that the policy can be saved and loaded correctly."""
    root = tmp_path / "test_multi_task_dit_save_and_load"

    state_dim = 10
    action_dim = 10
    batch_size = 2
    n_obs_steps = 2
    horizon = 16
    n_action_steps = 8

    config = create_config(
        state_dim=state_dim,
        action_dim=action_dim,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
    )

    policy = MultiTaskDiTPolicy(config=config)
    policy.eval()

    # Get device before saving
    device = next(policy.parameters()).device

    policy.save_pretrained(root)
    loaded_policy = MultiTaskDiTPolicy.from_pretrained(root, config=config)

    # Explicitly move loaded_policy to the same device
    loaded_policy.to(device)
    loaded_policy.eval()

    batch = create_train_batch(
        batch_size=batch_size,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    # Move batch to the same device as the policy
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    with torch.no_grad():
        with seeded_context(12):
            # Collect policy values before saving
            loss, _ = policy.forward(batch)

            observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
            # Move observation batch to device
            for key in observation_batch:
                if isinstance(observation_batch[key], torch.Tensor):
                    observation_batch[key] = observation_batch[key].to(device)
            actions = policy.select_action(observation_batch)

        with seeded_context(12):
            # Collect policy values after loading
            loaded_loss, _ = loaded_policy.forward(batch)

            loaded_observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
            # Move observation batch to device
            for key in loaded_observation_batch:
                if isinstance(loaded_observation_batch[key], torch.Tensor):
                    loaded_observation_batch[key] = loaded_observation_batch[key].to(device)
            loaded_actions = loaded_policy.select_action(loaded_observation_batch)

        # Compare state dicts
        assert policy.state_dict().keys() == loaded_policy.state_dict().keys()
        for k in policy.state_dict():
            assert torch.allclose(policy.state_dict()[k], loaded_policy.state_dict()[k], atol=1e-6)

        # Compare values before and after saving and loading
        assert torch.allclose(loss, loaded_loss)
        assert torch.allclose(actions, loaded_actions)


def test_multi_task_dit_policy_get_optim_params():
    """Test that the policy returns correct optimizer parameter groups."""
    config = create_config(
        state_dim=10,
        action_dim=10,
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
    )

    policy = MultiTaskDiTPolicy(config=config)
    param_groups = policy.get_optim_params()

    # Should have 2 parameter groups: non-vision and vision encoder
    assert len(param_groups) == 2

    # First group is non-vision params (no lr specified, will use default)
    assert "params" in param_groups[0]
    assert len(param_groups[0]["params"]) > 0

    # Second group is vision encoder params with different lr
    assert "params" in param_groups[1]
    assert "lr" in param_groups[1]
    expected_lr = config.optimizer_lr * config.observation_encoder.vision.lr_multiplier
    assert param_groups[1]["lr"] == expected_lr

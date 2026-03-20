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

# ruff: noqa: E402

"""Test script for Multi-Task DiT policy.

To run tests locally:
    python -m pytest tests/policies/multi_task_dit/test_multi_task_dit.py -v
"""

import os

import pytest
import torch
from torch import Tensor

pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="This test requires local transformers installation and is not meant for CI",
)

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
from lerobot.policies.multi_task_dit.processor_multi_task_dit import (
    make_multi_task_dit_pre_post_processors,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
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
        # Use smaller model for faster tests
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
    )

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

    # Use preprocessor to handle tokenization
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    preprocessor, _ = make_multi_task_dit_pre_post_processors(config=config, dataset_stats=None)

    batch = create_train_batch(
        batch_size=batch_size,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    # Process batch through preprocessor to tokenize task text
    processed_batch = preprocessor(batch)

    # Test forward pass
    loss, _ = policy.forward(processed_batch)
    assert loss is not None
    assert loss.item() is not None
    assert loss.shape == ()

    # Test backward pass
    loss.backward()


def test_multi_task_dit_pre_post_processors():
    """Test pre and post processors for Multi-Task DiT policy."""
    state_dim = 10
    action_dim = 8
    n_obs_steps = 2
    horizon = 16

    config = create_config(
        state_dim=state_dim,
        action_dim=action_dim,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=8,
    )
    config.device = "cpu"

    # Set normalization mode to match the stats we're providing
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,  # Use MEAN_STD since we provide mean/std stats
        "ACTION": NormalizationMode.MIN_MAX,
    }

    # Create dataset stats for normalization
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(state_dim),
            "std": torch.ones(state_dim),
        },
        "action": {
            "min": torch.full((action_dim,), -1.0),
            "max": torch.ones(action_dim),
        },
    }

    # Create processors
    preprocessor, postprocessor = make_multi_task_dit_pre_post_processors(
        config=config, dataset_stats=dataset_stats
    )

    # Test preprocessor with sample data
    batch = {
        "observation.state": torch.randn(state_dim),
        f"{OBS_IMAGES}.laptop": torch.rand(3, 224, 224),
        ACTION: torch.randn(action_dim),
        "task": "pick up the cube",
    }

    processed_batch = preprocessor(batch)

    # Check that data is batched
    assert processed_batch["observation.state"].shape == (1, state_dim)
    assert processed_batch[f"{OBS_IMAGES}.laptop"].shape == (1, 3, 224, 224)
    assert processed_batch[ACTION].shape == (1, action_dim)
    # Check that task text was tokenized
    assert OBS_LANGUAGE_TOKENS in processed_batch
    assert OBS_LANGUAGE_ATTENTION_MASK in processed_batch
    assert processed_batch[OBS_LANGUAGE_TOKENS].shape[0] == 1  # batch dimension
    assert processed_batch[OBS_LANGUAGE_ATTENTION_MASK].shape[0] == 1  # batch dimension

    # Check that data is on correct device
    assert processed_batch["observation.state"].device.type == "cpu"
    assert processed_batch[f"{OBS_IMAGES}.laptop"].device.type == "cpu"
    assert processed_batch[ACTION].device.type == "cpu"

    # Test postprocessor with sample action (PolicyAction is just a torch.Tensor)
    action = torch.randn(1, action_dim)
    processed_action = postprocessor(action)

    # Check that action is unnormalized and on CPU
    assert processed_action.shape == (1, action_dim)
    assert processed_action.device.type == "cpu"


def test_multi_task_dit_pre_post_processors_normalization():
    """Test that normalization and unnormalization work correctly with simple sanity check numbers."""
    state_dim = 3
    action_dim = 2

    config = create_config(
        state_dim=state_dim,
        action_dim=action_dim,
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
    )
    config.device = "cpu"

    # Set normalization mode to match the stats we're providing
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.MEAN_STD,  # Use MEAN_STD since we provide mean/std stats
        "ACTION": NormalizationMode.MIN_MAX,
    }

    # Use simple stats that will actually transform the values
    dataset_stats = {
        "observation.state": {
            "mean": torch.full((state_dim,), 5.0),
            "std": torch.full((state_dim,), 2.0),
        },
        "action": {
            "min": torch.zeros(action_dim),
            "max": torch.full((action_dim,), 2.0),
        },
    }

    # Create processors
    preprocessor, postprocessor = make_multi_task_dit_pre_post_processors(
        config=config, dataset_stats=dataset_stats
    )

    # Use simple input values
    input_state = torch.tensor([7.0, 5.0, 3.0])  # Will normalize to [1.0, 0.0, -1.0]
    input_action = torch.tensor([1.0, 2.0])  # Will normalize to [0.0, 1.0]

    batch = {
        "observation.state": input_state,
        f"{OBS_IMAGES}.laptop": torch.rand(3, 224, 224),
        ACTION: input_action,
        "task": "test task",
    }

    # Process through preprocessor
    processed_batch = preprocessor(batch)

    # State normalization: (x - mean) / std
    expected_normalized_state = torch.tensor([1.0, 0.0, -1.0])
    assert torch.allclose(processed_batch["observation.state"][0], expected_normalized_state, atol=1e-5)

    # Action normalization: (x - min) / (max - min) * 2 - 1
    expected_normalized_action = torch.tensor([0.0, 1.0])
    assert torch.allclose(processed_batch[ACTION][0], expected_normalized_action, atol=1e-5)

    # Test unnormalization: should recover original values
    normalized_action_tensor = processed_batch[ACTION][0:1]  # Keep batch dimension
    unnormalized_action = postprocessor(normalized_action_tensor)

    # Should recover original action values
    assert torch.allclose(unnormalized_action[0], input_action, atol=1e-4)


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

    # Create processors - use IDENTITY normalization when no stats provided
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    preprocessor, postprocessor = make_multi_task_dit_pre_post_processors(config=config, dataset_stats=None)

    with torch.no_grad():
        observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
        # Process observation through preprocessor
        processed_obs = preprocessor(observation_batch)
        selected_action = policy.select_action(processed_obs)
        # Process action through postprocessor (PolicyAction is just a torch.Tensor)
        processed_action = postprocessor(selected_action)
        assert processed_action.shape == (batch_size, action_dim)


def test_multi_task_dit_policy_diffusion_objective():
    """Test policy with diffusion objective."""
    batch_size = 2
    state_dim = 10
    action_dim = 10
    n_obs_steps = 2
    horizon = 16
    n_action_steps = 8

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        f"{OBS_IMAGES}.laptop": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }

    config = MultiTaskDiTConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
        # Use diffusion objective
        objective="diffusion",
        noise_scheduler_type="DDPM",
        num_train_timesteps=100,
        num_inference_steps=10,
        # Smaller model for tests
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
    )
    config.validate_features()

    policy = MultiTaskDiTPolicy(config=config)
    policy.train()

    # Use preprocessor to handle tokenization
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    preprocessor, _ = make_multi_task_dit_pre_post_processors(config=config, dataset_stats=None)

    batch = create_train_batch(
        batch_size=batch_size,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    # Process batch through preprocessor to tokenize task text
    processed_batch = preprocessor(batch)

    # Test forward pass
    loss, _ = policy.forward(processed_batch)
    assert loss is not None
    assert loss.item() is not None

    # Test inference
    policy.eval()
    # Use IDENTITY normalization when no stats provided
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    preprocessor, postprocessor = make_multi_task_dit_pre_post_processors(config=config, dataset_stats=None)
    with torch.no_grad():
        observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
        # Process observation through preprocessor
        processed_obs = preprocessor(observation_batch)
        selected_action = policy.select_action(processed_obs)
        # Process action through postprocessor (PolicyAction is just a torch.Tensor)
        processed_action = postprocessor(selected_action)
        assert processed_action.shape == (batch_size, action_dim)


def test_multi_task_dit_policy_flow_matching_objective():
    """Test policy with flow matching objective."""
    batch_size = 2
    state_dim = 10
    action_dim = 10
    n_obs_steps = 2
    horizon = 16
    n_action_steps = 8

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        f"{OBS_IMAGES}.laptop": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }

    config = MultiTaskDiTConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_action_steps=n_action_steps,
        # Use flow matching objective
        objective="flow_matching",
        sigma_min=0.0,
        num_integration_steps=10,  # Fewer steps for faster tests
        integration_method="euler",
        # Smaller model for tests
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
    )
    config.validate_features()

    policy = MultiTaskDiTPolicy(config=config)
    policy.train()

    # Use preprocessor to handle tokenization
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    preprocessor, _ = make_multi_task_dit_pre_post_processors(config=config, dataset_stats=None)

    batch = create_train_batch(
        batch_size=batch_size,
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    # Process batch through preprocessor to tokenize task text
    processed_batch = preprocessor(batch)

    # Test forward pass
    loss, _ = policy.forward(processed_batch)
    assert loss is not None
    assert loss.item() is not None

    # Test inference
    policy.eval()
    # Use IDENTITY normalization when no stats provided
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    preprocessor, postprocessor = make_multi_task_dit_pre_post_processors(config=config, dataset_stats=None)
    with torch.no_grad():
        observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
        # Process observation through preprocessor
        processed_obs = preprocessor(observation_batch)
        selected_action = policy.select_action(processed_obs)
        # Process action through postprocessor (PolicyAction is just a torch.Tensor)
        processed_action = postprocessor(selected_action)
        assert processed_action.shape == (batch_size, action_dim)


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

    # Use preprocessor to handle tokenization
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }
    preprocessor, postprocessor = make_multi_task_dit_pre_post_processors(config=config, dataset_stats=None)

    with torch.no_grad():
        with seeded_context(12):
            # Process batch through preprocessor
            processed_batch = preprocessor(batch)
            # Move batch to the same device as the policy
            for key in processed_batch:
                if isinstance(processed_batch[key], torch.Tensor):
                    processed_batch[key] = processed_batch[key].to(device)
            # Collect policy values before saving
            loss, _ = policy.forward(processed_batch)

            observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
            # Process observation through preprocessor
            processed_obs = preprocessor(observation_batch)
            actions = policy.select_action(processed_obs)

        with seeded_context(12):
            # Process batch through preprocessor
            processed_batch = preprocessor(batch)
            # Collect policy values after loading
            loaded_loss, _ = loaded_policy.forward(processed_batch)

            loaded_observation_batch = create_observation_batch(batch_size=batch_size, state_dim=state_dim)
            processed_obs = preprocessor(loaded_observation_batch)
            loaded_actions = loaded_policy.select_action(processed_obs)

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
    expected_lr = config.optimizer_lr * config.vision_encoder_lr_multiplier
    assert param_groups[1]["lr"] == expected_lr

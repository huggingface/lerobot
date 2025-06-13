# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import torch

from lerobot.common.constants import (
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
)
from lerobot.common.optim.optimizers import (
    AdamConfig,
    AdamWConfig,
    MultiAdamConfig,
    SGDConfig,
    load_optimizer_state,
    save_optimizer_state,
)


@pytest.mark.parametrize(
    "config_cls, expected_class",
    [
        (AdamConfig, torch.optim.Adam),
        (AdamWConfig, torch.optim.AdamW),
        (SGDConfig, torch.optim.SGD),
        (MultiAdamConfig, dict),
    ],
)
def test_optimizer_build(config_cls, expected_class, model_params):
    config = config_cls()
    if config_cls == MultiAdamConfig:
        params_dict = {"default": model_params}
        optimizer = config.build(params_dict)
        assert isinstance(optimizer, expected_class)
        assert isinstance(optimizer["default"], torch.optim.Adam)
        assert optimizer["default"].defaults["lr"] == config.lr
    else:
        optimizer = config.build(model_params)
        assert isinstance(optimizer, expected_class)
        assert optimizer.defaults["lr"] == config.lr


def test_save_optimizer_state(optimizer, tmp_path):
    save_optimizer_state(optimizer, tmp_path)
    assert (tmp_path / OPTIMIZER_STATE).is_file()
    assert (tmp_path / OPTIMIZER_PARAM_GROUPS).is_file()


def test_save_and_load_optimizer_state(model_params, optimizer, tmp_path):
    save_optimizer_state(optimizer, tmp_path)
    loaded_optimizer = AdamConfig().build(model_params)
    loaded_optimizer = load_optimizer_state(loaded_optimizer, tmp_path)

    torch.testing.assert_close(optimizer.state_dict(), loaded_optimizer.state_dict())


@pytest.fixture
def base_params_dict():
    return {
        "actor": [torch.nn.Parameter(torch.randn(10, 10))],
        "critic": [torch.nn.Parameter(torch.randn(5, 5))],
        "temperature": [torch.nn.Parameter(torch.randn(3, 3))],
    }


@pytest.mark.parametrize(
    "config_params, expected_values",
    [
        # Test 1: Basic configuration with different learning rates
        (
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "optimizer_groups": {
                    "actor": {"lr": 1e-4},
                    "critic": {"lr": 5e-4},
                    "temperature": {"lr": 2e-3},
                },
            },
            {
                "actor": {"lr": 1e-4, "weight_decay": 1e-4, "betas": (0.9, 0.999)},
                "critic": {"lr": 5e-4, "weight_decay": 1e-4, "betas": (0.9, 0.999)},
                "temperature": {"lr": 2e-3, "weight_decay": 1e-4, "betas": (0.9, 0.999)},
            },
        ),
        # Test 2: Different weight decays and beta values
        (
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "optimizer_groups": {
                    "actor": {"lr": 1e-4, "weight_decay": 1e-5},
                    "critic": {"lr": 5e-4, "weight_decay": 1e-6},
                    "temperature": {"lr": 2e-3, "betas": (0.95, 0.999)},
                },
            },
            {
                "actor": {"lr": 1e-4, "weight_decay": 1e-5, "betas": (0.9, 0.999)},
                "critic": {"lr": 5e-4, "weight_decay": 1e-6, "betas": (0.9, 0.999)},
                "temperature": {"lr": 2e-3, "weight_decay": 1e-4, "betas": (0.95, 0.999)},
            },
        ),
        # Test 3: Epsilon parameter customization
        (
            {
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "optimizer_groups": {
                    "actor": {"lr": 1e-4, "eps": 1e-6},
                    "critic": {"lr": 5e-4, "eps": 1e-7},
                    "temperature": {"lr": 2e-3, "eps": 1e-8},
                },
            },
            {
                "actor": {"lr": 1e-4, "weight_decay": 1e-4, "betas": (0.9, 0.999), "eps": 1e-6},
                "critic": {"lr": 5e-4, "weight_decay": 1e-4, "betas": (0.9, 0.999), "eps": 1e-7},
                "temperature": {"lr": 2e-3, "weight_decay": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8},
            },
        ),
    ],
)
def test_multi_adam_configuration(base_params_dict, config_params, expected_values):
    # Create config with the given parameters
    config = MultiAdamConfig(**config_params)
    optimizers = config.build(base_params_dict)

    # Verify optimizer count and keys
    assert len(optimizers) == len(expected_values)
    assert set(optimizers.keys()) == set(expected_values.keys())

    # Check that all optimizers are Adam instances
    for opt in optimizers.values():
        assert isinstance(opt, torch.optim.Adam)

    # Verify hyperparameters for each optimizer
    for name, expected in expected_values.items():
        optimizer = optimizers[name]
        for param, value in expected.items():
            assert optimizer.defaults[param] == value


@pytest.fixture
def multi_optimizers(base_params_dict):
    config = MultiAdamConfig(
        lr=1e-3,
        optimizer_groups={
            "actor": {"lr": 1e-4},
            "critic": {"lr": 5e-4},
            "temperature": {"lr": 2e-3},
        },
    )
    return config.build(base_params_dict)


def test_save_multi_optimizer_state(multi_optimizers, tmp_path):
    # Save optimizer states
    save_optimizer_state(multi_optimizers, tmp_path)

    # Verify that directories were created for each optimizer
    for name in multi_optimizers:
        assert (tmp_path / name).is_dir()
        assert (tmp_path / name / OPTIMIZER_STATE).is_file()
        assert (tmp_path / name / OPTIMIZER_PARAM_GROUPS).is_file()


def test_save_and_load_multi_optimizer_state(base_params_dict, multi_optimizers, tmp_path):
    # Option 1: Add a minimal backward pass to populate optimizer states
    for name, params in base_params_dict.items():
        if name in multi_optimizers:
            # Create a dummy loss and do backward
            dummy_loss = params[0].sum()
            dummy_loss.backward()
            # Perform an optimization step
            multi_optimizers[name].step()
            # Zero gradients for next steps
            multi_optimizers[name].zero_grad()

    # Save optimizer states
    save_optimizer_state(multi_optimizers, tmp_path)

    # Create new optimizers with the same config
    config = MultiAdamConfig(
        lr=1e-3,
        optimizer_groups={
            "actor": {"lr": 1e-4},
            "critic": {"lr": 5e-4},
            "temperature": {"lr": 2e-3},
        },
    )
    new_optimizers = config.build(base_params_dict)

    # Load optimizer states
    loaded_optimizers = load_optimizer_state(new_optimizers, tmp_path)

    # Verify state dictionaries match
    for name in multi_optimizers:
        torch.testing.assert_close(multi_optimizers[name].state_dict(), loaded_optimizers[name].state_dict())


def test_save_and_load_empty_multi_optimizer_state(base_params_dict, tmp_path):
    """Test saving and loading optimizer states even when the state is empty (no backward pass)."""
    # Create config and build optimizers
    config = MultiAdamConfig(
        lr=1e-3,
        optimizer_groups={
            "actor": {"lr": 1e-4},
            "critic": {"lr": 5e-4},
            "temperature": {"lr": 2e-3},
        },
    )
    optimizers = config.build(base_params_dict)

    # Save optimizer states without any backward pass (empty state)
    save_optimizer_state(optimizers, tmp_path)

    # Create new optimizers with the same config
    new_optimizers = config.build(base_params_dict)

    # Load optimizer states
    loaded_optimizers = load_optimizer_state(new_optimizers, tmp_path)

    # Verify hyperparameters match even with empty state
    for name, optimizer in optimizers.items():
        assert optimizer.defaults["lr"] == loaded_optimizers[name].defaults["lr"]
        assert optimizer.defaults["weight_decay"] == loaded_optimizers[name].defaults["weight_decay"]
        assert optimizer.defaults["betas"] == loaded_optimizers[name].defaults["betas"]

        # Verify state dictionaries match (they will be empty)
        torch.testing.assert_close(
            optimizer.state_dict()["param_groups"], loaded_optimizers[name].state_dict()["param_groups"]
        )

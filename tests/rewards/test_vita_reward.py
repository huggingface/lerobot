# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.factory import (
    get_reward_model_class,
    make_reward_model_config,
    make_reward_pre_post_processors,
)
from lerobot.rewards.vita.configuration_vita import VitaConfig
from lerobot.rewards.vita.modeling_vita import DummyVitaBackbone, VitaRewardModel


def _make_model() -> VitaRewardModel:
    config = VitaConfig(
        image_feature_dim=4,
        text_feature_dim=4,
        adaptation_dim=4,
        reward_hidden_dim=8,
        adaptation_lr=0.05,
    )
    backbone = DummyVitaBackbone(
        image_input_dim=4,
        text_input_dim=4,
        image_output_dim=4,
        text_output_dim=4,
    )
    return VitaRewardModel(config=config, backbone=backbone)


def test_vita_config_registry_and_factory_visibility():
    known = RewardModelConfig.get_known_choices()
    assert "vita" in known
    assert RewardModelConfig.get_choice_class("vita") is VitaConfig
    assert isinstance(make_reward_model_config("vita"), VitaConfig)

    reward_cls = get_reward_model_class("vita")
    assert reward_cls is VitaRewardModel

    pre_processor, post_processor = make_reward_pre_post_processors(VitaConfig())
    assert pre_processor.name == "vita_preprocessor"
    assert post_processor.name == "vita_postprocessor"


def test_adaptation_state_init_update_reset_without_mutating_base_weights():
    model = _make_model()
    base_weight_before = model.adaptation_module.base_weight.detach().clone()
    batch = {
        "image_features": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]),
        "text_features": torch.tensor([[0.5, 0.6, 0.7, 0.8], [0.8, 0.7, 0.6, 0.5]]),
    }

    _ = model.compute_reward(batch)
    assert model.adaptation_state is not None
    state = model.adaptation_state
    assert torch.equal(state.num_updates, torch.ones(2, dtype=torch.long))

    base_expanded = model.adaptation_module.base_weight.detach().unsqueeze(0).expand_as(state.fast_weights)
    assert not torch.allclose(state.fast_weights, base_expanded)
    assert torch.allclose(model.adaptation_module.base_weight.detach(), base_weight_before)

    model.reset_adaptation_state(batch_size=2, device=torch.device("cpu"))
    reset_state = model.adaptation_state
    assert reset_state is not None
    assert torch.equal(reset_state.num_updates, torch.zeros(2, dtype=torch.long))
    reset_base_expanded = model.adaptation_module.base_weight.detach().unsqueeze(0).expand_as(reset_state.fast_weights)
    assert torch.allclose(reset_state.fast_weights, reset_base_expanded)


def test_compute_reward_shape_type_and_determinism_with_dummy_backbone():
    model = _make_model()
    batch = {
        "image_features": torch.tensor([[0.2, 0.0, 0.1, 0.3], [0.9, 0.1, 0.2, 0.4]]),
        "text_features": torch.tensor([[0.5, 0.1, 0.2, 0.0], [0.0, 0.3, 0.2, 0.1]]),
    }

    model.reset_adaptation_state()
    reward_first = model.compute_reward(batch)
    model.reset_adaptation_state()
    reward_second = model.compute_reward(batch)

    assert reward_first.shape == (2,)
    assert reward_first.dtype == torch.float32
    assert torch.allclose(reward_first, reward_second)


def test_episode_start_resets_selected_adaptation_states():
    model = _make_model()
    batch_step1 = {
        "image_features": torch.tensor([[0.1, 0.2, 0.1, 0.0], [0.0, 0.1, 0.2, 0.3]]),
        "text_features": torch.tensor([[0.3, 0.2, 0.1, 0.0], [0.4, 0.3, 0.2, 0.1]]),
    }
    batch_step2 = {
        "image_features": torch.tensor([[0.5, 0.1, 0.0, 0.3], [0.2, 0.2, 0.2, 0.2]]),
        "text_features": torch.tensor([[0.0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4]]),
        "episode_start": torch.tensor([True, False]),
    }

    _ = model.compute_reward(batch_step1)
    _ = model.compute_reward(batch_step2)
    assert model.adaptation_state is not None
    updates = model.adaptation_state.num_updates
    assert updates.tolist() == [1, 2]


def test_forward_meta_learning_outer_loop_outputs_loss_and_metrics():
    model = _make_model()
    model.config.meta_enabled = True
    model.config.support_len = 1
    model.config.query_len = 1
    model.config.inner_steps = 2
    model.config.inner_lr = 0.05
    model.config.target_reward_key = "reward"

    batch = {
        "image_features": torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.1, 0.0]]]),
        "text_features": torch.tensor([[[0.5, 0.6, 0.7, 0.8], [0.8, 0.7, 0.6, 0.5]]]),
        "reward": torch.tensor([[0.0, 1.0]]),
    }

    loss, metrics = model.forward(batch)
    assert loss.ndim == 0
    assert "loss" in metrics
    assert "outer_loss" in metrics
    assert "inner_loss" in metrics

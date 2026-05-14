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
from lerobot.optim.schedulers import LRSchedulerConfig
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
    default_cfg = VitaConfig()
    assert default_cfg.backbone_type == "openclip"
    assert default_cfg.first_order is False
    assert default_cfg.force_temporal_progress_targets is True


def test_vita_clip_mode_validation_and_processor_creation():
    config = VitaConfig(backbone_type="clip", raw_image_key="observation.images.top", raw_text_key="task")
    config.validate_features()
    pre_processor, post_processor = make_reward_pre_post_processors(config)
    assert pre_processor.name == "vita_preprocessor"
    assert post_processor.name == "vita_postprocessor"


def test_vita_openclip_mode_validation_and_processor_creation():
    config = VitaConfig(backbone_type="openclip", raw_image_key="observation.images.top", raw_text_key="task")
    config.validate_features()
    pre_processor, post_processor = make_reward_pre_post_processors(config)
    assert pre_processor.name == "vita_preprocessor"
    assert post_processor.name == "vita_postprocessor"


def test_adaptation_state_init_update_reset_without_mutating_base_weights():
    model = _make_model()
    base_weights_before = model.adaptation_module.base_fast_weights()
    base_w1_before = base_weights_before.w1.detach().clone()
    base_b1_before = base_weights_before.b1.detach().clone()
    base_w2_before = base_weights_before.w2.detach().clone()
    base_b2_before = base_weights_before.b2.detach().clone()
    batch = {
        "image_features": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]),
        "text_features": torch.tensor([[0.5, 0.6, 0.7, 0.8], [0.8, 0.7, 0.6, 0.5]]),
    }

    _ = model.compute_reward(batch)
    assert model.adaptation_state is not None
    state = model.adaptation_state
    assert torch.equal(state.num_updates, torch.ones(2, dtype=torch.long))

    base_weights = model.adaptation_module.base_fast_weights()
    base_w1_expanded = base_weights.w1.detach().unsqueeze(0).expand_as(state.fast_weights.w1)
    assert not torch.allclose(state.fast_weights.w1, base_w1_expanded)
    assert torch.allclose(model.adaptation_module.base_fast_weights().w1.detach(), base_w1_before)
    assert torch.allclose(model.adaptation_module.base_fast_weights().b1.detach(), base_b1_before)
    assert torch.allclose(model.adaptation_module.base_fast_weights().w2.detach(), base_w2_before)
    assert torch.allclose(model.adaptation_module.base_fast_weights().b2.detach(), base_b2_before)

    model.reset_adaptation_state(batch_size=2, device=torch.device("cpu"))
    reset_state = model.adaptation_state
    assert reset_state is not None
    assert torch.equal(reset_state.num_updates, torch.zeros(2, dtype=torch.long))
    reset_base = model.adaptation_module.base_fast_weights()
    reset_w1_expanded = reset_base.w1.detach().unsqueeze(0).expand_as(reset_state.fast_weights.w1)
    assert torch.allclose(reset_state.fast_weights.w1, reset_w1_expanded)


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
    assert "self_supervised_loss" in metrics


def test_forward_dissimilarity_sampling_runs_and_returns_scalar_loss():
    model = _make_model()
    model.config.meta_enabled = True
    model.config.support_len = 2
    model.config.query_len = 2
    model.config.sampling_strategy = "dissimilarity"
    model.config.sampling_window_size = 2
    model.config.sampling_num_windows = 2
    model.config.sampling_stride = 1
    model.config.target_reward_key = "reward"

    batch = {
        "image_features": torch.tensor(
            [
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ]
        ),
        "text_features": torch.tensor(
            [
                [
                    [0.0, 0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.0, 0.1],
                    [0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0, 0.0],
                ]
            ]
        ),
        "reward": torch.tensor([[0.0, 0.2, 0.4, 0.6, 1.0]]),
    }

    loss, metrics = model.forward(batch)
    assert loss.ndim == 0
    assert metrics["outer_loss"].ndim == 0
    assert metrics["inner_loss"].ndim == 0


def test_forward_generates_progress_targets_when_missing_reward_key():
    model = _make_model()
    model.config.meta_enabled = True
    model.config.target_reward_key = "missing_reward"
    model.config.force_temporal_progress_targets = True
    batch = {
        "image_features": torch.tensor([[[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.1, 0.0]]]),
        "text_features": torch.tensor([[[0.5, 0.6, 0.7, 0.8], [0.8, 0.7, 0.6, 0.5]]]),
    }

    loss, metrics = model.forward(batch)
    assert loss.ndim == 0
    assert "outer_loss" in metrics


def test_scheduler_uses_ten_percent_warmup_ratio():
    cfg = VitaConfig()
    scheduler_cfg = cfg.get_scheduler_preset()
    assert scheduler_cfg is not None
    assert "vita_paper_cosine" in LRSchedulerConfig.get_known_choices()
    assert LRSchedulerConfig.get_choice_class("vita_paper_cosine") is type(scheduler_cfg)
    assert hasattr(scheduler_cfg, "warmup_ratio")
    assert scheduler_cfg.warmup_ratio == 0.1

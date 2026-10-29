#!/usr/bin/env python
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
"""Tests for the subclassing hooks on DiffusionPolicy and DiffusionModel.

Two properties: the hooks change nothing for Diffusion Policy itself, and a
subclass can use them to condition on an extra per-observation-step input
without rebuilding the U-Net. CPU-only, a few seconds.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalResidualBlock1d,
    DiffusionModel,
    DiffusionPolicy,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

NORM = dict.fromkeys(("STATE", "ACTION", "VISUAL", "ENV"), NormalizationMode.MEAN_STD)
STATE_DIM, ENV_DIM, ACTION_DIM, EXTRA_DIM = 6, 4, 6, 5
OBS_EXTRA = "observation.extra"


def make_config(with_camera: bool = False) -> DiffusionConfig:
    config = DiffusionConfig(horizon=8, n_action_steps=4, n_obs_steps=2, down_dims=(64, 128))
    config.input_features = {OBS_STATE: PolicyFeature(FeatureType.STATE, (STATE_DIM,))}
    if with_camera:
        config.input_features["observation.images.top"] = PolicyFeature(FeatureType.VISUAL, (3, 96, 96))
    else:
        config.input_features[OBS_ENV_STATE] = PolicyFeature(FeatureType.ENV, (ENV_DIM,))
    config.output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))}
    config.normalization_mapping = NORM
    return config


def make_batch(config: DiffusionConfig, batch_size: int = 2) -> dict[str, torch.Tensor]:
    steps = config.n_obs_steps
    return {
        OBS_STATE: torch.randn(batch_size, steps, STATE_DIM),
        OBS_ENV_STATE: torch.randn(batch_size, steps, ENV_DIM),
        ACTION: torch.randn(batch_size, config.horizon, ACTION_DIM),
        "action_is_pad": torch.zeros(batch_size, config.horizon, dtype=torch.bool),
    }


class WiderModel(DiffusionModel):
    """Conditions on one extra per-observation-step input, through the hooks only."""

    def _extra_global_cond_dim(self, config: DiffusionConfig) -> int:
        return EXTRA_DIM

    def _extra_global_cond_feats(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        return batch[OBS_EXTRA]


class WiderPolicy(DiffusionPolicy):
    def _make_diffusion_model(self, config: DiffusionConfig) -> DiffusionModel:
        return WiderModel(config)


def test_hooks_are_no_ops_for_diffusion_policy():
    """The shipped policy must be provably untouched by the hooks."""
    config = make_config(with_camera=True)
    policy = DiffusionPolicy(config)
    assert type(policy.diffusion) is DiffusionModel
    assert policy.diffusion._extra_global_cond_dim(config) == 0
    assert policy.diffusion._extra_global_cond_feats({}) is None
    assert policy.diffusion._has_conditioning_input({"observation.images": 1}) is True
    assert policy.diffusion._has_conditioning_input({"observation.state": 1}) is False


def test_subclass_model_is_built_by_the_policy():
    policy = WiderPolicy(make_config())
    assert type(policy.diffusion) is WiderModel


def film_input_widths(model: DiffusionModel) -> set[int]:
    """Input width of every FiLM projection: diffusion_step_embed_dim + global_cond_dim."""
    return {
        m.cond_encoder[1].in_features
        for m in model.unet.modules()
        if isinstance(m, DiffusionConditionalResidualBlock1d)
    }


def test_extra_width_sizes_the_unet():
    """The U-Net is built once, with the extra width times n_obs_steps already in it."""
    config = make_config()
    (plain,) = film_input_widths(DiffusionModel(config))
    (wider,) = film_input_widths(WiderModel(config))
    assert wider - plain == EXTRA_DIM * config.n_obs_steps


def test_extra_features_reach_the_loss():
    """A training step runs, and the gradient flows back into the extra input."""
    config = make_config()
    model = WiderModel(config)
    batch = make_batch(config)
    batch[OBS_EXTRA] = torch.randn(2, config.n_obs_steps, EXTRA_DIM, requires_grad=True)
    loss = model.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert batch[OBS_EXTRA].grad is not None
    assert batch[OBS_EXTRA].grad.abs().sum() > 0


def test_missing_conditioning_raises_rather_than_asserts():
    """The input check is a ValueError, so it survives `python -O`."""
    config = make_config()
    model = DiffusionModel(config)
    batch = make_batch(config)
    del batch[OBS_ENV_STATE]
    with pytest.raises(ValueError, match="images or an environment state"):
        model.compute_loss(batch)

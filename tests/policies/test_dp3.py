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
"""Tests for DP3, 3D Diffusion Policy.

These cover construction and a training step, not just tensor shapes. Two bugs
in this PR were invisible to shape-level tests and only appeared when the policy
actually ran: `compute_loss` asserted that images or an environment state were
present, which rejects the point-cloud-only variant the paper mostly reports;
and the parent flattens its conditioning to (B, D) before returning, so
per-observation-step point features could not be concatenated onto it. Both have
a regression test here.

Everything runs CPU-only in well under a minute.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
    DiffusionPolicy,
)
from lerobot.policies.dp3.configuration_dp3 import DP3Config
from lerobot.policies.dp3.modeling_dp3 import OBS_POINTCLOUD, DP3Model, DP3Policy

NORM = dict.fromkeys(("STATE", "ACTION", "VISUAL"), NormalizationMode.MEAN_STD)
STATE_DIM, ACTION_DIM = 6, 6


def make_config(with_camera: bool = False, **kwargs) -> DP3Config:
    """A small DP3 config that trains in seconds on a CPU."""
    config = DP3Config(
        pointcloud_num_points=kwargs.pop("num_points", 64),
        pointcloud_feature_dim=kwargs.pop("feature_dim", 32),
        horizon=8,
        n_action_steps=4,
        n_obs_steps=2,
        down_dims=(64, 128),
        num_inference_steps=2,
        **kwargs,
    )
    config.input_features = {"observation.state": PolicyFeature(FeatureType.STATE, (STATE_DIM,))}
    if with_camera:
        config.input_features["observation.images.top"] = PolicyFeature(FeatureType.VISUAL, (3, 96, 96))
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))}
    config.normalization_mapping = NORM
    return config


def make_batch(config: DP3Config, batch_size: int = 2) -> dict[str, torch.Tensor]:
    steps, points = config.n_obs_steps, config.pointcloud_num_points
    return {
        "observation.state": torch.randn(batch_size, steps, STATE_DIM),
        OBS_POINTCLOUD: torch.randn(batch_size, steps, points, 3),
        "action": torch.randn(batch_size, config.horizon, ACTION_DIM),
        "action_is_pad": torch.zeros(batch_size, config.horizon, dtype=torch.bool),
    }


def test_config_is_registered():
    from lerobot.configs import PreTrainedConfig

    assert PreTrainedConfig.get_choice_class("dp3") is DP3Config
    assert DP3Policy.name == "dp3"


def test_config_rejects_bad_point_settings():
    with pytest.raises(ValueError, match="pointcloud_channels"):
        DP3Config(pointcloud_channels=4)
    with pytest.raises(ValueError, match="pointcloud_num_points"):
        DP3Config(pointcloud_num_points=0)


@pytest.mark.parametrize("with_camera", [False, True])
def test_constructs_in_both_variants(with_camera):
    """Point-cloud-only and RGB-D. The former is the paper's main variant."""
    policy = DP3Policy(make_config(with_camera=with_camera))
    assert isinstance(policy.diffusion, DP3Model)
    assert sum(p.numel() for p in policy.parameters()) > 0


def test_exactly_one_unet_is_built():
    """The hook exists so the U-Net is sized once, not built and rebuilt.

    A Diffusion Policy U-Net is hundreds of millions of parameters; constructing
    it twice to discover the conditioning width is wrong would double peak
    memory at init for nothing.
    """
    policy = DP3Policy(make_config())
    n_unets = sum(1 for m in policy.modules() if isinstance(m, DiffusionConditionalUnet1d))
    assert n_unets == 1


def test_global_cond_dim_is_the_expected_arithmetic():
    """Spelled out, because a silent mismatch here is a shape error deep in the U-Net."""
    config = make_config(feature_dim=32)
    state_width = config.robot_state_feature.shape[0]
    point_width = config.pointcloud_feature_dim
    expected_per_step = state_width + point_width
    expected_total = expected_per_step * config.n_obs_steps

    assert state_width == STATE_DIM
    assert point_width == 32
    assert expected_total == (6 + 32) * 2 == 76

    model = DP3Model(config)
    assert model._extra_global_cond_dim(config) == point_width


def test_forward_and_backward_reach_the_point_encoder():
    policy = DP3Policy(make_config())
    loss, _ = policy.forward(make_batch(policy.config))
    assert loss.ndim == 0 and torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in policy.diffusion.pointcloud_encoder.parameters()]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads), "point encoder received no gradient"


def test_point_cloud_only_batch_is_accepted():
    """Regression: `compute_loss` used to require images or an environment state.

    That assertion rejected the point-cloud-only configuration outright, so the
    paper's main variant could be constructed but never trained.
    """
    policy = DP3Policy(make_config())
    batch = make_batch(policy.config)
    assert "observation.images" not in batch
    loss, _ = policy.forward(batch)
    assert torch.isfinite(loss)


def test_conditioning_keeps_one_row_per_observation_step():
    """Regression: the parent flattens conditioning to (B, D) before returning.

    Point features are (B, n_obs_steps, D) and must be appended before that
    flatten, both so the concatenation is legal and so each observation step's
    features stay contiguous.
    """
    policy = DP3Policy(make_config(feature_dim=32))
    batch = make_batch(policy.config)
    extra = policy.diffusion._extra_global_cond_feats(batch)
    assert extra.shape == (2, policy.config.n_obs_steps, 32)


def test_each_observation_step_is_encoded_independently():
    """Mixing steps would leak a later observation into an earlier step."""
    policy = DP3Policy(make_config(feature_dim=32))
    policy.eval()
    batch = make_batch(policy.config, batch_size=3)
    with torch.no_grad():
        folded = policy.diffusion._extra_global_cond_feats(batch)
        for b in (0, 2):
            for s in range(policy.config.n_obs_steps):
                alone = policy.diffusion.pointcloud_encoder(batch[OBS_POINTCLOUD][b, s][None])[0]
                assert torch.allclose(alone, folded[b, s], atol=1e-5)


def test_select_action_survives_a_queue_refill():
    """The action queue holds n_action_steps; stepping past it must re-plan."""
    policy = DP3Policy(make_config())
    policy.reset()
    obs = {
        "observation.state": torch.randn(2, STATE_DIM),
        OBS_POINTCLOUD: torch.randn(2, policy.config.pointcloud_num_points, 3),
    }
    actions = []
    for _ in range(policy.config.n_action_steps + 3):
        with torch.no_grad():
            actions.append(policy.select_action(dict(obs)))
    assert len(actions) == policy.config.n_action_steps + 3
    assert all(a.shape == (2, ACTION_DIM) for a in actions)
    assert all(torch.isfinite(a).all() for a in actions)


def test_overfits_a_single_batch():
    """If the conditioning is wired wrong, no amount of training will show it.

    The action here is a deterministic function of the cloud, so a correctly
    conditioned model must be able to memorise it.
    """
    torch.manual_seed(0)
    policy = DP3Policy(make_config(num_points=64, feature_dim=32))
    policy.train()
    optimiser = torch.optim.AdamW(policy.parameters(), lr=1e-3)

    cloud = torch.randn(4, policy.config.n_obs_steps, 64, 3)
    target = cloud[:, -1].mean(1)
    action = target.repeat(1, 2)[:, None, :].repeat(1, policy.config.horizon, 1)
    batch = {
        "observation.state": torch.randn(4, policy.config.n_obs_steps, STATE_DIM),
        OBS_POINTCLOUD: cloud,
        "action": action,
        "action_is_pad": torch.zeros(4, policy.config.horizon, dtype=torch.bool),
    }

    first = None
    for step in range(300):
        loss, _ = policy.forward({k: v.clone() for k, v in batch.items()})
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step == 0:
            first = loss.item()
    assert loss.item() < first * 0.25, f"loss only fell {first:.3f} -> {loss.item():.3f}"


def test_hooks_are_no_ops_for_diffusion_policy():
    """The shipped policy must be provably untouched by this PR's hooks."""
    config = DiffusionConfig(horizon=8, n_action_steps=4, n_obs_steps=2, down_dims=(64, 128))
    config.input_features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (STATE_DIM,)),
        "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
    }
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))}
    config.normalization_mapping = NORM

    policy = DiffusionPolicy(config)
    assert type(policy.diffusion) is DiffusionModel
    assert policy.diffusion._extra_global_cond_dim(config) == 0
    assert policy.diffusion._extra_global_cond_feats({}) is None
    assert policy.diffusion._has_conditioning_input({"observation.images": 1}) is True
    assert policy.diffusion._has_conditioning_input({"observation.state": 1}) is False


# --- factory integration -----------------------------------------------------
#
# Every test above constructs DP3Policy directly, which imports the config
# module as a side effect and so registers it. That masked two real gaps: the
# config was not imported from `lerobot.policies.__init__`, so `dp3` was absent
# from the registry and `--policy.type=dp3` could not resolve it at all; and
# there was no `processor_dp3` module, so the factory's convention lookup found
# no processors. The documented training command would have failed on a clean
# checkout while the whole suite passed. These tests go through the factory.


def test_dp3_is_registered_by_the_policies_package():
    """`--policy.type=dp3` resolves only if `lerobot.policies` imports the config."""
    import lerobot.policies as policies

    assert hasattr(policies, "DP3Config")

    from lerobot.configs import PreTrainedConfig

    assert "dp3" in PreTrainedConfig.get_known_choices()


def test_factory_resolves_the_policy_class():
    from lerobot.policies.factory import get_policy_class

    assert get_policy_class("dp3") is DP3Policy


def test_factory_builds_processors():
    """The convention lookup needs `processor_dp3.make_dp3_pre_post_processors`."""
    from lerobot.policies.factory import make_pre_post_processors

    config = make_config()
    preprocessor, postprocessor = make_pre_post_processors(config)
    assert preprocessor is not None and postprocessor is not None


def test_depth_step_is_in_the_pipeline_by_default():
    """`--policy.type=dp3` on a depth dataset must not need a hand-built pipeline."""
    from lerobot.policies.dp3.processor_dp3 import make_dp3_pre_post_processors
    from lerobot.processor.depth_processor import DepthToPointCloudStep

    preprocessor, _ = make_dp3_pre_post_processors(make_config())
    assert any(isinstance(s, DepthToPointCloudStep) for s in preprocessor.steps)


def test_depth_step_can_be_turned_off():
    from lerobot.policies.dp3.processor_dp3 import make_dp3_pre_post_processors
    from lerobot.processor.depth_processor import DepthToPointCloudStep

    config = make_config()
    config.pointcloud_from_depth = False
    preprocessor, _ = make_dp3_pre_post_processors(config)
    assert not any(isinstance(s, DepthToPointCloudStep) for s in preprocessor.steps)


def test_depth_step_defaults_to_millimetres():
    """LeRobotDataset dequantises depth to millimetres; the default must match."""
    config = make_config()
    assert config.pointcloud_depth_scale == 1e-3


def test_config_rejects_bad_pointcloud_frame():
    with pytest.raises(ValueError, match="pointcloud_frame"):
        DP3Config(pointcloud_frame="elbow")


def test_config_rejects_half_a_workspace_crop():
    with pytest.raises(ValueError, match="must be set together"):
        DP3Config(pointcloud_workspace_extent=0.6)

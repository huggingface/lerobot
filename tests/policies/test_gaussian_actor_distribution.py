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

import pytest
import torch
from torch.distributions import Independent, Normal, TanhTransform, TransformedDistribution

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import (
    ActorNetworkConfig,
    GaussianActorConfig,
    PolicyConfig,
)
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import (
    GaussianActorPolicy,
    TanhMultivariateNormalDiag,
)
from lerobot.utils.constants import ACTION, OBS_STATE


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
def test_tanh_gaussian_uses_diagonal_standard_deviations(dtype, batch_shape):
    loc = torch.zeros(*batch_shape, 3, dtype=dtype)
    scale_diag = torch.tensor([0.125, 0.5, 2.0], dtype=dtype)
    dist = TanhMultivariateNormalDiag(loc, scale_diag)

    expected_std = scale_diag.expand_as(loc)
    torch.testing.assert_close(dist.base_dist.stddev, expected_std)
    torch.testing.assert_close(dist.base_dist.covariance_matrix, torch.diag_embed(expected_std.square()))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_tanh_gaussian_matches_independent_normals_and_gradients(dtype, batch_shape):
    loc = torch.tensor([-0.2, 0.1, 0.3], dtype=dtype).expand(*batch_shape, 3).clone().requires_grad_()
    log_std = torch.tensor([0.25, 0.5, 1.5], dtype=dtype).log().requires_grad_()
    std = log_std.exp()
    actual = TanhMultivariateNormalDiag(loc, std)
    expected = TransformedDistribution(Independent(Normal(loc, std), 1), [TanhTransform(cache_size=1)])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        actions = actual.rsample((5,))
        torch.manual_seed(7)
        expected_actions = expected.rsample((5,))

    torch.testing.assert_close(actions, expected_actions)
    log_prob = actual.log_prob(actions)
    expected_log_prob = expected.log_prob(expected_actions)
    torch.testing.assert_close(log_prob, expected_log_prob)
    assert log_prob.shape == (5, *batch_shape)

    loss = actions.sum() + log_prob.sum()
    expected_loss = expected_actions.sum() + expected_log_prob.sum()
    grads = torch.autograd.grad(loss, (loc, log_std), retain_graph=True)
    expected_grads = torch.autograd.grad(expected_loss, (loc, log_std))
    for grad, expected_grad in zip(grads, expected_grads, strict=True):
        torch.testing.assert_close(grad, expected_grad)
        assert torch.isfinite(grad).all()


@pytest.mark.parametrize("std_mode", ["learned", "clipped", "fixed"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_gaussian_actor_forward_uses_standard_deviations(std_mode, batch_size):
    policy_kwargs = PolicyConfig(std_min=0.5, std_max=1.0) if std_mode == "clipped" else PolicyConfig()
    config = GaussianActorConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        actor_network_kwargs=ActorNetworkConfig(hidden_dims=[8]),
        policy_kwargs=policy_kwargs,
        latent_dim=8,
        shared_encoder=False,
        device="cpu",
    )
    policy = GaussianActorPolicy(config)
    std = torch.tensor([0.25, 0.5, 1.5])
    with torch.no_grad():
        policy.actor.mean_layer.weight.zero_()
        policy.actor.mean_layer.bias.copy_(torch.tensor([-0.2, 0.1, 0.3]))
        policy.actor.std_layer.weight.zero_()
        policy.actor.std_layer.bias.copy_(std.log())
    if std_mode == "fixed":
        policy.actor.fixed_std = std
    else:
        std = std.clamp(policy_kwargs.std_min, policy_kwargs.std_max)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        output = policy({OBS_STATE: torch.zeros(batch_size, 2)})
        expected = TransformedDistribution(
            Independent(Normal(output["action_mean"], std), 1), [TanhTransform(cache_size=1)]
        )
        torch.manual_seed(7)
        expected_actions = expected.rsample()

    torch.testing.assert_close(output["action"], expected_actions)
    torch.testing.assert_close(output["log_prob"], expected.log_prob(expected_actions))
    assert output["action"].shape == (batch_size, 3)
    assert output["log_prob"].shape == (batch_size,)

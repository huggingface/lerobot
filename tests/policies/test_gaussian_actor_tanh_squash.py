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

"""Regression: Policy.use_tanh_squash must actually select the action distribution."""

import torch
from torch import nn

from lerobot.policies.gaussian_actor.modeling_gaussian_actor import MLP, Policy


class _PassthroughEncoder(nn.Module):
    """Minimal encoder stub: returns the observation tensor unchanged."""

    def __init__(self, dim: int):
        super().__init__()
        self._out_dim = dim
        self.has_images = False

    def forward(self, observations, cache=None, detach: bool = False):
        if isinstance(observations, dict):
            return next(iter(observations.values()))
        return observations

    @property
    def output_dim(self) -> int:
        return self._out_dim


def _make_policy(*, use_tanh_squash: bool, feat_dim: int = 4, action_dim: int = 2) -> Policy:
    encoder = _PassthroughEncoder(feat_dim)
    network = MLP(input_dim=feat_dim, hidden_dims=[32, 32], activate_final=True)
    return Policy(
        encoder=encoder,
        network=network,
        action_dim=action_dim,
        std_min=1e-5,
        std_max=10.0,
        init_final=0.05,
        use_tanh_squash=use_tanh_squash,
        encoder_is_shared=False,
    )


def test_use_tanh_squash_false_allows_actions_outside_unit_interval():
    """With large pre-squash means, False escapes [-1, 1]; True stays bounded."""
    torch.manual_seed(0)
    policy_squash = _make_policy(use_tanh_squash=True)
    policy_plain = _make_policy(use_tanh_squash=False)
    policy_plain.load_state_dict(policy_squash.state_dict())
    # Flag lives on the module, not in state_dict — restore after weight copy.
    policy_plain.use_tanh_squash = False

    # Push means far outside [-1, 1] so squashing vs not is unambiguous.
    with torch.no_grad():
        policy_squash.mean_layer.bias.fill_(5.0)
        policy_plain.mean_layer.bias.fill_(5.0)
        policy_squash.std_layer.bias.fill_(-2.0)  # std ≈ exp(-2) ≈ 0.14
        policy_plain.std_layer.bias.fill_(-2.0)

    policy_squash.eval()
    policy_plain.eval()

    batch_size, feat_dim = 512, 4
    observations = {"obs": torch.zeros(batch_size, feat_dim)}

    with torch.no_grad():
        actions_squash, log_probs_squash, means_squash = policy_squash(observations)
        actions_plain, log_probs_plain, means_plain = policy_plain(observations)

    assert torch.allclose(means_squash, means_plain)
    assert (means_squash.abs() > 1.0).all()

    assert (actions_squash.abs() <= 1.0 + 1e-5).all()
    assert (actions_plain.abs() > 1.0).any(), (
        f"expected unsquashed samples outside [-1, 1], max|a|={actions_plain.abs().max().item()}"
    )
    # Tanh Jacobian correction changes the log-density; distributions must differ.
    assert not torch.allclose(log_probs_squash, log_probs_plain, atol=1e-3)


def test_use_tanh_squash_true_is_default_bounded_behavior():
    """Default True path still returns tanh-bounded actions for large means."""
    torch.manual_seed(1)
    policy = _make_policy(use_tanh_squash=True)
    with torch.no_grad():
        policy.mean_layer.bias.fill_(8.0)
        policy.std_layer.bias.fill_(0.0)

    observations = {"obs": torch.randn(64, 4)}
    with torch.no_grad():
        actions, _, _ = policy(observations)

    assert (actions.abs() <= 1.0 + 1e-5).all()

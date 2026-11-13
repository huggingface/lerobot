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
"""Regression tests for `ACTPolicy.forward(reduction=...)`.

`update_policy` calls `policy.forward(batch, reduction="none")` whenever a `SampleWeighter`
is configured (see `lerobot.utils.sample_weighting`), so the per-sample contract of ACT's
loss is load-bearing for RA-BC style training.

These tests pin the per-sample shape, the `action_is_pad` masking, the fully-padded edge
case (finite, zero), the VAE KL routing, the accepted values of `reduction`, and the
documented (non-)relationship between the scalar and per-sample reductions.

The policy is built with `object.__new__` and a stubbed `model`, so no weights are
constructed or loaded and the loss is a pure function of the batch - the same idiom as the
molmoact2 reduction tests.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION

JUNK = 99.0  # value written into padded slots; must never contribute to the loss


class _StubACT(nn.Module):
    """Returns a canned action chunk so the loss is a pure function of the batch."""

    def __init__(self, actions_hat: torch.Tensor, mu=None, log_sigma_x2=None):
        super().__init__()
        self.actions_hat = actions_hat
        self.mu = mu
        self.log_sigma_x2 = log_sigma_x2

    def forward(self, batch: dict) -> tuple:
        del batch
        return self.actions_hat, (self.mu, self.log_sigma_x2)


def _make_policy(
    actions_hat: torch.Tensor,
    *,
    use_vae: bool = False,
    kl_weight: float = 10.0,
    mu: torch.Tensor | None = None,
    log_sigma_x2: torch.Tensor | None = None,
) -> ACTPolicy:
    policy = object.__new__(ACTPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(image_features=None, use_vae=use_vae, kl_weight=kl_weight)
    policy.model = _StubACT(actions_hat, mu, log_sigma_x2)
    return policy


def _batch(actions_hat: torch.Tensor, action_is_pad: torch.Tensor) -> dict:
    # Targets are all zeros, so abs_err == |actions_hat|. `forward` only reads ACTION and
    # action_is_pad here because the model is stubbed.
    return {ACTION: torch.zeros_like(actions_hat), "action_is_pad": action_is_pad}


def _pad(*rows: int, length: int = 4) -> torch.Tensor:
    """`*rows` gives the number of valid (unpadded) leading timesteps per sample."""
    return torch.tensor([[t >= n for t in range(length)] for n in rows])


def test_reduction_none_returns_per_sample_losses():
    actions_hat = torch.tensor([[[0.0], [1.0], [1.0], [1.0]], [[2.0], [2.0], [2.0], [2.0]]])
    policy = _make_policy(actions_hat)

    loss, metrics = policy.forward(_batch(actions_hat, _pad(4, 4)), reduction="none")

    assert loss.shape == (2,)
    assert loss.dtype == torch.float32
    assert loss[0].item() == pytest.approx(0.75)
    assert loss[1].item() == pytest.approx(2.0)
    # metrics["l1_loss"] is the mean of the per-sample losses in this branch.
    assert metrics["l1_loss"] == pytest.approx(loss.mean().item())


def test_padded_timesteps_are_excluded_from_each_samples_loss():
    # Sample 0's valid errors are [0, 1] and sample 1's are [2, 2]; the trailing JUNK
    # entries are padded and must be masked out of both the numerator and the denominator.
    actions_hat = torch.tensor([[[0.0], [1.0], [JUNK], [JUNK]], [[2.0], [2.0], [JUNK], [JUNK]]])
    policy = _make_policy(actions_hat)
    batch = _batch(actions_hat, _pad(2, 2))

    per_sample, _ = policy.forward(batch, reduction="none")
    scalar, _ = policy.forward(batch)

    assert per_sample[0].item() == pytest.approx((0.0 + 1.0) / 2)
    assert per_sample[1].item() == pytest.approx((2.0 + 2.0) / 2)
    # Both samples have the same number of valid entries, so the two reductions agree here.
    assert scalar.item() == pytest.approx(per_sample.mean().item())
    assert scalar.item() == pytest.approx((0.0 + 1.0 + 2.0 + 2.0) / 4)


@pytest.mark.parametrize("use_vae", [False, True])
def test_padded_values_never_influence_the_loss(use_vae):
    """The padded slots hold arbitrary data; changing them must not move either reduction."""
    observed = []
    for junk in (0.0, JUNK, -3.5):
        actions_hat = torch.tensor(
            [[[0.0], [1.0], [junk], [junk]], [[2.0], [2.0], [junk], [junk]]], dtype=torch.float32
        )
        mu = torch.tensor([[1.0, 2.0], [0.0, 0.0]]) if use_vae else None
        log_sigma_x2 = torch.zeros(2, 2) if use_vae else None
        policy = _make_policy(actions_hat, use_vae=use_vae, mu=mu, log_sigma_x2=log_sigma_x2)
        batch = _batch(actions_hat, _pad(2, 2))

        per_sample, _ = policy.forward(batch, reduction="none")
        scalar, _ = policy.forward(batch)
        observed.append((per_sample.tolist(), scalar.item()))

    assert all(result == observed[0] for result in observed)


@pytest.mark.parametrize("use_vae", [False, True])
def test_fully_padded_sample_contributes_no_l1_and_stays_finite(use_vae):
    # Sample 0 is entirely padded (a degenerate episode tail): its reconstruction loss must be
    # exactly zero and finite rather than NaN from dividing by an empty mask. The per-sample KL
    # is still applied, mirroring the scalar path where every sample contributes its KL to the
    # batch mean. Sample 0's KL here is 2.5 (mu=(1, 2), log_sigma_x2=0) and sample 1's is 0.
    kld0 = 2.5
    actions_hat = torch.tensor(
        [[[JUNK], [JUNK], [JUNK], [JUNK]], [[1.0], [1.0], [1.0], [1.0]]], dtype=torch.float32
    )
    mu = torch.tensor([[1.0, 2.0], [0.0, 0.0]]) if use_vae else None
    log_sigma_x2 = torch.zeros(2, 2) if use_vae else None
    policy = _make_policy(actions_hat, use_vae=use_vae, kl_weight=10.0, mu=mu, log_sigma_x2=log_sigma_x2)

    loss, _ = policy.forward(_batch(actions_hat, _pad(0, 4)), reduction="none")

    assert torch.isfinite(loss).all()
    # No l1 contribution at all, so only the KL survives for the fully padded sample.
    assert loss[0].item() == pytest.approx(10.0 * kld0 if use_vae else 0.0)
    assert loss[1].item() == pytest.approx(1.0)


def test_fully_padded_batch_yields_finite_zero_loss():
    actions_hat = torch.full((2, 4, 1), JUNK)
    policy = _make_policy(actions_hat)
    batch = _batch(actions_hat, _pad(0, 0))

    per_sample, _ = policy.forward(batch, reduction="none")
    scalar, _ = policy.forward(batch)

    assert torch.isfinite(per_sample).all()
    assert per_sample.tolist() == [0.0, 0.0]
    assert scalar.item() == 0.0


@pytest.mark.parametrize("use_vae", [False, True])
def test_reduction_none_matches_scalar_when_padding_is_uniform(use_vae):
    actions_hat = torch.tensor([[[1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [1.0]]])
    mu = torch.tensor([[1.0, 2.0], [0.0, 0.0]]) if use_vae else None
    log_sigma_x2 = torch.zeros(2, 2) if use_vae else None
    policy = _make_policy(actions_hat, use_vae=use_vae, mu=mu, log_sigma_x2=log_sigma_x2)
    batch = _batch(actions_hat, _pad(4, 4))

    per_sample, _ = policy.forward(batch, reduction="none")
    scalar, _ = policy.forward(batch)

    assert per_sample.shape == (2,)
    assert scalar.ndim == 0
    assert per_sample.mean().item() == pytest.approx(scalar.item())


def test_vae_kl_is_routed_per_sample_and_scaled_by_kl_weight():
    # l1 is 1.0 for both samples; kld is 2.5 for sample 0 and 0.0 for sample 1
    # (mu=0, log_sigma_x2=0 gives zero KL).
    actions_hat = torch.tensor([[[1.0], [1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0], [1.0]]])
    mu = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    log_sigma_x2 = torch.zeros(2, 2)
    policy = _make_policy(actions_hat, use_vae=True, kl_weight=10.0, mu=mu, log_sigma_x2=log_sigma_x2)
    batch = _batch(actions_hat, _pad(4, 4))

    per_sample, metrics = policy.forward(batch, reduction="none")
    scalar, scalar_metrics = policy.forward(batch)

    assert per_sample[0].item() == pytest.approx(1.0 + 10.0 * 2.5)
    assert per_sample[1].item() == pytest.approx(1.0 + 10.0 * 0.0)
    # KL is already reduced over the batch in the scalar path, so its per-sample mean is exact.
    assert metrics["kld_loss"] == pytest.approx((2.5 + 0.0) / 2)
    assert scalar_metrics["kld_loss"] == pytest.approx(metrics["kld_loss"])
    assert scalar.item() == pytest.approx(per_sample.mean().item())


def test_reduction_none_does_not_add_kl_without_vae():
    actions_hat = torch.ones(2, 4, 1)
    policy = _make_policy(actions_hat, use_vae=False)

    _, metrics = policy.forward(_batch(actions_hat, _pad(4, 4)), reduction="none")

    assert set(metrics) == {"l1_loss"}


def test_per_sample_mean_differs_from_scalar_under_ragged_padding():
    """Documents the intended relationship: the two reductions are NOT interchangeable.

    The scalar path weights every valid (time, action) entry equally; the per-sample path
    weights every sample equally, so their batch means only coincide for uniform padding.
    """
    # Sample 0 has 2 valid entries with zero error; sample 1 has all 4 valid with error 1.
    actions_hat = torch.tensor([[[0.0], [0.0], [JUNK], [JUNK]], [[1.0], [1.0], [1.0], [1.0]]])
    policy = _make_policy(actions_hat)
    batch = _batch(actions_hat, _pad(2, 4))

    per_sample, _ = policy.forward(batch, reduction="none")
    scalar, _ = policy.forward(batch)

    assert per_sample.tolist() == [0.0, 1.0]
    assert per_sample.mean().item() == pytest.approx(0.5)
    # Global masked mean over the 6 valid entries: 4 / 6, not 0.5.
    assert scalar.item() == pytest.approx(4 / 6)
    assert not torch.isclose(per_sample.mean(), scalar)


def test_default_reduction_is_the_scalar_mean():
    actions_hat = torch.tensor([[[0.0], [1.0], [JUNK], [JUNK]], [[2.0], [2.0], [JUNK], [JUNK]]])
    policy = _make_policy(actions_hat)
    batch = _batch(actions_hat, _pad(2, 2))

    implicit, implicit_metrics = policy.forward(batch)
    explicit, explicit_metrics = policy.forward(batch, reduction="mean")

    assert implicit.ndim == 0
    assert implicit.item() == explicit.item() == pytest.approx(1.25)
    assert implicit_metrics == explicit_metrics


@pytest.mark.parametrize("bad_reduction", ["sum", "batchmean", "NONE", "None", ""])
def test_invalid_reduction_raises_value_error(bad_reduction):
    actions_hat = torch.ones(2, 4, 1)
    policy = _make_policy(actions_hat)

    with pytest.raises(ValueError, match="Unsupported reduction"):
        policy.forward(_batch(actions_hat, _pad(4, 4)), reduction=bad_reduction)

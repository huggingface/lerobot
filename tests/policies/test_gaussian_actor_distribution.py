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
from torch.distributions import TanhTransform, TransformedDistribution

from lerobot.policies.gaussian_actor.modeling_gaussian_actor import TanhMultivariateNormalDiag


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (3,)])
@pytest.mark.parametrize("low,high", [(2.0, 4.0), ([-2.0, 1.0], [0.0, 5.0])])
def test_tanh_normal_custom_bounds(
    dtype: torch.dtype, batch_shape: tuple[int, ...], low: float | list[float], high: float | list[float]
) -> None:
    loc = torch.tensor([0.0, 0.25], dtype=dtype).expand(*batch_shape, 2).clone().requires_grad_()
    dist = TanhMultivariateNormalDiag(loc, torch.ones_like(loc), low=low, high=high)
    low = torch.as_tensor(low, dtype=dtype)
    high = torch.as_tensor(high, dtype=dtype)
    half_range = (high - low) / 2
    midpoint = (high + low) / 2

    torch.testing.assert_close(dist.mode(), midpoint + half_range * loc.tanh())
    assert dist.support.check(dist.mode()).all()

    z = torch.tensor([0.1, -0.2], dtype=dtype).expand_as(loc)
    actions = midpoint + half_range * z.tanh()
    expected_log_prob = dist.base_dist.log_prob(z) - (
        torch.log(half_range) + torch.log1p(-z.tanh().square())
    ).sum(dim=-1)
    actual_log_prob = dist.log_prob(actions)
    assert actual_log_prob.shape == batch_shape
    torch.testing.assert_close(actual_log_prob, expected_log_prob)

    samples = dist.rsample((4,))
    assert samples.dtype == dtype
    assert ((samples >= low) & (samples <= high)).all()
    samples.sum().backward()
    assert loc.grad is not None
    assert torch.isfinite(loc.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tanh_normal_default_bounds(dtype: torch.dtype) -> None:
    loc = torch.tensor([[0.1, -0.2]], dtype=dtype)
    dist = TanhMultivariateNormalDiag(loc, torch.ones_like(loc))
    reference = TransformedDistribution(dist.base_dist, [TanhTransform(cache_size=1)])
    torch.testing.assert_close(dist.mode(), loc.tanh())
    torch.testing.assert_close(dist.log_prob(loc.tanh()), reference.log_prob(loc.tanh()))

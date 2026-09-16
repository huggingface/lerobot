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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.policies.pi052.modeling_pi052 import PI052Policy  # noqa: E402
from lerobot.utils.logging_utils import MetricsTracker  # noqa: E402


@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_component_metrics_are_numeric_without_detaching_training_loss(monkeypatch, reduction):
    policy = PI052Policy.__new__(PI052Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(flow_loss_weight=10.0, text_loss_weight=1.0)
    shape = () if reduction == "mean" else (2,)
    flow = torch.full(shape, 0.2, requires_grad=True)
    text = torch.full(shape, 0.3, requires_grad=True)
    monkeypatch.setattr(policy, "_compute_all_losses_fused", lambda *args, **kwargs: (flow, text, None))
    loss, metrics = policy(
        {"predict_actions": torch.tensor([True]), "text_labels": torch.tensor([[1]])}, reduction=reduction
    )
    tracker = MetricsTracker(1, 10, 1, {})
    tracker.update_metrics(metrics)
    assert all(isinstance(v, float) for v in metrics.values())
    assert tracker.flow_loss.avg == pytest.approx(0.2)
    assert tracker.text_loss.avg == pytest.approx(0.3)
    assert loss.shape == shape
    loss.sum().backward()
    torch.testing.assert_close(flow.grad, torch.full(shape, 10.0))
    torch.testing.assert_close(text.grad, torch.ones(shape))

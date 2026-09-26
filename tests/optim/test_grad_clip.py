"""Group clipping isolates expert spikes and rejects bad updates before mutation."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from lerobot.optim.grad_clip import clip_grad_norm_with_groups_


def parameter_with_grad(values):
    parameter = nn.Parameter(torch.ones(len(values)))
    parameter.grad = torch.tensor(values, dtype=torch.float32)
    return parameter


def test_large_finite_conditioning_gradient_does_not_suppress_backbone():
    conditioning = parameter_with_grad([1e30, -1e30])
    expert = parameter_with_grad([1e20])
    backbone = parameter_with_grad([3.0, 4.0])
    params = [conditioning, expert, backbone]
    groups = [
        {"params": [conditioning], "name": "conditioning", "grad_clip_norm": 0.1},
        {"params": [expert], "name": "action_expert", "grad_clip_norm": 1.0},
        {"params": [backbone], "name": "backbone"},
    ]
    norm, metrics = clip_grad_norm_with_groups_(params, groups, 1.0)
    assert torch.isfinite(norm) and norm > 1e30
    assert conditioning.grad.norm() <= 0.1
    assert expert.grad.norm() <= 1.0
    assert backbone.grad.norm() > 0.95
    combined = torch.cat([p.grad for p in params])
    assert combined.norm().item() == pytest.approx(1.0, abs=1e-6)
    assert metrics["grad_scale_backbone"] > 0.19


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_nonfinite_in_later_group_does_not_mutate_any_gradients(bad):
    good, faulty = parameter_with_grad([3.0, 4.0]), parameter_with_grad([bad])
    with pytest.raises(FloatingPointError, match="refusing optimizer"):
        clip_grad_norm_with_groups_(
            [good, faulty], [{"params": [good], "grad_clip_norm": 0.1}, {"params": [faulty]}], 1.0
        )
    torch.testing.assert_close(good.grad, torch.tensor([3.0, 4.0]))


def test_group_only_clipping_when_global_clipping_disabled():
    a, b = parameter_with_grad([3.0, 4.0]), parameter_with_grad([10.0])
    clip_grad_norm_with_groups_([a, b], [{"params": [a], "grad_clip_norm": 0.1}, {"params": [b]}], 0)
    assert a.grad.norm().item() == pytest.approx(0.1)
    assert b.grad.item() == 10


def test_group_coverage_and_overlap_are_rejected():
    a, b = parameter_with_grad([1.0]), parameter_with_grad([2.0])
    with pytest.raises(ValueError, match="cover every"):
        clip_grad_norm_with_groups_([a, b], [{"params": [a]}], 1)
    with pytest.raises(ValueError, match="not overlap"):
        clip_grad_norm_with_groups_([a], [{"params": [a]}, {"params": [a]}], 1)


def test_trainer_refuses_nonfinite_update_and_preserves_optimizer_scheduler():
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.logging_utils import MetricsTracker

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))

        def forward(self, _batch):
            return self.weight.sum(), {}

    policy = Policy()
    optimizer = torch.optim.AdamW([{"params": policy.parameters(), "grad_clip_norm": 0.1}], lr=1e-3)
    scheduler = Mock()
    accelerator = SimpleNamespace(
        distributed_type=SimpleNamespace(value="NO"),
        scaler=None,
        sync_gradients=True,
        accumulate=lambda model: nullcontext(),
        autocast=nullcontext,
        backward=lambda loss: loss.backward(),
        unscale_gradients=lambda optimizer: None,
    )
    with pytest.raises(FloatingPointError):
        update_policy(MetricsTracker(1, 1, 1, {}), policy, {}, optimizer, 1, accelerator, scheduler)
    torch.testing.assert_close(policy.weight, torch.ones(2))
    assert not optimizer.state
    scheduler.step.assert_not_called()


def _ddp_worker(rank, init_path):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        model = nn.Linear(1, 1)
        model.weight.data.zero_()
        model.bias.data.zero_()
        ddp = DistributedDataParallel(model)
        # One replica sees the spike; clipping must use the averaged gradient.
        ddp(torch.tensor([[1e20 if rank == 0 else 1.0]])).sum().backward()
        raw, _ = clip_grad_norm_with_groups_(
            ddp.parameters(),
            [{"params": [model.weight], "grad_clip_norm": 0.1}, {"params": [model.bias]}],
            1.0,
        )
        assert raw.item() == pytest.approx(5e19, rel=1e-5)
        assert model.bias.grad.item() > 0.99
        flat = torch.cat([p.grad.flatten() for p in model.parameters()])
        gathered = [torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(gathered, flat)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_ddp_clips_synchronized_gradients_identically(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo unavailable")
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path / "ddp-init"),), nprocs=2, join=True)


def test_trainer_applies_limits_and_reports_raw_norm_on_success():
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.expert = nn.Parameter(torch.zeros(1))
            self.backbone = nn.Parameter(torch.zeros(1))

        def forward(self, _batch):
            return (self.expert * 1e20 + self.backbone).sum(), {}

    policy = Policy()
    optimizer = torch.optim.SGD(
        [
            {"params": [policy.expert], "name": "expert", "grad_clip_norm": 0.1},
            {"params": [policy.backbone], "name": "backbone"},
        ],
        lr=0.01,
    )
    scheduler = Mock()
    accelerator = SimpleNamespace(
        distributed_type=SimpleNamespace(value="NO"),
        scaler=None,
        sync_gradients=True,
        accumulate=lambda model: nullcontext(),
        autocast=nullcontext,
        backward=lambda loss: loss.backward(),
        unscale_gradients=lambda optimizer: None,
        unwrap_model=lambda model, **kwargs: model,
    )
    meters = {name: AverageMeter(name) for name in ("loss", "grad_norm", "lr", "update_s", "gpu_mem_gb")}
    _, metrics = update_policy(
        MetricsTracker(1, 1, 1, meters), policy, {}, optimizer, 1, accelerator, scheduler
    )
    assert metrics["grad_norm_expert"] > 1e19
    assert metrics["grad_scale_backbone"] > 0.99
    assert abs(policy.backbone.item()) > 0.0099
    assert abs(policy.expert.item()) < 0.001
    scheduler.step.assert_called_once()

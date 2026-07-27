#!/usr/bin/env python

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

import pytest
import torch

from lerobot.utils.logging_utils import AverageMeter, MetricsTracker


@pytest.fixture
def mock_metrics():
    return {"loss": AverageMeter("loss", ":.3f"), "accuracy": AverageMeter("accuracy", ":.2f")}


class MockAccelerator:
    def __init__(self, num_processes: int, reduce_fn=None):
        self.num_processes = num_processes
        self.device = torch.device("cpu")
        self._reduce_fn = reduce_fn

    def reduce(self, tensor, reduction="mean"):
        # In single-process tests we just want a deterministic stand-in for accelerate's reduce.
        if self._reduce_fn is not None:
            return self._reduce_fn(tensor, reduction)
        return tensor


def test_average_meter_initialization():
    meter = AverageMeter("loss", ":.2f")
    assert meter.name == "loss"
    assert meter.fmt == ":.2f"
    assert meter.val == 0.0
    assert meter.avg == 0.0
    assert meter.sum == 0.0
    assert meter.count == 0.0


def test_average_meter_update():
    meter = AverageMeter("accuracy")
    meter.update(5, n=2)
    assert meter.val == 5
    assert meter.sum == 10
    assert meter.count == 2
    assert meter.avg == 5


def test_average_meter_reset():
    meter = AverageMeter("loss")
    meter.update(3, 4)
    meter.reset()
    assert meter.val == 0.0
    assert meter.avg == 0.0
    assert meter.sum == 0.0
    assert meter.count == 0.0


def test_average_meter_str():
    meter = AverageMeter("metric", ":.1f")
    meter.update(4.567, 3)
    assert str(meter) == "metric:4.6"


def test_metrics_tracker_initialization(mock_metrics):
    tracker = MetricsTracker(
        batch_size=32, num_frames=1000, num_episodes=50, metrics=mock_metrics, initial_step=10
    )
    assert tracker.steps == 10
    assert tracker.samples == 10 * 32
    assert tracker.episodes == tracker.samples / (1000 / 50)
    assert tracker.epochs == tracker.samples / 1000
    assert "loss" in tracker.metrics
    assert "accuracy" in tracker.metrics


def test_metrics_tracker_step(mock_metrics):
    tracker = MetricsTracker(
        batch_size=32, num_frames=1000, num_episodes=50, metrics=mock_metrics, initial_step=5
    )
    tracker.step()
    assert tracker.steps == 6
    assert tracker.samples == 6 * 32
    assert tracker.episodes == tracker.samples / (1000 / 50)
    assert tracker.epochs == tracker.samples / 1000


def test_metrics_tracker_initialization_with_accelerator(mock_metrics):
    tracker = MetricsTracker(
        batch_size=32,
        num_frames=1000,
        num_episodes=50,
        metrics=mock_metrics,
        initial_step=10,
        accelerator=MockAccelerator(num_processes=2),
    )
    assert tracker.steps == 10
    assert tracker.samples == 10 * 32 * 2
    assert tracker.episodes == tracker.samples / (1000 / 50)
    assert tracker.epochs == tracker.samples / 1000


def test_metrics_tracker_step_with_accelerator(mock_metrics):
    tracker = MetricsTracker(
        batch_size=32,
        num_frames=1000,
        num_episodes=50,
        metrics=mock_metrics,
        initial_step=5,
        accelerator=MockAccelerator(num_processes=2),
    )
    tracker.step()
    assert tracker.steps == 6
    assert tracker.samples == (5 * 32 * 2) + (32 * 2)
    assert tracker.episodes == tracker.samples / (1000 / 50)
    assert tracker.epochs == tracker.samples / 1000


def test_metrics_tracker_getattr(mock_metrics):
    tracker = MetricsTracker(batch_size=32, num_frames=1000, num_episodes=50, metrics=mock_metrics)
    assert tracker.loss == mock_metrics["loss"]
    assert tracker.accuracy == mock_metrics["accuracy"]
    with pytest.raises(AttributeError):
        _ = tracker.non_existent_metric


def test_metrics_tracker_setattr(mock_metrics):
    tracker = MetricsTracker(batch_size=32, num_frames=1000, num_episodes=50, metrics=mock_metrics)
    tracker.loss = 2.0
    assert tracker.loss.val == 2.0


def test_metrics_tracker_str(mock_metrics):
    tracker = MetricsTracker(batch_size=32, num_frames=1000, num_episodes=50, metrics=mock_metrics)
    tracker.loss.update(3.456, 1)
    tracker.accuracy.update(0.876, 1)
    output = str(tracker)
    assert "loss:3.456" in output
    assert "accuracy:0.88" in output


def test_metrics_tracker_to_dict(mock_metrics):
    tracker = MetricsTracker(batch_size=32, num_frames=1000, num_episodes=50, metrics=mock_metrics)
    tracker.loss.update(5, 2)
    metrics_dict = tracker.to_dict()
    assert isinstance(metrics_dict, dict)
    assert metrics_dict["loss"] == 5  # average value
    assert metrics_dict["steps"] == tracker.steps


def test_metrics_tracker_reset_averages(mock_metrics):
    tracker = MetricsTracker(batch_size=32, num_frames=1000, num_episodes=50, metrics=mock_metrics)
    tracker.loss.update(10, 3)
    tracker.accuracy.update(0.95, 5)
    tracker.reset_averages()
    assert tracker.loss.avg == 0.0
    assert tracker.accuracy.avg == 0.0


def test_average_meter_invalid_reduction():
    with pytest.raises(ValueError):
        AverageMeter("loss", reduction="median")


def test_average_meter_reduction_stored():
    meter = AverageMeter("updt_s", reduction="max")
    assert meter.reduction == "max"


def test_metrics_tracker_reduce_across_ranks_no_accelerator():
    metrics = {"update_s": AverageMeter("update_s", reduction="max")}
    tracker = MetricsTracker(batch_size=32, num_frames=1000, num_episodes=50, metrics=metrics)
    tracker.update_s = 0.5
    tracker.reduce_across_ranks()  # no-op without accelerator
    assert tracker.update_s.avg == 0.5


def test_metrics_tracker_reduce_across_ranks_single_process():
    metrics = {"update_s": AverageMeter("update_s", reduction="max")}
    tracker = MetricsTracker(
        batch_size=32,
        num_frames=1000,
        num_episodes=50,
        metrics=metrics,
        accelerator=MockAccelerator(num_processes=1),
    )
    tracker.update_s = 0.5
    tracker.reduce_across_ranks()  # no-op when world size is 1
    assert tracker.update_s.avg == 0.5


def test_metrics_tracker_reduce_across_ranks_invokes_reduce():
    captured = {}

    def fake_reduce(tensor, reduction):
        captured["reduction"] = reduction
        captured["values"] = tensor.clone()
        # Pretend the slowest rank reported 0.9 instead of this rank's 0.4.
        return torch.tensor([0.9], dtype=tensor.dtype, device=tensor.device)

    metrics = {
        "loss": AverageMeter("loss"),  # reduction="none" -> not touched
        "update_s": AverageMeter("update_s", reduction="max"),
    }
    tracker = MetricsTracker(
        batch_size=32,
        num_frames=1000,
        num_episodes=50,
        metrics=metrics,
        accelerator=MockAccelerator(num_processes=4, reduce_fn=fake_reduce),
    )
    tracker.loss = 1.0
    tracker.update_s = 0.4
    tracker.reduce_across_ranks()

    assert captured["reduction"] == "max"
    assert torch.allclose(captured["values"], torch.tensor([0.4]))
    assert tracker.update_s.avg == pytest.approx(0.9)
    # Metrics without a reduction stay untouched.
    assert tracker.loss.avg == 1.0
    # Invariant: avg == sum / count must hold after reduce, so subsequent .update() calls
    # accumulate against the cluster view rather than the stale per-rank sum.
    meter = tracker.update_s
    assert meter.sum / meter.count == pytest.approx(meter.avg)


def test_metrics_tracker_update_metrics_registers_and_averages():
    tracker = MetricsTracker(batch_size=32, num_frames=1000, num_episodes=50, metrics={})
    tracker.update_metrics({"latent_loss": 0.2, "action_loss": 0.4})
    tracker.update_metrics({"latent_loss": 0.4, "action_loss": 0.6})

    # New keys are auto-registered as mean-reduced meters and averaged over the window.
    assert tracker.metrics["latent_loss"].reduction == "mean"
    assert tracker.metrics["latent_loss"].avg == pytest.approx(0.3)
    assert tracker.metrics["action_loss"].avg == pytest.approx(0.5)
    assert tracker.to_dict()["latent_loss"] == pytest.approx(0.3)


def test_metrics_tracker_update_metrics_skips_non_numeric():
    tracker = MetricsTracker(batch_size=32, num_frames=1000, num_episodes=50, metrics={})
    tracker.update_metrics({"loss": 0.5, "head_mode": "sparse", "enabled": True})

    # strings and bools ignored
    assert "loss" in tracker.metrics
    assert "head_mode" not in tracker.metrics
    assert "enabled" not in tracker.metrics


def test_metrics_tracker_update_metrics_does_not_override_caller_meter():
    # A policy that echoes "loss" in its output dict must not overwrite the caller-owned,
    # already-aggregated loss meter.
    metrics = {"loss": AverageMeter("loss", ":.3f", reduction="mean")}
    tracker = MetricsTracker(batch_size=32, num_frames=1000, num_episodes=50, metrics=metrics)
    tracker.loss = 1.0  # caller-set optimized loss
    tracker.update_metrics({"loss": 99.0, "latent_loss": 0.2})

    assert tracker.metrics["loss"].avg == pytest.approx(1.0)  # snapshot ignored
    assert tracker.metrics["latent_loss"].avg == pytest.approx(0.2)

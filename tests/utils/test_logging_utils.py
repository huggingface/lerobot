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

from lerobot.utils.logging_utils import AverageMeter, MetricsTracker


@pytest.fixture
def mock_metrics():
    return {"loss": AverageMeter("loss", ":.3f"), "accuracy": AverageMeter("accuracy", ":.2f")}


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

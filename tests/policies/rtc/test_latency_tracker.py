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

"""Tests for RTC LatencyTracker module."""

import pytest

from lerobot.policies.rtc.latency_tracker import LatencyTracker

# ====================== Fixtures ======================


@pytest.fixture
def tracker():
    """Create a LatencyTracker with default maxlen."""
    return LatencyTracker(maxlen=100)


@pytest.fixture
def small_tracker():
    """Create a LatencyTracker with small maxlen for overflow testing."""
    return LatencyTracker(maxlen=5)


# ====================== Initialization Tests ======================


def test_latency_tracker_initialization():
    """Test LatencyTracker initializes correctly."""
    tracker = LatencyTracker(maxlen=50)
    assert len(tracker) == 0
    assert tracker.max_latency == 0.0
    assert tracker.max() == 0.0


def test_latency_tracker_default_maxlen():
    """Test LatencyTracker uses default maxlen."""
    tracker = LatencyTracker()
    # Should accept default maxlen=100
    assert len(tracker) == 0


# ====================== add() Tests ======================


def test_add_single_latency(tracker):
    """Test adding a single latency value."""
    tracker.add(0.5)
    assert len(tracker) == 1
    assert tracker.max() == 0.5


def test_add_multiple_latencies(tracker):
    """Test adding multiple latency values."""
    latencies = [0.1, 0.5, 0.3, 0.8, 0.2]
    for lat in latencies:
        tracker.add(lat)

    assert len(tracker) == 5
    assert tracker.max() == 0.8


def test_add_negative_latency_ignored(tracker):
    """Test that negative latencies are ignored."""
    tracker.add(0.5)
    tracker.add(-0.1)
    tracker.add(0.3)

    # Should only have 2 valid latencies
    assert len(tracker) == 2
    assert tracker.max() == 0.5


def test_add_zero_latency(tracker):
    """Test adding zero latency."""
    tracker.add(0.0)
    assert len(tracker) == 1
    assert tracker.max() == 0.0


def test_add_converts_to_float(tracker):
    """Test add() converts input to float."""
    tracker.add(5)  # Integer
    tracker.add("3.5")  # String

    assert len(tracker) == 2
    assert tracker.max() == 5.0


def test_add_updates_max_latency(tracker):
    """Test that max_latency is updated correctly."""
    tracker.add(0.5)
    assert tracker.max_latency == 0.5

    tracker.add(0.3)
    assert tracker.max_latency == 0.5  # Should not decrease

    tracker.add(0.9)
    assert tracker.max_latency == 0.9  # Should increase


# ====================== reset() Tests ======================


def test_reset_clears_values(tracker):
    """Test reset() clears all values."""
    tracker.add(0.5)
    tracker.add(0.8)
    tracker.add(0.3)
    assert len(tracker) == 3

    tracker.reset()
    assert len(tracker) == 0
    assert tracker.max_latency == 0.0


def test_reset_clears_max_latency(tracker):
    """Test reset() resets max_latency."""
    tracker.add(1.5)
    assert tracker.max_latency == 1.5

    tracker.reset()
    assert tracker.max_latency == 0.0


def test_reset_allows_new_values(tracker):
    """Test that tracker works correctly after reset."""
    tracker.add(0.5)
    tracker.reset()

    tracker.add(0.3)
    assert len(tracker) == 1
    assert tracker.max() == 0.3


# ====================== max() Tests ======================


def test_max_returns_zero_when_empty(tracker):
    """Test max() returns 0.0 when tracker is empty."""
    assert tracker.max() == 0.0


def test_max_returns_maximum_value(tracker):
    """Test max() returns the maximum latency."""
    latencies = [0.2, 0.8, 0.3, 0.5, 0.1]
    for lat in latencies:
        tracker.add(lat)

    assert tracker.max() == 0.8


def test_max_persists_after_sliding_window(small_tracker):
    """Test max() persists even after values slide out of window."""
    # Add values that will exceed maxlen=5
    small_tracker.add(0.1)
    small_tracker.add(0.9)  # This is max
    small_tracker.add(0.2)
    small_tracker.add(0.3)
    small_tracker.add(0.4)
    small_tracker.add(0.5)  # This pushes out 0.1

    # Max should still be 0.9 even though only last 5 values kept
    assert small_tracker.max() == 0.9


def test_max_after_reset(tracker):
    """Test max() returns 0.0 after reset."""
    tracker.add(1.5)
    tracker.reset()
    assert tracker.max() == 0.0


# ====================== p95() Tests ======================


def test_p95_returns_zero_when_empty(tracker):
    """Test p95() returns 0.0 when tracker is empty."""
    assert tracker.p95() == 0.0


def test_p95_returns_95th_percentile(tracker):
    """Test p95() returns the 95th percentile."""
    # Add 100 values
    for i in range(100):
        tracker.add(i / 100.0)

    p95 = tracker.p95()
    assert 0.93 <= p95 <= 0.96


def test_p95_equals_percentile_95(tracker):
    """Test p95() equals percentile(0.95)."""
    for i in range(50):
        tracker.add(i / 50.0)

    assert tracker.p95() == tracker.percentile(0.95)


# ====================== Edge Cases Tests ======================


def test_single_value(tracker):
    """Test tracker behavior with single value."""
    tracker.add(0.75)

    assert len(tracker) == 1
    assert tracker.max() == 0.75
    assert tracker.percentile(0.0) == 0.75
    assert tracker.percentile(0.5) == 0.75
    assert tracker.percentile(1.0) == 0.75


def test_all_same_values(tracker):
    """Test tracker with all identical values."""
    for _ in range(10):
        tracker.add(0.5)

    assert len(tracker) == 10
    assert tracker.max() == 0.5
    assert tracker.percentile(0.0) == 0.5
    assert tracker.percentile(0.5) == 0.5
    assert tracker.percentile(1.0) == 0.5


def test_very_small_values(tracker):
    """Test tracker with very small float values."""
    tracker.add(1e-10)
    tracker.add(2e-10)
    tracker.add(3e-10)

    assert len(tracker) == 3
    assert tracker.max() == pytest.approx(3e-10)


def test_very_large_values(tracker):
    """Test tracker with very large float values."""
    tracker.add(1e10)
    tracker.add(2e10)
    tracker.add(3e10)

    assert len(tracker) == 3
    assert tracker.max() == pytest.approx(3e10)


# ====================== Integration Tests ======================


def test_typical_usage_pattern(tracker):
    """Test a typical usage pattern of the tracker."""
    # Simulate adding latencies over time
    latencies = [0.05, 0.08, 0.12, 0.07, 0.15, 0.09, 0.11, 0.06, 0.14, 0.10]

    for lat in latencies:
        tracker.add(lat)

    # Check statistics
    assert len(tracker) == 10
    assert tracker.max() == 0.15

    # p95 should be close to max since we have only 10 values
    p95 = tracker.p95()
    assert p95 >= tracker.percentile(0.5)  # p95 should be >= median
    assert p95 <= tracker.max()  # p95 should be <= max


def test_reset_and_reuse(tracker):
    """Test resetting and reusing tracker."""
    # First batch
    tracker.add(1.0)
    tracker.add(2.0)
    assert tracker.max() == 2.0

    # Reset
    tracker.reset()

    # Second batch
    tracker.add(0.5)
    tracker.add(0.8)
    assert len(tracker) == 2
    assert tracker.max() == 0.8
    assert tracker.percentile(0.5) <= 0.8


# ====================== Type Conversion Tests ======================


def test_add_with_integer(tracker):
    """Test adding integer values."""
    tracker.add(5)
    assert len(tracker) == 1
    assert tracker.max() == 5.0


def test_add_with_string_number(tracker):
    """Test adding string representation of number."""
    tracker.add("3.14")
    assert len(tracker) == 1
    assert tracker.max() == pytest.approx(3.14)


def test_percentile_converts_q_to_float(tracker):
    """Test percentile converts q parameter to float."""
    tracker.add(0.5)
    tracker.add(0.8)

    # Pass integer q
    result = tracker.percentile(1)
    assert result == 0.8

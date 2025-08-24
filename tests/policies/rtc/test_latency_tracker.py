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

"""Unit tests for LatencyTracker class."""

import numpy as np
import pytest

from lerobot.policies.rtc.latency_tracker import LatencyTracker


class TestLatencyTracker:
    """Test suite for LatencyTracker functionality."""

    def test_initialization(self):
        """Test LatencyTracker initialization."""
        # Test with default maxlen
        tracker = LatencyTracker()
        assert len(tracker) == 0
        assert tracker.max() == 0.0
        assert tracker.percentile(0.5) == 0.0
        assert tracker.p95() == 0.0
        assert tracker.percentile(0.0) == 0.0
        assert tracker.percentile(1.0) == 0.0

        # Test with custom maxlen
        tracker_custom = LatencyTracker(maxlen=50)
        assert len(tracker_custom) == 0
        assert tracker_custom.max() == 0.0
        assert tracker_custom.percentile(0.5) == 0.0
        assert tracker_custom.p95() == 0.0

    def test_add_single_latency(self):
        """Test adding a single latency value."""
        tracker = LatencyTracker()

        # Add valid latency
        tracker.add(0.5)
        assert len(tracker) == 1
        assert tracker.max() == 0.5
        assert np.isclose(tracker.percentile(0.5), 0.5)
        assert np.isclose(tracker.p95(), 0.5)
        assert np.isclose(tracker.percentile(0.0), 0.5)
        assert np.isclose(tracker.percentile(1.0), 0.5)

        # Add another latency
        tracker.add(0.3)
        assert len(tracker) == 2
        assert tracker.max() == 0.5
        assert np.isclose(tracker.percentile(0.5), 0.4)
        assert np.isclose(tracker.p95(), 0.49)
        assert np.isclose(tracker.percentile(0.0), 0.3)
        assert np.isclose(tracker.percentile(1.0), 0.5)

    def test_add_invalid_latencies(self):
        """Test handling of invalid latency values."""
        tracker = LatencyTracker()

        # Test None value
        tracker.add(None)
        assert len(tracker) == 0
        assert tracker.max() == 0.0
        assert tracker.percentile(0.5) == 0.0
        assert tracker.p95() == 0.0
        assert tracker.percentile(0.0) == 0.0
        assert tracker.percentile(1.0) == 0.0

        # Test negative value
        tracker.add(-0.5)
        assert len(tracker) == 0
        assert tracker.max() == 0.0
        assert tracker.percentile(0.5) == 0.0
        assert tracker.p95() == 0.0
        assert tracker.percentile(0.0) == 0.0
        assert tracker.percentile(1.0) == 0.0

        # Test non-numeric value
        tracker.add("not_a_number")
        assert len(tracker) == 0
        assert tracker.max() == 0.0
        assert tracker.percentile(0.5) == 0.0
        assert tracker.p95() == 0.0
        assert tracker.percentile(0.0) == 0.0
        assert tracker.percentile(1.0) == 0.0

        # Verify valid values still work
        tracker.add(0.5)
        assert len(tracker) == 1
        assert tracker.max() == 0.5
        assert tracker.percentile(0.5) == 0.5
        assert tracker.p95() == 0.5
        assert tracker.percentile(0.0) == 0.5
        assert tracker.percentile(1.0) == 0.5

    def test_max_latency_tracking(self):
        """Test that max_latency is properly tracked."""
        tracker = LatencyTracker()

        values = [0.1, 0.5, 0.3, 0.9, 0.2]
        for val in values:
            tracker.add(val)

        assert tracker.max() == 0.9

        # Add a new maximum
        tracker.add(1.5)
        assert tracker.max() == 1.5

    def test_maxlen_sliding_window(self):
        """Test sliding window behavior with maxlen."""
        tracker = LatencyTracker(maxlen=3)

        # Add more values than maxlen
        for i in range(5):
            tracker.add(float(i))

        # Should only keep last 3 values
        assert len(tracker) == 3
        # Max should still be tracked correctly
        assert tracker.max() == 4.0
        assert np.isclose(tracker.percentile(0.5), 3.0)
        assert np.isclose(tracker.p95(), 3.9)
        assert np.isclose(tracker.percentile(0.0), 2.0)
        assert np.isclose(tracker.percentile(1.0), 4.0)

    def test_reset(self):
        """Test resetting the tracker."""
        tracker = LatencyTracker()
        tracker.add(0.1)
        tracker.add(0.2)
        tracker.add(0.3)
        assert len(tracker) == 3
        assert tracker.max() == 0.3
        assert np.isclose(tracker.percentile(0.5), 0.2)

        # Reset
        tracker.reset()
        assert len(tracker) == 0
        assert tracker.max() == 0.0
        assert tracker.percentile(0.5) == 0.0
        assert tracker.p95() == 0.0
        assert tracker.percentile(0.0) == 0.0
        assert tracker.percentile(1.0) == 0.0

    def test_percentile_calculations(self):
        """Test percentile calculations."""
        tracker = LatencyTracker()

        # Test with empty tracker
        assert tracker.percentile(0.5) == 0.0
        assert tracker.p95() == 0.0
        assert tracker.percentile(0.0) == 0.0
        assert tracker.percentile(1.0) == 0.0

        # Add values
        values = list(range(1, 101))  # 1 to 100
        for val in values:
            tracker.add(val)

        # Test various percentiles
        assert tracker.percentile(0.0) == 1.0  # min
        assert tracker.percentile(1.0) == 100.0  # max
        assert abs(tracker.percentile(0.5) - 50.5) < 1.0  # median (approximately)
        assert abs(tracker.percentile(0.25) - 25.75) < 1.0  # Q1 (approximately)
        assert abs(tracker.percentile(0.75) - 75.25) < 1.0  # Q3 (approximately)

        # Test p95 specifically
        p95 = tracker.p95()
        assert p95 is not None
        assert abs(p95 - 95.05) < 1.0  # approximately 95th percentile

    def test_percentile_edge_cases(self):
        """Test percentile with edge cases."""
        tracker = LatencyTracker()

        # Single value
        tracker.add(5.0)
        assert tracker.percentile(0.0) == 5.0
        assert tracker.percentile(0.5) == 5.0
        assert tracker.percentile(1.0) == 5.0

        # Two values
        tracker.reset()
        tracker.add(1.0)
        tracker.add(2.0)
        assert tracker.percentile(0.0) == 1.0
        assert tracker.percentile(1.0) == tracker.max()

    def test_percentile_with_sliding_window(self):
        """Test percentile calculations with sliding window."""
        tracker = LatencyTracker(maxlen=10)

        # Add 20 values, only last 10 should be kept
        for i in range(20):
            tracker.add(float(i))

        assert len(tracker) == 10
        # Should contain values 10-19
        assert tracker.percentile(0.0) == 10.0
        assert tracker.percentile(1.0) == 19.0

        # Median should be around 14.5
        median = tracker.percentile(0.5)
        assert abs(median - 14.5) < 1.0

    def test_consistent_max_after_sliding(self):
        """Test that max_latency remains consistent even after sliding window removes old max."""
        tracker = LatencyTracker(maxlen=3)

        # Add a high value that will be removed
        tracker.add(10.0)
        assert tracker.max() == 10.0

        # Add more values to push out the 10.0
        tracker.add(1.0)
        tracker.add(2.0)
        tracker.add(3.0)

        # max_latency should still be 10.0 (it's never decreased in current implementation)
        assert tracker.max() == 10.0

        # But percentile(1.0) should reflect current values
        assert tracker.percentile(1.0) == 10.0  # This will still be 10.0 due to max_latency

    def test_float_conversion(self):
        """Test that various numeric types are properly converted to float."""
        tracker = LatencyTracker()

        # Test integer
        tracker.add(5)
        assert len(tracker) == 1

        # Test numpy types
        tracker.add(np.float32(0.5))
        tracker.add(np.float64(0.6))
        tracker.add(np.int32(1))

        assert len(tracker) == 4
        """Test handling of zero values."""
        tracker = LatencyTracker()

        # Zero is a valid latency
        tracker.add(0.0)
        assert len(tracker) == 1
        assert tracker.max() == 0.0

        tracker.add(0.5)
        assert tracker.max() == 0.5
        assert tracker.percentile(0.0) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

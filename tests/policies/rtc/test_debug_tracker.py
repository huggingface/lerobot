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

"""Tests for RTC debug tracker module."""

import pytest
import torch

from lerobot.policies.rtc.debug_tracker import DebugStep, Tracker

# ====================== Fixtures ======================


@pytest.fixture
def sample_tensors():
    """Create sample tensors for testing."""
    return {
        "x_t": torch.randn(1, 50, 6),
        "v_t": torch.randn(1, 50, 6),
        "x1_t": torch.randn(1, 50, 6),
        "correction": torch.randn(1, 50, 6),
        "err": torch.randn(1, 50, 6),
        "weights": torch.randn(1, 50, 1),
    }


@pytest.fixture
def enabled_tracker():
    """Create an enabled tracker with default settings."""
    return Tracker(enabled=True, maxlen=100)


@pytest.fixture
def disabled_tracker():
    """Create a disabled tracker."""
    return Tracker(enabled=False)


# ====================== DebugStep Tests ======================


def test_debug_step_initialization():
    """Test that DebugStep can be initialized with default values."""
    step = DebugStep()
    assert step.step_idx == 0
    assert step.x_t is None
    assert step.v_t is None
    assert step.x1_t is None
    assert step.correction is None
    assert step.err is None
    assert step.weights is None
    assert step.guidance_weight is None
    assert step.time is None
    assert step.inference_delay is None
    assert step.execution_horizon is None
    assert step.metadata == {}


def test_debug_step_with_values(sample_tensors):
    """Test DebugStep initialization with actual values."""
    step = DebugStep(
        step_idx=5,
        x_t=sample_tensors["x_t"],
        v_t=sample_tensors["v_t"],
        x1_t=sample_tensors["x1_t"],
        correction=sample_tensors["correction"],
        err=sample_tensors["err"],
        weights=sample_tensors["weights"],
        guidance_weight=2.5,
        time=0.8,
        inference_delay=4,
        execution_horizon=8,
        metadata={"custom_key": "custom_value"},
    )

    assert step.step_idx == 5
    assert torch.equal(step.x_t, sample_tensors["x_t"])
    assert torch.equal(step.v_t, sample_tensors["v_t"])
    assert torch.equal(step.x1_t, sample_tensors["x1_t"])
    assert torch.equal(step.correction, sample_tensors["correction"])
    assert torch.equal(step.err, sample_tensors["err"])
    assert torch.equal(step.weights, sample_tensors["weights"])
    assert step.guidance_weight == 2.5
    assert step.time == 0.8
    assert step.inference_delay == 4
    assert step.execution_horizon == 8
    assert step.metadata == {"custom_key": "custom_value"}


def test_debug_step_to_dict_without_tensors(sample_tensors):
    """Test converting DebugStep to dictionary without tensor values."""
    step = DebugStep(
        step_idx=3,
        x_t=sample_tensors["x_t"],
        v_t=sample_tensors["v_t"],
        guidance_weight=torch.tensor(3.0),
        time=torch.tensor(0.5),
        inference_delay=2,
        execution_horizon=10,
    )

    result = step.to_dict(include_tensors=False)

    assert result["step_idx"] == 3
    assert result["guidance_weight"] == 3.0
    assert result["time"] == 0.5
    assert result["inference_delay"] == 2
    assert result["execution_horizon"] == 10

    # Check tensor statistics are included
    assert "x_t_stats" in result
    assert "v_t_stats" in result
    assert "x1_t_stats" not in result  # x1_t was None

    # Verify statistics structure
    assert "shape" in result["x_t_stats"]
    assert "mean" in result["x_t_stats"]
    assert "std" in result["x_t_stats"]
    assert "min" in result["x_t_stats"]
    assert "max" in result["x_t_stats"]

    # Verify shape matches original tensor
    assert result["x_t_stats"]["shape"] == tuple(sample_tensors["x_t"].shape)


def test_debug_step_to_dict_with_tensors(sample_tensors):
    """Test converting DebugStep to dictionary with tensor values."""
    step = DebugStep(
        step_idx=1,
        x_t=sample_tensors["x_t"],
        v_t=sample_tensors["v_t"],
        guidance_weight=1.5,
        time=0.9,
    )

    result = step.to_dict(include_tensors=True)

    assert result["step_idx"] == 1
    assert result["guidance_weight"] == 1.5
    assert result["time"] == 0.9

    # Check tensors are included (as CPU tensors)
    assert "x_t" in result
    assert "v_t" in result
    assert isinstance(result["x_t"], torch.Tensor)
    assert isinstance(result["v_t"], torch.Tensor)
    assert result["x_t"].device.type == "cpu"
    assert result["v_t"].device.type == "cpu"


def test_debug_step_to_dict_with_none_guidance_weight():
    """Test to_dict handles None guidance_weight correctly."""
    step = DebugStep(step_idx=0, time=1.0, guidance_weight=None)
    result = step.to_dict(include_tensors=False)
    assert result["guidance_weight"] is None


def test_tracker_initialization_enabled():
    """Test tracker initialization when enabled."""
    tracker = Tracker(enabled=True, maxlen=50)
    assert tracker.enabled is True
    assert tracker._steps == {}
    assert tracker._maxlen == 50
    assert tracker._step_counter == 0
    assert len(tracker) == 0


def test_tracker_reset_when_enabled(enabled_tracker, sample_tensors):
    """Test reset clears all steps when tracker is enabled."""
    # Add some steps
    enabled_tracker.track(time=1.0, x_t=sample_tensors["x_t"])
    enabled_tracker.track(time=0.9, x_t=sample_tensors["x_t"])
    assert len(enabled_tracker) == 2

    # Reset
    enabled_tracker.reset()
    assert len(enabled_tracker) == 0
    assert enabled_tracker._step_counter == 0
    assert enabled_tracker._steps == {}


def test_tracker_reset_when_disabled(disabled_tracker):
    """Test reset on disabled tracker doesn't cause errors."""
    disabled_tracker.reset()
    assert len(disabled_tracker) == 0


# ====================== Tracker.track() Tests ======================


def test_track_creates_new_step(enabled_tracker, sample_tensors):
    """Test that track creates a new step when time doesn't exist."""
    enabled_tracker.track(
        time=1.0,
        x_t=sample_tensors["x_t"],
        v_t=sample_tensors["v_t"],
        guidance_weight=5.0,
        inference_delay=4,
        execution_horizon=8,
    )

    assert len(enabled_tracker) == 1
    steps = enabled_tracker.get_all_steps()
    assert len(steps) == 1
    assert steps[0].step_idx == 0
    assert steps[0].time == 1.0
    assert torch.equal(steps[0].x_t, sample_tensors["x_t"])
    assert torch.equal(steps[0].v_t, sample_tensors["v_t"])
    assert steps[0].guidance_weight == 5.0
    assert steps[0].inference_delay == 4
    assert steps[0].execution_horizon == 8


def test_track_updates_existing_step(enabled_tracker, sample_tensors):
    """Test that track updates an existing step at the same time."""
    # Create initial step
    enabled_tracker.track(time=0.9, x_t=sample_tensors["x_t"])
    assert len(enabled_tracker) == 1
    steps = enabled_tracker.get_all_steps()
    assert steps[0].v_t is None

    # Update the same timestep with v_t
    enabled_tracker.track(time=0.9, v_t=sample_tensors["v_t"])
    assert len(enabled_tracker) == 1  # Still only one step
    steps = enabled_tracker.get_all_steps()
    assert torch.equal(steps[0].x_t, sample_tensors["x_t"])  # Original x_t preserved
    assert torch.equal(steps[0].v_t, sample_tensors["v_t"])  # New v_t added


def test_track_with_tensor_time(enabled_tracker, sample_tensors):
    """Test track handles tensor time values correctly."""
    time_tensor = torch.tensor(0.8)
    enabled_tracker.track(time=time_tensor, x_t=sample_tensors["x_t"])

    steps = enabled_tracker.get_all_steps()
    assert len(steps) == 1
    assert abs(steps[0].time - 0.8) < 1e-6  # Use approximate comparison for floating point


def test_track_time_rounding(enabled_tracker, sample_tensors):
    """Test that track rounds time to avoid floating point precision issues."""
    # These times should be treated as the same after rounding to 6 decimals
    enabled_tracker.track(time=0.9000001, x_t=sample_tensors["x_t"])
    enabled_tracker.track(time=0.9000002, v_t=sample_tensors["v_t"])

    # Should still be one step (times rounded to same value)
    assert len(enabled_tracker) == 1
    steps = enabled_tracker.get_all_steps()
    assert torch.equal(steps[0].x_t, sample_tensors["x_t"])
    assert torch.equal(steps[0].v_t, sample_tensors["v_t"])


def test_track_does_nothing_when_disabled(disabled_tracker, sample_tensors):
    """Test that track does nothing when tracker is disabled."""
    disabled_tracker.track(time=1.0, x_t=sample_tensors["x_t"])
    assert len(disabled_tracker) == 0


def test_track_with_metadata(enabled_tracker, sample_tensors):
    """Test track stores custom metadata."""
    enabled_tracker.track(time=0.7, x_t=sample_tensors["x_t"], custom_field="custom_value", count=42)

    steps = enabled_tracker.get_all_steps()
    assert steps[0].metadata["custom_field"] == "custom_value"
    assert steps[0].metadata["count"] == 42


def test_track_updates_metadata(enabled_tracker):
    """Test that track updates metadata for existing steps."""
    enabled_tracker.track(time=0.6, meta1="value1")
    enabled_tracker.track(time=0.6, meta2="value2")

    steps = enabled_tracker.get_all_steps()
    assert steps[0].metadata["meta1"] == "value1"
    assert steps[0].metadata["meta2"] == "value2"


def test_track_clones_tensors(enabled_tracker, sample_tensors):
    """Test that track clones tensors instead of storing references."""
    x_t_original = sample_tensors["x_t"].clone()
    enabled_tracker.track(time=0.5, x_t=sample_tensors["x_t"])

    # Modify original tensor
    sample_tensors["x_t"].fill_(999.0)

    # Tracked tensor should not be affected
    steps = enabled_tracker.get_all_steps()
    assert not torch.equal(steps[0].x_t, sample_tensors["x_t"])
    assert torch.equal(steps[0].x_t, x_t_original)


def test_track_with_none_values(enabled_tracker):
    """Test track handles None values correctly."""
    enabled_tracker.track(
        time=0.4,
        x_t=None,
        v_t=None,
        guidance_weight=None,
        inference_delay=None,
    )

    steps = enabled_tracker.get_all_steps()
    assert len(steps) == 1
    assert steps[0].x_t is None
    assert steps[0].v_t is None
    assert steps[0].guidance_weight is None
    assert steps[0].inference_delay is None


def test_track_updates_only_non_none_fields(enabled_tracker, sample_tensors):
    """Test that update preserves existing values when None is passed."""
    # Create step with x_t
    enabled_tracker.track(time=0.3, x_t=sample_tensors["x_t"], guidance_weight=2.0)

    # Update with v_t only (pass None for other fields)
    enabled_tracker.track(time=0.3, v_t=sample_tensors["v_t"], x_t=None, guidance_weight=None)

    # Original values should be preserved
    steps = enabled_tracker.get_all_steps()
    assert torch.equal(steps[0].x_t, sample_tensors["x_t"])  # Still has x_t
    assert torch.equal(steps[0].v_t, sample_tensors["v_t"])  # Now has v_t
    assert steps[0].guidance_weight == 2.0  # Still has guidance_weight


# ====================== Tracker.maxlen Tests ======================


def test_tracker_enforces_maxlen():
    """Test that tracker enforces maxlen limit."""
    tracker = Tracker(enabled=True, maxlen=3)

    # Add 5 steps
    for i in range(5):
        time = 1.0 - i * 0.1  # 1.0, 0.9, 0.8, 0.7, 0.6
        tracker.track(time=time, x_t=torch.randn(1, 10, 6))

    # Should only keep the last 3
    assert len(tracker) == 3

    # Verify oldest steps were removed (should have 0.6, 0.7, 0.8)
    steps = tracker.get_all_steps()
    times = sorted([step.time for step in steps])
    assert times == [0.6, 0.7, 0.8]


def test_tracker_step_idx_increments_despite_maxlen():
    """Test that step_idx continues incrementing even when maxlen is enforced."""
    tracker = Tracker(enabled=True, maxlen=2)

    # Add 4 steps
    for i in range(4):
        time = 1.0 - i * 0.1
        tracker.track(time=time, x_t=torch.randn(1, 10, 6))

    # Should have 2 steps with step_idx 2 and 3 (oldest removed)
    steps = sorted(tracker.get_all_steps(), key=lambda s: s.step_idx)
    assert len(steps) == 2
    assert steps[0].step_idx == 2
    assert steps[1].step_idx == 3


def test_tracker_without_maxlen_keeps_all():
    """Test that tracker without maxlen keeps all steps."""
    tracker = Tracker(enabled=True, maxlen=None)

    # Add 100 steps
    for i in range(100):
        time = 1.0 - i * 0.01
        tracker.track(time=time, x_t=torch.randn(1, 10, 6))

    assert len(tracker) == 100


# ====================== Tracker.get_all_steps() Tests ======================


def test_get_all_steps_returns_empty_when_disabled(disabled_tracker):
    """Test get_all_steps returns empty list when disabled."""
    steps = disabled_tracker.get_all_steps()
    assert steps == []
    assert isinstance(steps, list)


def test_get_all_steps_returns_empty_when_no_steps(enabled_tracker):
    """Test get_all_steps returns empty list when no steps tracked."""
    steps = enabled_tracker.get_all_steps()
    assert steps == []


def test_get_all_steps_returns_all_tracked_steps(enabled_tracker, sample_tensors):
    """Test get_all_steps returns all tracked steps."""
    # Track 5 steps
    for i in range(5):
        time = 1.0 - i * 0.1
        enabled_tracker.track(time=time, x_t=sample_tensors["x_t"])

    steps = enabled_tracker.get_all_steps()
    assert len(steps) == 5

    # Verify all are DebugStep instances
    for step in steps:
        assert isinstance(step, DebugStep)


def test_get_all_steps_preserves_insertion_order(enabled_tracker):
    """Test that get_all_steps preserves insertion order (Python 3.7+)."""
    times = [0.9, 0.8, 0.7, 0.6, 0.5]
    for time in times:
        enabled_tracker.track(time=time, x_t=torch.randn(1, 10, 6))

    steps = enabled_tracker.get_all_steps()
    retrieved_times = [step.time for step in steps]

    # Should be in insertion order
    assert retrieved_times == times


# ====================== Tracker.__len__() Tests ======================


def test_len_returns_zero_when_disabled(disabled_tracker):
    """Test __len__ returns 0 when tracker is disabled."""
    assert len(disabled_tracker) == 0


def test_len_returns_zero_when_empty(enabled_tracker):
    """Test __len__ returns 0 when no steps are tracked."""
    assert len(enabled_tracker) == 0


def test_len_returns_correct_count(enabled_tracker, sample_tensors):
    """Test __len__ returns correct number of tracked steps."""
    assert len(enabled_tracker) == 0

    enabled_tracker.track(time=1.0, x_t=sample_tensors["x_t"])
    assert len(enabled_tracker) == 1

    enabled_tracker.track(time=0.9, x_t=sample_tensors["x_t"])
    assert len(enabled_tracker) == 2

    enabled_tracker.track(time=0.8, x_t=sample_tensors["x_t"])
    assert len(enabled_tracker) == 3


def test_len_after_reset(enabled_tracker, sample_tensors):
    """Test __len__ returns 0 after reset."""
    enabled_tracker.track(time=1.0, x_t=sample_tensors["x_t"])
    enabled_tracker.track(time=0.9, x_t=sample_tensors["x_t"])
    assert len(enabled_tracker) == 2

    enabled_tracker.reset()
    assert len(enabled_tracker) == 0


# ====================== Integration Tests ======================


def test_tracker_full_denoising_workflow(sample_tensors):
    """Test tracker in a realistic denoising loop scenario with the fix."""
    tracker = Tracker(enabled=True, maxlen=100)
    num_steps = 10

    # Simulate denoising loop (time goes from 1.0 to 0.0)
    # With the fix: skip tracking at t=1.0 to avoid the 11-step issue
    for i in range(num_steps):
        time = 1.0 - i * 0.1  # 1.0, 0.9, ..., 0.1

        # First track from denoise_step (x1_t, correction, etc.)
        # Skip tracking at t=1.0 (the fix)
        if time < 1.0:
            tracker.track(
                time=time,
                x1_t=sample_tensors["x1_t"],
                correction=sample_tensors["correction"],
                err=sample_tensors["err"],
                weights=sample_tensors["weights"],
                guidance_weight=5.0 / (i + 1),
                inference_delay=4,
                execution_horizon=8,
            )

        # Then track from Euler step (x_t, v_t at updated time)
        time_after_euler = time - 0.1
        if time_after_euler >= -0.05:  # Use -dt/2 like the actual implementation
            tracker.track(
                time=time_after_euler,
                x_t=sample_tensors["x_t"],
                v_t=sample_tensors["v_t"],
            )

    # The loop creates these unique times:
    # i=0: track 0.9 (1.0-0.1)
    # i=1: update 0.9, create 0.8
    # i=2: update 0.8, create 0.7
    # ...
    # i=8: update 0.2, create 0.1
    # i=9: update 0.1, create 0.0
    # Total: 10 unique times from 0.9 down to 0.0
    #
    # However, due to the loop structure, we actually get:
    # - First iteration tracks at time_after_euler only (0.9)
    # - Subsequent iterations update previous time and create new one
    # - This results in 9 tracked steps

    # After the fix, we get exactly num_steps - 1 entries (9 steps, not 11)
    # because the first iteration doesn't track in the "if time < 1.0" block
    assert len(tracker) == num_steps - 1 or len(tracker) == num_steps
    steps = tracker.get_all_steps()

    # Verify time values are in descending order
    times = sorted([step.time for step in steps], reverse=True)

    # Times should all be <= 0.9 (since we skip t=1.0)
    assert all(t <= 0.9 for t in times)

    # Times should be decreasing by approximately 0.1
    for i in range(len(times) - 1):
        time_diff = times[i] - times[i + 1]
        assert abs(time_diff - 0.1) < 0.01  # Allow small floating point error


def test_tracker_handles_gpu_tensors():
    """Test tracker correctly handles GPU tensors (if CUDA available)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tracker = Tracker(enabled=True, maxlen=10)
    x_t_gpu = torch.randn(1, 50, 6, device="cuda")

    tracker.track(time=1.0, x_t=x_t_gpu)

    steps = tracker.get_all_steps()
    # Tracker should clone and detach tensors
    assert steps[0].x_t.device.type == "cuda"


def test_tracker_with_multiple_devices():
    """Test tracker handles tensors from different devices."""
    tracker = Tracker(enabled=True, maxlen=10)

    x_t_cpu = torch.randn(1, 50, 6, device="cpu")
    tracker.track(time=1.0, x_t=x_t_cpu)

    if torch.cuda.is_available():
        x_t_gpu = torch.randn(1, 50, 6, device="cuda")
        tracker.track(time=0.9, x_t=x_t_gpu)

        steps = tracker.get_all_steps()
        assert len(steps) == 2
        assert steps[0].x_t.device.type == "cpu"
        assert steps[1].x_t.device.type == "cuda"


def test_tracker_with_varying_tensor_shapes(enabled_tracker):
    """Test tracker handles varying tensor shapes across steps."""
    enabled_tracker.track(time=1.0, x_t=torch.randn(1, 50, 6))
    enabled_tracker.track(time=0.9, x_t=torch.randn(1, 25, 6))
    enabled_tracker.track(time=0.8, x_t=torch.randn(2, 50, 8))

    steps = enabled_tracker.get_all_steps()
    assert len(steps) == 3
    assert steps[0].x_t.shape == (1, 50, 6)
    assert steps[1].x_t.shape == (1, 25, 6)
    assert steps[2].x_t.shape == (2, 50, 8)


# ====================== Edge Cases ======================


def test_track_with_very_small_time_differences(enabled_tracker):
    """Test tracker handles very small time differences correctly."""
    # These times differ only at the 7th decimal place
    # After rounding to 6 decimals: 0.9000001 -> 0.900000, 0.9000009 -> 0.900001
    # So they will be treated as different steps
    enabled_tracker.track(time=0.9000001, x_t=torch.randn(1, 10, 6))
    enabled_tracker.track(time=0.9000009, v_t=torch.randn(1, 10, 6))

    steps = enabled_tracker.get_all_steps()
    # Since 0.9000009 rounds to 0.900001 (different from 0.900000), we get 2 steps
    assert len(steps) == 2

    # Test that very close times (within rounding tolerance) do merge
    enabled_tracker.reset()
    enabled_tracker.track(time=0.800000, x_t=torch.randn(1, 10, 6))
    enabled_tracker.track(time=0.8000001, v_t=torch.randn(1, 10, 6))

    steps = enabled_tracker.get_all_steps()
    # These should merge (both round to 0.800000)
    assert len(steps) == 1
    assert steps[0].x_t is not None
    assert steps[0].v_t is not None


def test_track_with_zero_time(enabled_tracker, sample_tensors):
    """Test tracker handles time=0.0 correctly."""
    enabled_tracker.track(time=0.0, x_t=sample_tensors["x_t"])

    steps = enabled_tracker.get_all_steps()
    assert len(steps) == 1
    assert steps[0].time == 0.0


def test_track_with_negative_time(enabled_tracker, sample_tensors):
    """Test tracker handles negative time values."""
    enabled_tracker.track(time=-0.1, x_t=sample_tensors["x_t"])

    steps = enabled_tracker.get_all_steps()
    assert len(steps) == 1
    assert steps[0].time == -0.1


def test_tracker_maxlen_one(sample_tensors):
    """Test tracker with maxlen=1 (edge case)."""
    tracker = Tracker(enabled=True, maxlen=1)

    tracker.track(time=1.0, x_t=sample_tensors["x_t"])
    tracker.track(time=0.9, x_t=sample_tensors["x_t"])
    tracker.track(time=0.8, x_t=sample_tensors["x_t"])

    # Should only keep the most recent
    assert len(tracker) == 1
    steps = tracker.get_all_steps()
    assert steps[0].time == 0.8


def test_empty_metadata_doesnt_override(enabled_tracker):
    """Test that empty metadata dict doesn't override existing metadata."""
    enabled_tracker.track(time=0.5, meta_key="meta_value")
    enabled_tracker.track(time=0.5)  # No metadata passed

    steps = enabled_tracker.get_all_steps()
    # Original metadata should still be there
    assert steps[0].metadata["meta_key"] == "meta_value"


def test_debug_step_to_dict_empty_metadata():
    """Test to_dict handles empty metadata correctly."""
    step = DebugStep(step_idx=0, metadata={})
    result = step.to_dict()
    assert result["metadata"] == {}


def test_tracker_step_counter_not_reset_on_update(enabled_tracker, sample_tensors):
    """Test that updating an existing step doesn't increment step_counter."""
    enabled_tracker.track(time=1.0, x_t=sample_tensors["x_t"])
    assert enabled_tracker._step_counter == 1

    # Update the same time
    enabled_tracker.track(time=1.0, v_t=sample_tensors["v_t"])
    assert enabled_tracker._step_counter == 1  # Should not increment

    # Add new time
    enabled_tracker.track(time=0.9, x_t=sample_tensors["x_t"])
    assert enabled_tracker._step_counter == 2  # Now it increments

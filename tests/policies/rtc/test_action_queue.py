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

"""Tests for RTC ActionQueue module."""

import threading
import time

import pytest
import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig

# ====================== Fixtures ======================


@pytest.fixture
def rtc_config_enabled():
    """Create an RTC config with RTC enabled."""
    return RTCConfig(enabled=True, execution_horizon=10, max_guidance_weight=1.0)


@pytest.fixture
def rtc_config_disabled():
    """Create an RTC config with RTC disabled."""
    return RTCConfig(enabled=False, execution_horizon=10, max_guidance_weight=1.0)


@pytest.fixture
def sample_actions():
    """Create sample action tensors for testing."""
    return {
        "original": torch.randn(50, 6),  # (time_steps, action_dim)
        "processed": torch.randn(50, 6),
        "short": torch.randn(10, 6),
        "longer": torch.randn(100, 6),
    }


@pytest.fixture
def action_queue_rtc_enabled(rtc_config_enabled):
    """Create an ActionQueue with RTC enabled."""
    return ActionQueue(rtc_config_enabled)


@pytest.fixture
def action_queue_rtc_disabled(rtc_config_disabled):
    """Create an ActionQueue with RTC disabled."""
    return ActionQueue(rtc_config_disabled)


# ====================== Initialization Tests ======================


def test_action_queue_initialization_rtc_enabled(rtc_config_enabled):
    """Test ActionQueue initializes correctly with RTC enabled."""
    queue = ActionQueue(rtc_config_enabled)
    assert queue.queue is None
    assert queue.original_queue is None
    assert queue.last_index == 0
    assert queue.cfg.enabled is True


def test_action_queue_initialization_rtc_disabled(rtc_config_disabled):
    """Test ActionQueue initializes correctly with RTC disabled."""
    queue = ActionQueue(rtc_config_disabled)
    assert queue.queue is None
    assert queue.original_queue is None
    assert queue.last_index == 0
    assert queue.cfg.enabled is False


# ====================== get() Tests ======================


def test_get_returns_none_when_empty(action_queue_rtc_enabled):
    """Test get() returns None when queue is empty."""
    action = action_queue_rtc_enabled.get()
    assert action is None


def test_get_returns_actions_sequentially(action_queue_rtc_enabled, sample_actions):
    """Test get() returns actions in sequence."""
    # Initialize queue with actions
    action_queue_rtc_enabled.merge(sample_actions["original"], sample_actions["processed"], real_delay=0)

    # Get first action
    action1 = action_queue_rtc_enabled.get()
    assert action1 is not None
    assert action1.shape == (6,)
    assert torch.equal(action1, sample_actions["processed"][0])

    # Get second action
    action2 = action_queue_rtc_enabled.get()
    assert action2 is not None
    assert torch.equal(action2, sample_actions["processed"][1])


def test_get_returns_none_after_exhaustion(action_queue_rtc_enabled, sample_actions):
    """Test get() returns None after all actions are consumed."""
    # Use short action sequence
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

    # Consume all actions
    for _ in range(10):
        action = action_queue_rtc_enabled.get()
        assert action is not None

    # Next get should return None
    action = action_queue_rtc_enabled.get()
    assert action is None


def test_get_increments_last_index(action_queue_rtc_enabled, sample_actions):
    """Test get() increments last_index correctly."""
    action_queue_rtc_enabled.merge(sample_actions["original"], sample_actions["processed"], real_delay=0)

    assert action_queue_rtc_enabled.last_index == 0
    action_queue_rtc_enabled.get()
    assert action_queue_rtc_enabled.last_index == 1
    action_queue_rtc_enabled.get()
    assert action_queue_rtc_enabled.last_index == 2


# ====================== qsize() Tests ======================


def test_qsize_returns_zero_when_empty(action_queue_rtc_enabled):
    """Test qsize() returns 0 when queue is empty."""
    assert action_queue_rtc_enabled.qsize() == 0


def test_qsize_returns_correct_size(action_queue_rtc_enabled, sample_actions):
    """Test qsize() returns correct number of remaining actions."""
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)
    assert action_queue_rtc_enabled.qsize() == 10

    action_queue_rtc_enabled.get()
    assert action_queue_rtc_enabled.qsize() == 9

    action_queue_rtc_enabled.get()
    assert action_queue_rtc_enabled.qsize() == 8


def test_qsize_after_exhaustion(action_queue_rtc_enabled, sample_actions):
    """Test qsize() returns 0 after queue is exhausted."""
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

    # Consume all actions
    for _ in range(10):
        action_queue_rtc_enabled.get()

    assert action_queue_rtc_enabled.qsize() == 0


# ====================== empty() Tests ======================


def test_empty_returns_true_when_empty(action_queue_rtc_enabled):
    """Test empty() returns True when queue is empty."""
    assert action_queue_rtc_enabled.empty() is True


def test_empty_returns_false_when_not_empty(action_queue_rtc_enabled, sample_actions):
    """Test empty() returns False when queue has actions."""
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)
    assert action_queue_rtc_enabled.empty() is False


def test_empty_after_partial_consumption(action_queue_rtc_enabled, sample_actions):
    """Test empty() returns False after partial consumption."""
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

    action_queue_rtc_enabled.get()
    action_queue_rtc_enabled.get()

    assert action_queue_rtc_enabled.empty() is False


def test_empty_after_full_consumption(action_queue_rtc_enabled, sample_actions):
    """Test empty() returns True after all actions consumed."""
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

    # Consume all
    for _ in range(10):
        action_queue_rtc_enabled.get()

    assert action_queue_rtc_enabled.empty() is True


# ====================== get_action_index() Tests ======================


def test_get_action_index_initial_value(action_queue_rtc_enabled):
    """Test get_action_index() returns 0 initially."""
    assert action_queue_rtc_enabled.get_action_index() == 0


def test_get_action_index_after_consumption(action_queue_rtc_enabled, sample_actions):
    """Test get_action_index() tracks consumption correctly."""
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

    assert action_queue_rtc_enabled.get_action_index() == 0
    action_queue_rtc_enabled.get()
    assert action_queue_rtc_enabled.get_action_index() == 1
    action_queue_rtc_enabled.get()
    action_queue_rtc_enabled.get()
    assert action_queue_rtc_enabled.get_action_index() == 3


# ====================== get_left_over() Tests ======================


def test_get_left_over_returns_none_when_empty(action_queue_rtc_enabled):
    """Test get_left_over() returns None when queue is empty."""
    leftover = action_queue_rtc_enabled.get_left_over()
    assert leftover is None


def test_get_left_over_returns_all_when_unconsumed(action_queue_rtc_enabled, sample_actions):
    """Test get_left_over() returns all original actions when none consumed."""
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

    leftover = action_queue_rtc_enabled.get_left_over()
    assert leftover is not None
    assert leftover.shape == (10, 6)
    assert torch.equal(leftover, sample_actions["short"])


def test_get_left_over_returns_remaining_after_consumption(action_queue_rtc_enabled, sample_actions):
    """Test get_left_over() returns only remaining original actions."""
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

    # Consume 3 actions
    action_queue_rtc_enabled.get()
    action_queue_rtc_enabled.get()
    action_queue_rtc_enabled.get()

    leftover = action_queue_rtc_enabled.get_left_over()
    assert leftover is not None
    assert leftover.shape == (7, 6)
    assert torch.equal(leftover, sample_actions["short"][3:])


def test_get_left_over_returns_empty_after_exhaustion(action_queue_rtc_enabled, sample_actions):
    """Test get_left_over() returns empty tensor after all consumed."""
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

    # Consume all
    for _ in range(10):
        action_queue_rtc_enabled.get()

    leftover = action_queue_rtc_enabled.get_left_over()
    assert leftover is not None
    assert leftover.shape == (0, 6)


# ====================== merge() with RTC Enabled Tests ======================


def test_merge_replaces_queue_when_rtc_enabled(action_queue_rtc_enabled, sample_actions):
    """Test merge() replaces queue when RTC is enabled."""
    # Add initial actions
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)
    assert action_queue_rtc_enabled.qsize() == 10

    # Consume some actions
    action_queue_rtc_enabled.get()
    action_queue_rtc_enabled.get()
    assert action_queue_rtc_enabled.qsize() == 8

    # Merge new actions - should replace, not append
    action_queue_rtc_enabled.merge(sample_actions["original"], sample_actions["processed"], real_delay=5)

    # Queue should be replaced with new actions minus delay
    # Original has 50 actions, delay is 5, so remaining is 45
    assert action_queue_rtc_enabled.qsize() == 45
    assert action_queue_rtc_enabled.get_action_index() == 0


def test_merge_respects_real_delay(action_queue_rtc_enabled, sample_actions):
    """Test merge() correctly applies real_delay when RTC is enabled."""
    delay = 10
    action_queue_rtc_enabled.merge(sample_actions["original"], sample_actions["processed"], real_delay=delay)

    # Queue should have original length minus delay
    expected_size = len(sample_actions["original"]) - delay
    assert action_queue_rtc_enabled.qsize() == expected_size

    # First action should be the one at index [delay]
    first_action = action_queue_rtc_enabled.get()
    assert torch.equal(first_action, sample_actions["processed"][delay])


def test_merge_resets_last_index_when_rtc_enabled(action_queue_rtc_enabled, sample_actions):
    """Test merge() resets last_index to 0 when RTC is enabled."""
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)
    action_queue_rtc_enabled.get()
    action_queue_rtc_enabled.get()
    assert action_queue_rtc_enabled.last_index == 2

    # Merge new actions
    action_queue_rtc_enabled.merge(sample_actions["original"], sample_actions["processed"], real_delay=5)

    assert action_queue_rtc_enabled.last_index == 0


def test_merge_with_zero_delay(action_queue_rtc_enabled, sample_actions):
    """Test merge() with zero delay keeps all actions."""
    action_queue_rtc_enabled.merge(sample_actions["original"], sample_actions["processed"], real_delay=0)

    assert action_queue_rtc_enabled.qsize() == len(sample_actions["original"])


def test_merge_with_large_delay(action_queue_rtc_enabled, sample_actions):
    """Test merge() with delay larger than action sequence."""
    # Delay is larger than sequence length
    delay = 100
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=delay)

    # Queue should be empty (delay >= length)
    assert action_queue_rtc_enabled.qsize() == 0


# ====================== merge() with RTC Disabled Tests ======================


def test_merge_appends_when_rtc_disabled(action_queue_rtc_disabled, sample_actions):
    """Test merge() appends actions when RTC is disabled."""
    # Add initial actions
    action_queue_rtc_disabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)
    initial_size = action_queue_rtc_disabled.qsize()
    assert initial_size == 10

    # Merge more actions
    action_queue_rtc_disabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

    # Should have appended
    assert action_queue_rtc_disabled.qsize() == initial_size + 10


def test_merge_removes_consumed_actions_when_appending(action_queue_rtc_disabled, sample_actions):
    """Test merge() removes consumed actions before appending when RTC is disabled."""
    # Add initial actions
    action_queue_rtc_disabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)
    assert action_queue_rtc_disabled.qsize() == 10

    # Consume 3 actions
    action_queue_rtc_disabled.get()
    action_queue_rtc_disabled.get()
    action_queue_rtc_disabled.get()
    assert action_queue_rtc_disabled.qsize() == 7

    # Merge more actions
    action_queue_rtc_disabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

    # Should have 7 remaining + 10 new = 17
    assert action_queue_rtc_disabled.qsize() == 17


def test_merge_resets_last_index_after_append(action_queue_rtc_disabled, sample_actions):
    """Test merge() resets last_index after appending when RTC is disabled."""
    action_queue_rtc_disabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)
    action_queue_rtc_disabled.get()
    action_queue_rtc_disabled.get()
    assert action_queue_rtc_disabled.last_index == 2

    # Merge more actions
    action_queue_rtc_disabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

    # last_index should be reset to 0
    assert action_queue_rtc_disabled.last_index == 0


def test_merge_ignores_delay_when_rtc_disabled(action_queue_rtc_disabled, sample_actions):
    """Test merge() ignores real_delay parameter when RTC is disabled."""
    action_queue_rtc_disabled.merge(sample_actions["original"], sample_actions["processed"], real_delay=10)

    # All actions should be in queue (delay ignored)
    assert action_queue_rtc_disabled.qsize() == len(sample_actions["original"])


def test_merge_first_call_with_rtc_disabled(action_queue_rtc_disabled, sample_actions):
    """Test merge() on first call with RTC disabled."""
    action_queue_rtc_disabled.merge(sample_actions["original"], sample_actions["processed"], real_delay=0)

    assert action_queue_rtc_disabled.qsize() == len(sample_actions["original"])
    assert action_queue_rtc_disabled.last_index == 0


# ====================== merge() with Different Action Shapes Tests ======================


def test_merge_with_different_action_dims():
    """Test merge() handles actions with different dimensions."""
    cfg = RTCConfig(enabled=True, execution_horizon=10)
    queue = ActionQueue(cfg)

    # Actions with 4 dimensions instead of 6
    actions_4d = torch.randn(20, 4)
    queue.merge(actions_4d, actions_4d, real_delay=5)

    action = queue.get()
    assert action.shape == (4,)


def test_merge_with_different_lengths():
    """Test merge() handles action sequences of varying lengths."""
    cfg = RTCConfig(enabled=False, execution_horizon=10)
    queue = ActionQueue(cfg)

    # Add sequences of different lengths
    queue.merge(torch.randn(10, 6), torch.randn(10, 6), real_delay=0)
    assert queue.qsize() == 10

    queue.merge(torch.randn(25, 6), torch.randn(25, 6), real_delay=0)
    assert queue.qsize() == 35


# ====================== merge() Delay Validation Tests ======================


def test_merge_validates_delay_consistency(action_queue_rtc_enabled, sample_actions, caplog):
    """Test merge() validates that real_delay matches action index difference."""
    import logging

    caplog.set_level(logging.WARNING)

    # Initialize queue
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

    # Consume 5 actions
    for _ in range(5):
        action_queue_rtc_enabled.get()

    # Merge with mismatched delay (should log warning)
    # We consumed 5 actions, so index is 5. If we pass action_index_before_inference=0,
    # then indexes_diff=5, but if real_delay=3, it will warn
    action_queue_rtc_enabled.merge(
        sample_actions["original"],
        sample_actions["processed"],
        real_delay=3,
        action_index_before_inference=0,
    )

    # Check warning was logged
    assert "Indexes diff is not equal to real delay" in caplog.text


def test_merge_no_warning_when_delays_match(action_queue_rtc_enabled, sample_actions, caplog):
    """Test merge() doesn't warn when delays are consistent."""
    import logging

    caplog.set_level(logging.WARNING)

    # Initialize queue
    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

    # Consume 5 actions
    for _ in range(5):
        action_queue_rtc_enabled.get()

    # Merge with matching delay
    action_queue_rtc_enabled.merge(
        sample_actions["original"],
        sample_actions["processed"],
        real_delay=5,
        action_index_before_inference=0,
    )

    # Should not have warning
    assert "Indexes diff is not equal to real delay" not in caplog.text


def test_merge_skips_validation_when_action_index_none(action_queue_rtc_enabled, sample_actions, caplog):
    """Test merge() skips delay validation when action_index_before_inference is None."""
    import logging

    caplog.set_level(logging.WARNING)

    action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

    for _ in range(5):
        action_queue_rtc_enabled.get()

    # Pass None for action_index_before_inference
    action_queue_rtc_enabled.merge(
        sample_actions["original"],
        sample_actions["processed"],
        real_delay=999,  # Doesn't matter
        action_index_before_inference=None,
    )

    # Should not warn (validation skipped)
    assert "Indexes diff is not equal to real delay" not in caplog.text


# ====================== Thread Safety Tests ======================


def test_get_is_thread_safe(action_queue_rtc_enabled, sample_actions):
    """Test get() is thread-safe with multiple consumers."""
    action_queue_rtc_enabled.merge(sample_actions["longer"], sample_actions["longer"], real_delay=0)

    results = []
    errors = []

    def consumer():
        try:
            for _ in range(25):
                action = action_queue_rtc_enabled.get()
                if action is not None:
                    results.append(action)
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=consumer) for _ in range(4)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Should not have errors
    assert len(errors) == 0

    # Should have consumed all actions (100 total, 4 threads * 25 each)
    assert len(results) == 100

    # All results should be unique (no duplicate consumption)
    # We can verify by checking that indices are not duplicated
    # Since we don't track indices in results, we check total count is correct
    assert action_queue_rtc_enabled.qsize() == 0


def test_merge_is_thread_safe(action_queue_rtc_disabled, sample_actions):
    """Test merge() is thread-safe with multiple producers."""
    errors = []

    def producer():
        try:
            for _ in range(5):
                action_queue_rtc_disabled.merge(
                    sample_actions["short"], sample_actions["short"], real_delay=0
                )
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=producer) for _ in range(3)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Should not have errors
    assert len(errors) == 0

    # Should have accumulated all actions (3 threads * 5 merges * 10 actions = 150)
    assert action_queue_rtc_disabled.qsize() == 150


def test_concurrent_get_and_merge(action_queue_rtc_disabled, sample_actions):
    """Test concurrent get() and merge() operations."""
    errors = []
    consumed_count = [0]

    def consumer():
        try:
            for _ in range(50):
                action = action_queue_rtc_disabled.get()
                if action is not None:
                    consumed_count[0] += 1
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    def producer():
        try:
            for _ in range(10):
                action_queue_rtc_disabled.merge(
                    sample_actions["short"], sample_actions["short"], real_delay=0
                )
                time.sleep(0.005)
        except Exception as e:
            errors.append(e)

    consumer_threads = [threading.Thread(target=consumer) for _ in range(2)]
    producer_threads = [threading.Thread(target=producer) for _ in range(2)]

    for t in consumer_threads + producer_threads:
        t.start()

    for t in consumer_threads + producer_threads:
        t.join()

    # Should not have errors
    assert len(errors) == 0

    # Should have consumed some or all actions (non-deterministic due to timing)
    # Total produced: 2 producers * 10 merges * 10 actions = 200
    # Total consumed attempts: 2 consumers * 50 = 100
    assert consumed_count[0] <= 200


# ====================== get_left_over() Thread Safety Tests ======================


def test_get_left_over_is_thread_safe(action_queue_rtc_enabled, sample_actions):
    """Test get_left_over() is thread-safe with concurrent access."""
    action_queue_rtc_enabled.merge(sample_actions["longer"], sample_actions["longer"], real_delay=0)

    errors = []
    leftovers = []

    def reader():
        try:
            for _ in range(20):
                leftover = action_queue_rtc_enabled.get_left_over()
                if leftover is not None:
                    leftovers.append(leftover.shape[0])
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(3)]

    # Also consume some actions concurrently
    def consumer():
        try:
            for _ in range(10):
                action_queue_rtc_enabled.get()
                time.sleep(0.002)
        except Exception as e:
            errors.append(e)

    consumer_thread = threading.Thread(target=consumer)

    all_threads = threads + [consumer_thread]

    for t in all_threads:
        t.start()

    for t in all_threads:
        t.join()

    # Should not have errors
    assert len(errors) == 0

    # Leftovers should be monotonically decreasing or stable
    # (as actions are consumed, leftover size decreases)
    assert len(leftovers) > 0


# ====================== Edge Cases Tests ======================


def test_queue_with_single_action(action_queue_rtc_enabled):
    """Test queue behavior with a single action."""
    single_action_original = torch.randn(1, 6)
    single_action_processed = torch.randn(1, 6)

    action_queue_rtc_enabled.merge(single_action_original, single_action_processed, real_delay=0)

    assert action_queue_rtc_enabled.qsize() == 1
    action = action_queue_rtc_enabled.get()
    assert action is not None
    assert action.shape == (6,)
    assert action_queue_rtc_enabled.qsize() == 0


def test_queue_behavior_after_multiple_merge_cycles(action_queue_rtc_enabled, sample_actions):
    """Test queue maintains correct state through multiple merge cycles."""
    for _ in range(5):
        action_queue_rtc_enabled.merge(sample_actions["short"], sample_actions["short"], real_delay=0)

        # Consume half
        for _ in range(5):
            action_queue_rtc_enabled.get()

        # Merge again
        action_queue_rtc_enabled.merge(sample_actions["original"], sample_actions["processed"], real_delay=3)

        assert action_queue_rtc_enabled.qsize() > 0


def test_queue_with_all_zeros_actions(action_queue_rtc_enabled):
    """Test queue handles all-zero action tensors."""
    zeros_actions = torch.zeros(20, 6)
    action_queue_rtc_enabled.merge(zeros_actions, zeros_actions, real_delay=0)

    action = action_queue_rtc_enabled.get()
    assert torch.all(action == 0)


def test_queue_clones_input_tensors(action_queue_rtc_enabled, sample_actions):
    """Test that merge() clones input tensors, not storing references."""
    original_copy = sample_actions["original"].clone()
    processed_copy = sample_actions["processed"].clone()

    action_queue_rtc_enabled.merge(sample_actions["original"], sample_actions["processed"], real_delay=0)

    # Modify original tensors
    sample_actions["original"].fill_(999.0)
    sample_actions["processed"].fill_(-999.0)

    # Queue should have cloned values
    action = action_queue_rtc_enabled.get()
    assert not torch.equal(action, sample_actions["processed"][0])
    assert torch.equal(action, processed_copy[0])

    leftover = action_queue_rtc_enabled.get_left_over()
    assert not torch.equal(leftover, sample_actions["original"][1:])
    assert torch.equal(leftover, original_copy[1:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_queue_handles_gpu_tensors():
    """Test queue correctly handles GPU tensors."""
    cfg = RTCConfig(enabled=True, execution_horizon=10)
    queue = ActionQueue(cfg)

    actions_gpu = torch.randn(20, 6, device="cuda")
    queue.merge(actions_gpu, actions_gpu, real_delay=0)

    action = queue.get()
    assert action.device.type == "cuda"

    leftover = queue.get_left_over()
    assert leftover.device.type == "cuda"


def test_queue_handles_different_dtypes():
    """Test queue handles actions with different dtypes."""
    cfg = RTCConfig(enabled=True, execution_horizon=10)
    queue = ActionQueue(cfg)

    # Use float64 instead of default float32
    actions_f64 = torch.randn(20, 6, dtype=torch.float64)
    queue.merge(actions_f64, actions_f64, real_delay=0)

    action = queue.get()
    assert action.dtype == torch.float64


def test_empty_with_none_queue(action_queue_rtc_enabled):
    """Test empty() correctly handles None queue."""
    assert action_queue_rtc_enabled.queue is None
    assert action_queue_rtc_enabled.empty() is True


def test_qsize_with_none_queue(action_queue_rtc_enabled):
    """Test qsize() correctly handles None queue."""
    assert action_queue_rtc_enabled.queue is None
    assert action_queue_rtc_enabled.qsize() == 0


# ====================== Integration Tests ======================


def test_typical_rtc_workflow(action_queue_rtc_enabled, sample_actions):
    """Test a typical RTC workflow: merge, consume, merge with delay."""
    # First inference
    action_queue_rtc_enabled.merge(sample_actions["original"], sample_actions["processed"], real_delay=0)
    initial_size = action_queue_rtc_enabled.qsize()
    assert initial_size == 50

    # Consume 10 actions (execution_horizon)
    for _ in range(10):
        action = action_queue_rtc_enabled.get()
        assert action is not None

    assert action_queue_rtc_enabled.qsize() == 40

    # Second inference with delay
    action_index_before = action_queue_rtc_enabled.get_action_index()

    action_queue_rtc_enabled.merge(
        sample_actions["original"],
        sample_actions["processed"],
        real_delay=5,
        action_index_before_inference=action_index_before,
    )

    # Queue should be replaced, minus delay
    assert action_queue_rtc_enabled.qsize() == 45
    assert action_queue_rtc_enabled.get_action_index() == 0


def test_typical_non_rtc_workflow(action_queue_rtc_disabled, sample_actions):
    """Test a typical non-RTC workflow: merge, consume, merge again."""
    # First inference
    action_queue_rtc_disabled.merge(sample_actions["original"], sample_actions["processed"], real_delay=0)
    assert action_queue_rtc_disabled.qsize() == 50

    # Consume 40 actions
    for _ in range(40):
        action = action_queue_rtc_disabled.get()
        assert action is not None

    assert action_queue_rtc_disabled.qsize() == 10

    # Second inference (should append)
    action_queue_rtc_disabled.merge(sample_actions["original"], sample_actions["processed"], real_delay=0)

    # Should have 10 remaining + 50 new = 60
    assert action_queue_rtc_disabled.qsize() == 60

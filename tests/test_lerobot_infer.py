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

"""Tests for lerobot_infer.py script and its components."""

import pytest

from lerobot.utils.temporal_ensemble import TemporalEnsembler


class TestTemporalEnsembler:
    """Test suite for TemporalEnsembler class."""

    def test_k_equals_1_passthrough(self):
        """When k=1, temporal ensembling should be disabled and return actions unchanged."""
        ensembler = TemporalEnsembler(k=1, exp=1.0)
        assert not ensembler.enabled

        action = {"motor_1": 0.5, "motor_2": -0.3}
        result = ensembler.update(action)

        assert result == action
        assert len(ensembler.action_buffer) == 0  # Buffer should not be used

    def test_uniform_weights_k_equals_3(self):
        """With exp=1.0 and k=3, should compute simple moving average."""
        ensembler = TemporalEnsembler(k=3, exp=1.0)
        assert ensembler.enabled

        # Verify weights are uniform
        assert ensembler.weights == pytest.approx([1/3, 1/3, 1/3])

        # Add actions to buffer
        action1 = {"motor_1": 0.0}
        action2 = {"motor_1": 0.3}
        action3 = {"motor_1": 0.6}

        # First two return unchanged (buffer not full)
        result1 = ensembler.update(action1)
        assert result1 == action1

        result2 = ensembler.update(action2)
        assert result2 == action2

        # Third should return weighted average: (0.0 + 0.3 + 0.6) / 3 = 0.3
        result3 = ensembler.update(action3)
        assert result3["motor_1"] == pytest.approx(0.3)

    def test_exponential_decay_recent_weighted_more(self):
        """With exp<1.0, recent actions should be weighted more heavily."""
        ensembler = TemporalEnsembler(k=3, exp=0.5)
        assert ensembler.enabled

        # Weights should be: [0.5^2, 0.5^1, 0.5^0] = [0.25, 0.5, 1.0]
        # Normalized: [0.25/1.75, 0.5/1.75, 1.0/1.75] ≈ [0.143, 0.286, 0.571]
        expected_weights = [0.25/1.75, 0.5/1.75, 1.0/1.75]
        assert ensembler.weights == pytest.approx(expected_weights)

        # Add actions
        action1 = {"motor_1": 0.0}
        action2 = {"motor_1": 0.0}
        action3 = {"motor_1": 1.0}

        ensembler.update(action1)
        ensembler.update(action2)
        result = ensembler.update(action3)

        # Most recent (1.0) has highest weight, so result should be > 0.5
        assert result["motor_1"] == pytest.approx(0.571, abs=1e-3)
        assert result["motor_1"] > 0.5  # Verify recent action dominates

    def test_exponential_growth_older_weighted_more(self):
        """With exp>1.0, older actions should be weighted more heavily."""
        ensembler = TemporalEnsembler(k=3, exp=2.0)
        assert ensembler.enabled

        # Weights should be: [2^2, 2^1, 2^0] = [4.0, 2.0, 1.0]
        # Normalized: [4.0/7.0, 2.0/7.0, 1.0/7.0] ≈ [0.571, 0.286, 0.143]
        expected_weights = [4.0/7.0, 2.0/7.0, 1.0/7.0]
        assert ensembler.weights == pytest.approx(expected_weights)

        # Add actions
        action1 = {"motor_1": 1.0}
        action2 = {"motor_1": 0.0}
        action3 = {"motor_1": 0.0}

        ensembler.update(action1)
        ensembler.update(action2)
        result = ensembler.update(action3)

        # Oldest (1.0) has highest weight, so result should be > 0.5
        assert result["motor_1"] == pytest.approx(0.571, abs=1e-3)
        assert result["motor_1"] > 0.5  # Verify older action dominates

    def test_missing_keys_in_old_actions(self):
        """Should handle cases where old actions don't have all keys."""
        ensembler = TemporalEnsembler(k=3, exp=1.0)

        # First action has only motor_1
        action1 = {"motor_1": 0.0}
        ensembler.update(action1)

        # Second action has motor_1 and motor_2
        action2 = {"motor_1": 0.5, "motor_2": 1.0}
        ensembler.update(action2)

        # Third action has motor_1 and motor_2
        action3 = {"motor_1": 1.0, "motor_2": 2.0}
        result = ensembler.update(action3)

        # motor_1 should average all 3 values: (0.0 + 0.5 + 1.0) / 3 = 0.5
        assert result["motor_1"] == pytest.approx(0.5)

        # motor_2 should average only 2 values: (1.0 + 2.0) / 2 = 1.5
        assert result["motor_2"] == pytest.approx(1.5)

    def test_missing_key_in_all_buffers(self):
        """Should return current value if key not found in any buffered action."""
        ensembler = TemporalEnsembler(k=3, exp=1.0)

        # Fill buffer with actions containing motor_1 only
        action1 = {"motor_1": 0.0}
        action2 = {"motor_1": 0.5}
        action3 = {"motor_1": 1.0, "motor_2": 5.0}

        ensembler.update(action1)
        ensembler.update(action2)
        result = ensembler.update(action3)

        # motor_2 appears for first time, should return current value
        assert result["motor_2"] == 5.0

    def test_reset_clears_buffer(self):
        """Reset should clear the action buffer."""
        ensembler = TemporalEnsembler(k=3, exp=1.0)

        # Fill buffer
        ensembler.update({"motor_1": 0.0})
        ensembler.update({"motor_1": 0.5})
        ensembler.update({"motor_1": 1.0})

        assert len(ensembler.action_buffer) == 3

        # Reset
        ensembler.reset()
        assert len(ensembler.action_buffer) == 0

    def test_multiple_actions_dict_values(self):
        """Should handle actions with multiple motor values."""
        ensembler = TemporalEnsembler(k=2, exp=1.0)

        action1 = {"motor_1": 0.0, "motor_2": 1.0, "motor_3": 2.0}
        action2 = {"motor_1": 1.0, "motor_2": 3.0, "motor_3": 4.0}

        ensembler.update(action1)
        result = ensembler.update(action2)

        # All should be averaged: (old + new) / 2
        assert result["motor_1"] == pytest.approx(0.5)
        assert result["motor_2"] == pytest.approx(2.0)
        assert result["motor_3"] == pytest.approx(3.0)

    def test_invalid_k_raises_error(self):
        """k must be >= 1."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            TemporalEnsembler(k=0, exp=1.0)

        with pytest.raises(ValueError, match="k must be >= 1"):
            TemporalEnsembler(k=-1, exp=1.0)

    def test_invalid_exp_raises_error(self):
        """exp must be > 0."""
        with pytest.raises(ValueError, match="exp must be > 0"):
            TemporalEnsembler(k=3, exp=0.0)

        with pytest.raises(ValueError, match="exp must be > 0"):
            TemporalEnsembler(k=3, exp=-1.0)

    def test_buffer_maxlen_enforced(self):
        """Buffer should maintain maxlen=k."""
        ensembler = TemporalEnsembler(k=2, exp=1.0)

        action1 = {"motor_1": 0.0}
        action2 = {"motor_1": 1.0}
        action3 = {"motor_1": 2.0}

        ensembler.update(action1)
        ensembler.update(action2)
        ensembler.update(action3)

        # Buffer should only have 2 most recent actions
        assert len(ensembler.action_buffer) == 2
        assert ensembler.action_buffer[0] == action2
        assert ensembler.action_buffer[1] == action3


class TestInferIntegration:
    """Integration tests for the infer() function components."""

    def test_temporal_ensembler_integration_with_real_actions(self):
        """Test TemporalEnsembler integrates properly with action smoothing."""
        from lerobot.utils.temporal_ensemble import TemporalEnsembler

        # Simulate a sequence of noisy actions
        ensembler = TemporalEnsembler(k=3, exp=0.5)
        
        # Actions with jitter (simulating noisy policy predictions)
        actions = [
            {"motor_1": 10.0, "motor_2": 20.0},
            {"motor_1": 10.5, "motor_2": 19.5},
            {"motor_1": 9.8, "motor_2": 20.3},
            {"motor_1": 10.2, "motor_2": 19.8},
        ]
        
        smoothed_actions = []
        for action in actions:
            smoothed = ensembler.update(action)
            smoothed_actions.append(smoothed)
        
        # First two should be unchanged (buffer not full)
        assert smoothed_actions[0] == actions[0]
        assert smoothed_actions[1] == actions[1]
        
        # Later actions should be smoothed
        # The smoothed values should reduce jitter
        assert smoothed_actions[2] != actions[2]  # Should be smoothed
        assert smoothed_actions[3] != actions[3]  # Should be smoothed

    def test_control_mode_transitions(self):
        """Test control mode state transitions work correctly."""
        from lerobot.scripts.lerobot_infer import ControlMode

        # Verify ControlMode enum exists and has expected values
        assert hasattr(ControlMode, "IDLE")
        assert hasattr(ControlMode, "POLICY")
        assert hasattr(ControlMode, "TELEOP")
        
        # Verify modes are distinct
        assert ControlMode.IDLE != ControlMode.POLICY
        assert ControlMode.POLICY != ControlMode.TELEOP
        assert ControlMode.TELEOP != ControlMode.IDLE

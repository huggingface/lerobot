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

"""Tests for QuasiStaticWrenchEstimatorStep."""

import pytest
import torch

from lerobot.processor.quasistatic_wrench_processor import QuasiStaticWrenchEstimatorStep
from lerobot.processor.core import EnvTransition, TransitionKey


def test_quasistatic_wrench_estimator_basic():
    """Test basic contact detection functionality."""
    # Create processor with minimal config
    processor = QuasiStaticWrenchEstimatorStep(
        ema_alpha=0.1,
        threshold_on=0.5,
        threshold_off=0.3,
        min_consecutive_frames=2,
        debug=True,
        enable_wrench=False,  # Disable wrench for CI safety
    )
    
    # Create fake transition with effort data
    batch_size = 3
    n_joints = 2
    
    # Initial stable effort (low values)
    stable_effort = torch.zeros(batch_size, n_joints) + 0.1
    
    transition = {
        TransitionKey.OBSERVATION: {
            "effort": stable_effort,
        }
    }
    
    # Process initial frames to establish baseline
    for _ in range(5):
        transition = processor(transition)
    
    # Verify no contact initially
    obs = transition[TransitionKey.OBSERVATION]
    assert "contact_score" in obs
    assert "contact_flag" in obs
    assert "effort_residual" in obs  # debug output
    assert torch.all(~obs["contact_flag"])
    
    # Inject effort spike (contact event)
    spike_effort = torch.zeros(batch_size, n_joints) + 2.0
    transition[TransitionKey.OBSERVATION]["effort"] = spike_effort
    
    # Process spike frames - should require min_consecutive_frames
    for i in range(3):
        transition = processor(transition)
        obs = transition[TransitionKey.OBSERVATION]
        
        if i < 1:  # First frame - not enough consecutive frames yet
            # Some samples might already trigger due to batch processing
            # Check that at least some samples are still False
            assert torch.any(~obs["contact_flag"])
        else:  # After min_consecutive_frames
            # All samples should now be True
            assert torch.all(obs["contact_flag"])
    
    # Return to normal effort
    transition[TransitionKey.OBSERVATION]["effort"] = stable_effort
    
    # Process recovery - should deassert below threshold_off
    # Need more frames to allow EMA baseline to adapt back down
    for _ in range(10):  # More frames for EMA to adapt
        transition = processor(transition)
        obs = transition[TransitionKey.OBSERVATION]
    
    # Check that contact flags are eventually deasserted
    assert torch.all(~obs["contact_flag"])


def test_quasistatic_wrench_estimator_missing_effort():
    """Test graceful handling of missing effort data."""
    processor = QuasiStaticWrenchEstimatorStep(
        strict=False,  # Graceful mode
        enable_wrench=False,
    )
    
    # Transition without effort data
    transition = {
        TransitionKey.OBSERVATION: {
            "some_other_key": torch.ones(1, 3),
        }
    }
    
    # Should not raise error
    result = processor(transition)
    obs = result[TransitionKey.OBSERVATION]
    
    # Should have default outputs
    assert "contact_score" in obs
    assert "contact_flag" in obs
    assert torch.all(~obs["contact_flag"])


def test_quasistatic_wrench_estimator_strict_mode():
    """Test strict mode raises error for missing effort."""
    processor = QuasiStaticWrenchEstimatorStep(
        strict=True,  # Strict mode
        enable_wrench=False,
    )
    
    # Transition without effort data
    transition = {
        TransitionKey.OBSERVATION: {
            "some_other_key": torch.ones(1, 3),
        }
    }
    
    # Should raise error
    with pytest.raises(ValueError, match="No effort data found"):
        processor(transition)


def test_quasistatic_wrench_estimator_state_dict():
    """Test state dict serialization and loading."""
    processor = QuasiStaticWrenchEstimatorStep(
        ema_alpha=0.1,
        threshold_on=0.5,
        min_consecutive_frames=2,
        enable_wrench=False,
    )
    
    # Process some data to build up state
    transition = {
        TransitionKey.OBSERVATION: {
            "effort": torch.ones(1, 2) * 0.2,
        }
    }
    
    for _ in range(3):
        processor(transition)
    
    # Save state
    state = processor.state_dict()
    assert "_ema_baseline" in state
    assert "_consecutive_counters" in state
    assert "_contact_active" in state
    
    # Create new processor and load state
    new_processor = QuasiStaticWrenchEstimatorStep(
        ema_alpha=0.1,
        threshold_on=0.5,
        min_consecutive_frames=2,
        enable_wrench=False,
    )
    
    new_processor.load_state_dict(state)
    
    # Verify state is preserved
    assert new_processor._ema_baseline is not None
    assert torch.equal(new_processor._consecutive_counters, processor._consecutive_counters)
    assert torch.equal(new_processor._contact_active, processor._contact_active)


def test_quasistatic_wrench_estimator_reset():
    """Test reset functionality."""
    processor = QuasiStaticWrenchEstimatorStep(enable_wrench=False)
    
    # Process some data to build up state
    transition = {
        TransitionKey.OBSERVATION: {
            "effort": torch.ones(1, 2),
        }
    }
    
    processor(transition)
    
    # Verify state exists
    assert processor._ema_baseline is not None
    
    # Reset
    processor.reset()
    
    # Verify state is cleared
    assert processor._ema_baseline is None
    assert processor._consecutive_counters is None
    assert processor._contact_active is None


def test_quasistatic_wrench_estimator_config():
    """Test configuration serialization."""
    config = {
        "effort_keys_candidates": ["custom_effort", "current"],
        "strict": True,
        "ema_alpha": 0.05,
        "score_mode": "l1mean",
        "threshold_on": 0.8,
        "threshold_off": 0.4,
        "min_consecutive_frames": 3,
        "debug": True,
        "enable_wrench": False,
        "damping": 1e-5,
        "tau_scale": 2.0,
        "q_key": "joint_angles",
    }
    
    processor = QuasiStaticWrenchEstimatorStep(**config)
    
    # Get config and verify
    retrieved_config = processor.get_config()
    
    for key, value in config.items():
        assert retrieved_config[key] == value


def test_quasistatic_wrench_estimator_effort_extraction():
    """Test effort extraction from different formats."""
    processor = QuasiStaticWrenchEstimatorStep(enable_wrench=False)
    
    # Test tensor format
    transition_tensor = {
        TransitionKey.OBSERVATION: {
            "Present_Current": torch.ones(2, 3),
        }
    }
    
    result = processor(transition_tensor)
    assert "contact_score" in result[TransitionKey.OBSERVATION]
    
    # Test dict format
    transition_dict = {
        TransitionKey.OBSERVATION: {
            "motor_load": {
                "joint_1": torch.tensor([[1.0]]),
                "joint_2": torch.tensor([[2.0]]),
            }
        }
    }
    
    processor.reset()  # Reset to clear previous state
    result = processor(transition_dict)
    assert "contact_score" in result[TransitionKey.OBSERVATION]


def test_quasistatic_wrench_estimator_score_modes():
    """Test different scoring modes."""
    # Test L2 norm mode
    processor_l2 = QuasiStaticWrenchEstimatorStep(
        score_mode="l2",
        ema_alpha=0.0,  # Disable EMA to get exact scores
        enable_wrench=False,
    )
    
    # Test L1 mean mode
    processor_l1 = QuasiStaticWrenchEstimatorStep(
        score_mode="l1mean",
        ema_alpha=0.0,  # Disable EMA to get exact scores
        enable_wrench=False,
    )
    
    # Create transitions with effort (use separate transitions)
    transition_l2 = {
        TransitionKey.OBSERVATION: {
            "effort": torch.tensor([[1.0, 2.0]]),
        }
    }
    
    transition_l1 = {
        TransitionKey.OBSERVATION: {
            "effort": torch.tensor([[1.0, 2.0]]),
        }
    }
    
    # Process with both modes
    result_l2 = processor_l2(transition_l2)
    result_l1 = processor_l1(transition_l1)
    
    # Both should produce contact scores
    assert "contact_score" in result_l2[TransitionKey.OBSERVATION]
    assert "contact_score" in result_l1[TransitionKey.OBSERVATION]
    
    # Scores should be different due to different computation methods
    score_l2 = result_l2[TransitionKey.OBSERVATION]["contact_score"]
    score_l1 = result_l1[TransitionKey.OBSERVATION]["contact_score"]
    
    # With EMA disabled, scores should be computed from zero baseline
    # L2 norm should be sqrt(1^2 + 2^2) = sqrt(5) ≈ 2.236
    # L1 mean should be (|1| + |2|) / 2 = 1.5
    expected_l2 = torch.sqrt(torch.tensor(5.0)).reshape(1, 1)
    expected_l1 = torch.tensor(1.5).reshape(1, 1)
    
    torch.testing.assert_close(score_l2, expected_l2, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(score_l1, expected_l1, atol=1e-3, rtol=1e-3)


def test_quasistatic_wrench_estimator_invalid_score_mode():
    """Test error handling for invalid score mode."""
    with pytest.raises(ValueError, match="score_mode must be 'l2' or 'l1mean'"):
        QuasiStaticWrenchEstimatorStep(score_mode="invalid")


def test_quasistatic_wrench_estimator_wrench_disabled():
    """Test that wrench estimation is properly disabled."""
    processor = QuasiStaticWrenchEstimatorStep(enable_wrench=False)
    
    transition = {
        TransitionKey.OBSERVATION: {
            "effort": torch.ones(1, 2),
            "q": torch.ones(1, 2),  # Joint positions available
        }
    }
    
    result = processor(transition)
    obs = result[TransitionKey.OBSERVATION]
    
    # Should have contact outputs
    assert "contact_score" in obs
    assert "contact_flag" in obs
    
    # Should NOT have wrench outputs
    assert "ee_wrench_hat" not in obs
    assert "wrench_unavailable" not in obs

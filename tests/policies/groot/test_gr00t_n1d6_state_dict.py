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

"""Test script for Gr00tN1d6ProcessStep.state_dict() and load_state_dict() methods.

This verifies that normalization statistics can be correctly serialized and restored,
which is critical for saving/loading checkpoints.
"""

import numpy as np
import pytest
import torch

from lerobot.policies.gr00t_n1d6.processor_gr00t_n1d6 import (
    Gr00tN1d6ProcessStep,
    Gr00tN1d6Processor,
)
from lerobot.policies.gr00t_n1d6.utils import ModalityConfig


# Test constants
STATE_DIM = 7
ACTION_DIM = 7
ACTION_HORIZON = 16
EMBODIMENT_TAG = "test_embodiment"


def create_test_statistics() -> dict[str, dict[str, dict[str, dict[str, list[float]]]]]:
    """Create test statistics with realistic structure."""
    return {
        EMBODIMENT_TAG: {
            "state": {
                "state": {
                    "min": [-1.0] * STATE_DIM,
                    "max": [1.0] * STATE_DIM,
                    "mean": [0.0] * STATE_DIM,
                    "std": [0.5] * STATE_DIM,
                    "q01": [-0.9] * STATE_DIM,
                    "q99": [0.9] * STATE_DIM,
                }
            },
            "action": {
                "action": {
                    "min": [-2.0] * ACTION_DIM,
                    "max": [2.0] * ACTION_DIM,
                    "mean": [0.1] * ACTION_DIM,
                    "std": [0.8] * ACTION_DIM,
                    "q01": [-1.8] * ACTION_DIM,
                    "q99": [1.8] * ACTION_DIM,
                }
            },
        }
    }


def create_test_processor(statistics: dict | None = None) -> Gr00tN1d6Processor:
    """Create a test processor with the given statistics."""
    modality_configs = {
        EMBODIMENT_TAG: {
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=["state"],
            ),
            "action": ModalityConfig(
                delta_indices=list(range(ACTION_HORIZON)),
                modality_keys=["action"],
            ),
            "video": ModalityConfig(
                delta_indices=[0],
                modality_keys=["image"],
            ),
        }
    }

    return Gr00tN1d6Processor(
        modality_configs=modality_configs,
        statistics=statistics,
        max_state_dim=STATE_DIM,
        max_action_dim=ACTION_DIM,
        max_action_horizon=ACTION_HORIZON,
        use_relative_action=False,
        formalize_language=False,
        embodiment_id_mapping={EMBODIMENT_TAG: 0},
    )


class TestGr00tN1d6ProcessStepStateDict:
    """Test suite for state_dict() and load_state_dict() methods."""

    def test_state_dict_returns_empty_dict_when_no_statistics(self):
        """state_dict() should return empty dict when processor has no statistics."""
        processor = create_test_processor(statistics=None)
        step = Gr00tN1d6ProcessStep(processor=processor)

        state = step.state_dict()

        assert state == {}
        assert isinstance(state, dict)

    def test_state_dict_serializes_statistics_to_flat_tensors(self):
        """state_dict() should serialize nested statistics to flat tensor dict."""
        statistics = create_test_statistics()
        processor = create_test_processor(statistics=statistics)
        step = Gr00tN1d6ProcessStep(processor=processor)

        state = step.state_dict()

        # Verify we have the expected keys
        expected_keys = [
            f"{EMBODIMENT_TAG}.state.state.min",
            f"{EMBODIMENT_TAG}.state.state.max",
            f"{EMBODIMENT_TAG}.state.state.mean",
            f"{EMBODIMENT_TAG}.state.state.std",
            f"{EMBODIMENT_TAG}.state.state.q01",
            f"{EMBODIMENT_TAG}.state.state.q99",
            f"{EMBODIMENT_TAG}.action.action.min",
            f"{EMBODIMENT_TAG}.action.action.max",
            f"{EMBODIMENT_TAG}.action.action.mean",
            f"{EMBODIMENT_TAG}.action.action.std",
            f"{EMBODIMENT_TAG}.action.action.q01",
            f"{EMBODIMENT_TAG}.action.action.q99",
        ]
        assert set(state.keys()) == set(expected_keys)

        # Verify all values are tensors
        for key, value in state.items():
            assert isinstance(value, torch.Tensor), f"Expected tensor for {key}, got {type(value)}"
            assert value.device.type == "cpu", f"Expected CPU tensor for {key}"

        # Verify values are correct
        state_min = state[f"{EMBODIMENT_TAG}.state.state.min"]
        assert state_min.shape == (STATE_DIM,)
        assert torch.allclose(state_min, torch.tensor([-1.0] * STATE_DIM))

        action_mean = state[f"{EMBODIMENT_TAG}.action.action.mean"]
        assert action_mean.shape == (ACTION_DIM,)
        assert torch.allclose(action_mean, torch.tensor([0.1] * ACTION_DIM))

    def test_load_state_dict_restores_statistics(self):
        """load_state_dict() should restore statistics from flat tensor dict."""
        # Create processor with statistics and get state_dict
        original_statistics = create_test_statistics()
        processor_with_stats = create_test_processor(statistics=original_statistics)
        step_with_stats = Gr00tN1d6ProcessStep(processor=processor_with_stats)
        saved_state = step_with_stats.state_dict()

        # Create a new processor without statistics
        processor_without_stats = create_test_processor(statistics=None)
        step_without_stats = Gr00tN1d6ProcessStep(processor=processor_without_stats)

        # Verify no statistics initially
        assert not processor_without_stats.state_action_processor.statistics

        # Load the saved state
        step_without_stats.load_state_dict(saved_state)

        # Verify statistics were restored
        restored_stats = processor_without_stats.state_action_processor.statistics
        assert restored_stats is not None
        assert EMBODIMENT_TAG in restored_stats
        assert "state" in restored_stats[EMBODIMENT_TAG]
        assert "action" in restored_stats[EMBODIMENT_TAG]

        # Verify values match original
        restored_state_min = restored_stats[EMBODIMENT_TAG]["state"]["state"]["min"]
        original_state_min = original_statistics[EMBODIMENT_TAG]["state"]["state"]["min"]
        assert np.allclose(restored_state_min, original_state_min)

        restored_action_mean = restored_stats[EMBODIMENT_TAG]["action"]["action"]["mean"]
        original_action_mean = original_statistics[EMBODIMENT_TAG]["action"]["action"]["mean"]
        assert np.allclose(restored_action_mean, original_action_mean)

    def test_load_state_dict_with_empty_state_is_noop(self):
        """load_state_dict() with empty dict should be a no-op."""
        statistics = create_test_statistics()
        processor = create_test_processor(statistics=statistics)
        step = Gr00tN1d6ProcessStep(processor=processor)

        original_stats = processor.state_action_processor.statistics.copy()

        # Load empty state
        step.load_state_dict({})

        # Verify statistics unchanged
        current_stats = processor.state_action_processor.statistics
        assert current_stats == original_stats

    def test_pending_state_pattern_for_lazy_initialization(self):
        """load_state_dict() should store pending state when processor not initialized."""
        # Create state dict from a step with statistics
        original_statistics = create_test_statistics()
        processor_with_stats = create_test_processor(statistics=original_statistics)
        step_with_stats = Gr00tN1d6ProcessStep(processor=processor_with_stats)
        saved_state = step_with_stats.state_dict()

        # Create a step without processor (lazy initialization)
        step_lazy = Gr00tN1d6ProcessStep(processor=None, processor_config_path=None)

        # Load state before processor exists
        step_lazy.load_state_dict(saved_state)

        # Verify state is stored as pending
        assert step_lazy._pending_state is not None
        assert step_lazy._pending_state == saved_state

    def test_roundtrip_preserves_statistics(self):
        """Full roundtrip: create -> state_dict -> new step -> load_state_dict should preserve stats."""
        # Create original step with statistics
        original_statistics = create_test_statistics()
        processor1 = create_test_processor(statistics=original_statistics)
        step1 = Gr00tN1d6ProcessStep(processor=processor1)

        # Get state dict
        saved_state = step1.state_dict()

        # Create new step and load state
        processor2 = create_test_processor(statistics=None)
        step2 = Gr00tN1d6ProcessStep(processor=processor2)
        step2.load_state_dict(saved_state)

        # Get state dict from new step
        restored_state = step2.state_dict()

        # Verify state dicts are identical
        assert saved_state.keys() == restored_state.keys()
        for key in saved_state:
            assert torch.allclose(saved_state[key], restored_state[key]), f"Mismatch for key {key}"

    def test_state_dict_handles_multiple_embodiment_tags(self):
        """state_dict() should handle multiple embodiment tags."""
        # Create statistics with multiple embodiment tags
        statistics = {
            "embodiment_a": {
                "state": {
                    "state": {
                        "min": [-1.0] * STATE_DIM,
                        "max": [1.0] * STATE_DIM,
                        "mean": [0.0] * STATE_DIM,
                        "std": [0.5] * STATE_DIM,
                        "q01": [-0.9] * STATE_DIM,
                        "q99": [0.9] * STATE_DIM,
                    }
                },
            },
            "embodiment_b": {
                "action": {
                    "action": {
                        "min": [-2.0] * ACTION_DIM,
                        "max": [2.0] * ACTION_DIM,
                        "mean": [0.0] * ACTION_DIM,
                        "std": [0.8] * ACTION_DIM,
                        "q01": [-1.8] * ACTION_DIM,
                        "q99": [1.8] * ACTION_DIM,
                    }
                },
            },
        }

        # Create modality configs for both embodiments
        modality_configs = {
            "embodiment_a": {
                "state": ModalityConfig(delta_indices=[0], modality_keys=["state"]),
                "action": ModalityConfig(delta_indices=list(range(ACTION_HORIZON)), modality_keys=["action"]),
                "video": ModalityConfig(delta_indices=[0], modality_keys=["image"]),
            },
            "embodiment_b": {
                "state": ModalityConfig(delta_indices=[0], modality_keys=["state"]),
                "action": ModalityConfig(delta_indices=list(range(ACTION_HORIZON)), modality_keys=["action"]),
                "video": ModalityConfig(delta_indices=[0], modality_keys=["image"]),
            },
        }

        processor = Gr00tN1d6Processor(
            modality_configs=modality_configs,
            statistics=statistics,
            max_state_dim=STATE_DIM,
            max_action_dim=ACTION_DIM,
            max_action_horizon=ACTION_HORIZON,
            use_relative_action=False,
            formalize_language=False,
            embodiment_id_mapping={"embodiment_a": 0, "embodiment_b": 1},
        )

        step = Gr00tN1d6ProcessStep(processor=processor)
        state = step.state_dict()

        # Verify keys from both embodiments are present
        assert "embodiment_a.state.state.min" in state
        assert "embodiment_a.state.state.max" in state
        assert "embodiment_b.action.action.min" in state
        assert "embodiment_b.action.action.max" in state

    def test_load_state_dict_updates_norm_params(self):
        """load_state_dict() should trigger _compute_normalization_parameters."""
        original_statistics = create_test_statistics()
        processor_with_stats = create_test_processor(statistics=original_statistics)
        step_with_stats = Gr00tN1d6ProcessStep(processor=processor_with_stats)
        saved_state = step_with_stats.state_dict()

        # Create new processor without statistics
        processor_without_stats = create_test_processor(statistics=None)
        step_without_stats = Gr00tN1d6ProcessStep(processor=processor_without_stats)

        # Verify no norm_params initially
        assert not processor_without_stats.state_action_processor.norm_params

        # Load state
        step_without_stats.load_state_dict(saved_state)

        # Verify norm_params were computed
        norm_params = processor_without_stats.state_action_processor.norm_params
        assert EMBODIMENT_TAG in norm_params
        assert "state" in norm_params[EMBODIMENT_TAG]
        assert "action" in norm_params[EMBODIMENT_TAG]

        # Verify specific norm param values (e.g., min, scale)
        state_params = norm_params[EMBODIMENT_TAG]["state"]["state"]
        assert "min" in state_params or "offset" in state_params  # depends on implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


from __future__ import annotations

import pytest
import torch

from lerobot.policies.safediff_vla.libero_contracts import (
    LIBERO_ACTION_CONTRACT,
    LIBERO_SAFETY_ACTION_CONTRACT,
    LiberoActionContract,
    compare_dataset_contracts,
    convert_libero_action,
)


def test_action_contract_round_trip() -> None:
    action = torch.tensor([[0.4, -0.2, 0.1, 0.3, -0.4, 0.2, -1.0]])
    physical = LIBERO_ACTION_CONTRACT.command_to_physical_delta(action)
    torch.testing.assert_close(physical, torch.tensor([[0.02, -0.01, 0.005, 0.15, -0.2, 0.1, -1.0]]))
    torch.testing.assert_close(LIBERO_ACTION_CONTRACT.physical_delta_to_command(physical), action)


def test_conversion_supports_step_and_velocity_semantics() -> None:
    action = torch.ones(2, 4, 7)
    per_step = convert_libero_action(
        action, LIBERO_ACTION_CONTRACT, LIBERO_SAFETY_ACTION_CONTRACT, semantics="per_step"
    )
    velocity = convert_libero_action(
        action, LIBERO_ACTION_CONTRACT, LIBERO_SAFETY_ACTION_CONTRACT, semantics="velocity"
    )
    torch.testing.assert_close(per_step[..., :3], torch.full((2, 4, 3), 0.025))
    torch.testing.assert_close(per_step[..., 3:6], torch.full((2, 4, 3), 0.25))
    torch.testing.assert_close(velocity[..., :3], torch.full((2, 4, 3), 0.0125))
    torch.testing.assert_close(velocity[..., 3:6], torch.full((2, 4, 3), 0.125))
    torch.testing.assert_close(per_step[..., 6], action[..., 6])
    torch.testing.assert_close(velocity[..., 6], action[..., 6])


def test_conversion_validation_and_clipping() -> None:
    converted = convert_libero_action(
        torch.tensor([[100.0, 0, 0, 0, 0, 0, -1.0]]),
        LIBERO_SAFETY_ACTION_CONTRACT,
        LIBERO_ACTION_CONTRACT,
        clip=True,
    )
    assert converted.max() == 1
    with pytest.raises(ValueError, match="dimension 7"):
        LIBERO_ACTION_CONTRACT.command_to_physical_delta(torch.ones(3))
    with pytest.raises(ValueError, match="fps"):
        LiberoActionContract("bad", 0, (1, 1, 1, 1, 1, 1))


def test_comparison_distinguishes_shape_from_semantics() -> None:
    common = {
        "observation.state": {"shape": [8], "dtype": "float32"},
        "observation.image": {"shape": [3, 256, 256], "dtype": "video"},
        "observation.wrist_image": {"shape": [3, 256, 256], "dtype": "video"},
    }
    ordinary = {"fps": 10, "features": {**common, "action": {"shape": [7], "dtype": "float32"}}}
    safety = {"fps": 20, "features": {**common, "actions": {"shape": [7], "dtype": "float32"}}}
    report = compare_dataset_contracts(
        ordinary, safety, LIBERO_ACTION_CONTRACT, LIBERO_SAFETY_ACTION_CONTRACT
    )
    assert report["compatible_by_shape_only"] is True
    assert report["semantically_interchangeable"] is False
    assert report["target_to_source_controller_scale_ratio"] == {
        "translation": 40.0,
        "rotation": 4.0,
    }

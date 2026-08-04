from __future__ import annotations

import copy

import numpy as np
import pytest

from run_remote_sim_gate12 import (
    EXPECTED_READY_POSE_TOLERANCE,
    validate_object_spawn,
    validate_ready_pose_evidence,
)


READY_STATE = [
    7.4285712242126465,
    -98.32967376708984,
    45.010990142822266,
    92.21977996826172,
    1.8461538553237915,
    19.765840530395508,
]


def ready_contract() -> dict:
    return {
        "profile_id": "task1_real24_ready_pose_reset_v1",
        "state_dataset_units": list(READY_STATE),
    }


def ready_evidence() -> dict:
    return {
        "contract": ready_contract(),
        "application": {
            "requested_state_dataset_units": list(READY_STATE),
            "observed_tick0_state_dataset_units": list(READY_STATE),
            "per_joint_delta_dataset_units": [0.0] * 6,
            "absolute_tolerance_dataset_units": (
                EXPECTED_READY_POSE_TOLERANCE
            ),
            "within_tolerance": True,
            "robot_qvel_zero": True,
            "object_pose_unchanged": True,
            "simulation_time_advanced": False,
            "nexus_init_noise_overwritten": True,
            "env_step_after_override": 0,
            "last_step_result_cleared": True,
        },
    }


def test_ready_pose_evidence_accepts_exact_tick0() -> None:
    result = validate_ready_pose_evidence(
        ready_evidence(),
        np.asarray(READY_STATE, dtype=np.float32),
        ready_contract(),
    )
    assert result["ready_pose_tick0_valid"] is True
    assert result["maximum_absolute_tick0_delta"] == 0.0


def test_ready_pose_evidence_rejects_out_of_tolerance_delta() -> None:
    evidence = copy.deepcopy(ready_evidence())
    evidence["application"]["observed_tick0_state_dataset_units"][0] += 0.001
    evidence["application"]["per_joint_delta_dataset_units"][0] = 0.001
    with pytest.raises(RuntimeError, match="exceeds tolerance"):
        validate_ready_pose_evidence(
            evidence,
            np.asarray(
                evidence["application"]["observed_tick0_state_dataset_units"],
                dtype=np.float32,
            ),
            ready_contract(),
        )


def test_object_spawn_requires_exact_frozen_plan_fields() -> None:
    trial = {"spawn": {"profile_id": "gate", "seed": 9000, "spawn_id": 0}}
    task = {
        "spawn": {
            **trial["spawn"],
            "actual_initial_pose": {"x": 0.1},
            "placement_method": "frozen",
        }
    }
    result = validate_object_spawn(trial, task)
    assert result["object_spawn_plan_valid"] is True
    task["spawn"]["seed"] = 9999
    with pytest.raises(RuntimeError, match="differs"):
        validate_object_spawn(trial, task)
